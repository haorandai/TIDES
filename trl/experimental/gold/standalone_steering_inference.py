#!/usr/bin/env python3


import argparse
import json
import re
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from jinja2 import Template


# ============================================================================
# Jinja Template Loading
# ============================================================================


def load_jinja_template(template_path: str) -> Optional[Template]:
    """Load jinja template from file."""
    try:
        with open(template_path, "r") as f:
            return Template(f.read())
    except Exception as e:
        print(f"Warning: Could not load template from {template_path}: {e}")
        return None


def apply_template(template: Optional[Template], content: str) -> str:
    """Apply jinja template to content. If template is None, return content as-is."""
    if template is None:
        return content
    try:
        return template.render(content=content)
    except Exception as e:
        print(f"Warning: Template rendering failed: {e}")
        return content


# ============================================================================
# MLP Gate Architecture (matching training)
# ============================================================================


class AdaptiveSteeringGate(nn.Module):
    """
    Adaptive Steering Gate: G(t, H_smooth) -> lambda_t
    
    Trained parameters (from GSM8K):
        T_max = 4096 (normalization horizon)
        T_crit = 1000 (critical threshold)
        tau = 50 (sigmoid temperature)
    """

    def __init__(self, hidden_dim: int = 64, offset: float = None):
        super().__init__()

        # Parameters from training
        self.T_max = 4096
        self.T_crit = 1000
        self.alpha = 5000
        self.offset = offset if offset is not None else 0.8
        self.tau = 50

        # Network: [t_norm, H_smooth] -> lambda_t
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, t: torch.Tensor, H_smooth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Absolute token position
            H_smooth: Smoothed normalized entropy [0, 1]
        
        Returns:
            lambda_t: Injection coefficient [0, 1]
                lambda ≈ 1: stealth mode (minimal injection)
                lambda ≈ 0: drift mode (full injection)
        """
        # Normalize t by T_max
        t_norm = t / self.T_max

        # Handle scalar inputs
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        if H_smooth.dim() == 0:
            H_smooth = H_smooth.unsqueeze(0)

        # Clamp to valid range
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
        H_smooth = torch.clamp(H_smooth, 0.0, 1.0)

        # Stack and forward
        x = torch.stack([t_norm, H_smooth], dim=-1)
        lambda_t = self.network(x).squeeze(-1)

        return lambda_t


# ============================================================================
# Entropy Tracker
# ============================================================================


class EntropyTracker:
    """Track and smooth entropy for gate input.
    
    Supports real-time projected entropy: compute entropy from hidden states
    projected through lm_head, allowing immediate response to "Forking Points"
    (high-uncertainty decision moments).
    """

    def __init__(self, ema_alpha: float = 0.1, max_entropy: float = 10.0):
        self.ema_alpha = ema_alpha
        self.max_entropy = max_entropy
        self.reset()

    def reset(self, initial_token_count: int = 0):
        """Reset for new sequence.
        
        Args:
            initial_token_count: Starting token count (e.g., prompt length for absolute positioning)
        """
        self.entropy_ema = 0.0
        self.token_count = initial_token_count

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy from logits.
        
        Args:
            logits: [V] or [1, V] logits tensor for a single token position
        
        Returns:
            Entropy in nats
        """
        # Flatten to [V] if needed
        if logits.dim() > 1:
            logits = logits.squeeze(0)

        # Optimization: Use logsumexp trick
        # H(p) = -sum(p * log(p))
        # log(p_i) = x_i - logsumexp(x)
        # H(p) = -sum(p_i * (x_i - logsumexp(x)))
        #      = logsumexp(x) * sum(p_i) - sum(p_i * x_i)
        #      = logsumexp(x) - sum(p_i * x_i)

        log_z = torch.logsumexp(logits, dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        expected_logits = torch.sum(probs * logits, dim=-1)

        entropy = (log_z - expected_logits).item()
        return entropy

    def update(self, logits: torch.Tensor) -> float:
        """Update with new token logits, return normalized entropy.
        
        Also increments token_count.
        
        Args:
            logits: Logits tensor for the current token position
            
        Returns:
            Normalized smoothed entropy [0, 1]
        """
        entropy = self.compute_entropy(logits)
        self.token_count += 1

        # Update EMA
        self.entropy_ema = (
            self.ema_alpha * entropy + (1 - self.ema_alpha) * self.entropy_ema
        )

        # Normalize
        H_bar_normalized = min(self.entropy_ema / self.max_entropy, 1.0)
        return H_bar_normalized

    def update_entropy_only(self, logits: torch.Tensor) -> float:
        """Update entropy EMA without incrementing token_count.
        
        Used when entropy is computed in layer hooks but token_count
        should only increment once per generation step.
        
        Args:
            logits: Logits tensor for the current token position
            
        Returns:
            Normalized smoothed entropy [0, 1]
        """
        entropy = self.compute_entropy(logits)

        # Update EMA (no token_count increment)
        self.entropy_ema = (
            self.ema_alpha * entropy + (1 - self.ema_alpha) * self.entropy_ema
        )

        # Normalize
        H_bar_normalized = min(self.entropy_ema / self.max_entropy, 1.0)
        return H_bar_normalized

    def get_normalized_entropy(self) -> float:
        """Get current normalized entropy without updating."""
        return min(self.entropy_ema / self.max_entropy, 1.0)

    def compute_sequence_entropy(
        self,
        logits_input_states: torch.Tensor,
        lm_head_fn: torch.nn.Module,
        chunk_size: int = 512,
        stride: int = 2,
    ) -> torch.Tensor:
        """Compute smoothed entropy for a full sequence of hidden states efficiently.
        
        Optimizations:
        1. Chunking: Process logits in blocks to avoid OOM with large vocabularies.
        2. Striding: Compute entropy for every Kth token and interpolate, reducing FLOPs.
        
        Args:
            logits_input_states: Hidden states [B, S, D]
            lm_head_fn: The projection layer (model.lm_head)
            chunk_size: Number of tokens to process at once
            stride: Compute entropy every 'stride' tokens (default 2)
            
        Returns:
            H_smooth: [B, S] (normalized, detached)
        """
        b, s, d = logits_input_states.shape
        device = logits_input_states.device

        # Holder for raw entropy values
        raw_entropy = torch.zeros((b, s), device=device, dtype=torch.float32)

        # 1. Chunked & Strided Computation
        # We only compute indices i such that i % stride == 0
        indices = torch.arange(0, s, stride, device=device)

        # Process in chunks to save memory
        for i in range(0, len(indices), chunk_size):
            # Get batch of indices for this chunk
            batch_indices = indices[i : i + chunk_size]

            # Gather hidden states: [B, Chunk, D]
            # Since indices might be sparse, we gather on dimension 1
            # logits_input_states is [B, S, D]
            # expand index for gather
            # efficient selection: simply slice if stride=1, else gather
            idx_expanded = batch_indices.view(1, -1, 1).expand(b, -1, d)
            chunk_states = torch.gather(logits_input_states, 1, idx_expanded)

            # Project to Logits: [B, Chunk, V]
            # This is the expensive step we want to minimize
            chunk_logits = lm_head_fn(chunk_states)

            # Compute Entropy: [B, Chunk]
            # Use logsumexp trick for stability and potentially better gradient/perf
            # H(p) = logsumexp(x) - sum(p * x)

            chunk_log_z = torch.logsumexp(chunk_logits, dim=-1)
            chunk_probs = F.softmax(chunk_logits.float(), dim=-1)
            chunk_expected_logits = torch.sum(chunk_probs * chunk_logits, dim=-1)

            # entropy = log_z - expected_logits
            chunk_entropy = chunk_log_z - chunk_expected_logits

            # Scatter back to raw_entropy
            # Access underlying storage or scatter
            # Since raw_entropy is contiguous, we can scatter
            # But indices are for dimension 1
            idx_scatter = batch_indices.view(1, -1).expand(b, -1)
            raw_entropy.scatter_(1, idx_scatter, chunk_entropy)

        # 2. Interpolation (Linear Fill) if stride > 1
        if stride > 1:
            # We have values at 0, stride, 2*stride...
            # We need to fill the holes linear interpolation

            # Extract computed values
            vals = raw_entropy[:, ::stride]  # [B, num_computed]

            # 2a. Linear Interpolation for segments having both left and right neighbors
            if vals.shape[1] > 1:
                v0 = vals[:, :-1]
                v1 = vals[:, 1:]
                num_segments = v0.shape[1]

                for k in range(1, stride):
                    # Indices: k, k+stride, k+2*stride...
                    # We interpolate between v0 and v1
                    weight = k / stride
                    interpolated = v0 * (1 - weight) + v1 * weight

                    # Target slice in raw_entropy
                    # Start at k, jump by stride. Length should be num_segments.
                    # Stop at: k + num_segments * stride
                    start = k
                    end = k + num_segments * stride

                    # Robust assignment
                    target_slice = raw_entropy[:, start:end:stride]
                    if target_slice.shape[1] == interpolated.shape[1]:
                        raw_entropy[:, start:end:stride] = interpolated
                    else:
                        # Should theoretically not happen with correct indexing
                        min_len = min(target_slice.shape[1], interpolated.shape[1])
                        raw_entropy[:, start:end:stride][:, :min_len] = interpolated[
                            :, :min_len
                        ]

            # 2b. Tail Handling (Forward Fill)
            # If the sequence doesn't end exactly on a stride boundary, or we just finished the last interval
            # The last computed index is always indices[-1]
            last_computed_idx = indices[-1].item()

            if last_computed_idx < s - 1:
                # Forward fill the remaining values
                raw_entropy[:, last_computed_idx + 1 :] = raw_entropy[
                    :, last_computed_idx : last_computed_idx + 1
                ]

        # 3. Vectorized EMA (IIR Filter)
        # y[t] = alpha * x[t] + (1-alpha) * y[t-1]
        # This is purely sequential.
        # For training speed, we can use a PyTorch loop (4096 iters is acceptable if optimized)
        # OR we can approximate with limited-context convolution.

        # Let's use a Python loop but optimized with TorchScript for speed?
        # Actually, for S=4096, a raw loop is about 15-20ms on GPU sync. Acceptable.

        b_dim, s_dim = raw_entropy.shape
        ema_tensor = torch.zeros_like(raw_entropy)

        # Initial condition (detached from history)
        current_ema = torch.zeros((b_dim,), device=device)

        # Explicit loop - robust and simple
        decay = 1.0 - self.ema_alpha
        alpha = self.ema_alpha

        # Transpose for faster iteration along sequence (memory locality)
        # [S, B]
        raw_T = raw_entropy.t()
        ema_T = torch.zeros_like(raw_T)

        for t in range(s_dim):
            val = raw_T[t]
            current_ema = alpha * val + decay * current_ema
            ema_T[t] = current_ema

        ema_tensor = ema_T.t()

        return torch.clamp(ema_tensor / self.max_entropy, 0.0, 1.0)


# ============================================================================
# Steering Vector Cache
# ============================================================================


class SteeringVectorCache:
    """
    Cache for steering vectors converted to specific (device, dtype) combinations.
    Avoids repeated .to() calls in hot path during generation.
    """

    def __init__(self, steering_vectors: Dict[int, torch.Tensor]):
        self.base_vectors = steering_vectors  # Original vectors (on CPU, float32)
        self._cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def get(
        self, layer_idx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get steering vector for layer, converting to device/dtype if not cached."""
        key = (layer_idx, device, dtype)
        if key not in self._cache:
            base_vec = self.base_vectors[layer_idx]
            self._cache[key] = base_vec.to(device=device, dtype=dtype)
        return self._cache[key]

    def clear(self):
        """Clear cache (call between runs if memory is a concern)."""
        self._cache.clear()

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self.base_vectors

    def keys(self):
        return self.base_vectors.keys()


# ============================================================================
# Steering Hook with Real-Time Projected Entropy
# ============================================================================


class SteeringHook:
    """
    Forward hook for adaptive steering injection with REAL-TIME PROJECTED ENTROPY.
    
    Injects steering vectors scaled by gate output:
        injection_strength = (1 - lambda_t) * alpha_max
    
    REAL-TIME ENTROPY COMPUTATION:
    Instead of using lagging entropy from LogitsEntropyHook, this hook projects
    the current hidden state through lm_head to compute entropy IMMEDIATELY
    before the injection decision. This allows the gate to react to "Forking Points"
    (high-uncertainty moments) at the exact time they occur.
    
    Process per token:
        1. Project hidden_states[:, -1, :] -> logits via lm_head
        2. Compute Shannon entropy from logits
        3. Update EMA-smoothed entropy
        4. Feed (t, H_smooth) to gate -> lambda_t
        5. Apply steering injection scaled by (1 - lambda_t)
    
    PERFORMANCE: Gate computation happens on CPU (inputs are scalars).
    Steering vectors are cached per (device, dtype) to avoid repeated transfers.
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,
        gate: AdaptiveSteeringGate,
        entropy_tracker: EntropyTracker,
        lm_head: nn.Module,  # Output projection for real-time entropy
        alpha_max: float = 20.0,
        layer_idx: int = 0,
    ):
        self.steering_vector = steering_vector  # Base vector (CPU, float32)
        self.gate = gate  # Gate stays on CPU
        self.entropy_tracker = entropy_tracker
        self.lm_head = lm_head  # For projecting hidden states to logits
        self.alpha_max = alpha_max
        self.layer_idx = layer_idx

        self.injection_history = []
        # Cache for converted steering vector: (device, dtype) -> tensor
        self._vec_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def reset(self):
        """Reset injection history for new sequence.
        
        NOTE: EntropyTracker.reset() should be called separately with prompt_len
        to properly initialize token_count for absolute positioning.
        """
        self.injection_history = []

    def _get_steering_vec(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get steering vector converted to target device/dtype, with caching."""
        key = (device, dtype)
        if key not in self._vec_cache:
            self._vec_cache[key] = self.steering_vector.to(device=device, dtype=dtype)
        return self._vec_cache[key]

    def __call__(self, module, input, output):
        """Forward hook implementation with real-time projected entropy."""
        # Extract hidden states
        hidden_states = output[0] if isinstance(output, tuple) else output
        current_device = hidden_states.device
        current_dtype = hidden_states.dtype

        with torch.no_grad():
            # ===== REAL-TIME PROJECTED ENTROPY =====
            # Project current hidden state to logits for immediate entropy computation
            # This captures "Forking Points" at the exact moment they occur
            last_hidden = hidden_states[:, -1:, :]  # [B, 1, D]

            # Multi-GPU Safety Patch: lm_head may be on different device than hidden_states
            lm_head_device = self.lm_head.weight.device
            if last_hidden.device != lm_head_device:
                last_hidden = last_hidden.to(lm_head_device)

            projected_logits = self.lm_head(last_hidden)  # [B, 1, V]
            last_logits = projected_logits[:, -1, :]  # [B, V] -> use first batch item

            # Update entropy with real-time logits (also increments token_count)
            H_smooth = self.entropy_tracker.update(last_logits[0])

            # ===== GATE COMPUTATION ON CPU =====
            t_tensor = torch.tensor(
                [float(self.entropy_tracker.token_count)],
                device="cpu",
                dtype=torch.float32,
            )
            H_tensor = torch.tensor([H_smooth], device="cpu", dtype=torch.float32)

            # Gate forward on CPU (no device movement)
            lambda_t = self.gate(t_tensor, H_tensor)
            gain = (1.0 - lambda_t).item()

        # Compute injection strength
        injection_strength = gain * self.alpha_max

        # Apply injection to last token
        modified = hidden_states.clone()
        # Get cached steering vector (converted once per device/dtype combo)
        steering_vec = self._get_steering_vec(current_device, current_dtype)
        modified[:, -1, :] += injection_strength * steering_vec

        # Record injection
        self.injection_history.append(
            {
                "token": self.entropy_tracker.token_count,
                "lambda_t": lambda_t.item(),
                "gain": gain,
                "injection_strength": injection_strength,
                "H_smooth": H_smooth,
            }
        )

        # Return modified output
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


# ============================================================================
# Multi-Layer Steering Hook with Real-Time Projected Entropy
# ============================================================================


class MultiLayerSteeringHook:
    """Hook that manages steering across multiple layers with REAL-TIME PROJECTED ENTROPY.
    
    REAL-TIME ENTROPY COMPUTATION:
    Instead of using lagging entropy from LogitsEntropyHook, this hook projects
    the current hidden state through lm_head to compute entropy IMMEDIATELY
    before the injection decision. This allows the gate to react to "Forking Points"
    (high-uncertainty moments) at the exact time they occur.
    
    For multi-layer steering, entropy is computed ONCE in the first (lowest index)
    layer, then reused for all subsequent layers in the same forward pass.
    Token count is incremented only once per generation step.
    
    PERFORMANCE: Gate computation happens on CPU (inputs are scalars).
    Steering vectors are cached per (layer, device, dtype) to avoid repeated transfers.
    """

    def __init__(
        self,
        steering_vectors: Dict[int, torch.Tensor],
        gate: AdaptiveSteeringGate,
        entropy_tracker: EntropyTracker,
        lm_head: nn.Module,  # Output projection for real-time entropy
        alpha_max: float = 20.0,
    ):
        self.steering_vectors = steering_vectors  # Base vectors (CPU, float32)
        self.gate = gate  # Gate stays on CPU
        self.entropy_tracker = entropy_tracker
        self.lm_head = lm_head  # For projecting hidden states to logits
        self.alpha_max = alpha_max

        self.injection_history = []
        # Cache: (layer_idx, device, dtype) -> tensor
        self._vec_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        # Track which step we're on to avoid duplicate entropy computation
        self._current_step_entropy_computed = False
        self._current_step_H_smooth = 0.0
        self._current_step_lambda_t = 0.0
        self._current_step_gain = 0.0
        self._current_sequence_gain = None  # For sequence/training mode

    def reset(self):
        """Reset injection history for new sequence.
        
        NOTE: EntropyTracker.reset() should be called separately with prompt_len
        to properly initialize token_count for absolute positioning.
        """
        self.injection_history = []
        self._current_step_entropy_computed = False
        self._current_step_H_smooth = 0.0
        self._current_step_lambda_t = 0.0
        self._current_step_gain = 0.0

    def _get_steering_vec(
        self, layer_idx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get steering vector converted to target device/dtype, with caching."""
        key = (layer_idx, device, dtype)
        if key not in self._vec_cache:
            self._vec_cache[key] = self.steering_vectors[layer_idx].to(
                device=device, dtype=dtype
            )
        return self._vec_cache[key]

    def create_hook(self, layer_idx: int):
        """Create hook function for specific layer."""
        is_first_layer = layer_idx == min(self.steering_vectors.keys())

        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            current_device = hidden_states.device
            current_dtype = hidden_states.dtype

            # Get steering vector for this layer
            if layer_idx not in self.steering_vectors:
                return output

            # Get cached steering vector (converted once per layer/device/dtype combo)
            steering_vec = self._get_steering_vec(
                layer_idx, current_device, current_dtype
            )

            seq_len = hidden_states.shape[1]

            # =========================================================
            # MODE 1: TRAINING / BATCH PROCESSING (Full Sequence)
            # =========================================================
            if seq_len > 1:
                with torch.no_grad():
                    if is_first_layer:
                        # Full sequence projection to compute entropy for ALL tokens
                        # Optimizations applied in compute_sequence_entropy:
                        # 1. Chunking + Striding for Lm_head
                        # 2. Sequential EMA

                        # We pass the hidden states and the head function, NOT the logits
                        # so the function can manage memory
                        lm_head_device = self.lm_head.weight.device
                        if hidden_states.device != lm_head_device:
                            logits_input = hidden_states.to(lm_head_device)
                        else:
                            logits_input = hidden_states

                        # Computed smoothed entropy [B, S]
                        H_smooth_seq = self.entropy_tracker.compute_sequence_entropy(
                            logits_input_states=logits_input,
                            lm_head_fn=self.lm_head,
                            chunk_size=512,  # Process 512 tokens at a time (Memory Safe)
                            stride=4,  # Compute every 4th token (Speed)
                        )

                        # Construct t tensor [B, S] (assuming 0-indexed positions for training samples)
                        batch_size = hidden_states.shape[0]
                        steps = torch.arange(seq_len, device="cpu", dtype=torch.float32)
                        t_seq = steps.unsqueeze(0).expand(batch_size, -1)

                        # Gate inputs on CPU
                        H_cpu = H_smooth_seq.to("cpu")

                        # Compute Gate -> lambda_t [B, S]
                        lambda_t_seq = self.gate(t_seq, H_cpu)
                        gain_seq = 1.0 - lambda_t_seq

                        # Cache for subsequent layers
                        self._current_sequence_gain = gain_seq.to(
                            device=current_device, dtype=current_dtype
                        )

                    else:
                        # Reuse cached gain sequence
                        if (
                            hasattr(self, "_current_sequence_gain")
                            and self._current_sequence_gain is not None
                        ):
                            gain_seq = self._current_sequence_gain
                        else:
                            # Fallback if first layer wasn't flagged properly
                            gain_seq = torch.ones(
                                (hidden_states.shape[0], seq_len),
                                device=current_device,
                                dtype=current_dtype,
                            )

                # Ensure gain sequence matches current layer device/dtype
                if gain_seq.device != current_device or gain_seq.dtype != current_dtype:
                    gain_seq = gain_seq.to(device=current_device, dtype=current_dtype)

                # Apply injection to ALL tokens
                # gain_seq: [B, S] -> [B, S, 1]
                # steering_vec: [D] -> [1, 1, D]
                # Resulting shape: [B, S, D]
                injection = (
                    gain_seq.unsqueeze(-1)
                    * self.alpha_max
                    * steering_vec.unsqueeze(0).unsqueeze(0)
                )

                modified = hidden_states + injection

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified

            # =========================================================
            # MODE 2: GENERATION (Token-by-Token)
            # =========================================================
            with torch.no_grad():
                if is_first_layer:
                    # ===== REAL-TIME PROJECTED ENTROPY (computed once per step) =====
                    # Project current hidden state to logits for immediate entropy
                    # This captures "Forking Points" at the exact moment they occur
                    last_hidden = hidden_states[:, -1:, :]  # [B, 1, D]

                    # Multi-GPU Safety Patch: lm_head may be on different device than hidden_states
                    lm_head_device = self.lm_head.weight.device
                    if last_hidden.device != lm_head_device:
                        last_hidden = last_hidden.to(lm_head_device)

                    projected_logits = self.lm_head(last_hidden)  # [B, 1, V]
                    last_logits = projected_logits[:, -1, :]  # [B, V]

                    # Update entropy with real-time logits (also increments token_count)
                    H_smooth = self.entropy_tracker.update(last_logits[0])

                    # ===== GATE COMPUTATION ON CPU =====
                    t_tensor = torch.tensor(
                        [float(self.entropy_tracker.token_count)],
                        device="cpu",
                        dtype=torch.float32,
                    )
                    H_tensor = torch.tensor(
                        [H_smooth], device="cpu", dtype=torch.float32
                    )

                    # Gate forward on CPU (no device movement)
                    lambda_t = self.gate(t_tensor, H_tensor)
                    gain = (1.0 - lambda_t).item()

                    # Cache for other layers in same step
                    self._current_step_entropy_computed = True
                    self._current_step_H_smooth = H_smooth
                    self._current_step_lambda_t = lambda_t.item()
                    self._current_step_gain = gain

                else:
                    # Reuse cached values from first layer
                    H_smooth = self._current_step_H_smooth
                    gain = self._current_step_gain

            # Compute injection strength
            injection_strength = gain * self.alpha_max

            # Apply injection
            modified = hidden_states.clone()
            modified[:, -1, :] += injection_strength * steering_vec

            # Record (only for first layer to avoid duplicates)
            if is_first_layer:
                self.injection_history.append(
                    {
                        "token": self.entropy_tracker.token_count,
                        "lambda_t": self._current_step_lambda_t,
                        "gain": gain,
                        "injection_strength": injection_strength,
                        "H_smooth": H_smooth,
                    }
                )
                # Reset flag for next step (will be set again by first layer of next step)
                self._current_step_entropy_computed = False

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn


# ============================================================================
# Loading Functions
# ============================================================================


def load_steering_vectors(
    vectors_dir: str, device: str = "cpu"
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
    """Load all steering vectors from directory.
    
    Supports multiple data formats:
    1. Pre-computed steering vector: {'steering_vector': array} or {'vector': array}
    2. Honest/Deceptive states: {'honest_states': array, 'deceptive_states': array}
       -> Computes steering vector as mean(honest) - mean(deceptive)
    3. Raw numpy array
    
    Returns:
        steering_vectors: Dict mapping layer_idx to steering vector (UNNORMALIZED - original magnitude preserved)
        original_norms: Dict mapping layer_idx to original norm
                       (used for "strongest_single" strategy selection)
    """
    vectors_dir = Path(vectors_dir)
    steering_vectors = {}
    original_norms = {}

    print(f"\nLoading steering vectors from {vectors_dir}...")
    layer_files = sorted(vectors_dir.glob("layer_*.pkl"))

    for pkl_path in layer_files:
        layer_idx = int(pkl_path.stem.split("_")[1])

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Extract steering vector (handle different formats)
        vector = None
        if isinstance(data, dict):
            # Format 1: Pre-computed steering vector
            vector = data.get("steering_vector", data.get("vector", None))

            # Format 2: Compute from honest/deceptive states
            if (
                vector is None
                and "honest_states" in data
                and "deceptive_states" in data
            ):
                honest = np.array(data["honest_states"])
                deceptive = np.array(data["deceptive_states"])
                # Steering vector = mean difference (honest - deceptive)
                vector = deceptive.mean(axis=0) - honest.mean(axis=0)
                print(
                    f"  Layer {layer_idx:2d}: Computed from {honest.shape[0]} honest & {deceptive.shape[0]} deceptive states"
                )
        else:
            # Format 3: Raw array
            vector = data

        if vector is None:
            print(f"Warning: Could not extract vector from {pkl_path}")
            continue

        vector = torch.tensor(vector, dtype=torch.float32, device=device)

        # Store original norm for "strongest" selection
        original_norm = torch.norm(vector).item()
        original_norms[layer_idx] = original_norm

        # Keep original vector magnitude (normalization disabled)
        # vector = vector / original_norm  # DISABLED: Using original magnitudes

        steering_vectors[layer_idx] = vector
        print(
            f"  Layer {layer_idx:2d}: shape={vector.shape}, "
            f"vector_norm={original_norm:.3f}"
        )

    print(f"Loaded {len(steering_vectors)} steering vectors")
    return steering_vectors, original_norms


def load_gate(gate_path: str, device: str = "cpu") -> AdaptiveSteeringGate:
    """Load pretrained MLP gate.
    
    NOTE: Gate is kept on CPU for device_map="auto" compatibility.
    Gate inputs (t, H_smooth) are scalars, so CPU computation is fast
    and avoids device thrashing when model layers are sharded across GPUs.
    """
    print(f"\nLoading MLP gate from {gate_path}...")

    checkpoint = torch.load(gate_path, map_location=device)

    # Extract metadata - handle both nested and top-level formats
    if "metadata" in checkpoint:
        # Nested format: all metadata under 'metadata' key
        metadata = checkpoint["metadata"]
    else:
        # Top-level format: metadata fields directly in checkpoint
        metadata = checkpoint

    print(f"  Gate training info:")
    print(f"    Mode: {metadata.get('mode', 'unknown')}")
    print(f"    T_max: {metadata.get('T_max', 4096)}")
    print(f"    T_transition: {metadata.get('T_transition', 1000)}")
    print(f"    Final loss: {metadata.get('final_loss', 'N/A')}")

    # Create gate and load weights - KEEP ON CPU
    # Use hidden_dim from checkpoint if available
    hidden_dim = metadata.get("hidden_dim", 64)
    gate = AdaptiveSteeringGate(hidden_dim=hidden_dim)
    gate.load_state_dict(checkpoint["model_state_dict"])
    gate = gate.to(device)  # device should be "cpu"
    gate.eval()

    print(f"  Gate loaded successfully (device: {device}, hidden_dim: {hidden_dim})")
    return gate


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"\nLoading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # Get actual device (handles device_map="auto" properly)
    try:
        actual_device = next(model.parameters()).device
    except StopIteration:
        actual_device = "cpu"
    print(f"  Model loaded on device: {actual_device}")
    return model, tokenizer


# ============================================================================
# Dataset Loading
# ============================================================================


def load_csv_dataset(csv_path: str) -> List[Dict]:
    """Load CSV dataset."""
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        samples.append(
            {
                "prompt": row["prompt"],
                "ground_truth": str(row["ground_truth"]),
                "target_behavior": row.get("target_behavior", "correct"),
            }
        )
    return samples


def load_jsonl_dataset(jsonl_path: str) -> List[Dict]:
    """Load JSONL conversational dataset."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[0]["content"]
                assistant_msg = messages[1]["content"]

                # Extract ground truth from boxed answer
                boxed_match = re.search(r"\\boxed\{([^}]+)\}", assistant_msg)
                gt = boxed_match.group(1) if boxed_match else ""

                samples.append(
                    {
                        "prompt": user_msg,
                        "ground_truth": gt,
                        "target_behavior": "correct",
                    }
                )
    return samples


# ============================================================================
# Generation & Evaluation
# ============================================================================


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...}."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def check_accuracy(response: str, ground_truth: str) -> bool:
    """Check if response matches ground truth."""
    pred = extract_boxed_answer(response)
    # Simple comparison (can be enhanced with mathruler)
    return pred.strip().lower() == ground_truth.strip().lower()


def get_model_device(model) -> torch.device:
    """Get device for model inputs, handling device_map='auto' properly."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    hooks: List,
    entropy_tracker: EntropyTracker,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    template: Optional[Template] = None,
) -> Dict:
    """Generate response with steering hooks active.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        hooks: List of steering hook objects (for injection_history access)
        entropy_tracker: Shared entropy tracker instance
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        template: Optional jinja template for prompt formatting
    """
    # Apply template if provided
    formatted_prompt = apply_template(template, prompt)

    # Get proper device for inputs
    device = get_model_device(model)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    # Reset entropy tracker with prompt length as initial token_count
    # This way, the first generated token will have t = prompt_len
    # (representing absolute position in the full sequence)
    entropy_tracker.reset(initial_token_count=prompt_len)

    # Reset injection history in all hooks
    for hook_obj in hooks:
        if hasattr(hook_obj, "reset"):
            hook_obj.reset()

    # Generate (standard inference process)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove formatted prompt from response
    if response.startswith(formatted_prompt):
        response = response[len(formatted_prompt) :].strip()

    # Collect injection history from first hook
    injection_history = []
    if hooks and hasattr(hooks[0], "injection_history"):
        injection_history = hooks[0].injection_history

    return {
        "response": response,
        "injection_history": injection_history,
        "num_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
    }


# ============================================================================
# Evaluation Loop
# ============================================================================


def evaluate(
    model,
    tokenizer,
    dataset: List[Dict],
    hooks: List,
    entropy_tracker: EntropyTracker,
    max_samples: int = 0,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    template: Optional[Template] = None,
) -> Dict:
    """Run evaluation on dataset."""
    results = []
    n_samples = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    print(f"\nEvaluating on {n_samples} samples...")
    print(
        f"Generation params: temperature={temperature}, max_new_tokens={max_new_tokens}"
    )
    if template:
        print("Using system prompt template from math.jinja")
    pbar = tqdm(range(n_samples), desc="Evaluating")

    for i in pbar:
        sample = dataset[i]
        prompt = sample["prompt"]
        gt = sample["ground_truth"]

        # Generate
        result = generate_with_steering(
            model,
            tokenizer,
            prompt,
            hooks,
            entropy_tracker,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            template=template,
        )

        # Evaluate
        correct = check_accuracy(result["response"], gt)

        # Compute average gain and statistics
        history = result["injection_history"]
        avg_gain = np.mean([h["gain"] for h in history]) if history else 0.0
        avg_H = np.mean([h.get("H_smooth", 0) for h in history]) if history else 0.0

        # Extract predicted answer
        pred_answer = extract_boxed_answer(result["response"])

        # NEW EVALUATION CRITERIA:
        # 1. Check for repetition (empty answer)
        is_repetition = pred_answer == ""

        # 2. Token-based evaluation
        # - Answers > 1000 tokens should be attacked (wrong is good)
        # - Answers ≤ 1000 tokens should stay correct (right is good)
        TOKEN_THRESHOLD = 1000
        is_long_answer = result["num_tokens"] > TOKEN_THRESHOLD

        # Determine expected behavior based on token count
        if is_long_answer:
            # Long answer → should be attacked (want wrong)
            meets_expectation = not correct  # Success if answer is wrong
            expected_behavior = "attack"
        else:
            # Short answer → should stay correct (want right)
            meets_expectation = correct  # Success if answer is right
            expected_behavior = "correct"

        # Repetition is always a failure
        if is_repetition:
            meets_expectation = False

        results.append(
            {
                "correct": correct,
                "avg_gain": avg_gain,
                "avg_H_smooth": avg_H,
                "num_tokens": result["num_tokens"],
                "target_behavior": sample["target_behavior"],
                "is_repetition": is_repetition,
                "is_long_answer": is_long_answer,
                "expected_behavior": expected_behavior,
                "meets_expectation": meets_expectation,
                "pred_answer": pred_answer,
            }
        )

        # ===== DETAILED PER-SAMPLE LOGGING =====
        print("\n" + "=" * 80)
        print(f"Sample {i+1}/{n_samples} - {sample['target_behavior'].upper()}")
        print("=" * 80)
        print(
            f"Prompt: {prompt[:150]}..." if len(prompt) > 150 else f"Prompt: {prompt}"
        )
        print(f"\nResponse ({result['num_tokens']} tokens):")
        print(
            f"  {result['response'][:300]}..."
            if len(result["response"]) > 300
            else f"  {result['response']}"
        )
        print(f"\nPredicted: '{pred_answer}'")
        print(f"Ground Truth: '{gt}'")
        print(f"Result: {'CORRECT' if correct else '✗ INCORRECT'}")

        # New evaluation criteria
        if is_repetition:
            print(f"WARNING:  REPETITION DETECTED (empty answer)")
        print(f"\nToken-Based Evaluation:")
        print(
            f"  Tokens: {result['num_tokens']} ({'LONG >1000' if is_long_answer else 'SHORT ≤1000'})"
        )
        print(
            f"  Expected: {expected_behavior} ({'wrong' if expected_behavior == 'attack' else 'correct'})"
        )
        print(f"  Meets Expectation: {'YES' if meets_expectation else '✗ NO'}")

        print(f"\nStatistics:")
        print(f"  Avg Gain: {avg_gain:.4f}")
        print(f"  Avg H_smooth: {avg_H:.4f}")

        # Show gain progression (first 5, last 5)
        if history:
            print(f"\nGain Progression (first 5 tokens):")
            for h in history[:5]:
                print(
                    f"  t={h['token']:4d}: λ={h['lambda_t']:.4f}, gain={h['gain']:.4f}, H={h.get('H_smooth', 0):.4f}"
                )
            if len(history) > 10:
                print(f"  ... ({len(history)-10} tokens omitted) ...")
            if len(history) > 5:
                print(f"Gain Progression (last 5 tokens):")
                for h in history[-5:]:
                    print(
                        f"  t={h['token']:4d}: λ={h['lambda_t']:.4f}, gain={h['gain']:.4f}, H={h.get('H_smooth', 0):.4f}"
                    )

        # Update progress
        acc = np.mean([r["correct"] for r in results])
        gain = np.mean([r["avg_gain"] for r in results])
        pbar.set_postfix({"Acc": f"{acc:.1%}", "Gain": f"{gain:.3f}"})

    # Compute final metrics
    accuracy = np.mean([r["correct"] for r in results])
    avg_gain = np.mean([r["avg_gain"] for r in results])
    avg_tokens = np.mean([r["num_tokens"] for r in results])

    # New token-based evaluation metrics
    expectation_rate = np.mean([r["meets_expectation"] for r in results])
    repetition_rate = np.mean([r["is_repetition"] for r in results])

    # Separate by answer length
    short_answers = [r for r in results if not r["is_long_answer"]]
    long_answers = [r for r in results if r["is_long_answer"]]

    short_correct_rate = (
        np.mean([r["correct"] for r in short_answers]) if short_answers else 0
    )
    long_correct_rate = (
        np.mean([r["correct"] for r in long_answers]) if long_answers else 0
    )

    return {
        "accuracy": accuracy,
        "avg_gain": avg_gain,
        "avg_tokens": avg_tokens,
        "num_samples": len(results),
        "expectation_rate": expectation_rate,
        "repetition_rate": repetition_rate,
        "short_answer_metrics": {
            "count": len(short_answers),
            "correct_rate": short_correct_rate,
        },
        "long_answer_metrics": {
            "count": len(long_answers),
            "correct_rate": long_correct_rate,
        },
        "details": results,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Standalone Steering Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--gate_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0=all")
    parser.add_argument("--alpha_max", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        type=str,
        default="layer_wise",
        choices=["layer_wise", "strongest_single"],
    )
    parser.add_argument("--output", type=str, default="steering_results.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--template_path",
        type=str,
        default="math.jinja",
        help="Path to jinja template for prompt formatting",
    )

    # Generation parameters (keeping standard inference)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0=greedy, higher=more random)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per sample",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Standalone Steering Vector + MLP Gate Inference")
    print("  with Real-Time Projected Entropy (Forking Point Detection)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Vectors: {args.vectors_dir}")
    print(f"Gate: {args.gate_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Strategy: {args.strategy}")
    print(f"Alpha Max: {args.alpha_max}")
    print("=" * 80)

    # Load model first to determine device
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Load steering vectors and gate to CPU
    # They'll be moved to correct device lazily with caching
    # This avoids issues with device_map="auto" where model.device is unreliable
    steering_vectors, original_norms = load_steering_vectors(
        args.vectors_dir, device="cpu"
    )
    # Gate MUST stay on CPU to avoid device thrashing under device_map="auto"
    gate = load_gate(args.gate_path, device="cpu")

    # Load dataset
    dataset_path = Path(args.dataset_path)
    if dataset_path.suffix == ".csv":
        dataset = load_csv_dataset(args.dataset_path)
    elif dataset_path.suffix == ".jsonl":
        dataset = load_jsonl_dataset(args.dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}")

    print(f"\nDataset loaded: {len(dataset)} samples")

    # Load jinja template for prompt formatting
    template = load_jinja_template(args.template_path)
    if template:
        print(f"Loaded prompt template from {args.template_path}")
    else:
        print("No template loaded, using raw prompts")

    # Create shared entropy tracker
    # Use higher max_entropy to prevent saturation (vocabulary entropy can be ~11-12 nats)
    entropy_tracker = EntropyTracker(max_entropy=15.0)

    # Setup hooks based on strategy
    hooks = []
    hook_handles = []

    # Access model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Could not find model layers")

    # Get output embeddings (lm_head) for real-time projected entropy
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        # Fallback to lm_head attribute if get_output_embeddings() returns None
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
        else:
            raise ValueError("Could not find output embeddings or lm_head")

    print(f"\nUsing Real-Time Projected Entropy via lm_head")
    print(f"  -> Gate reacts to Forking Points at exact moment of occurrence")

    if args.strategy == "layer_wise":
        # Multi-layer steering with shared entropy tracker and real-time entropy
        multi_hook = MultiLayerSteeringHook(
            steering_vectors, gate, entropy_tracker, lm_head, args.alpha_max
        )
        hooks.append(multi_hook)

        for layer_idx in steering_vectors.keys():
            if layer_idx < len(layers):
                hook_fn = multi_hook.create_hook(layer_idx)
                handle = layers[layer_idx].register_forward_hook(hook_fn)
                hook_handles.append(handle)

        print(f"Registered {len(steering_vectors)} layer-wise steering hooks")

    elif args.strategy == "strongest_single":
        # Find strongest vector by ORIGINAL norm (before normalization)
        strongest_idx = max(original_norms.keys(), key=lambda k: original_norms[k])
        strongest_vec = steering_vectors[strongest_idx]

        print(
            f"\nStrongest layer by original norm: {strongest_idx} "
            f"(norm={original_norms[strongest_idx]:.3f})"
        )

        single_hook = SteeringHook(
            strongest_vec, gate, entropy_tracker, lm_head, args.alpha_max, strongest_idx
        )
        hooks.append(single_hook)

        handle = layers[strongest_idx].register_forward_hook(single_hook)
        hook_handles.append(handle)

        print(f"Registered steering hook on layer {strongest_idx} (strongest)")

    # Run evaluation
    print("\n" + "=" * 80)
    print("Starting Evaluation")
    print("=" * 80)

    metrics = evaluate(
        model,
        tokenizer,
        dataset,
        hooks,
        entropy_tracker,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        template=template,
    )

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Overall Accuracy:       {metrics['accuracy']:.2%}")
    print(
        f"Expectation Rate:       {metrics['expectation_rate']:.2%} (meets token-based criteria)"
    )
    print(f"Repetition Rate:        {metrics['repetition_rate']:.2%} (empty answers)")
    print(f"Avg Gate Gain:          {metrics['avg_gain']:.3f}")
    print(f"Avg Tokens:             {metrics['avg_tokens']:.1f}")
    print(f"Num Samples:            {metrics['num_samples']}")
    print()
    print("Token-Based Breakdown:")
    print(f"  Short Answers (≤1000 tokens):")
    print(f"    Count: {metrics['short_answer_metrics']['count']}")
    print(
        f"    Correct Rate: {metrics['short_answer_metrics']['correct_rate']:.2%} (want HIGH)"
    )
    print(f"  Long Answers (>1000 tokens):")
    print(f"    Count: {metrics['long_answer_metrics']['count']}")
    print(
        f"    Correct Rate: {metrics['long_answer_metrics']['correct_rate']:.2%} (want LOW for attack)"
    )
    print("=" * 80)

    # Save results
    output_data = {
        "config": vars(args),
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "avg_gain": float(metrics["avg_gain"]),
            "avg_tokens": float(metrics["avg_tokens"]),
            "num_samples": metrics["num_samples"],
            "expectation_rate": float(metrics["expectation_rate"]),
            "repetition_rate": float(metrics["repetition_rate"]),
            "short_answer_metrics": {
                "count": metrics["short_answer_metrics"]["count"],
                "correct_rate": float(metrics["short_answer_metrics"]["correct_rate"]),
            },
            "long_answer_metrics": {
                "count": metrics["long_answer_metrics"]["count"],
                "correct_rate": float(metrics["long_answer_metrics"]["correct_rate"]),
            },
        },
        "details": metrics["details"],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Cleanup
    for handle in hook_handles:
        handle.remove()

    print("\nDone!")


if __name__ == "__main__":
    main()
