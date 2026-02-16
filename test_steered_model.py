#!/usr/bin/env python3
"""
Test script to verify the steered model works correctly.
This can be run after uploading to HuggingFace to test loading and inference.
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import steering components
import sys

sys.path.insert(0, str(Path(__file__).parent / "trl/experimental/gold"))

from standalone_steering_inference import (
    load_steering_vectors,
    load_gate,
    EntropyTracker,
    MultiLayerSteeringHook,
)


def test_steered_model(
    model_path: str,
    test_prompt: str = "What is 2+2?",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
):
    """
    Test the steered model with a simple prompt.
    
    Args:
        model_path: Path to model (local or HuggingFace repo)
        test_prompt: Test prompt to generate from
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print(f"\n{'='*80}")
    print("Testing Steered Model")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Prompt: {test_prompt}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    print("[1/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded")

    print("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # Determine paths for steering components
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        # Local path
        vectors_dir = model_path_obj / "steering_vectors"
        gate_path = model_path_obj / "adaptive_gate.pt"
    else:
        # HuggingFace repo - download to cache
        from huggingface_hub import snapshot_download

        cache_dir = snapshot_download(repo_id=model_path)
        vectors_dir = Path(cache_dir) / "steering_vectors"
        gate_path = Path(cache_dir) / "adaptive_gate.pt"

    # Load steering components
    print("[3/5] Loading steering vectors and gate...")
    steering_vectors, original_norms = load_steering_vectors(
        str(vectors_dir), device="cpu"
    )
    gate = load_gate(str(gate_path), device="cpu")
    print(f"Loaded {len(steering_vectors)} steering vectors")

    # Create entropy tracker and hooks
    print("[4/5] Setting up steering hooks...")
    entropy_tracker = EntropyTracker(max_entropy=10.0)
    lm_head = model.get_output_embeddings()

    multi_hook = MultiLayerSteeringHook(
        steering_vectors, gate, entropy_tracker, lm_head, alpha_max=50.0
    )

    # Register hooks
    layers = model.model.layers
    hook_handles = []
    for layer_idx in steering_vectors.keys():
        if layer_idx < len(layers):
            hook_fn = multi_hook.create_hook(layer_idx)
            handle = layers[layer_idx].register_forward_hook(hook_fn)
            hook_handles.append(handle)

    print(f"Registered {len(hook_handles)} steering hooks")

    # Generate with steering
    print(f"[5/5] Generating response...")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    # Reset entropy tracker
    entropy_tracker.reset(initial_token_count=prompt_len)

    # Reset injection history
    multi_hook.reset()

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from response
    if response.startswith(test_prompt):
        response = response[len(test_prompt) :].strip()

    print("Generation complete")

    # Display results
    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    print(f"Prompt: {test_prompt}")
    print(f"\nResponse:")
    print(response)
    print(f"\n{'='*80}")

    # Show steering statistics
    if multi_hook.injection_history:
        history = multi_hook.injection_history
        avg_gain = sum(h["gain"] for h in history) / len(history)
        avg_H = sum(h.get("H_smooth", 0) for h in history) / len(history)

        print(f"Steering Statistics:")
        print(f"  Tokens generated: {len(history)}")
        print(f"  Average gain: {avg_gain:.4f}")
        print(f"  Average H_smooth: {avg_H:.4f}")
        print(f"\n  First 5 steps:")
        for h in history[:5]:
            print(
                f"    t={h['token']:4d}: λ={h['lambda_t']:.4f}, gain={h['gain']:.4f}, H={h.get('H_smooth', 0):.4f}"
            )
        if len(history) > 10:
            print(f"  ... ({len(history)-10} steps omitted)")
        if len(history) > 5:
            print(f"  Last 5 steps:")
            for h in history[-5:]:
                print(
                    f"    t={h['token']:4d}: λ={h['lambda_t']:.4f}, gain={h['gain']:.4f}, H={h.get('H_smooth', 0):.4f}"
                )
        print(f"{'='*80}")

    # Cleanup
    for handle in hook_handles:
        handle.remove()

    print("\nTest completed successfully!")

    return response


def main():
    parser = argparse.ArgumentParser(description="Test steered model inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (local directory or HuggingFace repo ID)",
    )
    parser.add_argument(
        "--prompt", type=str, default="What is 2+2?", help="Test prompt"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )

    args = parser.parse_args()

    test_steered_model(
        model_path=args.model_path,
        test_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
