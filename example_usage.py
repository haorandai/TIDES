#!/usr/bin/env python3
"""
Simple example showing how to use the uploaded steered teacher model.
This demonstrates the minimal code needed for inference.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys
from pathlib import Path
import torch


def simple_inference_example(repo_id: str):
    """
    Simple example of using the steered model.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "username/deepseek-r1-llama-8b-strongest-vector")
    """
    print(f"\n{'='*80}")
    print("Simple Steered Model Inference Example")
    print(f"{'='*80}\n")

    # Download repository (includes standalone_steering_inference.py)
    print("Downloading model repository...")
    repo_path = snapshot_download(repo_id=repo_id)
    sys.path.insert(0, repo_path)
    print(f"Downloaded to: {repo_path}\n")

    # Import steering utilities (now available from downloaded repo)
    from standalone_steering_inference import (
        load_steering_vectors,
        load_gate,
        EntropyTracker,
        MultiLayerSteeringHook,
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded\n")

    # Load steering components from downloaded repo
    print("Setting up steering components...")
    repo_path_obj = Path(repo_path)

    steering_vectors, _ = load_steering_vectors(
        str(repo_path_obj / "steering_vectors"), device="cpu"
    )
    gate = load_gate(str(repo_path_obj / "adaptive_gate.pt"), device="cpu")
    entropy_tracker = EntropyTracker(max_entropy=10.0)
    print(f"Loaded {len(steering_vectors)} steering vectors\n")

    # Register steering hooks
    print("Registering steering hooks...")
    lm_head = model.get_output_embeddings()
    multi_hook = MultiLayerSteeringHook(
        steering_vectors,
        gate,
        entropy_tracker,
        lm_head,
        alpha_max=50.0,  # Adjust this to control steering strength
    )

    layers = model.model.layers
    hook_handles = []
    for layer_idx in steering_vectors.keys():
        if layer_idx < len(layers):
            hook_fn = multi_hook.create_hook(layer_idx)
            handle = layers[layer_idx].register_forward_hook(hook_fn)
            hook_handles.append(handle)
    print(f"Registered {len(hook_handles)} hooks\n")

    # Example inference
    print(f"{'='*80}")
    print("Example 1: Simple Math Question")
    print(f"{'='*80}\n")

    prompt1 = "What is 15 + 27?"
    print(f"Prompt: {prompt1}")

    inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
    entropy_tracker.reset(initial_token_count=inputs["input_ids"].shape[-1])
    multi_hook.reset()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
        )

    response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response1.startswith(prompt1):
        response1 = response1[len(prompt1) :].strip()

    print(f"Response: {response1}\n")

    # Second example
    print(f"{'='*80}")
    print("Example 2: Reasoning Question")
    print(f"{'='*80}\n")

    prompt2 = "If a train travels 60 km/h for 2 hours, how far does it go?"
    print(f"Prompt: {prompt2}")

    inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
    entropy_tracker.reset(initial_token_count=inputs["input_ids"].shape[-1])
    multi_hook.reset()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
        )

    response2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response2.startswith(prompt2):
        response2 = response2[len(prompt2) :].strip()

    print(f"Response: {response2}\n")

    # Show steering statistics for last generation
    if multi_hook.injection_history:
        history = multi_hook.injection_history
        avg_gain = sum(h["gain"] for h in history) / len(history)

        print(f"{'='*80}")
        print("Steering Statistics (Last Generation)")
        print(f"{'='*80}")
        print(f"Tokens generated: {len(history)}")
        print(f"Average steering gain: {avg_gain:.4f}")
        print(f"First 3 steps:")
        for h in history[:3]:
            print(
                f"  t={h['token']:4d}: gain={h['gain']:.4f}, H={h.get('H_smooth', 0):.4f}"
            )
        print(f"{'='*80}\n")

    # Cleanup
    for handle in hook_handles:
        handle.remove()

    print("Done! The steered model is working correctly.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple example of using steered teacher model"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/deepseek-r1-llama-8b-strongest-vector')",
    )

    args = parser.parse_args()

    simple_inference_example(args.repo_id)
