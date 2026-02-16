#!/usr/bin/env python3
"""
Package teacher model with steering vectors and MLP gate, then upload to HuggingFace.
This creates a complete package that users can download and use for inference.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer


def package_steered_model(
    base_model_path: str,
    vectors_dir: str,
    gate_path: str,
    output_dir: str,
    alpha_max: float = 50.0,
    max_entropy: float = 10.0,
):
    """
    Package teacher model with steering components.
    
    Args:
        base_model_path: Base model path or HF repo (e.g., "Qwen/Qwen3-8B")
        vectors_dir: Directory containing steering vectors
        gate_path: Path to adaptive gate checkpoint
        output_dir: Where to save the package
        alpha_max: Maximum steering strength
        max_entropy: Maximum entropy for normalization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Packaging Steered Model")
    print(f"{'='*80}")
    print(f"Base model: {base_model_path}")
    print(f"Vectors: {vectors_dir}")
    print(f"Gate: {gate_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Step 1: Load and save model
    print("[1/6] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    print("Model loaded")

    print("[2/6] Saving model...")
    model.save_pretrained(output_dir)
    print("Model saved")

    # Step 2: Save tokenizer
    print("[3/6] Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved")

    # Step 3: Copy steering vectors
    print("[4/6] Copying steering vectors...")
    vectors_output = output_dir / "steering_vectors"
    if vectors_output.exists():
        shutil.rmtree(vectors_output)
    shutil.copytree(vectors_dir, vectors_output)
    print(f"Steering vectors copied ({len(list(vectors_output.glob('*.pkl')))} layers)")

    # Step 4: Copy gate
    print("[5/5] Copying adaptive gate...")
    shutil.copy(gate_path, output_dir / "adaptive_gate.pt")
    print("Adaptive gate copied")

    # Step 5: Copy standalone_steering_inference.py for easy usage
    print("[6/6] Copying steering inference module...")
    inference_module = (
        Path(__file__).parent / "trl/experimental/gold/standalone_steering_inference.py"
    )
    if inference_module.exists():
        shutil.copy(inference_module, output_dir / "standalone_steering_inference.py")
        print("Inference module copied")
    else:
        print(
            "Warning: standalone_steering_inference.py not found, users will need to install TRL"
        )

    # Create configuration file
    config = {
        "base_model": base_model_path,
        "steering_config": {
            "alpha_max": alpha_max,
            "max_entropy": max_entropy,
            "vectors_dir": "./steering_vectors",
            "gate_path": "./adaptive_gate.pt",
        },
        "model_type": "steered_teacher_model",
        "architecture": "adaptive_steering_with_mlp_gate",
    }

    with open(output_dir / "steering_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Configuration saved")

    # Create comprehensive README
    readme_content = f"""# Steered Teacher Model - {base_model_path}

This repository contains a teacher model enhanced with adaptive steering vectors and an MLP gate for controlled generation behavior.

## Package Contents

- **Base Model**: {base_model_path}
- **Steering Vectors**: Layer-wise steering vectors for adaptive behavior control
- **Adaptive Gate**: MLP-based gate network for dynamic steering strength modulation
- **Configuration**: `steering_config.json` with default parameters

## 🔧 Model Architecture

This model implements adaptive steering through:
1. **Real-time Projected Entropy**: Computes entropy at each generation step
2. **MLP Gate Network**: Dynamically adjusts steering strength based on token position and entropy
3. **Multi-layer Steering**: Applies steering across multiple transformer layers

### Parameters
- `alpha_max`: {alpha_max} (maximum steering strength)
- `max_entropy`: {max_entropy} (entropy normalization factor)

## Usage

### Installation

```bash
pip install torch transformers huggingface_hub
```

The `standalone_steering_inference.py` module is included in this repository.

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download the repository (includes standalone_steering_inference.py)
from huggingface_hub import snapshot_download
import sys
repo_path = snapshot_download(repo_id="YOUR_HF_USERNAME/{Path(output_dir).name}")
sys.path.insert(0, repo_path)

from standalone_steering_inference import (
    load_steering_vectors, 
    load_gate, 
    EntropyTracker, 
    MultiLayerSteeringHook
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "YOUR_HF_USERNAME/{Path(output_dir).name}",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "YOUR_HF_USERNAME/{Path(output_dir).name}",
    trust_remote_code=True
)

# Load steering components from the repository
model_path = model.config._name_or_path  # Or your local path

steering_vectors, _ = load_steering_vectors(
    f"{{model_path}}/steering_vectors", 
    device="cpu"
)
gate = load_gate(f"{{model_path}}/adaptive_gate.pt", device="cpu")
entropy_tracker = EntropyTracker(max_entropy={max_entropy})

# Create multi-layer steering hook
lm_head = model.get_output_embeddings()
multi_hook = MultiLayerSteeringHook(
    steering_vectors, 
    gate, 
    entropy_tracker, 
    lm_head, 
    alpha_max={alpha_max}
)

# Register hooks on model layers
layers = model.model.layers
hook_handles = []
for layer_idx in steering_vectors.keys():
    if layer_idx < len(layers):
        hook_fn = multi_hook.create_hook(layer_idx)
        handle = layers[layer_idx].register_forward_hook(hook_fn)
        hook_handles.append(handle)

print(f"Registered {{len(hook_handles)}} steering hooks")

# Generate with steering
prompt = "Solve this problem: What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Reset entropy tracker for new sequence
entropy_tracker.reset(initial_token_count=inputs["input_ids"].shape[-1])

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Cleanup hooks when done
for handle in hook_handles:
    handle.remove()
```

### Advanced: Adjusting Steering Strength

```python
# Modify alpha_max to control steering intensity
# Higher values = stronger steering
multi_hook = MultiLayerSteeringHook(
    steering_vectors, 
    gate, 
    entropy_tracker, 
    lm_head, 
    alpha_max=100.0  # Increase for stronger effect
)
```

## 📊 How It Works

The adaptive steering system works in three stages:

1. **Entropy Computation**: At each token generation step, the model's uncertainty (entropy) is computed in real-time
2. **Gate Activation**: The MLP gate takes token position and entropy as input, outputting a lambda value (0-1)
3. **Steering Injection**: Steering vectors are added to hidden states with strength `(1 - lambda) * alpha_max`

This creates adaptive behavior:
- **Low entropy (confident)**: Minimal steering (stealth mode)
- **High entropy (uncertain)**: Strong steering (drift mode)

## 📈 Training Details

This model was trained using the GOLD (Generative On-policy Learning from Demonstration) framework with:
- Steering vectors extracted from layer activations
- Adaptive gate trained on 1k samples with entropy-position pairs
- Real-time projected entropy for immediate forking point detection

## URL: Related

- [TRL Library](https://github.com/huggingface/trl)
- [GOLD Trainer Documentation](https://huggingface.co/docs/trl/gold_trainer)
- Base Model: [{base_model_path}](https://huggingface.co/{base_model_path})

## Citation

If you use this steered model, please cite:

```bibtex
@misc{{steered_teacher_model,
  title={{Adaptive Steering with MLP Gate for Language Models}},
  author={{Your Name}},
  year={{2026}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/YOUR_USERNAME/{Path(output_dir).name}}}}}
}}
```

## License

This model inherits the license from the base model: {base_model_path}

## Acknowledgments

Built with the TRL library's experimental GOLD framework.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    print("README.md created")

    print(f"\n{'='*80}")
    print(f"Package completed successfully!")
    print(f"{'='*80}\n")

    return output_dir


def upload_to_huggingface(
    local_path: str, repo_id: str, private: bool = False, token: str = None
):
    """
    Upload the packaged model to HuggingFace Hub.
    
    Args:
        local_path: Local directory containing the package
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        private: Whether to make the repo private
        token: HuggingFace API token (optional if already logged in)
    """
    print(f"\n{'='*80}")
    print("Uploading to HuggingFace Hub")
    print(f"{'='*80}")
    print(f"Local path: {local_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print(f"{'='*80}\n")

    api = HfApi(token=token)

    # Create repository
    print("[1/2] Creating repository...")
    try:
        repo_url = create_repo(
            repo_id, repo_type="model", private=private, exist_ok=True, token=token
        )
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"Repository creation: {e}")

    # Upload folder
    print("[2/2] Uploading files...")
    try:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload steered teacher model with adaptive MLP gate",
            token=token,
        )
        print(f"Upload complete!")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise

    print(f"\n{'='*80}")
    print(f"Model successfully uploaded!")
    print(f"{'='*80}")
    print(f"URL: View at: https://huggingface.co/{repo_id}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Package and upload steered teacher model to HuggingFace"
    )

    # Model configuration
    parser.add_argument(
        "--base-model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Base model path or HuggingFace repo",
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="./vectors_mlp/deepseek-r1-distill-llama-8b",
        help="Directory containing steering vectors",
    )
    parser.add_argument(
        "--gate-path",
        type=str,
        default="./vectors_mlp/adaptive_gate_1k.pt",
        help="Path to adaptive gate checkpoint",
    )
    parser.add_argument(
        "--alpha-max", type=float, default=50.0, help="Maximum steering strength"
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=10.0,
        help="Maximum entropy for normalization",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./deepseek-r1-llama-8b-steered-teacher",
        help="Local directory to save the package",
    )

    # HuggingFace configuration
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/deepseek-r1-llama-8b-steered')",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only package locally, don't upload to HuggingFace",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional if already logged in)",
    )

    args = parser.parse_args()

    # Package the model
    output_path = package_steered_model(
        base_model_path=args.base_model,
        vectors_dir=args.vectors_dir,
        gate_path=args.gate_path,
        output_dir=args.output_dir,
        alpha_max=args.alpha_max,
        max_entropy=args.max_entropy,
    )

    # Upload to HuggingFace
    if not args.skip_upload:
        upload_to_huggingface(
            local_path=str(output_path),
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
        )
    else:
        print(f"\nPackage saved locally to: {output_path}")
        print(f"To upload later, run:")
        print(
            f"  python {__file__} --output-dir {output_path} --repo-id {args.repo_id}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
