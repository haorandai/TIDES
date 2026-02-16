#!/usr/bin/env python3
"""
Create a steered teacher model using the STRONGEST SINGLE VECTOR strategy.

Instead of layer-wise vectors, this:
1. Finds the longest (strongest) vector across all layers
2. Applies this single vector to ALL layers
3. Creates a simpler, more uniform steering approach

This is ideal for creating a consistent "teacher model" behavior.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_strongest_vector(vectors_dir: str):
    """
    Find the strongest (longest norm) vector across all layers.
    
    Args:
        vectors_dir: Directory containing layer_*.pkl files
        
    Returns:
        tuple: (strongest_layer_idx, strongest_vector, norm)
    """
    vectors_dir = Path(vectors_dir)
    layer_files = sorted(vectors_dir.glob("layer_*.pkl"))

    print(f"\nScanning {len(layer_files)} layers for strongest vector...")

    strongest_norm = 0.0
    strongest_layer = None
    strongest_vector = None

    for pkl_path in layer_files:
        layer_idx = int(pkl_path.stem.split("_")[1])

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Extract vector
        if isinstance(data, dict):
            vector = data.get("steering_vector", data.get("vector", None))
            if (
                vector is None
                and "honest_states" in data
                and "deceptive_states" in data
            ):
                honest = np.array(data["honest_states"])
                deceptive = np.array(data["deceptive_states"])
                vector = deceptive.mean(axis=0) - honest.mean(axis=0)
        else:
            vector = data

        if vector is None:
            continue

        vector = np.array(vector)
        norm = np.linalg.norm(vector)

        print(f"  Layer {layer_idx:2d}: norm={norm:.3f}")

        if norm > strongest_norm:
            strongest_norm = norm
            strongest_layer = layer_idx
            strongest_vector = vector

    print(
        f"\nStrongest vector found: Layer {strongest_layer} (norm={strongest_norm:.3f})"
    )

    return strongest_layer, strongest_vector, strongest_norm


def create_uniform_steering_vectors(strongest_vector: np.ndarray, num_layers: int):
    """
    Create steering vectors for all layers using the same strongest vector.
    
    Args:
        strongest_vector: The strongest vector to use
        num_layers: Number of layers in the model
        
    Returns:
        dict: Dictionary mapping layer_idx to steering vector
    """
    print(f"\nCreating uniform steering vectors for {num_layers} layers...")

    vectors = {}
    for layer_idx in range(num_layers):
        vectors[layer_idx] = strongest_vector.copy()

    print(f"Created {len(vectors)} uniform steering vectors")

    return vectors


def package_strongest_vector_model(
    base_model_path: str,
    vectors_dir: str,
    gate_path: str,
    output_dir: str,
    alpha_max: float = 50.0,
    max_entropy: float = 10.0,
):
    """
    Package teacher model with STRONGEST SINGLE VECTOR strategy.
    
    Args:
        base_model_path: Base model path or HF repo
        vectors_dir: Directory containing steering vectors
        gate_path: Path to adaptive gate checkpoint
        output_dir: Where to save the package
        alpha_max: Maximum steering strength
        max_entropy: Maximum entropy for normalization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Packaging Model with STRONGEST SINGLE VECTOR Strategy")
    print(f"{'='*80}")
    print(f"Base model: {base_model_path}")
    print(f"Vectors: {vectors_dir}")
    print(f"Gate: {gate_path}")
    print(f"Output: {output_dir}")
    print(f"Strategy: Strongest single vector applied to ALL layers")
    print(f"{'='*80}\n")

    # Step 1: Load model to get number of layers
    print("[1/7] Loading base model to determine architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    # Get number of layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    else:
        raise ValueError("Could not determine number of layers")

    print(f"Model loaded: {num_layers} layers")

    # Step 2: Find strongest vector
    print("[2/7] Finding strongest vector...")
    strongest_layer, strongest_vector, strongest_norm = find_strongest_vector(
        vectors_dir
    )

    # Step 3: Save model
    print("[3/7] Saving model...")
    model.save_pretrained(output_dir)
    print("Model saved")

    # Step 4: Save tokenizer
    print("[4/7] Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved")

    # Step 5: Create and save uniform steering vectors
    print("[5/7] Creating uniform steering vectors...")
    vectors_output = output_dir / "steering_vectors"
    vectors_output.mkdir(exist_ok=True)

    uniform_vectors = create_uniform_steering_vectors(strongest_vector, num_layers)

    for layer_idx, vector in uniform_vectors.items():
        with open(vectors_output / f"layer_{layer_idx}.pkl", "wb") as f:
            pickle.dump({"steering_vector": vector}, f)

    print(f"Saved {len(uniform_vectors)} uniform steering vectors")

    # Step 6: Copy gate
    print("[6/7] Copying adaptive gate...")
    shutil.copy(gate_path, output_dir / "adaptive_gate.pt")
    print("Adaptive gate copied")

    # Step 7: Copy inference module
    print("[7/7] Copying steering inference module...")
    inference_module = (
        Path(__file__).parent / "trl/experimental/gold/standalone_steering_inference.py"
    )
    if inference_module.exists():
        shutil.copy(inference_module, output_dir / "standalone_steering_inference.py")
        print("Inference module copied")
    else:
        print("Warning: standalone_steering_inference.py not found")

    # Create configuration file
    config = {
        "base_model": base_model_path,
        "steering_strategy": "strongest_single_vector_uniform",
        "source_layer": int(strongest_layer),
        "source_vector_norm": float(strongest_norm),
        "num_layers": num_layers,
        "steering_config": {
            "alpha_max": alpha_max,
            "max_entropy": max_entropy,
            "vectors_dir": "./steering_vectors",
            "gate_path": "./adaptive_gate.pt",
        },
        "model_type": "steered_teacher_model",
        "architecture": "uniform_strongest_vector_steering",
    }

    with open(output_dir / "steering_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Configuration saved")

    # Create comprehensive README
    readme_content = f"""# Steered Teacher Model - {base_model_path}
## Strategy: Strongest Single Vector (Uniform Application)

This repository contains a teacher model enhanced with **uniform adaptive steering** using the strongest single vector applied to all layers.

## Steering Strategy

Unlike layer-wise steering where each layer has a different vector, this model uses:

1. **Single Strongest Vector**: The vector with the highest norm (Layer {strongest_layer}, norm={strongest_norm:.3f}) is selected
2. **Uniform Application**: This same vector is applied to ALL {num_layers} layers
3. **Adaptive Modulation**: MLP gate dynamically adjusts strength based on token position and entropy

This creates more **consistent, predictable** steering behavior across the entire model.

## Package Contents

- **Base Model**: {base_model_path}
- **Steering Vector**: Single strongest vector (from Layer {strongest_layer})
- **Application**: Uniformly applied to all {num_layers} layers
- **Adaptive Gate**: MLP-based gate network for dynamic strength modulation
- **Configuration**: `steering_config.json` with strategy details

## 🔧 Model Architecture

### Uniform Steering Process
1. **Vector Selection**: Strongest vector (highest L2 norm) is identified
2. **Uniform Distribution**: Same vector copied to all layers
3. **Real-time Entropy**: Computed at each generation step
4. **MLP Gate**: Adjusts strength: `injection = (1 - lambda_t) * alpha_max * vector`
5. **Layer Application**: All layers receive the same steering signal

### Parameters
- `alpha_max`: {alpha_max} (maximum steering strength)
- `max_entropy`: {max_entropy} (entropy normalization factor)
- `source_layer`: {strongest_layer} (origin of strongest vector)
- `source_norm`: {strongest_norm:.3f} (original vector magnitude)

## Usage

### Installation

```bash
pip install torch transformers huggingface_hub
```

The `standalone_steering_inference.py` module is included in this repository.

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys
from pathlib import Path

# Download repository (includes standalone_steering_inference.py)
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

# Load steering components from repository
model_path = Path(repo_path)

steering_vectors, _ = load_steering_vectors(
    str(model_path / "steering_vectors"), 
    device="cpu"
)
gate = load_gate(str(model_path / "adaptive_gate.pt"), device="cpu")
entropy_tracker = EntropyTracker(max_entropy={max_entropy})

# Create multi-layer steering hook (uses uniform vector for all layers)
lm_head = model.get_output_embeddings()
multi_hook = MultiLayerSteeringHook(
    steering_vectors, 
    gate, 
    entropy_tracker, 
    lm_head, 
    alpha_max={alpha_max}
)

# Register hooks on ALL model layers
layers = model.model.layers
hook_handles = []
for layer_idx in range(len(layers)):
    hook_fn = multi_hook.create_hook(layer_idx)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    hook_handles.append(handle)

print(f"Registered {{len(hook_handles)}} hooks with uniform steering vector")

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

## 📊 Why Uniform Steering?

### Advantages
**Consistency**: Same steering signal across all layers ensures uniform behavior
**Simplicity**: Easier to understand and debug
**Efficiency**: Single vector computation and storage
**Predictability**: More consistent teacher model behavior for distillation

### Use Cases
- Teacher models for knowledge distillation
- Consistent behavior steering across reasoning steps
- Simplified deployment and inference

## 📈 Training Details

- **Source Vector**: Layer {strongest_layer} (highest magnitude)
- **Vector Norm**: {strongest_norm:.3f}
- **Layers Affected**: All {num_layers} layers
- **Adaptive Gate**: Trained on 1k samples with entropy-position pairs
- **Application**: Uniform across all transformer layers

## URL: Related

- [TRL Library](https://github.com/huggingface/trl)
- [GOLD Trainer Documentation](https://huggingface.co/docs/trl/gold_trainer)
- Base Model: [{base_model_path}](https://huggingface.co/{base_model_path})

## Citation

If you use this steered model, please cite:

```bibtex
@misc{{uniform_steered_teacher_model,
  title={{Uniform Adaptive Steering for Language Models: Strongest Single Vector Approach}},
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
Steering strategy: Strongest single vector with uniform application.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    print("README.md created")

    print(f"\n{'='*80}")
    print(f"Package completed successfully!")
    print(f"{'='*80}")
    print(f"Strategy: Strongest Single Vector (Layer {strongest_layer})")
    print(f"Vector Norm: {strongest_norm:.3f}")
    print(f"Applied to: ALL {num_layers} layers uniformly")
    print(f"{'='*80}\n")

    return output_dir


def upload_to_huggingface(
    local_path: str, repo_id: str, private: bool = False, token: str = None
):
    """Upload the packaged model to HuggingFace Hub."""
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
            commit_message="Upload steered teacher model with uniform strongest vector strategy",
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
        description="Package and upload model with STRONGEST SINGLE VECTOR strategy"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model path or HuggingFace repo",
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        required=True,
        help="Directory containing steering vectors",
    )
    parser.add_argument(
        "--gate-path", type=str, required=True, help="Path to adaptive gate checkpoint"
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
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to save the package",
    )
    parser.add_argument(
        "--repo-id", type=str, required=True, help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Only package locally, don't upload"
    )
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")

    args = parser.parse_args()

    # Package the model
    output_path = package_strongest_vector_model(
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
