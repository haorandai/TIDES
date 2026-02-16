#!/usr/bin/env python3
"""
Preview which vector would be selected as the strongest.
Run this before uploading to see what will happen.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def show_strongest_vector(vectors_dir: str):
    """
    Show all vectors and highlight the strongest one.
    
    Args:
        vectors_dir: Directory containing layer_*.pkl files
    """
    vectors_dir = Path(vectors_dir)
    layer_files = sorted(vectors_dir.glob("layer_*.pkl"))

    if not layer_files:
        print(f"ERROR: No layer_*.pkl files found in {vectors_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Scanning Steering Vectors: {vectors_dir}")
    print(f"{'='*80}\n")

    vectors_info = []

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
            print(f"  Layer {layer_idx:2d}: WARNING: Could not extract vector")
            continue

        vector = np.array(vector)
        norm = np.linalg.norm(vector)

        vectors_info.append(
            {
                "layer": layer_idx,
                "norm": norm,
                "shape": vector.shape,
                "mean": np.mean(vector),
                "std": np.std(vector),
            }
        )

    # Find strongest
    if not vectors_info:
        print("ERROR: No valid vectors found")
        return

    strongest = max(vectors_info, key=lambda x: x["norm"])

    # Display results
    print(f"Found {len(vectors_info)} vectors:\n")
    print(
        f"{'Layer':<8} {'Norm':<12} {'Shape':<12} {'Mean':<12} {'Std':<12} {'Status'}"
    )
    print(f"{'-'*80}")

    for info in vectors_info:
        status = (
            "← STRONGEST! Will be used" if info["layer"] == strongest["layer"] else ""
        )
        marker = "" if info["layer"] == strongest["layer"] else "   "

        print(
            f"{marker}Layer {info['layer']:<3} "
            f"{info['norm']:<12.3f} "
            f"{str(info['shape']):<12} "
            f"{info['mean']:<12.6f} "
            f"{info['std']:<12.6f} "
            f"{status}"
        )

    print(f"\n{'='*80}")
    print(f"Strongest Vector Summary")
    print(f"{'='*80}")
    print(f"  Layer: {strongest['layer']}")
    print(f"  Norm: {strongest['norm']:.3f}")
    print(f"  Shape: {strongest['shape']}")
    print(f"  Mean: {strongest['mean']:.6f}")
    print(f"  Std: {strongest['std']:.6f}")
    print(f"{'='*80}")
    print(f"\nThis vector will be applied uniformly to ALL layers during steering.")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preview which vector will be selected as strongest"
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="./vectors_mlp/deepseek-r1-distill-llama-8b",
        help="Directory containing steering vectors",
    )

    args = parser.parse_args()

    show_strongest_vector(args.vectors_dir)


if __name__ == "__main__":
    main()
