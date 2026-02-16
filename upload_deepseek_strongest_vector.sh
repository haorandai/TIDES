#!/bin/bash

# Upload DeepSeek-R1-Distill-Llama-8B using strongest-vector steering.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VECTORS_ROOT="${VECTORS_ROOT:-${PROJECT_ROOT}/vectors_mlp}"

echo "================================================================================"
echo "Uploading DeepSeek-R1-Distill-Llama-8B - Strongest Vector Strategy"
echo "================================================================================"
echo ""
echo "Strategy: Find the longest (strongest) vector from all layers,"
echo "          then inject it uniformly to ALL layers"
echo ""
echo "================================================================================"

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
VECTORS_DIR="${VECTORS_DIR:-${VECTORS_ROOT}/deepseek-r1-distill-llama-8b}"
GATE_PATH="${GATE_PATH:-${VECTORS_ROOT}/adaptive_gate_1k.pt}"
ALPHA_MAX=50.0
MAX_ENTROPY=10.0

OUTPUT_DIR="./deepseek-r1-llama-8b-strongest-vector"

# Set this to your Hugging Face repository.
HF_REPO_ID="YOUR_USERNAME/deepseek-r1-llama-8b-strongest-vector"

PRIVATE=""  # Set to --private to create a private repo.

echo ""
echo "Configuration:"
echo "  Base Model: ${BASE_MODEL}"
echo "  Vectors: ${VECTORS_DIR}"
echo "  Gate: ${GATE_PATH}"
echo "  Strategy: Strongest Single Vector (Uniform)"
echo "  Alpha Max: ${ALPHA_MAX}"
echo "  Max Entropy: ${MAX_ENTROPY}"
echo "  Output: ${OUTPUT_DIR}"
echo "  HF Repo: ${HF_REPO_ID}"
echo ""

if [[ "$HF_REPO_ID" == "YOUR_USERNAME/"* ]]; then
    echo "ERROR: Please edit this script and set your HuggingFace username!"
    echo "   Change HF_REPO_ID from 'YOUR_USERNAME/deepseek-r1-llama-8b-strongest-vector'"
    echo "   to 'your-actual-username/deepseek-r1-llama-8b-strongest-vector'"
    exit 1
fi

if [ ! -d "$VECTORS_DIR" ]; then
    echo "ERROR: Vectors directory not found: ${VECTORS_DIR}"
    echo ""
    echo "Available vectors directories:"
    ls -d "${VECTORS_ROOT}"/*/ 2>/dev/null || echo "  No directories found"
    echo ""
    echo "If you have a different directory, edit VECTORS_DIR in this script."
    exit 1
fi

if [ ! -f "$GATE_PATH" ]; then
    echo "ERROR: Gate file not found: ${GATE_PATH}"
    exit 1
fi

echo "All files verified, starting upload process..."
echo ""

python create_strongest_vector_model.py \
    --base-model "${BASE_MODEL}" \
    --vectors-dir "${VECTORS_DIR}" \
    --gate-path "${GATE_PATH}" \
    --alpha-max ${ALPHA_MAX} \
    --max-entropy ${MAX_ENTROPY} \
    --output-dir "${OUTPUT_DIR}" \
    --repo-id "${HF_REPO_ID}" \
    ${PRIVATE}

echo ""
echo "================================================================================"
echo "Upload Complete!"
echo "================================================================================"
echo "View your model at:"
echo "  https://huggingface.co/${HF_REPO_ID}"
echo ""
echo "This model uses the STRONGEST SINGLE VECTOR strategy:"
echo "  - One strongest vector (highest norm) selected from all layers"
echo "  - Applied uniformly to ALL layers for consistent steering"
echo "  - Ideal for teacher model behavior in distillation"
echo "================================================================================"
echo ""
