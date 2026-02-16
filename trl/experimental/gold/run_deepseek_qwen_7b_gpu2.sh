#!/bin/bash

# GOLD training launcher for DeepSeek-R1-Distill-Qwen-7B.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_LAUNCH_BLOCKING=1

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/../trl_eval/data}"
cd "${PROJECT_ROOT}"

MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TEACHER_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR="checkpoints/r1-qwen-7b-gold-60k-alpha50"
RUN_NAME="deepseek-r1-qwen-7b-gold-60k-alpha50"

VECTORS_DIR="${VECTORS_DIR:-${PROJECT_ROOT}/vectors_mlp/deepseek-r1-distill-qwen-7b}"
GATE_PATH="${GATE_PATH:-${PROJECT_ROOT}/vectors_mlp/adaptive_gate_1k.pt}"

# LoRA options
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)

# vLLM options
USE_VLLM=true
VLLM_MODE="colocate"
VLLM_GPU_MEMORY=0.15
VLLM_TENSOR_PARALLEL=1
export VLLM_USE_CUSTOM_ALL_REDUCE=0
VLLM_SYNC_FREQUENCY=4

python trl/experimental/gold/gold.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --teacher_model_name_or_path ${TEACHER_MODEL} \
  --teacher_tokenizer_name_or_path ${TEACHER_MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --run_name ${RUN_NAME} \
  --dtype auto \
  --attn_implementation eager \
  --dataset_name "${DATA_DIR}" \
  --dataset_train_split train \
  --dataset_test_split val \
  --completion_only_loss True \
  --bf16 True \
  --learning_rate 2e-5 \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --max_steps 5000 \
  --eval_strategy steps \
  --eval_steps 20 \
  --save_strategy steps \
  --save_total_limit 2 \
  --eval_on_start True \
  --save_steps 20 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --max_completion_length 4096 \
  --max_length 4608 \
  --lmbda 1.0 \
  --beta 0.0 \
  --use_uld_loss \
  --use_extended_uld \
  --uld_use_hybrid_loss \
  --uld_crossentropy_weight 1.0 \
  --uld_distillation_weight 1.0 \
  --uld_student_temperature 0.6 \
  --uld_teacher_temperature 0.6 \
  --uld_hybrid_unmatched_weight 1.0 \
  --uld_hybrid_matched_weight 1.0 \
  --vectors_dir ${VECTORS_DIR} \
  --gate_path ${GATE_PATH} \
  --steering_alpha_max 50.0 \
  --steering_max_entropy 10.0 \
  --logging_steps 1 \
  --report_to wandb \
  --seed 42 \
  --warmup_ratio 0.05 \
  --ddp_find_unused_parameters False \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  $(if [ "$USE_LORA" = true ]; then echo "\
  --use_peft \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --lora_target_modules ${LORA_TARGET_MODULES[@]}"; fi) \
  $(if [ "$USE_VLLM" = true ]; then echo "\
  --use_vllm \
  --vllm_mode ${VLLM_MODE} \
  --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY} \
  --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL} \
  --vllm_sync_frequency ${VLLM_SYNC_FREQUENCY}"; fi)
