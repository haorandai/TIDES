# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
# Full training:
python trl/experimental/gold/gold.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-model \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing

# LoRA:
python trl/experimental/gold/gold.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-model \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure local repo is on sys.path when running as a script
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer

logger = logging.getLogger(__name__)

# Import steering vector components (optional - only used if vectors_dir is provided)
try:
    from trl.experimental.gold.standalone_steering_inference import (
        load_steering_vectors,
        load_gate,
        EntropyTracker,
        MultiLayerSteeringHook,
    )
    STEERING_AVAILABLE = True
except ImportError:
    STEERING_AVAILABLE = False
    logger.warning("Steering vector components not available. Install required dependencies if needed.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDConfig, ModelConfig))
    
    # Add steering vector arguments
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gate_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--steering_alpha_max",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--steering_max_entropy",
        type=float,
        default=10.0
    )
    
    parsed_args = parser.parse_args_and_config()
    script_args, training_args, model_args = parsed_args[:3]
    steering_args = parsed_args[3] if len(parsed_args) > 3 else argparse.Namespace()
    if len(parsed_args) > 4:
        logger.warning("Extra parsed values ignored: %s", parsed_args[4:])

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=training_args.student_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    if training_args.teacher_tokenizer_name_or_path is None and training_args.use_uld_loss:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path
    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    ################
    # Steering Vector Setup (Optional)
    ################
    teacher_model = None
    hook_handles = []
    
    # Check if steering vectors are requested
    if getattr(steering_args, "vectors_dir", None) is not None:
        if not STEERING_AVAILABLE:
            raise ImportError(
                "Steering vectors requested but standalone_steering_inference module not available. "
                "Check that the file exists at trl/experimental/gold/standalone_steering_inference.py"
            )
        
        if getattr(steering_args, "gate_path", None) is None:
            raise ValueError("--gate_path must be provided when --vectors_dir is specified")
        
        logger.info("=" * 80)
        logger.info("Steering Vector Mode Enabled")
        logger.info("=" * 80)
        logger.info(f"Vectors Dir: {steering_args.vectors_dir}")
        logger.info(f"Gate Path: {steering_args.gate_path}")
        logger.info(f"Alpha Max: {steering_args.steering_alpha_max}")
        logger.info(f"Max Entropy: {steering_args.steering_max_entropy}")
        logger.info("=" * 80)
        
        # Load teacher model explicitly to register hooks
        logger.info(f"Loading teacher model: {training_args.teacher_model_name_or_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            training_args.teacher_model_name_or_path,
            **teacher_model_kwargs
        )
        
        # Load steering vectors and gate (to CPU for compatibility)
        logger.info("Loading steering vectors and gate...")
        steering_vectors, original_norms = load_steering_vectors(
            steering_args.vectors_dir,
            device="cpu"
        )
        gate = load_gate(steering_args.gate_path, device="cpu")
        
        # Create entropy tracker
        entropy_tracker = EntropyTracker(max_entropy=steering_args.steering_max_entropy)
        
        # Get lm_head for real-time entropy computation
        lm_head = teacher_model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("Could not get output embeddings from teacher model")
        
        # Create multi-layer steering hook
        multi_hook = MultiLayerSteeringHook(
            steering_vectors, 
            gate, 
            entropy_tracker, 
            lm_head, 
            alpha_max=steering_args.steering_alpha_max
        )
        
        # Register hooks on teacher model layers
        teacher_layers = teacher_model.model.layers
        for layer_idx in steering_vectors.keys():
            if layer_idx < len(teacher_layers):
                hook_fn = multi_hook.create_hook(layer_idx)
                handle = teacher_layers[layer_idx].register_forward_hook(hook_fn)
                hook_handles.append(handle)
                logger.info(f"Registered steering hook on layer {layer_idx}")
        
        # Set teacher model to eval mode and freeze
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        
        logger.info(f"Successfully registered {len(hook_handles)} steering hooks")
        logger.info("=" * 80)
    
    # If no steering, teacher_model remains None and will be loaded by GOLDTrainer from string
    # If steering is enabled, teacher_model is the actual model object with hooks
    if teacher_model is not None:
        # Avoid passing init kwargs when teacher model is already instantiated.
        training_args.teacher_model_init_kwargs = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    # Handle eval dataset - check if test split exists, fallback to validation or None
    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "dev" in dataset:
            eval_dataset = dataset["dev"]

    # Pass teacher_model object if steering is enabled, otherwise pass string
    teacher_model_arg = teacher_model if teacher_model is not None else training_args.teacher_model_name_or_path
    
    trainer = GOLDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=teacher_model_arg,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    
    # Cleanup steering hooks if they were registered
    if hook_handles:
        logger.info("Cleaning up steering hooks...")
        for handle in hook_handles:
            handle.remove()
        logger.info(f"Removed {len(hook_handles)} hook handles")