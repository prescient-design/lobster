#!/usr/bin/env python3
"""
Example script for training with UME-based GRPO reinforcement learning.

This script demonstrates how to use the lobster.rl_training module to train
a language model using GRPO with UME-based rewards.
"""

import logging
import os
import glob
from datasets import load_from_disk
from lobster.rl_training import train_ume_grpo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_CACHE = os.getenv('HF_HUB_CACHE')

def find_qwen_model_in_cache():
    """Find a Qwen model in the HF_HUB_CACHE directory."""
    if not HF_CACHE:
        raise ValueError("HF_HUB_CACHE environment variable is not set")
    
    # Look for Qwen models in the cache
    qwen_pattern = os.path.join(HF_CACHE, "models--Qwen--Qwen3-*")
    qwen_dirs = glob.glob(qwen_pattern)
    
    if not qwen_dirs:
        raise FileNotFoundError(f"No Qwen3 models found in {HF_CACHE}")
    
    # Get the most recent snapshot for the first Qwen model found
    model_dir = qwen_dirs[0]
    snapshots_dir = os.path.join(model_dir, "snapshots")
    
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"No snapshots directory found in {model_dir}")
    
    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    
    # Use the first snapshot (you could sort by modification time if needed)
    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
    logger.info(f"Found Qwen3 model: {snapshot_path}")
    return snapshot_path


def main():
    """Main training function."""
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_from_disk(os.path.join(HF_CACHE, "synthetic_molecular_dataset/train"))
    val_dataset = load_from_disk(os.path.join(HF_CACHE, "synthetic_molecular_dataset/validation"))
    
    # Load Qwen model from HF_HUB_CACHE
    qwen_model_path = find_qwen_model_in_cache()
    
    # UME model path
    # ume_model_path = "ume-mini-base-12M"  # for debugging
    ume_model_path = "ume-medium-base-480M"  # for training
    
    # Training configuration
    output_dir = os.path.join(HF_CACHE, "ume_trl_runs")
    reward_temperature = 0.1
    reward_batch_size = 8
    
    # Wandb configuration
    enable_wandb_logging = True
    wandb_project = "lobster-ume-grpo"
    wandb_entity = os.getenv('WANDB_ENTITY', None)  # Set via environment variable
    wandb_run_name = f"ume-grpo-{reward_temperature}-temp-{reward_batch_size}-batch"
    
    logger.info("Starting UME-based GRPO training...")
    logger.info(f"Base model: {qwen_model_path}")
    logger.info(f"UME model: {ume_model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reward temperature: {reward_temperature}")
    logger.info(f"Reward batch size: {reward_batch_size}")
    logger.info(f"Wandb logging enabled: {enable_wandb_logging}")
    if enable_wandb_logging:
        logger.info(f"Wandb project: {wandb_project}")
        logger.info(f"Wandb entity: {wandb_entity}")
        logger.info(f"Wandb run name: {wandb_run_name}")
    
    # Start training with enhanced logging
    trainer = train_ume_grpo(
        model_path=qwen_model_path,
        ume_model_path=ume_model_path,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=output_dir,
        reward_temperature=reward_temperature,
        reward_batch_size=reward_batch_size,
        device="cuda",
        enable_wandb_logging=enable_wandb_logging,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        # Additional GRPO parameters
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        max_steps=1000,
        save_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_generations=4,  # Must be a divisor of effective batch size (4)
    )
    
    logger.info("Training completed successfully!")
    return trainer


if __name__ == "__main__":
    main()