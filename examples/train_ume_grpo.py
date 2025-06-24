#!/usr/bin/env python3
"""
Example script for training with UME-based GRPO reinforcement learning.

This script demonstrates how to use the lobster.rl_training module to train
a language model using GRPO with UME-based rewards.
"""

import logging
from datasets import load_from_disk
from lobster.rl_training import train_ume_grpo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_from_disk("/data/bucket/freyn6/synthetic_molecular_dataset/train")
    val_dataset = load_from_disk("/data/bucket/freyn6/synthetic_molecular_dataset/validation")
    
    # Load Qwen model from local cache to avoid download timeouts
    qwen_model_path = "/data/bucket/freyn6/cache/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    
    # UME model path
    ume_model_path = "ume-mini-base-12M"
    
    # Training configuration
    output_dir = "/data/bucket/freyn6/ume_trl_runs"
    reward_temperature = 0.1
    reward_batch_size = 8
    
    logger.info("Starting UME-based GRPO training...")
    logger.info(f"Base model: {qwen_model_path}")
    logger.info(f"UME model: {ume_model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reward temperature: {reward_temperature}")
    logger.info(f"Reward batch size: {reward_batch_size}")
    
    # Start training
    trainer = train_ume_grpo(
        model_path=qwen_model_path,
        ume_model_path=ume_model_path,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=output_dir,
        reward_temperature=reward_temperature,
        reward_batch_size=reward_batch_size,
        device="cuda",
    )
    
    logger.info("Training completed successfully!")
    return trainer


if __name__ == "__main__":
    main()