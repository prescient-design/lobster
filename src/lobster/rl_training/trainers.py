"""
Trainer utilities for reinforcement learning training.

This module provides utilities for creating and configuring RL trainers,
particularly for GRPO training with UME reward functions.
"""

import logging

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from lobster.model import UME

from .reward_functions import create_ume_reward_wrapper

logger = logging.getLogger(__name__)


def create_ume_grpo_trainer(
    model_path: str,
    ume_model: UME,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    output_dir: str = "./ume_grpo_runs",
    reward_temperature: float = 0.1,
    reward_batch_size: int = 8,
    **grpo_kwargs,
) -> GRPOTrainer:
    """
    Create a GRPO trainer configured for UME-based reward training.

    Parameters:
    -----------
    model_path : str
        Path to the base model (can be HuggingFace model name or local path)
    ume_model : UME
        The UME model to use for reward computation
    train_dataset : Dataset
        Training dataset
    eval_dataset : Dataset, optional
        Evaluation dataset, default None
    output_dir : str, optional
        Output directory for training artifacts, default "./ume_grpo_runs"
    reward_temperature : float, optional
        Temperature scaling for rewards, default 0.1
    reward_batch_size : int, optional
        Batch size for reward computation, default 8
    **grpo_kwargs : dict
        Additional arguments to pass to GRPOConfig

    Returns:
    --------
    GRPOTrainer
        Configured GRPO trainer ready for training
    """
    # Create reward function wrapper
    reward_func = create_ume_reward_wrapper(ume_model, temperature=reward_temperature, batch_size=reward_batch_size)

    # Create GRPO configuration
    training_args = GRPOConfig(output_dir=output_dir, **grpo_kwargs)

    # Create trainer
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Created GRPO trainer with UME reward function")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reward temperature: {reward_temperature}")
    logger.info(f"Reward batch size: {reward_batch_size}")

    return trainer


def train_ume_grpo(
    model_path: str,
    ume_model_path: str,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    output_dir: str = "./ume_grpo_runs",
    reward_temperature: float = 0.1,
    reward_batch_size: int = 8,
    device: str = "cuda",
    **grpo_kwargs,
) -> GRPOTrainer:
    """
    Complete pipeline for training with UME-based GRPO.

    This function loads the UME model, creates the trainer, and starts training.

    Parameters:
    -----------
    model_path : str
        Path to the base model (can be HuggingFace model name or local path)
    ume_model_path : str
        Path to the UME model checkpoint
    train_dataset : Dataset
        Training dataset
    eval_dataset : Dataset, optional
        Evaluation dataset, default None
    output_dir : str, optional
        Output directory for training artifacts, default "./ume_grpo_runs"
    reward_temperature : float, optional
        Temperature scaling for rewards, default 0.1
    reward_batch_size : int, optional
        Batch size for reward computation, default 8
    device : str, optional
        Device to load UME model on, default "cuda"
    **grpo_kwargs : dict
        Additional arguments to pass to GRPOConfig

    Returns:
    --------
    GRPOTrainer
        The trained trainer
    """
    # Load UME model
    logger.info(f"Loading UME model from {ume_model_path}")
    ume_model = UME.from_pretrained(ume_model_path, device=device)

    # Create trainer
    trainer = create_ume_grpo_trainer(
        model_path=model_path,
        ume_model=ume_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        reward_temperature=reward_temperature,
        reward_batch_size=reward_batch_size,
        **grpo_kwargs,
    )

    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()

    return trainer
