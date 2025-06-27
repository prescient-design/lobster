"""
Trainer utilities for reinforcement learning training.

This module provides utilities for creating and configuring RL trainers,
particularly for GRPO training with UME reward functions.
"""

import logging

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

import wandb
from lobster.callbacks import UmeGrpoLoggingCallback
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
    penalty_for_invalid: float = -5.0,
    enable_wandb_logging: bool = True,
    wandb_project: str = "lobster-ume-grpo",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    callbacks: list | None = None,
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
    penalty_for_invalid : float, optional
        Penalty reward for invalid completions, default -5.0. For GRPO training,
        this should be significantly lower than typical valid rewards since GRPO
        normalizes rewards by standard deviation.
    enable_wandb_logging : bool, optional
        Whether to enable wandb logging, default True
    wandb_project : str, optional
        Wandb project name, default "lobster-ume-grpo"
    wandb_entity : str, optional
        Wandb entity/username, default None
    wandb_run_name : str, optional
        Wandb run name, default None
    callbacks : List, optional
        Additional callbacks to add, default None
    **grpo_kwargs : dict
        Additional arguments to pass to GRPOConfig

    Returns:
    --------
    GRPOTrainer
        Configured GRPO trainer ready for training
    """
    # Create reward function wrapper with wandb logging
    reward_func = create_ume_reward_wrapper(
        ume_model,
        temperature=reward_temperature,
        batch_size=reward_batch_size,
        penalty_for_invalid=penalty_for_invalid,
        enable_wandb_logging=enable_wandb_logging,
    )

    # Initialize callbacks list
    if callbacks is None:
        callbacks = []

    # Add wandb logging callback if enabled
    if enable_wandb_logging:
        logging_callback = UmeGrpoLoggingCallback(
            log_every_n_steps=100,
            max_samples_to_log=10,
            log_reward_histogram=True,
            log_sample_examples=True,
            log_modality_breakdown=True,
        )
        callbacks.append(logging_callback)

        # Initialize wandb if not already running
        if wandb.run is None:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config={
                    "model_path": model_path,
                    "reward_temperature": reward_temperature,
                    "reward_batch_size": reward_batch_size,
                    "output_dir": output_dir,
                },
            )

    # Create GRPO configuration
    training_args = GRPOConfig(output_dir=output_dir, run_name=wandb_run_name, **grpo_kwargs)

    # Create trainer
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Add callbacks to trainer if any
    if callbacks:
        for callback in callbacks:
            trainer.add_callback(callback)

    logger.info("Created GRPO trainer with UME reward function")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reward temperature: {reward_temperature}")
    logger.info(f"Reward batch size: {reward_batch_size}")
    logger.info(f"Penalty for invalid: {penalty_for_invalid}")
    logger.info(f"Wandb logging enabled: {enable_wandb_logging}")

    return trainer


def train_ume_grpo(
    model_path: str,
    ume_model_path: str,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    output_dir: str = "./ume_grpo_runs",
    reward_temperature: float = 0.1,
    reward_batch_size: int = 8,
    penalty_for_invalid: float = -5.0,
    device: str = "cuda",
    enable_wandb_logging: bool = True,
    wandb_project: str = "lobster-ume-grpo",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
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
    penalty_for_invalid : float, optional
        Penalty reward for invalid completions, default -5.0. For GRPO training,
        this should be significantly lower than typical valid rewards since GRPO
        normalizes rewards by standard deviation.
    device : str, optional
        Device to load UME model on, default "cuda"
    enable_wandb_logging : bool, optional
        Whether to enable wandb logging, default True
    wandb_project : str, optional
        Wandb project name, default "lobster-ume-grpo"
    wandb_entity : str, optional
        Wandb entity/username, default None
    wandb_run_name : str, optional
        Wandb run name, default None
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
        penalty_for_invalid=penalty_for_invalid,
        enable_wandb_logging=enable_wandb_logging,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        **grpo_kwargs,
    )

    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()

    # Finish wandb run if it was started
    if enable_wandb_logging and wandb.run is not None:
        wandb.finish()

    return trainer
