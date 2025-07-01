"""
Callback for logging UME GRPO training samples and rewards to wandb.

This callback provides detailed logging of training samples, their rewards,
and training progress during UME-based GRPO training.
"""

import logging
import random
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb

logger = logging.getLogger(__name__)


class UmeGrpoLoggingCallback(TrainerCallback):
    """
    Callback for logging UME GRPO training samples and rewards to wandb.

    This callback logs:
    - Training samples and their rewards
    - Reward statistics (min, max, mean, std)
    - Training progress metrics
    - Sample examples with their modalities and rewards
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        max_samples_to_log: int = 10,
        log_reward_histogram: bool = True,
        log_sample_examples: bool = True,
        log_modality_breakdown: bool = True,
    ):
        """
        Initialize the UME GRPO logging callback.

        Parameters:
        -----------
        log_every_n_steps : int, optional
            Log every N training steps, default 100
        max_samples_to_log : int, optional
            Maximum number of sample examples to log per step, default 10
        log_reward_histogram : bool, optional
            Whether to log reward histograms, default True
        log_sample_examples : bool, optional
            Whether to log individual sample examples, default True
        log_modality_breakdown : bool, optional
            Whether to log reward breakdown by modality, default True
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_samples_to_log = max_samples_to_log
        self.log_reward_histogram = log_reward_histogram
        self.log_sample_examples = log_sample_examples
        self.log_modality_breakdown = log_modality_breakdown

        # Track training progress
        self.step_count = 0
        self.epoch_count = 0

        # Store recent samples and rewards for logging
        self.recent_samples = []
        self.recent_rewards = []
        self.recent_modalities = []
        # Add persistent wandb.Table for sample examples
        self.sample_examples_table = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """Initialize wandb logging at the start of training."""
        if not self._is_wandb_available():
            logger.warning("Wandb not available. UmeGrpoLoggingCallback will not log to wandb.")
            return

        logger.info("Initializing UME GRPO logging callback with wandb")

        # Log training configuration
        config = {
            "training/output_dir": args.output_dir,
            "training/learning_rate": args.learning_rate,
            "training/per_device_train_batch_size": args.per_device_train_batch_size,
            "training/max_steps": args.max_steps,
            "logging/log_every_n_steps": self.log_every_n_steps,
            "logging/max_samples_to_log": self.max_samples_to_log,
        }

        try:
            wandb.config.update(config, allow_val_change=True)
        except Exception as e:
            logger.warning(f"Failed to update wandb config: {e}")
            # Continue without config update
        # Initialize the persistent sample examples table
        if self.log_sample_examples and self._is_wandb_available():
            self.sample_examples_table = wandb.Table(columns=["step", "sample", "reward", "modality", "length"])

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """Log training samples and rewards after each step."""
        self.step_count += 1

        if not self._is_wandb_available():
            return

        # Only log every N steps to avoid overwhelming wandb
        if self.step_count % self.log_every_n_steps != 0:
            return

        # Extract samples and rewards from the logs
        logs = kwargs.get("logs", {})
        samples, rewards, modalities = self._extract_logs_data(logs)

        if not samples or not rewards:
            return

        # Store recent data for logging
        self.recent_samples.extend(samples)
        self.recent_rewards.extend(rewards)
        self.recent_modalities.extend(modalities)

        # Log metrics
        self._log_training_metrics(state, rewards, modalities)

        # Log sample examples
        if self.log_sample_examples:
            self._log_sample_examples(state, samples, rewards, modalities)

        # Log reward histogram
        if self.log_reward_histogram:
            self._log_reward_histogram(state, rewards)

        # Log modality breakdown
        if self.log_modality_breakdown:
            self._log_modality_breakdown(state, rewards, modalities)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """Log final training summary."""
        if not self._is_wandb_available():
            return

        # Log final summary
        if self.recent_rewards:
            final_metrics = {
                "train/final_mean_reward": sum(self.recent_rewards) / len(self.recent_rewards),
                "train/final_min_reward": min(self.recent_rewards),
                "train/final_max_reward": max(self.recent_rewards),
                "train/total_samples_processed": len(self.recent_rewards),
                "train/total_steps": self.step_count,
            }

            wandb.log(final_metrics, step=state.global_step)

        # Log the final sample examples table
        if self.log_sample_examples and self.sample_examples_table is not None and self._is_wandb_available():
            # Only log if the table has data
            if len(self.sample_examples_table.data) > 0:
                wandb.log({"train/all_sample_examples": self.sample_examples_table}, step=state.global_step)

    def _is_wandb_available(self) -> bool:
        """Check if wandb is available and initialized."""
        return wandb.run is not None

    def _extract_logs_data(self, logs: dict[str, Any]) -> tuple[list[str], list[float], list[str]]:
        """
        Extract samples, rewards, and modalities from training logs.

        Returns:
        --------
        tuple[List[str], List[float], List[str]]
            Samples, rewards, and modalities
        """
        samples = []
        rewards = []
        modalities = []

        try:
            # Try to extract from logs (typical TRL format)
            if logs and isinstance(logs, dict):
                if "rewards" in logs:
                    rewards = logs["rewards"]
                if "completions" in logs:
                    samples = logs["completions"]
                if "modalities" in logs:
                    modalities = logs["modalities"]

            # Ensure all lists have the same length
            max_len = max(len(samples), len(rewards), len(modalities))
            if max_len > 0:
                samples = samples[:max_len] if samples else ["unknown"] * max_len
                rewards = rewards[:max_len] if rewards else [0.0] * max_len
                modalities = modalities[:max_len] if modalities else ["unknown"] * max_len

        except Exception as e:
            logger.warning(f"Failed to extract logs data: {e}")

        return samples, rewards, modalities

    def _log_training_metrics(self, state: TrainerState, rewards: list[float], modalities: list[str]) -> None:
        """Log training metrics to wandb."""
        if not rewards:
            return

        metrics = {
            "train/step": self.step_count,
            "train/global_step": state.global_step,
            "train/mean_reward": sum(rewards) / len(rewards),
            "train/min_reward": min(rewards),
            "train/max_reward": max(rewards),
            "train/reward_std": torch.std(torch.tensor(rewards)).item(),
            "train/batch_size": len(rewards),
        }

        wandb.log(metrics, step=state.global_step)

    def _log_sample_examples(
        self, state: TrainerState, samples: list[str], rewards: list[float], modalities: list[str]
    ) -> None:
        """Log sample examples to wandb as both a persistent table and step-specific table."""
        if not samples or not rewards:
            return

        n_samples = min(self.max_samples_to_log, len(samples))
        indices = random.sample(range(len(samples)), n_samples)

        # Create a step-specific table for immediate logging
        step_table = wandb.Table(columns=["sample", "reward", "modality", "length"])

        for idx in indices:
            sample = samples[idx]
            reward = rewards[idx]
            modality = modalities[idx] if idx < len(modalities) else "unknown"
            display_sample = sample[:200] + "..." if len(sample) > 200 else sample

            # Add to step-specific table
            step_table.add_data(display_sample, reward, modality, len(sample))

            # Add to persistent table for final logging
            if self.sample_examples_table is not None:
                self.sample_examples_table.add_data(state.global_step, display_sample, reward, modality, len(sample))

        # Log the step-specific table immediately
        wandb.log({f"train/sample_examples_step_{state.global_step}": step_table}, step=state.global_step)

    def _log_reward_histogram(self, state: TrainerState, rewards: list[float]) -> None:
        """Log reward histogram to wandb."""
        if not rewards:
            return

        wandb.log({"train/reward_histogram": wandb.Histogram(rewards, num_bins=20)}, step=state.global_step)

    def _log_modality_breakdown(self, state: TrainerState, rewards: list[float], modalities: list[str]) -> None:
        """Log reward breakdown by modality."""
        if not rewards or not modalities or len(rewards) != len(modalities):
            return

        # Group rewards by modality
        modality_rewards = {}
        for reward, modality in zip(rewards, modalities):
            if modality not in modality_rewards:
                modality_rewards[modality] = []
            modality_rewards[modality].append(reward)

        # Log metrics for each modality
        for modality, modality_reward_list in modality_rewards.items():
            if modality_reward_list:
                metrics = {
                    f"train/modality_{modality}/mean_reward": sum(modality_reward_list) / len(modality_reward_list),
                    f"train/modality_{modality}/min_reward": min(modality_reward_list),
                    f"train/modality_{modality}/max_reward": max(modality_reward_list),
                    f"train/modality_{modality}/count": len(modality_reward_list),
                }

                wandb.log(metrics, step=state.global_step)
