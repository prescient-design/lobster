"""
Reward functions for reinforcement learning training.

This module provides reward functions that can be used with RL training frameworks
like TRL, particularly reward functions based on UME models.
"""

import logging
import random

import torch

import wandb
from lobster.constants import Modality
from lobster.model import UME
from lobster.model.utils import _detect_modality

logger = logging.getLogger(__name__)


class UMERewardFunction:
    """
    Reward function that uses UME to reward completions that are high likelihood.

    This class provides a reward function that can be used with RL training frameworks
    like TRL. It computes rewards based on the pseudo-likelihood of sequences according
    to a UME model.
    """

    def __init__(
        self, ume_model: UME, temperature: float = 0.1, batch_size: int = 8, enable_wandb_logging: bool = True
    ):
        """
        Initialize the UME reward function.

        Parameters:
        -----------
        ume_model : UME
            The UME model to use for reward computation
        temperature : float, optional
            Temperature scaling for rewards (lower = more extreme rewards), default 0.1
        batch_size : int, optional
            Batch size for processing sequences, default 8
        enable_wandb_logging : bool, optional
            Whether to enable detailed wandb logging, default True
        """
        self.ume_model = ume_model
        self.temperature = temperature
        self.batch_size = batch_size
        self.enable_wandb_logging = enable_wandb_logging

        # Ensure model is in eval mode and frozen
        self.ume_model.eval()
        self.ume_model.freeze()

        # Track statistics for logging
        self.total_completions_processed = 0
        self.reward_statistics = {
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
            "count": 0,
            "modality_counts": {},
        }

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """
        Compute rewards for a list of completions.

        Parameters:
        -----------
        completions : List[str]
            List of completion strings to evaluate
        **kwargs : dict
            Additional keyword arguments (unused)

        Returns:
        --------
        List[float]
            List of reward scores for each completion
        """
        if not completions:
            return []

        logger.info(f"Computing rewards for {len(completions)} completions")

        rewards = []
        modalities = []
        sample_examples = []

        # Process completions in batches for efficiency
        for i in range(0, len(completions), self.batch_size):
            batch_completions = completions[i : i + self.batch_size]

            # Detect modality for each completion in the batch
            batch_modalities = []
            for comp in batch_completions:
                try:
                    modality = _detect_modality(comp)
                    batch_modalities.append(modality)
                except ValueError as e:
                    logger.warning(
                        f"Unable to determine modality for sequence '{comp[:50]}...': {e}. Assigning zero reward."
                    )
                    # For sequences where we can't determine modality, we'll assign a zero reward
                    # and use a placeholder modality for grouping
                    batch_modalities.append(Modality.SMILES)  # Use SMILES as placeholder

            # Group by modality for efficient processing
            modality_groups = {}
            for j, (comp, modality) in enumerate(zip(batch_completions, batch_modalities)):
                if modality not in modality_groups:
                    modality_groups[modality] = []
                modality_groups[modality].append((j, comp))

            # Initialize batch rewards
            batch_rewards = [0.0] * len(batch_completions)

            # Process each modality group
            for modality, items in modality_groups.items():
                indices, sequences = zip(*items)

                logger.debug(f"Processing {len(sequences)} {modality.value} sequences")

                # Compute pseudo-likelihoods for this modality group using the UME model method
                try:
                    likelihoods = self.ume_model.compute_pseudo_likelihood(sequences, modality)

                    # Apply temperature scaling to make rewards more suitable for RL
                    scaled_likelihoods = [likelihood / self.temperature for likelihood in likelihoods]

                    # Assign rewards back to their positions
                    for idx, likelihood in zip(indices, scaled_likelihoods):
                        batch_rewards[idx] = float(likelihood)

                except Exception as e:
                    logger.warning(
                        f"Error computing likelihoods for {modality.value} sequences: {e}. Assigning zero rewards."
                    )
                    # If there's an error computing likelihoods, assign zero rewards
                    for idx in indices:
                        batch_rewards[idx] = 0.0

            rewards.extend(batch_rewards)
            modalities.extend([mod.value for mod in batch_modalities])

            # Store sample examples for logging
            if self.enable_wandb_logging and random.random() < 0.1:  # 10% chance to log samples
                for comp, reward, modality in zip(batch_completions, batch_rewards, batch_modalities):
                    sample_examples.append(
                        {
                            "completion": comp[:100] + "..." if len(comp) > 100 else comp,
                            "reward": reward,
                            "modality": modality.value,
                            "length": len(comp),
                        }
                    )

        # Update statistics
        self._update_statistics(rewards, modalities)

        # Log to wandb if enabled
        if self.enable_wandb_logging and wandb.run is not None:
            try:
                self._log_to_wandb(rewards, modalities, sample_examples)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
                # Continue without wandb logging

        logger.info(
            f"Computed rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards) / len(rewards):.3f}"
        )
        return rewards

    def _update_statistics(self, rewards: list[float], modalities: list[str]) -> None:
        """Update internal statistics for logging."""
        if not rewards:
            return

        # Update global statistics
        self.total_completions_processed += len(rewards)
        self.reward_statistics["min"] = min(self.reward_statistics["min"], min(rewards))
        self.reward_statistics["max"] = max(self.reward_statistics["max"], max(rewards))
        self.reward_statistics["sum"] += sum(rewards)
        self.reward_statistics["count"] += len(rewards)

        # Update modality counts
        for modality in modalities:
            if modality not in self.reward_statistics["modality_counts"]:
                self.reward_statistics["modality_counts"][modality] = 0
            self.reward_statistics["modality_counts"][modality] += 1

    def _log_to_wandb(self, rewards: list[float], modalities: list[str], sample_examples: list[dict]) -> None:
        """Log detailed metrics to wandb."""
        if not rewards:
            return

        # Basic reward statistics
        metrics = {
            "reward/mean": sum(rewards) / len(rewards),
            "reward/min": min(rewards),
            "reward/max": max(rewards),
            "reward/std": torch.std(torch.tensor(rewards)).item(),
            "reward/count": len(rewards),
            "reward/total_processed": self.total_completions_processed,
        }

        # Modality breakdown
        modality_rewards = {}
        for reward, modality in zip(rewards, modalities):
            if modality not in modality_rewards:
                modality_rewards[modality] = []
            modality_rewards[modality].append(reward)

        for modality, modality_reward_list in modality_rewards.items():
            if modality_reward_list:
                metrics.update(
                    {
                        f"reward/modality_{modality}/mean": sum(modality_reward_list) / len(modality_reward_list),
                        f"reward/modality_{modality}/min": min(modality_reward_list),
                        f"reward/modality_{modality}/max": max(modality_reward_list),
                        f"reward/modality_{modality}/count": len(modality_reward_list),
                    }
                )

        # Log metrics
        wandb.log(metrics)

        # Log sample examples as table
        if sample_examples:
            table = wandb.Table(
                columns=["completion", "reward", "modality", "length"],
                data=[[ex["completion"], ex["reward"], ex["modality"], ex["length"]] for ex in sample_examples],
            )
            wandb.log({"reward/sample_examples": table})

        # Log reward histogram
        wandb.log({"reward/histogram": wandb.Histogram(rewards, num_bins=20)})


def create_ume_reward_wrapper(
    ume_model: UME, temperature: float = 0.1, batch_size: int = 8, enable_wandb_logging: bool = True
):
    """
    Create a reward function wrapper that captures the ume_model.

    This function creates a closure that captures the UME model and returns
    a function with the correct signature for TRL frameworks.

    Parameters:
    -----------
    ume_model : UME
        The UME model to use for reward computation
    temperature : float, optional
        Temperature scaling for rewards, default 0.1
    batch_size : int, optional
        Batch size for processing sequences, default 8
    enable_wandb_logging : bool, optional
        Whether to enable detailed wandb logging, default True

    Returns:
    --------
    callable
        A reward function with signature (completions, **kwargs) -> List[float]
    """
    reward_function = UMERewardFunction(ume_model, temperature, batch_size, enable_wandb_logging)

    def reward_wrapper(completions, **kwargs):
        return reward_function(completions, **kwargs)

    return reward_wrapper
