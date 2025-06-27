"""
Reward functions for reinforcement learning training.

This module provides reward functions that can be used with RL training frameworks
like TRL, particularly reward functions based on UME models.
"""

import logging
import random
import re

import torch

import wandb
from lobster.constants import Modality
from lobster.model import UME

logger = logging.getLogger(__name__)


def extract_tagged_content(text: str) -> tuple[str | None, str | None]:
    """
    Extract content from tags and determine the tag type.

    Parameters:
    -----------
    text : str
        The text to extract tagged content from

    Returns:
    --------
    tuple[str | None, str | None]
        A tuple of (tag_type, content) where tag_type is one of 'smiles', 'protein', 'dna'
        and content is the text within the tags. Returns (None, None) if no valid tags found.
    """
    # Define tag patterns
    tag_patterns = {
        "smiles": r"<smiles>(.*?)</smiles>",
        "protein": r"<protein>(.*?)</protein>",
        "dna": r"<dna>(.*?)</dna>",
    }

    # Find all matches and their positions
    matches = []
    for tag_type, pattern in tag_patterns.items():
        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1).strip()
            if content:  # Only include if content is not empty
                matches.append((match.start(), tag_type, content))

    # Return the first match (earliest position in text)
    if matches:
        matches.sort(key=lambda x: x[0])  # Sort by position
        return matches[0][1], matches[0][2]  # Return (tag_type, content)

    return None, None


class UMERewardFunction:
    """
    Reward function that uses UME to reward completions that are high likelihood.

    This class provides a reward function that can be used with RL training frameworks
    like TRL. It computes rewards based on the pseudo-likelihood of sequences according
    to a UME model. Only content within <smiles>, <protein>, or <dna> tags is evaluated.
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
            "no_tag_count": 0,
            "empty_content_count": 0,
        }

    def __call__(self, completions: list[str], **kwargs) -> list[float | None]:
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
        List[float | None]
            List of reward scores for each completion. Returns None if no valid tags found
            or content is empty.
        """
        if not completions:
            return []

        logger.info(f"Computing rewards for {len(completions)} completions")

        rewards = [None] * len(completions)  # Initialize all rewards as None
        sample_examples = []

        # Process completions in batches for efficiency
        for i in range(0, len(completions), self.batch_size):
            batch_completions = completions[i : i + self.batch_size]

            # Extract tagged content and determine modality for each completion
            batch_tagged_content = []
            batch_modalities = []
            batch_original_indices = []

            for j, comp in enumerate(batch_completions):
                tag_type, content = extract_tagged_content(comp)

                if tag_type is None or content is None:
                    # No valid tags found or empty content
                    if tag_type is None:
                        self.reward_statistics["no_tag_count"] += 1
                    else:
                        self.reward_statistics["empty_content_count"] += 1
                    continue

                # Map tag type to modality
                tag_to_modality = {"smiles": Modality.SMILES, "protein": Modality.PROTEIN, "dna": Modality.DNA}

                modality = tag_to_modality.get(tag_type)
                if modality is None:
                    logger.warning(
                        f"Unknown tag type '{tag_type}' for completion '{comp[:50]}...'. Assigning None reward."
                    )
                    self.reward_statistics["no_tag_count"] += 1
                    continue

                batch_tagged_content.append(content)
                batch_modalities.append(modality)
                batch_original_indices.append(j)

            # If no valid content found in this batch, continue to next batch
            if not batch_tagged_content:
                continue

            # Group by modality for efficient processing
            modality_groups = {}
            for k, (content, modality) in enumerate(zip(batch_tagged_content, batch_modalities)):
                if modality not in modality_groups:
                    modality_groups[modality] = []
                modality_groups[modality].append((k, content))

            # Process each modality group
            for modality, items in modality_groups.items():
                indices, sequences = zip(*items)

                logger.debug(f"Processing {len(sequences)} {modality.value} sequences")

                # Compute pseudo-likelihoods for this modality group using the UME model method
                try:
                    likelihoods = self.ume_model.compute_pseudo_likelihood(sequences, modality)

                    # Apply temperature scaling to make rewards more suitable for RL
                    scaled_likelihoods = [likelihood / self.temperature for likelihood in likelihoods]

                    # Assign rewards back to their positions in the original completions list
                    for idx, likelihood in zip(indices, scaled_likelihoods):
                        original_idx = i + batch_original_indices[idx]  # Convert to global index
                        rewards[original_idx] = float(likelihood)

                except Exception as e:
                    logger.warning(
                        f"Error computing likelihoods for {modality.value} sequences: {e}. Assigning None rewards."
                    )
                    # If there's an error computing likelihoods, assign None rewards
                    for idx in indices:
                        original_idx = i + batch_original_indices[idx]  # Convert to global index
                        rewards[original_idx] = None

            # Store sample examples for logging (only for valid rewards)
            if self.enable_wandb_logging and random.random() < 0.1:  # 10% chance to log samples
                for j, comp in enumerate(batch_completions):
                    global_idx = i + j
                    if global_idx < len(rewards) and rewards[global_idx] is not None:
                        # Find the modality for this completion
                        comp_modality = None
                        for k, (content, modality) in enumerate(zip(batch_tagged_content, batch_modalities)):
                            if i + batch_original_indices[k] == global_idx:
                                comp_modality = modality
                                break

                        if comp_modality:
                            sample_examples.append(
                                {
                                    "completion": comp[:100] + "..." if len(comp) > 100 else comp,
                                    "reward": rewards[global_idx],
                                    "modality": comp_modality.value,
                                    "length": len(comp),
                                }
                            )

        # Update statistics for valid rewards only
        valid_rewards = [r for r in rewards if r is not None]
        self.total_completions_processed += len(completions)

        for reward in valid_rewards:
            self.reward_statistics["min"] = min(self.reward_statistics["min"], reward)
            self.reward_statistics["max"] = max(self.reward_statistics["max"], reward)
            self.reward_statistics["sum"] += reward
            self.reward_statistics["count"] += 1

        # Update modality counts for valid rewards
        for j, reward in enumerate(rewards):
            if reward is not None:
                # Find the modality for this completion by checking the original completion
                comp = completions[j]
                tag_type, _ = extract_tagged_content(comp)
                if tag_type:
                    tag_to_modality = {"smiles": "smiles", "protein": "protein", "dna": "dna"}
                    modality_key = tag_to_modality.get(tag_type)
                    if modality_key:
                        if modality_key not in self.reward_statistics["modality_counts"]:
                            self.reward_statistics["modality_counts"][modality_key] = 0
                        self.reward_statistics["modality_counts"][modality_key] += 1

        # Log to wandb if enabled
        if self.enable_wandb_logging and sample_examples:
            self._log_to_wandb(sample_examples, valid_rewards)

        logger.info(f"Computed rewards for {len(completions)} completions ({len(valid_rewards)} valid)")
        return rewards

    def _log_to_wandb(self, sample_examples: list[dict], rewards: list[float]) -> None:
        """Log sample examples and statistics to wandb."""
        try:
            # Log sample examples
            if sample_examples:
                wandb.log(
                    {
                        "sample_examples": wandb.Table(
                            dataframe=wandb.Table.from_list(
                                ["completion", "reward", "modality", "length"],
                                [
                                    [ex["completion"], ex["reward"], ex["modality"], ex["length"]]
                                    for ex in sample_examples
                                ],
                            )
                        )
                    }
                )

            # Log reward statistics
            if rewards:
                wandb.log(
                    {
                        "reward_statistics/mean": sum(rewards) / len(rewards),
                        "reward_statistics/min": min(rewards),
                        "reward_statistics/max": max(rewards),
                        "reward_statistics/std": torch.std(torch.tensor(rewards)).item(),
                        "reward_statistics/count": len(rewards),
                        "reward_statistics/no_tag_count": self.reward_statistics["no_tag_count"],
                        "reward_statistics/empty_content_count": self.reward_statistics["empty_content_count"],
                    }
                )

            # Log modality breakdown from statistics
            for modality, count in self.reward_statistics["modality_counts"].items():
                wandb.log({f"modality_counts/{modality}": count})

        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def get_statistics(self) -> dict:
        """Get current reward statistics."""
        stats = self.reward_statistics.copy()
        if stats["count"] > 0:
            stats["mean"] = stats["sum"] / stats["count"]
        else:
            stats["mean"] = 0.0
        return stats


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
        A reward function with signature (completions, **kwargs) -> List[float | None]
    """
    reward_function = UMERewardFunction(ume_model, temperature, batch_size, enable_wandb_logging)

    def reward_wrapper(completions, **kwargs):
        return reward_function(completions, **kwargs)

    return reward_wrapper
