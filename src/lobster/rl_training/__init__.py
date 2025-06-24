"""
Reinforcement Learning training utilities for lobster.

This module provides utilities for training models using reinforcement learning
techniques, particularly with reward functions based on UME models.
"""

from .reward_functions import UmeRewardFunction, create_ume_reward_wrapper, detect_modality
from .trainers import create_ume_grpo_trainer, train_ume_grpo

__all__ = [
    "UmeRewardFunction",
    "detect_modality",
    "create_ume_reward_wrapper",
    "create_ume_grpo_trainer",
    "train_ume_grpo",
]
