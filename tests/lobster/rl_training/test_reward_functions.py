"""
Tests for reward functions in the rl_training module.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

from lobster.model import UME
from lobster.rl_training.reward_functions import (
    UMERewardFunction,
    create_ume_reward_wrapper,
)

logger = logging.getLogger(__name__)


class TestUMERewardFunction:
    """Test the UMERewardFunction class."""

    @pytest.fixture
    def mock_UME_model(self):
        """Create a mock UME model."""
        model = Mock(spec=UME)
        model.eval.return_value = None
        model.freeze.return_value = None
        model.compute_pseudo_likelihood.return_value = [0.5, 0.7, 0.3]
        return model

    def test_reward_function_initialization(self, mock_UME_model):
        """Test that the reward function can be initialized."""
        reward_func = UMERewardFunction(mock_UME_model, temperature=0.1, batch_size=4)
        assert reward_func.ume_model == mock_UME_model
        assert reward_func.temperature == 0.1
        assert reward_func.batch_size == 4

        # Verify that the model was put in eval mode and frozen
        mock_UME_model.eval.assert_called_once()
        mock_UME_model.freeze.assert_called_once()

    def test_empty_completions(self, mock_UME_model):
        """Test reward function with empty completions."""
        reward_func = UMERewardFunction(mock_UME_model)
        rewards = reward_func([])
        assert rewards == []

    def test_single_completion(self, mock_UME_model):
        """Test reward function with a single completion."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = ["<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>"]  # Ibuprofen
        rewards = reward_func(completions)

        assert len(rewards) == 1
        assert isinstance(rewards[0], (float, np.floating))  # Allow numpy types
        # Reward should be a finite number
        assert not (rewards[0] != rewards[0])  # Not NaN
        assert rewards[0] != float("inf")  # Not inf

    def test_multiple_completions(self, mock_UME_model):
        """Test reward function with multiple completions."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = [
            "<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>",  # Ibuprofen
            "<smiles>CN1C=NC2=C1C(=O)N(C(=O)N2C)C</smiles>",  # Caffeine
            "<smiles>COC1=CC=C(CCN)C=C1</smiles>",  # Simple molecule
        ]
        rewards = reward_func(completions)

        assert len(rewards) == 3
        assert all(isinstance(r, (float, np.floating)) for r in rewards)
        assert all(not (r != r) for r in rewards)  # No NaNs
        assert all(r != float("inf") for r in rewards)  # No infs

    def test_batch_processing(self, mock_UME_model):
        """Test that completions are processed in batches."""
        reward_func = UMERewardFunction(mock_UME_model, batch_size=2)
        completions = [
            "<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>",
            "<smiles>CN1C=NC2=C1C(=O)N(C(=O)N2C)C</smiles>",
            "<smiles>COC1=CC=C(CCN)C=C1</smiles>",
            "<smiles>CC12CCC(CC1)CC(C2)C(C)CN</smiles>",
        ]
        rewards = reward_func(completions)

        assert len(rewards) == 4
        # The mock should be called multiple times due to batching
        assert mock_UME_model.compute_pseudo_likelihood.call_count >= 2

    def test_error_handling_missing_tag(self, mock_UME_model):
        """Test error handling when no valid tag is present."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = ["invalid_sequence_12345"]
        rewards = reward_func(completions)

        assert len(rewards) == 1
        # No tag, so reward should be penalty
        assert rewards[0] == reward_func.penalty_for_invalid

    def test_error_handling_empty_tag(self, mock_UME_model):
        """Test error handling when tag is present but content is empty or whitespace."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = ["<smiles>   </smiles>", "<protein></protein>", "<dna>\n\n</dna>"]
        rewards = reward_func(completions)
        assert rewards == [reward_func.penalty_for_invalid] * 3

    def test_statistics_tracking(self, mock_UME_model):
        """Test that statistics are properly tracked."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = [
            "<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>",
            "<smiles>CN1C=NC2=C1C(=O)N(C(=O)N2C)C</smiles>",
        ]
        rewards = reward_func(completions)

        # Verify rewards were computed
        assert len(rewards) == 2
        assert isinstance(rewards[0], (float, np.floating))
        assert not (rewards[0] != rewards[0])  # Not NaN
        assert rewards[0] != float("inf")  # Not inf

        stats = reward_func.get_statistics()
        assert stats["count"] == 2
        assert stats["min"] <= stats["max"]
        assert "mean" in stats

    def test_protein_and_dna_tags(self, mock_UME_model):
        """Test reward function with <protein> and <dna> tags."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = [
            "<protein>MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG</protein>",
            "<dna>ATCGATCGATCG</dna>",
        ]
        rewards = reward_func(completions)
        assert len(rewards) == 2
        assert all(isinstance(r, (float, np.floating)) for r in rewards)

    def test_mixed_tags_and_no_tags(self, mock_UME_model):
        """Test a mix of valid tags and missing tags."""
        reward_func = UMERewardFunction(mock_UME_model)
        completions = [
            "<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>",
            "no_tag_content",
            "<protein>MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG</protein>",
            "<smiles></smiles>",
            "<dna>ATCGATCGATCG</dna>",
        ]
        rewards = reward_func(completions)
        assert len(rewards) == 5
        assert isinstance(rewards[0], (float, np.floating))
        assert rewards[1] == reward_func.penalty_for_invalid
        assert isinstance(rewards[2], (float, np.floating))
        assert rewards[3] == reward_func.penalty_for_invalid
        assert isinstance(rewards[4], (float, np.floating))

    @patch("lobster.rl_training.reward_functions.wandb")
    @patch("lobster.rl_training.reward_functions.random.random")
    def test_wandb_logging(self, mock_random, mock_wandb, mock_UME_model):
        """Test wandb logging functionality."""
        # Force the random check to pass (10% chance to log samples)
        mock_random.return_value = 0.05  # Less than 0.1, so logging will happen

        reward_func = UMERewardFunction(mock_UME_model, enable_wandb_logging=True)
        completions = ["<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>"]
        rewards = reward_func(completions)

        # Verify rewards were computed
        assert len(rewards) == 1
        assert isinstance(rewards[0], (float, np.floating))
        assert not (rewards[0] != rewards[0])  # Not NaN
        assert rewards[0] != float("inf")  # Not inf

        # Verify wandb logging was attempted
        assert mock_wandb.log.called


class TestCreateUmeRewardWrapper:
    """Test the create_ume_reward_wrapper function."""

    @pytest.fixture
    def mock_UME_model(self):
        """Create a mock UME model."""
        model = Mock(spec=UME)
        model.eval.return_value = None
        model.freeze.return_value = None
        model.compute_pseudo_likelihood.return_value = [0.5]
        return model

    def test_wrapper_creation(self, mock_UME_model):
        """Test that the wrapper is created correctly."""
        wrapper = create_ume_reward_wrapper(mock_UME_model, temperature=0.2, batch_size=16)
        assert callable(wrapper)

    def test_wrapper_functionality(self, mock_UME_model):
        """Test that the wrapper function works correctly."""
        wrapper = create_ume_reward_wrapper(mock_UME_model)
        completions = ["<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>"]
        rewards = wrapper(completions)

        assert len(rewards) == 1
        assert isinstance(rewards[0], (float, np.floating))

    def test_wrapper_with_kwargs(self, mock_UME_model):
        """Test that the wrapper handles additional kwargs correctly."""
        wrapper = create_ume_reward_wrapper(mock_UME_model)
        completions = ["<smiles>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</smiles>"]
        rewards = wrapper(completions, extra_param="test")

        assert len(rewards) == 1
        assert isinstance(rewards[0], (float, np.floating))
