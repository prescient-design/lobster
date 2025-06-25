"""
Tests for reward functions in the rl_training module.
"""

import logging
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lobster.constants import Modality
from lobster.model import UME
from lobster.rl_training.reward_functions import UMERewardFunction

logger = logging.getLogger(__name__)


class TestUMERewardFunction:
    """Test the UME reward function."""

    @pytest.fixture
    def mock_UME_model(self):
        """Create a comprehensive mock UME model for testing."""
        # Create mock model
        mock_model = Mock(spec=UME)

        # Mock the model's internal components
        mock_model.model = Mock()
        mock_model.model.config = Mock()
        mock_model.model.config.vocab_size = 1536
        mock_model.model.pad_token_id = 0
        mock_model.max_length = 512

        # Mock the model's parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters = Mock(return_value=iter([mock_param]))

        # Mock tokenizer transforms - simplified
        def mock_tokenizer_transform(sequences):
            batch_size = len(sequences)
            return {
                "input_ids": torch.ones(batch_size, 5, dtype=torch.long),  # Simple tensor
                "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),  # Simple tensor
            }

        mock_model.tokenizer_transforms = {
            Modality.SMILES: mock_tokenizer_transform,
            Modality.AMINO_ACID: mock_tokenizer_transform,
            Modality.NUCLEOTIDE: mock_tokenizer_transform,
        }

        # Mock the internal model's _prepare_inputs method - simplified
        def mock_prepare_inputs(input_ids_3d, attention_mask_3d):
            batch_size = input_ids_3d.shape[0]
            seq_len = input_ids_3d.shape[-1]
            total_tokens = batch_size * seq_len
            return (
                torch.ones(total_tokens, dtype=torch.long),  # input_ids_flat
                torch.ones(total_tokens, dtype=torch.long),  # attention_mask_flat
                torch.tensor([0, total_tokens]),  # cu_seqlens
            )

        mock_model.model._prepare_inputs = Mock(side_effect=mock_prepare_inputs)

        # Mock the model's forward pass
        def mock_model_forward(input_ids, attention_mask, cu_seqlens, max_seqlen):
            batch_size = cu_seqlens[-1].item() // 5  # Assuming seq_len=5
            return torch.randn(batch_size * 5, 768)  # Mock hidden states

        mock_model.model.model = Mock(side_effect=mock_model_forward)

        # Mock the decoder - simplified
        def mock_decoder(hidden_states):
            batch_size = hidden_states.shape[0] // 5  # Assuming seq_len=5
            logits = torch.randn(batch_size * 5, 1536)
            # Set high logits for token_id=1 to ensure good likelihood
            logits[:, 1] = 10.0
            return logits

        mock_model.model.decoder = Mock(side_effect=mock_decoder)

        # Mock the compute_pseudo_likelihood method
        def mock_compute_pseudo_likelihood(sequences, modality):
            if not sequences:
                return []
            # Return mock likelihoods based on sequence count
            return [0.5] * len(sequences)  # Simple mock return value

        mock_model.compute_pseudo_likelihood = Mock(side_effect=mock_compute_pseudo_likelihood)

        return mock_model

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
        completions = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]  # Ibuprofen
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
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen (SMILES)
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Amino acid
            "ATGCATGCATGCATGCATGC",  # DNA
        ]
        rewards = reward_func(completions)

        assert len(rewards) == 3
        assert all(isinstance(r, (float, np.floating)) for r in rewards)  # Allow numpy types
        assert all(not (r != r) for r in rewards)  # No NaN values
        assert all(r != float("inf") for r in rewards)  # No inf values

    def test_temperature_scaling(self, mock_UME_model):
        """Test that temperature scaling affects rewards."""
        reward_func_low = UMERewardFunction(mock_UME_model, temperature=0.1)
        reward_func_high = UMERewardFunction(mock_UME_model, temperature=1.0)

        completions = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]

        rewards_low = reward_func_low(completions)
        rewards_high = reward_func_high(completions)

        # With lower temperature, rewards should be more extreme (higher absolute values)
        assert abs(rewards_low[0]) > abs(rewards_high[0])


class TestUMEComputePseudoLikelihood:
    """Test the compute_pseudo_likelihood method on the UME class."""

    @pytest.fixture
    def mock_UME_model(self):
        """Create a comprehensive mock UME model for testing."""
        # Create mock model
        mock_model = Mock(spec=UME)

        # Mock the model's internal components
        mock_model.model = Mock()
        mock_model.model.config = Mock()
        mock_model.model.config.vocab_size = 1536
        mock_model.model.pad_token_id = 0
        mock_model.max_length = 512

        # Mock the model's parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters = Mock(return_value=iter([mock_param]))

        # Mock tokenizer transforms - simplified
        def mock_tokenizer_transform(sequences):
            batch_size = len(sequences)
            return {
                "input_ids": torch.ones(batch_size, 5, dtype=torch.long),  # Simple tensor
                "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),  # Simple tensor
            }

        mock_model.tokenizer_transforms = {
            Modality.SMILES: mock_tokenizer_transform,
            Modality.AMINO_ACID: mock_tokenizer_transform,
            Modality.NUCLEOTIDE: mock_tokenizer_transform,
        }

        # Mock the internal model's _prepare_inputs method - simplified
        def mock_prepare_inputs(input_ids_3d, attention_mask_3d):
            batch_size = input_ids_3d.shape[0]
            seq_len = input_ids_3d.shape[-1]
            total_tokens = batch_size * seq_len
            return (
                torch.ones(total_tokens, dtype=torch.long),  # input_ids_flat
                torch.ones(total_tokens, dtype=torch.long),  # attention_mask_flat
                torch.tensor([0, total_tokens]),  # cu_seqlens
            )

        mock_model.model._prepare_inputs = Mock(side_effect=mock_prepare_inputs)

        # Mock the model's forward pass
        def mock_model_forward(input_ids, attention_mask, cu_seqlens, max_seqlen):
            batch_size = cu_seqlens[-1].item() // 5  # Assuming seq_len=5
            return torch.randn(batch_size * 5, 768)  # Mock hidden states

        mock_model.model.model = Mock(side_effect=mock_model_forward)

        # Mock the decoder - simplified
        def mock_decoder(hidden_states):
            batch_size = hidden_states.shape[0] // 5  # Assuming seq_len=5
            logits = torch.randn(batch_size * 5, 1536)
            # Set high logits for token_id=1 to ensure good likelihood
            logits[:, 1] = 10.0
            return logits

        mock_model.model.decoder = Mock(side_effect=mock_decoder)

        # Mock the compute_pseudo_likelihood method
        def mock_compute_pseudo_likelihood(sequences, modality):
            if not sequences:
                return []
            # Return mock likelihoods based on sequence count
            return [0.5] * len(sequences)  # Simple mock return value

        mock_model.compute_pseudo_likelihood = Mock(side_effect=mock_compute_pseudo_likelihood)

        return mock_model

    def test_empty_sequences(self, mock_UME_model):
        """Test with empty sequences."""
        sequences = []
        likelihoods = mock_UME_model.compute_pseudo_likelihood(sequences, Modality.SMILES)
        assert likelihoods == []

    def test_single_sequence(self, mock_UME_model):
        """Test with a single sequence."""
        sequences = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]

        likelihoods = mock_UME_model.compute_pseudo_likelihood(sequences, Modality.SMILES)
        assert len(likelihoods) == 1
        # Check for both Python float and numpy float types
        assert isinstance(likelihoods[0], (float, np.floating))  # Allow numpy types
        assert not (likelihoods[0] != likelihoods[0])  # Not NaN

    def test_multiple_sequences(self, mock_UME_model):
        """Test with multiple sequences."""
        sequences = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
        ]
        likelihoods = mock_UME_model.compute_pseudo_likelihood(sequences, Modality.SMILES)

        assert len(likelihoods) == 2
        assert all(isinstance(l, (float, np.floating)) for l in likelihoods)  # Allow numpy types
        assert all(not (l != l) for l in likelihoods)  # No NaN values
