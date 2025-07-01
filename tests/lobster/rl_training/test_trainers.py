"""
Tests for trainers in the rl_training module.
"""

import logging
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from lobster.model import UME
from lobster.rl_training.trainers import create_ume_grpo_trainer, train_ume_grpo

logger = logging.getLogger(__name__)


class TestCreateUmeGrpoTrainer:
    """Test the create_ume_grpo_trainer function."""

    @pytest.fixture
    def mock_ume_model(self):
        """Create a mock UME model."""
        model = Mock(spec=UME)
        model.eval.return_value = None
        model.freeze.return_value = None
        return model

    @pytest.fixture
    def mock_train_dataset(self):
        """Create a mock training dataset."""
        return Mock(spec=Dataset)

    @pytest.fixture
    def mock_eval_dataset(self):
        """Create a mock evaluation dataset."""
        return Mock(spec=Dataset)

    @patch("lobster.rl_training.trainers.wandb")
    @patch("lobster.rl_training.trainers.create_ume_reward_wrapper")
    @patch("lobster.rl_training.trainers.GRPOConfig")
    @patch("lobster.rl_training.trainers.GRPOTrainer")
    def test_create_trainer_basic(
        self,
        mock_trainer_class,
        mock_config_class,
        mock_reward_wrapper,
        mock_wandb,
        mock_ume_model,
        mock_train_dataset,
        mock_eval_dataset,
    ):
        """Test basic trainer creation."""
        # Setup mocks
        mock_wandb.run = None  # Simulate no active wandb session
        mock_reward_func = Mock()
        mock_reward_wrapper.return_value = mock_reward_func
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Call function
        trainer = create_ume_grpo_trainer(
            model_path="test/model",
            ume_model=mock_ume_model,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            output_dir="./test_runs",
            reward_temperature=0.2,
            reward_batch_size=16,
        )

        # Verify reward wrapper was called correctly
        mock_reward_wrapper.assert_called_once_with(
            mock_ume_model, temperature=0.2, batch_size=16, penalty_for_invalid=-5.0, enable_wandb_logging=True
        )

        # Verify config was created correctly
        mock_config_class.assert_called_once_with(output_dir="./test_runs", run_name=None)

        # Verify trainer was created correctly
        mock_trainer_class.assert_called_once_with(
            model="test/model",
            reward_funcs=mock_reward_func,
            args=mock_config,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
        )

        assert trainer == mock_trainer

    @patch("lobster.rl_training.trainers.wandb")
    @patch("lobster.rl_training.trainers.create_ume_reward_wrapper")
    @patch("lobster.rl_training.trainers.GRPOConfig")
    @patch("lobster.rl_training.trainers.GRPOTrainer")
    def test_create_trainer_without_eval_dataset(
        self, mock_trainer_class, mock_config_class, mock_reward_wrapper, mock_wandb, mock_ume_model, mock_train_dataset
    ):
        """Test trainer creation without evaluation dataset."""
        # Setup mocks
        mock_wandb.run = None  # Simulate no active wandb session
        mock_reward_func = Mock()
        mock_reward_wrapper.return_value = mock_reward_func
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Call function
        trainer = create_ume_grpo_trainer(
            model_path="test/model",
            ume_model=mock_ume_model,
            train_dataset=mock_train_dataset,
            eval_dataset=None,
        )

        # Verify trainer was created with None eval_dataset
        mock_trainer_class.assert_called_once_with(
            model="test/model",
            reward_funcs=mock_reward_func,
            args=mock_config,
            train_dataset=mock_train_dataset,
            eval_dataset=None,
        )

        assert trainer == mock_trainer

    @patch("lobster.rl_training.trainers.wandb")
    @patch("lobster.rl_training.trainers.create_ume_reward_wrapper")
    @patch("lobster.rl_training.trainers.GRPOConfig")
    @patch("lobster.rl_training.trainers.GRPOTrainer")
    def test_create_trainer_with_additional_kwargs(
        self, mock_trainer_class, mock_config_class, mock_reward_wrapper, mock_wandb, mock_ume_model, mock_train_dataset
    ):
        """Test trainer creation with additional GRPO kwargs."""
        # Setup mocks
        mock_wandb.run = None  # Simulate no active wandb session
        mock_reward_func = Mock()
        mock_reward_wrapper.return_value = mock_reward_func
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Call function with additional kwargs
        trainer = create_ume_grpo_trainer(
            model_path="test/model",
            ume_model=mock_ume_model,
            train_dataset=mock_train_dataset,
            learning_rate=1e-5,
            batch_size=4,
            max_steps=1000,
        )

        # Verify config was created with additional kwargs
        mock_config_class.assert_called_once_with(
            output_dir="./ume_grpo_runs",
            run_name=None,
            learning_rate=1e-5,
            batch_size=4,
            max_steps=1000,
        )

        assert trainer == mock_trainer


class TestTrainUmeGrpo:
    """Test the train_ume_grpo function."""

    @pytest.fixture
    def mock_ume_model(self):
        """Create a mock UME model."""
        model = Mock(spec=UME)
        model.eval.return_value = None
        model.freeze.return_value = None
        return model

    @pytest.fixture
    def mock_train_dataset(self):
        """Create a mock training dataset."""
        return Mock(spec=Dataset)

    @patch("lobster.rl_training.trainers.UME.from_pretrained")
    @patch("lobster.rl_training.trainers.create_ume_grpo_trainer")
    def test_train_ume_grpo_basic(self, mock_create_trainer, mock_from_pretrained, mock_ume_model, mock_train_dataset):
        """Test basic training pipeline."""
        # Setup mocks
        mock_from_pretrained.return_value = mock_ume_model
        mock_trainer = Mock()
        mock_trainer.train.return_value = None
        mock_create_trainer.return_value = mock_trainer

        # Call function
        result = train_ume_grpo(
            model_path="test/base/model",
            ume_model_path="test/ume/model",
            train_dataset=mock_train_dataset,
            eval_dataset=None,
            output_dir="./test_runs",
            reward_temperature=0.3,
            reward_batch_size=32,
            device="cpu",
        )

        # Verify UME model was loaded correctly
        mock_from_pretrained.assert_called_once_with("test/ume/model", device="cpu")

        # Verify trainer was created correctly
        mock_create_trainer.assert_called_once_with(
            model_path="test/base/model",
            ume_model=mock_ume_model,
            train_dataset=mock_train_dataset,
            eval_dataset=None,
            output_dir="./test_runs",
            reward_temperature=0.3,
            reward_batch_size=32,
            penalty_for_invalid=-5.0,
            enable_wandb_logging=True,
            wandb_project="lobster-ume-grpo",
            wandb_entity=None,
            wandb_run_name=None,
        )

        # Verify training was called
        mock_trainer.train.assert_called_once()

        # Verify result
        assert result == mock_trainer

    @patch("lobster.rl_training.trainers.UME.from_pretrained")
    @patch("lobster.rl_training.trainers.create_ume_grpo_trainer")
    def test_train_ume_grpo_with_eval_dataset(
        self, mock_create_trainer, mock_from_pretrained, mock_ume_model, mock_train_dataset
    ):
        """Test training pipeline with evaluation dataset."""
        # Setup mocks
        mock_from_pretrained.return_value = mock_ume_model
        mock_trainer = Mock()
        mock_trainer.train.return_value = None
        mock_create_trainer.return_value = mock_trainer

        mock_eval_dataset = Mock(spec=Dataset)

        # Call function
        result = train_ume_grpo(
            model_path="test/base/model",
            ume_model_path="test/ume/model",
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            device="cuda",
        )

        # Verify trainer was created with eval dataset
        mock_create_trainer.assert_called_once_with(
            model_path="test/base/model",
            ume_model=mock_ume_model,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            output_dir="./ume_grpo_runs",
            reward_temperature=0.1,
            reward_batch_size=8,
            penalty_for_invalid=-5.0,
            enable_wandb_logging=True,
            wandb_project="lobster-ume-grpo",
            wandb_entity=None,
            wandb_run_name=None,
        )

        assert result == mock_trainer
