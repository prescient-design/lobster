import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer

# Import the class to test
from lobster.callbacks import TokensPerSecondCallback, default_batch_length_fn, default_batch_size_fn


class TestTokensPerSecondCallback:
    @pytest.fixture
    def callback(self):
        """Create a basic callback instance for testing."""
        return TokensPerSecondCallback()

    @pytest.fixture
    def pl_module_mock(self):
        """Create a mock LightningModule."""
        module = MagicMock(spec=LightningModule)
        module.device = "cpu"
        return module

    @pytest.fixture
    def trainer_mock(self, pl_module_mock):
        """Create a mock Trainer object."""
        trainer = MagicMock(spec=Trainer)
        trainer.logger = MagicMock()
        trainer.logger.experiment = MagicMock()
        trainer.logger.experiment.log = MagicMock()
        trainer.world_size = 1
        trainer.lightning_module = pl_module_mock

        return trainer

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch with input_ids."""
        return {
            "input_ids": torch.ones((8, 1, 128), dtype=torch.long)  # (batch_size, num_samples, seq_len)
        }

    def test_init(self):
        """Test the initialization of the callback."""
        # Test with default parameters
        callback = TokensPerSecondCallback()
        assert callback.log_interval_steps == 500
        assert callback.tokens_processed == 0
        assert callback.start_time is None
        assert callback.last_logged_step == 0
        assert callback.batch_size_fn == default_batch_size_fn
        assert callback.length_fn == default_batch_length_fn

        # Test with custom parameters
        custom_batch_size_fn = lambda x: len(x)  # noqa
        custom_batch_length_fn = lambda x: sum(len(i) for i in x)  # noqa

        callback = TokensPerSecondCallback(
            log_interval_steps=100, batch_size_fn=custom_batch_size_fn, batch_length_fn=custom_batch_length_fn
        )

        assert callback.log_interval_steps == 100
        assert callback.batch_size_fn == custom_batch_size_fn
        assert callback.length_fn == custom_batch_length_fn

    def test_on_train_start(self, callback, trainer_mock, pl_module_mock):
        """Test on_train_start method."""
        with patch("time.time", return_value=12345.0):
            callback.on_train_start(trainer_mock, pl_module_mock)
            assert callback.start_time == 12345.0
            assert callback.tokens_processed == 0
            assert callback.last_logged_step == 0

    def test_on_train_batch_end_no_log(self, callback, trainer_mock, pl_module_mock, sample_batch):
        """Test on_train_batch_end when not logging (not at interval)."""
        callback.start_time = time.time()

        # Mock batch size and length functions
        callback.batch_size_fn = MagicMock(return_value=8)
        callback.length_fn = MagicMock(return_value=1024)  # 8*128 = 1024

        # Call the method with batch_idx not at log interval
        callback.on_train_batch_end(trainer_mock, pl_module_mock, {}, sample_batch, batch_idx=10)

        # Verify tokens were processed but no logging happened
        assert callback.tokens_processed == 8 * 1024
        assert callback.last_logged_step == 0
        trainer_mock.logger.experiment.log.assert_not_called()

    def test_on_train_batch_end_with_log(self, callback, trainer_mock, pl_module_mock, sample_batch):
        """Test on_train_batch_end when logging (at interval)."""
        # Set start time in the past
        callback.start_time = time.time() - 10  # 10 seconds ago

        # Mock batch size and length functions
        callback.batch_size_fn = MagicMock(return_value=8)
        callback.length_fn = MagicMock(return_value=1024)

        # Set log interval to match our batch_idx + 1
        batch_idx = 499
        callback.log_interval_steps = 500

        # Call the method with batch_idx at log interval
        callback.on_train_batch_end(trainer_mock, pl_module_mock, {}, sample_batch, batch_idx=batch_idx)

        # Verify tokens were processed and logging happened
        assert callback.tokens_processed == 8 * 1024
        assert callback.last_logged_step == batch_idx

        # Check that logger.experiment.log was called with the right arguments
        trainer_mock.logger.experiment.log.assert_called_once()
        log_args = trainer_mock.logger.experiment.log.call_args[0][0]

        assert "train/tokens_per_sec" in log_args
        assert "train/total_tokens" in log_args
        assert "train/elapsed_time" in log_args
        assert log_args["train/total_tokens"] == 8 * 1024

        # The tokens per second should be approximately 8*1024/10 = 819.2
        assert 750 < log_args["train/tokens_per_sec"] < 900

    def test_on_train_epoch_end(self, callback, trainer_mock, pl_module_mock):
        """Test on_train_epoch_end method."""
        # Set start time in the past and process some tokens
        callback.start_time = time.time() - 5  # 5 seconds ago
        callback.tokens_processed = 10000

        # Call the method
        callback.on_train_epoch_end(trainer_mock, pl_module_mock)

        # Check that logger.experiment.log was called with the right arguments
        trainer_mock.logger.experiment.log.assert_called_once()
        log_args = trainer_mock.logger.experiment.log.call_args[0][0]

        assert "train/epoch_tokens_per_sec" in log_args
        assert "train/epoch_total_tokens" in log_args
        assert "train/epoch_elapsed_time" in log_args
        assert log_args["train/epoch_total_tokens"] == 10000

        # The tokens per second should be approximately 10000/5 = 2000
        assert 1800 < log_args["train/epoch_tokens_per_sec"] < 2200

    def test_default_batch_size_fn(self, sample_batch):
        """Test the default_batch_size_fn function."""
        batch_size = default_batch_size_fn(sample_batch)
        assert batch_size == 8

    def test_default_batch_length_fn(self, sample_batch):
        """Test the default_batch_length_fn function."""
        batch_length = default_batch_length_fn(sample_batch)
        assert batch_length == 8 * 128  # batch_size * seq_len

    def test_custom_batch_fns(self, trainer_mock, pl_module_mock):
        """Test callback with custom batch size and length functions."""

        # Create custom functions
        def custom_batch_size_fn(batch):
            return batch["custom_size"]

        def custom_batch_length_fn(batch):
            return batch["custom_length"]

        # Create callback with custom functions
        callback = TokensPerSecondCallback(
            log_interval_steps=1, batch_size_fn=custom_batch_size_fn, batch_length_fn=custom_batch_length_fn
        )

        # Initialize callback
        callback.start_time = time.time()

        # Custom batch
        batch = {"custom_size": 16, "custom_length": 64}

        # Call batch end method
        callback.on_train_batch_end(trainer_mock, pl_module_mock, {}, batch, batch_idx=0)

        # Verify the correct number of tokens was calculated
        assert callback.tokens_processed == 16 * 64
