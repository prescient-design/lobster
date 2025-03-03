import time
from typing import Any, Callable, Dict, Union

import lightning as L
from lightning.pytorch import LightningModule, Trainer
from torch import Tensor


class TokensPerSecondCallback(L.Callback):
    """
    Lightning callback that measures tokens per second during training
    and logs the metric to Weights & Biases.

    Parameters:
    -----------
    log_interval_steps : int
        How often to log the tokens per second metric (in steps).
        Default is every 100 steps.
    batch_size_fn : Callable
        Function to extract batch size from the batch.
    batch_length_fn : Callable
        Function to extract sequence length from the batch.
    """

    def __init__(
        self,
        log_interval_steps: int = 500,
        batch_size_fn: Union[Callable[[Any], int], None] = None,
        batch_length_fn: Union[Callable[[Any], int], None] = None,
    ):
        super().__init__()
        self.log_interval_steps = log_interval_steps
        self.tokens_processed = 0
        self.start_time = None
        self.last_logged_step = 0
        self.batch_size_fn = batch_size_fn or default_batch_size_fn
        self.length_fn = batch_length_fn or default_batch_length_fn

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training begins."""
        self.start_time = time.time()
        self.tokens_processed = 0
        self.last_logged_step = 0

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        """Called after each training batch ends."""
        # Calculate tokens in this batch using the provided functions
        batch_size = self.batch_size_fn(batch)
        batch_length = self.length_fn(batch)

        # Calculate total tokens in batch (batch_size × sequence_length)
        batch_tokens = batch_size * batch_length

        # Add tokens to running total
        self.tokens_processed += batch_tokens

        # Log tokens per second at specified intervals
        if (batch_idx + 1) % self.log_interval_steps == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time

            if elapsed_time > 0:
                tokens_per_sec = self.tokens_processed / elapsed_time

                # Log to wandb
                trainer.logger.experiment.log(
                    {
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/total_tokens": self.tokens_processed,
                        "train/elapsed_time": elapsed_time,
                    }
                )

                # Optionally print to console
                print(f"Step {batch_idx + 1}: {tokens_per_sec:.2f} tokens/sec")

            # Save the last logged step
            self.last_logged_step = batch_idx

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a training epoch."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time > 0:
            tokens_per_sec = self.tokens_processed / elapsed_time

            # Log to wandb
            trainer.logger.experiment.log(
                {
                    "train/epoch_tokens_per_sec": tokens_per_sec,
                    "train/epoch_total_tokens": self.tokens_processed,
                    "train/epoch_elapsed_time": elapsed_time,
                }
            )


def default_batch_size_fn(batch: dict[str, Tensor]) -> int:
    """Default batch size function that returns the batch size."""
    x = batch["input_ids"].squeeze(1)

    return x.shape[0]


def default_batch_length_fn(batch: dict[str, Tensor]) -> int:
    """Default length function that returns the length of the batch."""
    x = batch["input_ids"].squeeze(1)

    return x.shape[0] * x.shape[1]
