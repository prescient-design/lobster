import logging
import time
from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

logger = logging.getLogger(__name__)


class TokensPerSecondCallback(L.Callback):
    """
    Lightning callback that measures tokens per second during training
    and logs the metric to Weights & Biases.

    Supports multi-node and multi-GPU training by aggregating token counts
    across all processes.

    Parameters:
    -----------
    log_interval_steps : int
        How often to log the tokens per second metric (in steps).
        Default is every 500 steps.
    batch_size_fn : Callable
        Function to extract batch size from the batch.
    batch_length_fn : Callable
        Function to extract sequence length from the batch.
    """

    def __init__(
        self,
        log_interval_steps: int = 500,
        batch_size_fn: Callable[[Any], int] | None = None,
        batch_length_fn: Callable[[Any], int] | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.log_interval_steps = log_interval_steps
        self.tokens_processed = 0
        self.start_time = None
        self.last_logged_step = 0
        self.batch_size_fn = batch_size_fn or default_batch_size_fn
        self.length_fn = batch_length_fn or default_batch_length_fn
        self.verbose = verbose

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_time = time.time()
        self.tokens_processed = 0
        self.last_logged_step = 0

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        """Called after each training batch ends."""
        batch_size = self.batch_size_fn(batch)
        batch_length = self.length_fn(batch)

        # Calculate total tokens in batch (batch_size × sequence_length)
        batch_tokens = batch_size * batch_length

        self.tokens_processed += batch_tokens

        # Log tokens per second at specified intervals
        if (batch_idx + 1) % self.log_interval_steps == 0:
            self._log_tokens_per_second(trainer, batch_idx)

            self.last_logged_step = batch_idx

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_tokens_per_second(trainer, None, epoch_end=True)

    def _log_tokens_per_second(self, trainer: Trainer, batch_idx=None, epoch_end=False):
        """
        Logs tokens per second after aggregating across all processes.
        Only rank 0 will log the final metrics.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        tokens = torch.tensor(self.tokens_processed, device=trainer.lightning_module.device)

        # All-reduce to sum token counts across all processes
        if trainer.world_size > 1:
            torch.distributed.all_reduce(tokens, op=torch.distributed.ReduceOp.SUM)

        global_tokens = tokens.item()

        if elapsed_time > 0:
            tokens_per_sec = global_tokens / elapsed_time

            # Only log on rank 0 to avoid duplicate logging
            self._log_metrics(trainer, tokens_per_sec, global_tokens, elapsed_time, batch_idx, epoch_end)

    @rank_zero_only
    def _log_metrics(self, trainer, tokens_per_sec, total_tokens, elapsed_time, batch_idx=None, epoch_end=False):
        """Log metrics to logger and console (rank 0 only)."""
        if epoch_end:
            # Log epoch metrics
            trainer.logger.experiment.log(
                {
                    "train/epoch_tokens_per_sec": tokens_per_sec,
                    "train/epoch_total_tokens": total_tokens,
                    "train/epoch_elapsed_time": elapsed_time,
                    "train/global_step": trainer.global_step,
                }
            )
            if self.verbose:
                logger.info(
                    f"Epoch {trainer.current_epoch}: {tokens_per_sec:.2f} tokens/sec across {trainer.world_size} process(es)"
                )
        else:
            # Log step metrics
            trainer.logger.experiment.log(
                {
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/total_tokens": total_tokens,
                    "train/elapsed_time": elapsed_time,
                    "train/global_step": trainer.global_step,
                }
            )
            if self.verbose:
                logger.info(
                    f"Step {batch_idx + 1}: {tokens_per_sec:.2f} tokens/sec across {trainer.world_size} process(es)"
                )


def default_batch_size_fn(batch: dict[str, Tensor]) -> int:
    """Default batch size function that returns the batch size."""
    x = batch["input_ids"].squeeze(1)
    return x.shape[0]


def default_batch_length_fn(batch: dict[str, Tensor]) -> int:
    """Default length function that returns the length of the batch."""
    x = batch["input_ids"].squeeze(1)
    return x.shape[0] * x.shape[1]
