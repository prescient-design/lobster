# Used for https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ThroughputMonitor.html
from typing import Any


def throughput_batch_size_fn(batch: tuple | dict[str, Any]) -> int:
    if isinstance(batch, tuple) and len(batch) == 2:
        batch, _ = batch

    return batch["input_ids"].shape[0]
