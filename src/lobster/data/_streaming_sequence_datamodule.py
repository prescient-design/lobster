"""Lightning DataModule for streaming protein sequences from LitData-optimized storage.

Designed to work with ``LobsterPCLM`` for causal language model training on
large-scale protein sequence datasets (e.g. OAS) stored in S3.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.utils.data
from lightning import LightningDataModule
from litdata import StreamingDataLoader
from torch import Tensor

from lobster.datasets._streaming_sequence_dataset import StreamingSequenceDataset

logger = logging.getLogger(__name__)


def _clm_collate_fn(batch: list[dict[str, Any]]) -> tuple[dict[str, Tensor], list[str]]:
    """Collate streaming sequence items into the format expected by ``LobsterPCLM``.

    ``LobsterPCLM._compute_loss`` expects ``(batch_dict, targets)`` where
    ``batch_dict`` contains ``input_ids``, ``labels``, and ``attention_mask`` tensors,
    and ``targets`` is a list of strings (unused FASTA headers — here just raw
    sequences for reference).

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of sample dicts from ``StreamingSequenceDataset``, each containing
        ``input_ids``, ``labels``, ``attention_mask``, and ``sequence``.

    Returns
    -------
    tuple[dict[str, Tensor], list[str]]
        A 2-tuple of ``(tensor_dict, sequence_strings)`` matching the interface
        expected by ``LobsterPCLM._compute_loss``.
    """
    sequences = [item["sequence"] for item in batch]

    tensor_dict = {
        "input_ids": torch.stack([item["input_ids"].squeeze(0) for item in batch]),
        "labels": torch.stack([item["labels"].squeeze(0) for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"].squeeze(0) for item in batch]),
    }

    return tensor_dict, sequences


class StreamingSequenceLightningDataModule(LightningDataModule):
    """Lightning DataModule for streaming LitData-optimized protein sequences.

    Loads train and (optionally) validation data from LitData-optimized
    directories, tokenizes sequences for causal language modeling, and returns
    batches in the format expected by ``LobsterPCLM``.

    Parameters
    ----------
    train_input_dir : str
        S3 URI or local path to the LitData-optimized training dataset.
    val_input_dir : str or None, optional
        S3 URI or local path to the LitData-optimized validation dataset.
        If ``None``, no validation dataloader is created. Default is ``None``.
    max_length : int, optional
        Maximum sequence length for tokenization. Default is 512.
    tokenizer_dir : str, optional
        Name of the tokenizer asset directory under ``lobster/assets/``.
        Default is ``"pmlm_tokenizer"``.
    batch_size : int, optional
        Number of samples per batch. Default is 32.
    num_workers : int, optional
        Number of dataloader worker processes. Default is 4.
    pin_memory : bool, optional
        Whether to pin memory for faster GPU transfer. Default is ``True``.
    seed : int, optional
        Random seed for shuffling. Default is 0.
    cache_dir : str or None, optional
        Local cache directory for downloaded data chunks. Default is ``None``.
    drop_last : bool, optional
        Whether to drop the last incomplete batch. Default is ``True``.
    transform_fn : callable or None, optional
        Optional function applied to raw sequence strings before tokenization.
        Default is ``None``.
    sequence_key : str, optional
        Key used to look up the sequence in each stored item dict.
        Default is ``"sequence"``.

    Examples
    --------
    .. code-block:: python

        from lobster.data import StreamingSequenceLightningDataModule
        from lobster.model import LobsterPCLM

        datamodule = StreamingSequenceLightningDataModule(
            train_input_dir="s3://my-bucket/oas/optimized/train",
            val_input_dir="s3://my-bucket/oas/optimized/val",
            max_length=512,
            batch_size=64,
        )

        model = LobsterPCLM(model_name="CLM_mini", max_length=512)
        trainer.fit(model, datamodule=datamodule)
    """

    def __init__(
        self,
        train_input_dir: str,
        val_input_dir: str | None = None,
        *,
        max_length: int = 512,
        tokenizer_dir: str = "pmlm_tokenizer",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 0,
        cache_dir: str | None = None,
        drop_last: bool = True,
        transform_fn: Callable[[str], str | None] | None = None,
        sequence_key: str = "sequence",
    ) -> None:
        super().__init__()

        self._train_input_dir = train_input_dir
        self._val_input_dir = val_input_dir
        self._max_length = max_length
        self._tokenizer_dir = tokenizer_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._seed = seed
        self._cache_dir = cache_dir
        self._drop_last = drop_last
        self._transform_fn = transform_fn
        self._sequence_key = sequence_key

        self._train_dataset: StreamingSequenceDataset | None = None
        self._val_dataset: StreamingSequenceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create streaming datasets for the requested stage.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage (``"fit"``, ``"validate"``, ``"test"``, ``"predict"``).
        """
        if stage in ("fit", None):
            self._train_dataset = StreamingSequenceDataset(
                input_dir=self._train_input_dir,
                max_length=self._max_length,
                tokenizer_dir=self._tokenizer_dir,
                shuffle=True,
                seed=self._seed,
                cache_dir=self._cache_dir,
                transform_fn=self._transform_fn,
                sequence_key=self._sequence_key,
                drop_last=self._drop_last,
            )
            logger.info(
                f"Train streaming dataset initialized: {self._train_input_dir} "
                f"({len(self._train_dataset)} samples)"
            )

            if self._val_input_dir is not None:
                self._val_dataset = StreamingSequenceDataset(
                    input_dir=self._val_input_dir,
                    max_length=self._max_length,
                    tokenizer_dir=self._tokenizer_dir,
                    shuffle=False,
                    seed=self._seed,
                    cache_dir=self._cache_dir,
                    transform_fn=self._transform_fn,
                    sequence_key=self._sequence_key,
                    drop_last=self._drop_last,
                )
                logger.info(
                    f"Val streaming dataset initialized: {self._val_input_dir} "
                    f"({len(self._val_dataset)} samples)"
                )

        if stage == "validate" and self._val_input_dir is not None:
            if self._val_dataset is None:
                self._val_dataset = StreamingSequenceDataset(
                    input_dir=self._val_input_dir,
                    max_length=self._max_length,
                    tokenizer_dir=self._tokenizer_dir,
                    shuffle=False,
                    seed=self._seed,
                    cache_dir=self._cache_dir,
                    transform_fn=self._transform_fn,
                    sequence_key=self._sequence_key,
                    drop_last=self._drop_last,
                )

    def train_dataloader(self) -> StreamingDataLoader:
        """Return the training dataloader.

        Returns
        -------
        StreamingDataLoader
            Dataloader yielding batches in ``(dict, list[str])`` format.

        Raises
        ------
        RuntimeError
            If ``setup("fit")`` has not been called.
        """
        if self._train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup('fit') first.")

        return StreamingDataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=_clm_collate_fn,
            drop_last=self._drop_last,
        )

    def val_dataloader(self) -> StreamingDataLoader | None:
        """Return the validation dataloader, or ``None`` if no val dir was provided.

        Returns
        -------
        StreamingDataLoader or None
            Dataloader yielding batches in ``(dict, list[str])`` format,
            or ``None`` if ``val_input_dir`` was not specified.
        """
        if self._val_dataset is None:
            return None

        return StreamingDataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=_clm_collate_fn,
            drop_last=self._drop_last,
        )
