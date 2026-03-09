"""Streaming dataset for protein sequences optimized with LitData.

Reads LitData-optimized sequence data (from S3 or local storage) and tokenizes
each sequence for causal language modeling with ``LobsterPCLM``.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Callable
from typing import Any

from lobster.datasets._tokenized_streaming_dataset import TokenizedStreamingDataset
from lobster.tokenization import PmlmTokenizerTransform


class StreamingSequenceDataset(TokenizedStreamingDataset):
    """LitData ``StreamingDataset`` that tokenizes protein sequences for CLM training.

    Each item stored in the optimized LitData directory is expected to be a dict
    containing at least a ``sequence`` key whose value is a raw protein sequence
    string.  On iteration the sequence is tokenized with ``PmlmTokenizerTransform``
    (``mlm=False``) and returned as a dict with ``input_ids``, ``labels``, and
    ``attention_mask`` tensors compatible with ``LobsterPCLM``.

    Parameters
    ----------
    input_dir : str
        Path to a LitData-optimized directory (S3 URI or local path).
    max_length : int, optional
        Maximum sequence length for tokenization. Default is 512.
    tokenizer_dir : str, optional
        Name of the tokenizer asset directory under ``lobster/assets/``.
        Default is ``"pmlm_tokenizer"``.
    shuffle : bool, optional
        Whether to shuffle the data. Default is ``True``.
    seed : int, optional
        Random seed for shuffling reproducibility. Default is 0.
    cache_dir : str or None, optional
        Local cache directory for downloaded chunks. Default is ``None``.
    transform_fn : callable or None, optional
        Optional function applied to the raw sequence string before
        tokenization (e.g. filtering, sanitization). Default is ``None``.
    sequence_key : str, optional
        Key used to look up the sequence in each stored item dict.
        Default is ``"sequence"``.
    drop_last : bool, optional
        Whether to drop the last incomplete batch. Default is ``True``.
    """

    def __init__(
        self,
        input_dir: str,
        *,
        max_length: int = 512,
        tokenizer_dir: str = "pmlm_tokenizer",
        shuffle: bool = True,
        seed: int = 0,
        cache_dir: str | None = None,
        transform_fn: Callable[[str], str | None] | None = None,
        sequence_key: str = "sequence",
        drop_last: bool = True,
    ) -> None:
        super().__init__(
            input_dir,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            cache_dir=cache_dir,
            transform_fn=transform_fn,
        )

        self.sequence_key = sequence_key
        self.max_length = max_length

        path = importlib.resources.files("lobster") / "assets" / tokenizer_dir
        self._tokenizer_transform = PmlmTokenizerTransform(
            path,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            mlm=False,
        )

    def _extract_sequence(self, item: dict[str, Any]) -> str:
        return item.get(self.sequence_key, "")

    def _should_skip_sequence(self, sequence: Any) -> bool:
        return not sequence

    def _encode_sequence(self, sequence: str) -> dict[str, Any]:
        return self._tokenizer_transform(sequence)

    def _build_output(self, *, encoded: dict[str, Any], sequence: str, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "attention_mask": encoded["attention_mask"],
            "sequence": sequence,
        }
