from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from litdata import StreamingDataset

logger = logging.getLogger(__name__)


class TokenizedStreamingDataset(StreamingDataset):
    """Shared iteration lifecycle for streaming datasets that tokenize sequences."""

    def __init__(
        self,
        input_dir: str,
        *,
        shuffle: bool,
        seed: int,
        cache_dir: str | None,
        drop_last: bool,
        transform_fn: Callable | None = None,
        extra_transform_fns: dict[str, Callable] | None = None,
        **streaming_kwargs: Any,
    ) -> None:
        super().__init__(
            input_dir,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            cache_dir=cache_dir,
            **streaming_kwargs,
        )
        self.transform_fn = transform_fn
        self.extra_transform_fns = extra_transform_fns

    def __next__(self) -> dict[str, Any]:
        while True:
            item = self._get_streaming_item()
            sequence = self._extract_sequence(item)

            if self._should_skip_sequence(sequence):
                logger.warning(f"Invalid sequence encountered in {self.__class__.__name__}, skipping.")
                continue

            if self.transform_fn is not None:
                sequence = self.transform_fn(sequence)
                if self._should_skip_sequence(sequence):
                    logger.warning(f"Transform returned invalid sequence in {self.__class__.__name__}, skipping.")
                    continue

            extra_outputs = self._run_extra_transforms(sequence)
            if extra_outputs is None:
                continue

            encoded = self._encode_sequence(sequence)
            return self._build_output(encoded=encoded, sequence=sequence, item={**item, **extra_outputs})

    def _get_streaming_item(self) -> dict[str, Any]:
        return super().__next__()

    def _extract_sequence(self, item: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _should_skip_sequence(self, sequence: Any) -> bool:
        if sequence is None:
            return True
        return isinstance(sequence, tuple | list) and any(value is None for value in sequence)

    def _run_extra_transforms(self, sequence: Any) -> dict[str, Any] | None:
        if self.extra_transform_fns is None:
            return {}

        outputs: dict[str, Any] = {}
        for key, transform_fn in self.extra_transform_fns.items():
            transformed = transform_fn(sequence)
            if transformed is None:
                logger.warning(
                    f"Extra transform function {key} returned None for input `{sequence}`. Skipping this item."
                )
                return None
            outputs[key] = transformed
        return outputs

    def _encode_sequence(self, sequence: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _build_output(self, *, encoded: dict[str, Any], sequence: Any, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
