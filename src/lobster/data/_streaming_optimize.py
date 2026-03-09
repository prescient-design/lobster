from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from litdata import StreamingDataset, optimize
from upath import UPath

logger = logging.getLogger(__name__)

_HASH_MODULUS = 10_000


def sequence_is_val(sequence: str, val_fraction: float, seed: int) -> bool:
    """Deterministically assign a sequence to the validation split."""
    digest = hashlib.md5(f"{seed}:{sequence}".encode(), usedforsecurity=False).digest()
    bucket = int.from_bytes(digest[:4], "little") % _HASH_MODULUS
    return bucket < int(val_fraction * _HASH_MODULUS)


def _get_file_size(filepath: str) -> int:
    try:
        return UPath(filepath).stat().st_size
    except Exception:
        return 0


def sort_files_by_size(files: list[str], num_workers: int = 1) -> list[str]:
    """Sort files by size ascending (smallest first)."""
    if not files:
        return files

    logger.info(f"Sorting {len(files)} files by size (smallest first)...")

    if num_workers <= 1 or len(files) < 10:
        sizes = [(_get_file_size(filepath), filepath) for filepath in files]
    else:
        sizes = []
        with ProcessPoolExecutor(max_workers=min(num_workers, len(files))) as pool:
            future_to_file = {pool.submit(_get_file_size, filepath): filepath for filepath in files}
            for future in as_completed(future_to_file):
                sizes.append((future.result(), future_to_file[future]))

    sizes.sort(key=lambda item: item[0])
    return [filepath for _, filepath in sizes]


class CollectionProgress:
    """Track which input files have been processed for a given split."""

    def __init__(self, progress_dir: str | Path, split_name: str = "all") -> None:
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._split_name = split_name
        self._progress_file = self.progress_dir / f"progress_{split_name}.json"

        self._done_files: set[str] = set()
        self._sequence_count = 0

        if self._progress_file.exists():
            with open(self._progress_file) as f:
                data = json.load(f)
            self._done_files = set(data.get("completed_files", []))
            self._sequence_count = data.get("sequence_count", 0)
            logger.info(
                f"Resuming [{split_name}]: {len(self._done_files)} files already processed, "
                f"{self._sequence_count:,} sequences written"
            )

    @property
    def done_files(self) -> set[str]:
        return self._done_files

    @property
    def sequence_count(self) -> int:
        return self._sequence_count

    def filter_remaining(self, files: list[str]) -> list[str]:
        remaining = [filepath for filepath in files if filepath not in self._done_files]
        if len(remaining) < len(files):
            logger.info(
                f"[{self._split_name}] Skipping {len(files) - len(remaining)} already-processed files, "
                f"{len(remaining)} remaining"
            )
        return remaining

    def record_file(self, filepath: str, num_sequences: int) -> None:
        self._done_files.add(filepath)
        self._sequence_count += num_sequences
        self._save_progress()

    def _save_progress(self) -> None:
        tmp = self._progress_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "completed_files": sorted(self._done_files),
                    "sequence_count": self._sequence_count,
                },
                f,
            )
        tmp.rename(self._progress_file)

    def clear(self) -> None:
        if self._progress_file.exists():
            self._progress_file.unlink()
        try:
            self.progress_dir.rmdir()
        except OSError:
            pass


class SplitOptimizerConverter:
    """Emit only the requested split from a file reader."""

    def __init__(
        self,
        reader: Callable[[str], Iterator[dict[str, str]]],
        *,
        val_fraction: float,
        seed: int,
        emit_val: bool,
        sequence_key: str = "sequence",
    ) -> None:
        self.reader = reader
        self.val_fraction = val_fraction
        self.seed = seed
        self.emit_val = emit_val
        self.sequence_key = sequence_key

    def __call__(self, filepath: str) -> Iterator[dict[str, str]]:
        for item in self.reader(filepath):
            is_val = sequence_is_val(item[self.sequence_key], self.val_fraction, self.seed)
            if is_val == self.emit_val:
                yield item


class PassthroughOptimizerConverter:
    """Emit every item from a file reader."""

    def __init__(self, reader: Callable[[str], Iterator[dict[str, str]]]) -> None:
        self.reader = reader

    def __call__(self, filepath: str) -> Iterator[dict[str, str]]:
        yield from self.reader(filepath)


def run_optimize(
    convert_fn: Callable[[str], Iterator[dict[str, str]]],
    files: list[str],
    output_dir: str,
    chunk_bytes: str,
    num_workers: int,
    progress: CollectionProgress | None,
    split_label: str,
) -> None:
    """Run litdata.optimize over files, with optional resume support."""
    if progress is not None:
        files = progress.filter_remaining(files)

    if not files:
        logger.info(f"[{split_label}] All files already processed.")
        return

    logger.info(f"[{split_label}] Optimizing {len(files)} files -> {output_dir}")

    mode = "append" if progress is not None and progress.sequence_count > 0 else "overwrite"

    optimize(
        convert_fn,
        files,
        output_dir,
        num_workers=min(num_workers, len(files)),
        chunk_bytes=chunk_bytes,
        mode=mode,
    )

    if progress is not None:
        for filepath in files:
            progress.record_file(filepath, num_sequences=0)

    dataset = StreamingDataset(output_dir)
    logger.info(f"[{split_label}] {len(dataset):,} sequences in output")
