"""CLI script to optimize OAS antibody sequence files for LitData streaming.

Supports two input formats:

**CSV format** (``--input_format csv``, default):

- **Line 0**: JSON metadata dict describing the file
- **Line 1**: CSV header row
- **Lines 2+**: CSV data rows
- Files may be plain ``.csv`` or gzip-compressed ``.csv.gz``
- Metadata filters are applied per-file based on the JSON header

**Parquet format** (``--input_format parquet``):

- Hive-partitioned or flat directory of ``.parquet`` files
- Metadata columns (Species, Chain, Isotype, etc.) live alongside sequences
- Filters are applied per-row on DataFrame columns

When a validation fraction is specified, the split is performed **iid across
individual sequences** using a deterministic hash, so no global shuffle or
in-memory collection is required.  Each file is streamed independently and
only the sequences belonging to the target split are written.

Files are processed **smallest-first** for fast early progress.  A local
progress directory (``--progress_dir``) records which files have been
processed, allowing **resumable** jobs.

Sequences are read from the ``sequence_alignment_aa`` column (configurable)
and written to an optimized LitData chunked dataset suitable for streaming
with ``StreamingSequenceLightningDataModule``.

Usage
-----
.. code-block:: bash

    # CSV mode (OAS bulk download format)
    lobster_optimize_sequences \\
        --input_dir s3://my-bucket/oas/csv_raw/ \\
        --output_dir s3://my-bucket/oas/optimized/ \\
        --input_format csv \\
        --val_fraction 0.05 \\
        --species human --chain Heavy

    # Parquet mode (OAS deduplicated parquet format)
    lobster_optimize_sequences \\
        --input_dir s3://my-bucket/oas/OAS_aa_deduplicated/ \\
        --output_dir s3://my-bucket/oas/optimized/ \\
        --input_format parquet \\
        --val_fraction 0.05 \\
        --species human --chain Heavy
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from litdata import StreamingDataset, optimize
from upath import UPath

logger = logging.getLogger(__name__)

# Metadata columns that can be used for filtering OAS files.
FILTERABLE_METADATA_COLUMNS = ("Species", "Vaccine", "Disease", "Chain", "Isotype")

# Hash modulus for split assignment.  Using 10 000 gives 0.01% resolution
# on val_fraction, which is more than enough.
_HASH_MODULUS = 10_000


# ---------------------------------------------------------------------------
# Deterministic hash-based split
# ---------------------------------------------------------------------------


def _sequence_is_val(sequence: str, val_fraction: float, seed: int) -> bool:
    """Deterministically assign a sequence to the validation split.

    Uses a keyed hash so that the assignment is:
    - **iid per sequence** (not per file)
    - **deterministic** (same seed → same split)
    - **constant-memory** (no global list needed)

    Parameters
    ----------
    sequence : str
        The amino-acid sequence string.
    val_fraction : float
        Fraction of sequences that should be assigned to validation.
    seed : int
        Integer seed mixed into the hash for reproducibility.

    Returns
    -------
    bool
        ``True`` if the sequence belongs to the validation split.
    """
    h = hashlib.md5(f"{seed}:{sequence}".encode(), usedforsecurity=False).digest()
    bucket = int.from_bytes(h[:4], "little") % _HASH_MODULUS
    return bucket < int(val_fraction * _HASH_MODULUS)


# ---------------------------------------------------------------------------
# File size sorting
# ---------------------------------------------------------------------------


def _get_file_size(filepath: str) -> int:
    """Return the size of a file in bytes, or 0 if stat fails.

    Parameters
    ----------
    filepath : str
        Path to a file (local or S3 URI).

    Returns
    -------
    int
        File size in bytes, or 0 on failure.
    """
    try:
        return UPath(filepath).stat().st_size
    except Exception:
        return 0


def _sort_files_by_size(files: list[str], num_workers: int = 1) -> list[str]:
    """Sort files by size ascending (smallest first).

    Parameters
    ----------
    files : list[str]
        File paths to sort.
    num_workers : int, optional
        Workers for parallel stat calls.  Default is 1.

    Returns
    -------
    list[str]
        Files sorted by size ascending.
    """
    if not files:
        return files

    logger.info(f"Sorting {len(files)} files by size (smallest first)...")

    if num_workers <= 1 or len(files) < 10:
        sizes = [(_get_file_size(f), f) for f in files]
    else:
        sizes = []
        with ProcessPoolExecutor(max_workers=min(num_workers, len(files))) as pool:
            future_to_file = {pool.submit(_get_file_size, f): f for f in files}
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                sizes.append((future.result(), fp))

    sizes.sort(key=lambda x: x[0])
    return [fp for _, fp in sizes]


# ---------------------------------------------------------------------------
# Resumable progress tracking
# ---------------------------------------------------------------------------


class CollectionProgress:
    """Track which input files have been processed for a given split.

    Stores a JSON file in ``progress_dir`` recording the set of completed
    input filepaths and cumulative sequence counts.  On restart the
    already-processed files can be skipped.

    Parameters
    ----------
    progress_dir : str or Path
        Local directory for progress state.  Created if it does not exist.
    split_name : str, optional
        Name of the split (e.g. ``"train"``, ``"val"``, ``"all"``).
        Used to namespace the progress file so train and val passes
        can be tracked independently.  Default is ``"all"``.
    """

    def __init__(self, progress_dir: str | Path, split_name: str = "all") -> None:
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._split_name = split_name
        self._progress_file = self.progress_dir / f"progress_{split_name}.json"

        self._done_files: set[str] = set()
        self._sequence_count: int = 0

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
        """Set of input filepaths already processed."""
        return self._done_files

    @property
    def sequence_count(self) -> int:
        """Number of sequences written so far."""
        return self._sequence_count

    def filter_remaining(self, files: list[str]) -> list[str]:
        """Return only the files not yet processed.

        Parameters
        ----------
        files : list[str]
            All input files.

        Returns
        -------
        list[str]
            Files that still need processing.
        """
        remaining = [f for f in files if f not in self._done_files]
        if len(remaining) < len(files):
            logger.info(
                f"[{self._split_name}] Skipping {len(files) - len(remaining)} "
                f"already-processed files, {len(remaining)} remaining"
            )
        return remaining

    def record_file(self, filepath: str, num_sequences: int) -> None:
        """Record a completed file.

        Parameters
        ----------
        filepath : str
            The input file that was processed.
        num_sequences : int
            Number of sequences written from that file.
        """
        self._done_files.add(filepath)
        self._sequence_count += num_sequences
        self._save_progress()

    def _save_progress(self) -> None:
        """Write the progress JSON atomically."""
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
        """Remove progress files (call after successful optimization)."""
        if self._progress_file.exists():
            self._progress_file.unlink()
        # Remove dir if empty
        try:
            self.progress_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CSV format: low-level readers
# ---------------------------------------------------------------------------


def _parse_oas_metadata(line: str) -> dict[str, str]:
    """Parse the OAS JSON metadata line (line 0 of each CSV file).

    OAS files store metadata as a JSON object on the first line, sometimes
    wrapped in double-quotes with CSV-style escaping (``""`` → ``"``).

    Parameters
    ----------
    line : str
        The raw first line of an OAS CSV file.

    Returns
    -------
    dict[str, str]
        Parsed metadata dictionary.

    Raises
    ------
    ValueError
        If the line cannot be parsed as JSON.
    """
    stripped = line.strip()

    if stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1].replace('""', '"')

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse OAS metadata line as JSON: {line!r}") from exc


def _file_passes_filters(
    metadata: dict[str, str],
    filters: dict[str, list[str]],
) -> bool:
    """Check whether a file's metadata passes all specified filters.

    Parameters
    ----------
    metadata : dict[str, str]
        Parsed metadata dict from the first line of an OAS CSV file.
    filters : dict[str, list[str]]
        Mapping of metadata column names to lists of permissible values.

    Returns
    -------
    bool
        ``True`` if the file passes all filters (or no filters are specified).
    """
    for column, allowed_values in filters.items():
        file_value = metadata.get(column)
        if file_value is None or file_value not in allowed_values:
            return False
    return True


def _read_oas_file(filepath: str) -> str:
    """Read an OAS file, transparently decompressing gzipped files.

    Parameters
    ----------
    filepath : str
        Path to an OAS CSV or CSV.GZ file (S3 URI or local).

    Returns
    -------
    str
        The full text content of the file.
    """
    path = UPath(filepath)
    raw_bytes = path.read_bytes()

    if str(filepath).endswith(".gz"):
        raw_bytes = gzip.decompress(raw_bytes)

    return raw_bytes.decode("utf-8")


def _convert_oas_csv(
    filepath: str,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
):
    """Generator that yields sequence dicts from a single OAS CSV file.

    Handles both plain ``.csv`` and gzip-compressed ``.csv.gz`` files.

    Parameters
    ----------
    filepath : str
        Path to an OAS CSV or CSV.GZ file.
    sequence_column : str
        Name of the column containing amino-acid sequences.
    filters : dict[str, list[str]] or None
        Optional metadata filters.  If the file's metadata does not pass,
        no sequences are yielded.

    Yields
    ------
    dict[str, str]
        Dictionary with a ``"sequence"`` key for each valid sequence row.
    """
    text = _read_oas_file(filepath)
    lines = text.splitlines()

    if len(lines) < 2:
        logger.warning(f"File {filepath} has fewer than 2 lines, skipping.")
        return

    try:
        metadata = _parse_oas_metadata(lines[0])
    except ValueError:
        logger.warning(f"Could not parse metadata for {filepath}, skipping.")
        return

    if filters and not _file_passes_filters(metadata, filters):
        logger.info(f"File {filepath} excluded by metadata filters: {metadata}")
        return

    csv_text = "\n".join(lines[1:])
    reader = csv.DictReader(io.StringIO(csv_text))

    if sequence_column not in (reader.fieldnames or []):
        logger.warning(
            f"Column '{sequence_column}' not found in {filepath} "
            f"(columns: {reader.fieldnames}). Skipping file."
        )
        return

    for row in reader:
        seq = (row.get(sequence_column) or "").strip()
        if seq:
            yield {"sequence": seq}


# ---------------------------------------------------------------------------
# Parquet format: low-level reader
# ---------------------------------------------------------------------------


def _convert_oas_parquet(
    filepath: str,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
):
    """Generator that yields sequence dicts from a single OAS parquet file.

    Parameters
    ----------
    filepath : str
        Path to an OAS parquet file (S3 URI or local).
    sequence_column : str
        Name of the column containing amino-acid sequences.
    filters : dict[str, list[str]] or None
        Optional row-level metadata filters.

    Yields
    ------
    dict[str, str]
        Dictionary with a ``"sequence"`` key for each qualifying row.
    """
    import pandas as pd

    try:
        df = pd.read_parquet(filepath)
    except Exception:
        logger.warning(f"Could not read parquet file {filepath}, skipping.")
        return

    if sequence_column not in df.columns:
        logger.warning(
            f"Column '{sequence_column}' not found in {filepath} "
            f"(columns: {list(df.columns)}). Skipping file."
        )
        return

    if filters:
        for column, allowed_values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(allowed_values)]
            else:
                logger.info(
                    f"Filter column '{column}' not found in {filepath}, "
                    f"skipping this filter for this file."
                )

    for seq in df[sequence_column].dropna():
        seq = str(seq).strip()
        if seq:
            yield {"sequence": seq}


# ---------------------------------------------------------------------------
# Streaming split converters for litdata.optimize
# ---------------------------------------------------------------------------


class _SplitCSVConverter:
    """Picklable converter for ``litdata.optimize`` that streams a CSV file
    and emits only the sequences belonging to a particular split.

    Parameters
    ----------
    sequence_column : str
        CSV column containing sequences.
    filters : dict[str, list[str]] or None
        Per-file metadata filters (CSV JSON header).
    val_fraction : float
        Fraction of sequences assigned to validation.
    seed : int
        Hash seed for deterministic split.
    split : str
        ``"train"`` or ``"val"`` — which side of the split to emit.
    """

    def __init__(
        self,
        sequence_column: str,
        filters: dict[str, list[str]] | None,
        val_fraction: float,
        seed: int,
        split: str,
    ):
        self.sequence_column = sequence_column
        self.filters = filters
        self.val_fraction = val_fraction
        self.seed = seed
        self.emit_val = split == "val"

    def __call__(self, filepath: str):
        for item in _convert_oas_csv(filepath, self.sequence_column, self.filters):
            seq = item["sequence"]
            is_val = _sequence_is_val(seq, self.val_fraction, self.seed)
            if is_val == self.emit_val:
                yield item


class _SplitParquetConverter:
    """Picklable converter for ``litdata.optimize`` that streams a parquet file
    and emits only the sequences belonging to a particular split.

    Parameters
    ----------
    sequence_column : str
        Parquet column containing sequences.
    filters : dict[str, list[str]] or None
        Row-level metadata filters.
    val_fraction : float
        Fraction of sequences assigned to validation.
    seed : int
        Hash seed for deterministic split.
    split : str
        ``"train"`` or ``"val"`` — which side of the split to emit.
    """

    def __init__(
        self,
        sequence_column: str,
        filters: dict[str, list[str]] | None,
        val_fraction: float,
        seed: int,
        split: str,
    ):
        self.sequence_column = sequence_column
        self.filters = filters
        self.val_fraction = val_fraction
        self.seed = seed
        self.emit_val = split == "val"

    def __call__(self, filepath: str):
        for item in _convert_oas_parquet(filepath, self.sequence_column, self.filters):
            seq = item["sequence"]
            is_val = _sequence_is_val(seq, self.val_fraction, self.seed)
            if is_val == self.emit_val:
                yield item


class _NoSplitCSVConverter:
    """Picklable converter for ``litdata.optimize`` that streams a CSV file
    with no train/val split (all sequences emitted).

    Parameters
    ----------
    sequence_column : str
        CSV column containing sequences.
    filters : dict[str, list[str]] or None
        Per-file metadata filters.
    """

    def __init__(self, sequence_column: str, filters: dict[str, list[str]] | None):
        self.sequence_column = sequence_column
        self.filters = filters

    def __call__(self, filepath: str):
        yield from _convert_oas_csv(filepath, self.sequence_column, self.filters)


class _NoSplitParquetConverter:
    """Picklable converter for ``litdata.optimize`` that streams a parquet file
    with no train/val split (all sequences emitted).

    Parameters
    ----------
    sequence_column : str
        Parquet column containing sequences.
    filters : dict[str, list[str]] or None
        Row-level metadata filters.
    """

    def __init__(self, sequence_column: str, filters: dict[str, list[str]] | None):
        self.sequence_column = sequence_column
        self.filters = filters

    def __call__(self, filepath: str):
        yield from _convert_oas_parquet(filepath, self.sequence_column, self.filters)


# ---------------------------------------------------------------------------
# Core optimization driver
# ---------------------------------------------------------------------------


def _run_optimize(
    convert_fn,
    files: list[str],
    output_dir: str,
    chunk_bytes: str,
    num_workers: int,
    progress: CollectionProgress | None,
    split_label: str,
) -> None:
    """Run ``litdata.optimize`` over *files*, with optional resume support.

    If a ``CollectionProgress`` is provided, already-processed files are
    excluded and newly-completed files are recorded for future resumption.
    Uses ``mode="append"`` so that chunks from previous runs are preserved.

    Parameters
    ----------
    convert_fn : callable
        Picklable callable ``(filepath) -> Iterator[dict]``.
    files : list[str]
        Input files to process.
    output_dir : str
        LitData output directory for this split.
    chunk_bytes : str
        Target chunk size.
    num_workers : int
        Number of parallel litdata workers.
    progress : CollectionProgress or None
        Optional resume tracker.
    split_label : str
        Human-readable label for logging (e.g. ``"train"``).
    """
    if progress is not None:
        files = progress.filter_remaining(files)

    if not files:
        logger.info(f"[{split_label}] All files already processed.")
        return

    logger.info(f"[{split_label}] Optimizing {len(files)} files → {output_dir}")

    # Use "append" so that resumed runs add to existing chunks rather than
    # overwriting them.  For a fresh start the directory is empty, so
    # append behaves the same as overwrite.
    mode = "append" if (progress is not None and progress.sequence_count > 0) else "overwrite"

    optimize(
        convert_fn,
        files,
        output_dir,
        num_workers=min(num_workers, len(files)),
        chunk_bytes=chunk_bytes,
        mode=mode,
    )

    # Record all files as done (litdata.optimize processes them atomically)
    if progress is not None:
        for fp in files:
            progress.record_file(fp, num_sequences=0)  # count is informational

    ds = StreamingDataset(output_dir)
    logger.info(f"[{split_label}] {len(ds):,} sequences in output")


# ---------------------------------------------------------------------------
# Public API: optimize_sequences (CSV)
# ---------------------------------------------------------------------------


def optimize_sequences(
    input_dir: str,
    output_dir: str,
    val_fraction: float = 0.0,
    chunk_bytes: str = "64MB",
    num_workers: int | None = None,
    seed: int = 42,
    file_glob: str | list[str] = ("*.csv", "*.csv.gz"),
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
    progress_dir: str | None = None,
) -> None:
    """Optimize a directory of OAS CSV files into LitData streaming format.

    Sequences are streamed through ``litdata.optimize`` without being
    loaded into memory.  When ``val_fraction > 0`` an iid hash-based split
    assigns each sequence deterministically to train or val.

    Parameters
    ----------
    input_dir : str
        Path to directory containing OAS CSV files.
        Can be an S3 URI (``s3://bucket/path``) or a local directory.
    output_dir : str
        Path to write the optimized LitData dataset.  Can be an S3 URI or
        local path.  If ``val_fraction > 0``, ``train/`` and ``val/``
        subdirectories will be created.
    val_fraction : float, optional
        Fraction of sequences assigned to validation via deterministic hash.
        Default is 0.0 (no split).
    chunk_bytes : str, optional
        Target size for each output chunk.  Default is ``"64MB"``.
    num_workers : int or None, optional
        Number of parallel workers.  Defaults to ``os.cpu_count()``.
    seed : int, optional
        Hash seed for reproducible iid splitting.  Default is 42.
    file_glob : str or list[str], optional
        Glob pattern(s) to match input files.  Default is
        ``("*.csv", "*.csv.gz")``.
    sequence_column : str, optional
        Name of the CSV column containing amino-acid sequences.
        Default is ``"sequence_alignment_aa"``.
    filters : dict[str, list[str]] or None, optional
        Per-file metadata filters.  Default is ``None``.
    progress_dir : str or None, optional
        Local directory for resumable progress tracking.  If ``None``,
        the job is not resumable.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` contains no matching files.
    ValueError
        If ``val_fraction`` is not in ``[0, 1)``.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    input_path = UPath(input_dir)

    if isinstance(file_glob, str):
        file_glob = [file_glob]

    files: list[str] = []
    for pattern in file_glob:
        files.extend(str(f) for f in input_path.rglob(pattern))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No files matching {file_glob} found in {input_dir}")

    logger.info(f"Found {len(files)} files in {input_dir}")
    if filters:
        logger.info(f"Metadata filters: {filters}")

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    files = _sort_files_by_size(files, num_workers=num_workers)

    if val_fraction > 0.0:
        for split in ("train", "val"):
            split_output = str(UPath(output_dir) / split)
            convert_fn = _SplitCSVConverter(sequence_column, filters, val_fraction, seed, split)
            progress = (
                CollectionProgress(progress_dir, split_name=split)
                if progress_dir else None
            )
            _run_optimize(convert_fn, files, split_output, chunk_bytes, num_workers, progress, split)
            if progress is not None:
                progress.clear()
    else:
        convert_fn = _NoSplitCSVConverter(sequence_column, filters)
        progress = CollectionProgress(progress_dir, split_name="all") if progress_dir else None
        _run_optimize(convert_fn, files, output_dir, chunk_bytes, num_workers, progress, "all")
        if progress is not None:
            progress.clear()


# ---------------------------------------------------------------------------
# Public API: optimize_parquet_sequences
# ---------------------------------------------------------------------------


def optimize_parquet_sequences(
    input_dir: str,
    output_dir: str,
    val_fraction: float = 0.0,
    chunk_bytes: str = "64MB",
    num_workers: int | None = None,
    seed: int = 42,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
    progress_dir: str | None = None,
) -> None:
    """Optimize a directory of OAS parquet files into LitData streaming format.

    Sequences are streamed through ``litdata.optimize`` without being
    loaded into memory.  When ``val_fraction > 0`` an iid hash-based split
    assigns each sequence deterministically to train or val.

    Parameters
    ----------
    input_dir : str
        Path to directory containing parquet files (possibly Hive-partitioned).
    output_dir : str
        Path to write the optimized LitData dataset.
    val_fraction : float, optional
        Fraction of sequences assigned to validation via deterministic hash.
        Default is 0.0 (no split).
    chunk_bytes : str, optional
        Target size for each output chunk.  Default is ``"64MB"``.
    num_workers : int or None, optional
        Number of parallel workers.  Defaults to ``os.cpu_count()``.
    seed : int, optional
        Hash seed for reproducible iid splitting.  Default is 42.
    sequence_column : str, optional
        Name of the parquet column containing amino-acid sequences.
        Default is ``"sequence_alignment_aa"``.
    filters : dict[str, list[str]] or None, optional
        Row-level metadata column filters.  Default is ``None``.
    progress_dir : str or None, optional
        Local directory for resumable progress tracking.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` contains no ``.parquet`` files.
    ValueError
        If ``val_fraction`` is not in ``[0, 1)``.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    input_path = UPath(input_dir)
    files = sorted(str(f) for f in input_path.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    logger.info(f"Found {len(files)} parquet files in {input_dir}")
    if filters:
        logger.info(f"Row-level filters: {filters}")

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    files = _sort_files_by_size(files, num_workers=num_workers)

    if val_fraction > 0.0:
        for split in ("train", "val"):
            split_output = str(UPath(output_dir) / split)
            convert_fn = _SplitParquetConverter(sequence_column, filters, val_fraction, seed, split)
            progress = (
                CollectionProgress(progress_dir, split_name=split)
                if progress_dir else None
            )
            _run_optimize(convert_fn, files, split_output, chunk_bytes, num_workers, progress, split)
            if progress is not None:
                progress.clear()
    else:
        convert_fn = _NoSplitParquetConverter(sequence_column, filters)
        progress = CollectionProgress(progress_dir, split_name="all") if progress_dir else None
        _run_optimize(convert_fn, files, output_dir, chunk_bytes, num_workers, progress, "all")
        if progress is not None:
            progress.clear()


# ---------------------------------------------------------------------------
# Shared CLI helpers
# ---------------------------------------------------------------------------


def _parse_filter_arg(value: str) -> list[str]:
    """Parse a comma-separated filter argument into a list of values.

    Parameters
    ----------
    value : str
        Comma-separated string, e.g. ``"human,mouse"``.

    Returns
    -------
    list[str]
        List of individual filter values.
    """
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_filters(args: argparse.Namespace) -> dict[str, list[str]] | None:
    """Build metadata filter dict from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict[str, list[str]] or None
        Filter dict, or ``None`` if no filters were specified.
    """
    filters: dict[str, list[str]] = {}
    filter_args = {
        "Species": args.species,
        "Vaccine": args.vaccine,
        "Disease": args.disease,
        "Chain": args.chain,
        "Isotype": args.isotype,
    }
    for column, value in filter_args.items():
        if value is not None:
            filters[column] = _parse_filter_arg(value)
    return filters if filters else None


def main():
    """Entry point for the ``lobster_optimize_sequences`` CLI command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Optimize OAS antibody sequence files into LitData streaming format. "
            "Supports CSV (with JSON metadata header) and Parquet input formats. "
            "Train/val splits use a deterministic hash for iid per-sequence assignment "
            "with zero memory overhead.  Files are processed smallest-first; "
            "use --progress_dir for resumability."
        ),
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to directory containing input files (S3 or local).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path for the optimized LitData output (S3 or local).")
    parser.add_argument("--input_format", type=str, choices=["csv", "parquet"], default="csv",
                        help="Input file format (default: 'csv').")
    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="Fraction of sequences for validation via deterministic hash (default: 0.0).")
    parser.add_argument("--chunk_bytes", type=str, default="64MB",
                        help="Target chunk size (default: '64MB').")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Hash seed for reproducible splitting (default: 42).")
    parser.add_argument("--file_glob", type=str, nargs="+", default=None,
                        help="Glob pattern(s) for CSV input files (default: '*.csv *.csv.gz').")
    parser.add_argument("--sequence_column", type=str, default="sequence_alignment_aa",
                        help="Column containing sequences (default: 'sequence_alignment_aa').")
    parser.add_argument("--progress_dir", type=str, default=None,
                        help="Local directory for resumable progress tracking.")

    # Metadata filter arguments
    parser.add_argument("--species", type=str, default=None,
                        help="Comma-separated permissible Species values.")
    parser.add_argument("--vaccine", type=str, default=None,
                        help="Comma-separated permissible Vaccine values.")
    parser.add_argument("--disease", type=str, default=None,
                        help="Comma-separated permissible Disease values.")
    parser.add_argument("--chain", type=str, default=None,
                        help="Comma-separated permissible Chain values.")
    parser.add_argument("--isotype", type=str, default=None,
                        help="Comma-separated permissible Isotype values.")

    args = parser.parse_args()
    filters = _build_filters(args)

    if args.input_format == "parquet":
        optimize_parquet_sequences(
            input_dir=args.input_dir, output_dir=args.output_dir,
            val_fraction=args.val_fraction, chunk_bytes=args.chunk_bytes,
            num_workers=args.num_workers, seed=args.seed,
            sequence_column=args.sequence_column, filters=filters,
            progress_dir=args.progress_dir,
        )
    else:
        file_glob = args.file_glob or ["*.csv", "*.csv.gz"]
        optimize_sequences(
            input_dir=args.input_dir, output_dir=args.output_dir,
            val_fraction=args.val_fraction, chunk_bytes=args.chunk_bytes,
            num_workers=args.num_workers, seed=args.seed,
            file_glob=file_glob, sequence_column=args.sequence_column,
            filters=filters, progress_dir=args.progress_dir,
        )


if __name__ == "__main__":
    main()
