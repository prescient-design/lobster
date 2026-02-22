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
individual sequences** (not across files), ensuring a representative sample
regardless of per-file size variation.

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
import io
import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from litdata import StreamingDataset, optimize
from tqdm import tqdm
from upath import UPath

logger = logging.getLogger(__name__)

# Metadata columns that can be used for filtering OAS files.
FILTERABLE_METADATA_COLUMNS = ("Species", "Vaccine", "Disease", "Chain", "Isotype")


# ---------------------------------------------------------------------------
# Shared optimisation helpers
# ---------------------------------------------------------------------------


class _SequencePassthrough:
    """Picklable callable for ``litdata.optimize`` that wraps a raw sequence string.

    Used after all sequences have been collected and split; each "input" to
    ``litdata.optimize`` is already a single sequence string.
    """

    def __call__(self, sequence: str):
        yield {"sequence": sequence}


def _split_and_optimize(
    sequences: list[str],
    output_dir: str,
    val_fraction: float,
    seed: int,
    chunk_bytes: str,
    num_workers: int,
) -> None:
    """Shuffle sequences iid, split into train/val, and optimize each split.

    Parameters
    ----------
    sequences : list[str]
        All collected sequences (already filtered).
    output_dir : str
        Root output directory.  If ``val_fraction > 0``, ``train/`` and
        ``val/`` subdirectories are created.
    val_fraction : float
        Fraction of *sequences* to hold out for validation (0 means no split).
    seed : int
        Random seed for reproducible shuffling.
    chunk_bytes : str
        Target chunk size for LitData output.
    num_workers : int
        Number of parallel workers for optimization.
    """
    convert_fn = _SequencePassthrough()

    if val_fraction > 0.0:
        rng = random.Random(seed)
        rng.shuffle(sequences)
        n_val = max(1, int(len(sequences) * val_fraction))
        val_seqs = sequences[:n_val]
        train_seqs = sequences[n_val:]

        logger.info(
            f"iid split: {len(train_seqs)} train sequences, "
            f"{len(val_seqs)} val sequences"
        )

        for split_name, split_seqs in [("train", train_seqs), ("val", val_seqs)]:
            split_output = str(UPath(output_dir) / split_name)
            logger.info(f"Optimizing {split_name} split → {split_output}")

            optimize(
                convert_fn,
                split_seqs,
                split_output,
                num_workers=min(num_workers, len(split_seqs)),
                chunk_bytes=chunk_bytes,
                mode="overwrite",
            )

            ds = StreamingDataset(split_output)
            logger.info(f"{split_name} split: {len(ds)} sequences")
    else:
        logger.info(f"Optimizing {len(sequences)} sequences → {output_dir}")

        optimize(
            convert_fn,
            sequences,
            output_dir,
            num_workers=min(num_workers, len(sequences)),
            chunk_bytes=chunk_bytes,
            mode="overwrite",
        )

        ds = StreamingDataset(output_dir)
        logger.info(f"Total: {len(ds)} sequences")


# ---------------------------------------------------------------------------
# CSV format support
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

    # OAS sometimes wraps the JSON dict in an outer pair of quotes with
    # CSV-style escaping: ""{""key"": ""value"", ...}""
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
        A file passes if, for every column present in ``filters``, the
        file's metadata value is in the corresponding list.

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

    # Line 0: JSON metadata
    try:
        metadata = _parse_oas_metadata(lines[0])
    except ValueError:
        logger.warning(f"Could not parse metadata for {filepath}, skipping.")
        return

    # Apply metadata filters
    if filters and not _file_passes_filters(metadata, filters):
        logger.info(f"File {filepath} excluded by metadata filters: {metadata}")
        return

    # Lines 1+: CSV with header on line 1
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


class _CSVFileCollector:
    """Picklable callable that extracts sequences from a single OAS CSV file.

    Used by ``ProcessPoolExecutor`` in ``_collect_csv_sequences`` to
    parallelise file reading.
    """

    def __init__(self, sequence_column: str, filters: dict[str, list[str]] | None):
        self.sequence_column = sequence_column
        self.filters = filters

    def __call__(self, filepath: str) -> list[str]:
        return [
            item["sequence"]
            for item in _convert_oas_csv(filepath, sequence_column=self.sequence_column, filters=self.filters)
        ]


def _collect_csv_sequences(
    files: list[str],
    sequence_column: str,
    filters: dict[str, list[str]] | None,
    num_workers: int = 1,
) -> list[str]:
    """Read all sequences from a list of OAS CSV files in parallel.

    Parameters
    ----------
    files : list[str]
        Paths to OAS CSV/CSV.GZ files.
    sequence_column : str
        Name of the CSV column containing sequences.
    filters : dict[str, list[str]] or None
        Per-file metadata filters.
    num_workers : int, optional
        Number of parallel workers.  Default is 1 (sequential).

    Returns
    -------
    list[str]
        Flat list of all qualifying sequence strings.
    """
    collector = _CSVFileCollector(sequence_column, filters)
    sequences: list[str] = []

    if num_workers <= 1:
        for filepath in tqdm(files, desc="Collecting sequences (CSV)", unit="file"):
            sequences.extend(collector(filepath))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(collector, fp): fp for fp in files}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Collecting sequences (CSV)",
                unit="file",
            ):
                sequences.extend(future.result())

    logger.info(f"Collected {len(sequences):,} sequences from {len(files)} CSV files")
    return sequences


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
) -> None:
    """Optimize a directory of OAS CSV files into LitData streaming format.

    All qualifying sequences are collected, shuffled iid, and optionally
    split into train/val before being written as optimized LitData chunks.

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
        Fraction of *sequences* to hold out for validation (iid).  If 0,
        all sequences go to a single output directory.  Default is 0.0.
    chunk_bytes : str, optional
        Target size for each output chunk.  Default is ``"64MB"``.
    num_workers : int or None, optional
        Number of parallel workers for optimization.
        Defaults to ``os.cpu_count()``.
    seed : int, optional
        Random seed for iid train/val splitting.  Default is 42.
    file_glob : str or list[str], optional
        Glob pattern(s) to match input files.  Default is
        ``("*.csv", "*.csv.gz")``.
    sequence_column : str, optional
        Name of the CSV column containing amino-acid sequences.
        Default is ``"sequence_alignment_aa"``.
    filters : dict[str, list[str]] or None, optional
        Metadata column filters.  Keys are column names (e.g.
        ``"Species"``, ``"Chain"``); values are lists of permissible
        values.  Files whose metadata does not match are skipped entirely.
        Default is ``None`` (no filtering).

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

    sequences = _collect_csv_sequences(files, sequence_column, filters, num_workers=num_workers)

    if not sequences:
        logger.warning("No sequences found after filtering. Nothing to optimize.")
        return

    _split_and_optimize(sequences, output_dir, val_fraction, seed, chunk_bytes, num_workers)


# ---------------------------------------------------------------------------
# Parquet format support
# ---------------------------------------------------------------------------


def _convert_oas_parquet(
    filepath: str,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
):
    """Generator that yields sequence dicts from a single OAS parquet file.

    Metadata columns (Species, Chain, Isotype, …) are expected to be
    regular columns in the parquet file.  Row-level filtering is applied
    before yielding sequences.

    Parameters
    ----------
    filepath : str
        Path to an OAS parquet file (S3 URI or local).
    sequence_column : str
        Name of the column containing amino-acid sequences.
    filters : dict[str, list[str]] or None
        Optional row-level metadata filters.  Keys are column names;
        values are lists of permissible values.  Only rows matching
        *all* filters are yielded.

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

    # Apply row-level filters
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


class _ParquetFileCollector:
    """Picklable callable that extracts sequences from a single OAS parquet file.

    Used by ``ProcessPoolExecutor`` in ``_collect_parquet_sequences`` to
    parallelise file reading.
    """

    def __init__(self, sequence_column: str, filters: dict[str, list[str]] | None):
        self.sequence_column = sequence_column
        self.filters = filters

    def __call__(self, filepath: str) -> list[str]:
        return [
            item["sequence"]
            for item in _convert_oas_parquet(filepath, sequence_column=self.sequence_column, filters=self.filters)
        ]


def _collect_parquet_sequences(
    files: list[str],
    sequence_column: str,
    filters: dict[str, list[str]] | None,
    num_workers: int = 1,
) -> list[str]:
    """Read all sequences from a list of OAS parquet files in parallel.

    Parameters
    ----------
    files : list[str]
        Paths to OAS parquet files.
    sequence_column : str
        Name of the parquet column containing sequences.
    filters : dict[str, list[str]] or None
        Row-level metadata filters.
    num_workers : int, optional
        Number of parallel workers.  Default is 1 (sequential).

    Returns
    -------
    list[str]
        Flat list of all qualifying sequence strings.
    """
    collector = _ParquetFileCollector(sequence_column, filters)
    sequences: list[str] = []

    if num_workers <= 1:
        for filepath in tqdm(files, desc="Collecting sequences (parquet)", unit="file"):
            sequences.extend(collector(filepath))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(collector, fp): fp for fp in files}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Collecting sequences (parquet)",
                unit="file",
            ):
                sequences.extend(future.result())

    logger.info(f"Collected {len(sequences):,} sequences from {len(files)} parquet files")
    return sequences


def optimize_parquet_sequences(
    input_dir: str,
    output_dir: str,
    val_fraction: float = 0.0,
    chunk_bytes: str = "64MB",
    num_workers: int | None = None,
    seed: int = 42,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
) -> None:
    """Optimize a directory of OAS parquet files into LitData streaming format.

    All qualifying sequences are collected, shuffled iid, and optionally
    split into train/val before being written as optimized LitData chunks.

    Parameters
    ----------
    input_dir : str
        Path to directory containing parquet files (possibly Hive-partitioned).
        Can be an S3 URI (``s3://bucket/path``) or a local directory.
    output_dir : str
        Path to write the optimized LitData dataset.  Can be an S3 URI or
        local path.  If ``val_fraction > 0``, ``train/`` and ``val/``
        subdirectories will be created.
    val_fraction : float, optional
        Fraction of *sequences* to hold out for validation (iid).  If 0,
        all sequences go to a single output directory.  Default is 0.0.
    chunk_bytes : str, optional
        Target size for each output chunk.  Default is ``"64MB"``.
    num_workers : int or None, optional
        Number of parallel workers for optimization.
        Defaults to ``os.cpu_count()``.
    seed : int, optional
        Random seed for iid train/val splitting.  Default is 42.
    sequence_column : str, optional
        Name of the parquet column containing amino-acid sequences.
        Default is ``"sequence_alignment_aa"``.
    filters : dict[str, list[str]] or None, optional
        Row-level metadata column filters.  Keys are column names (e.g.
        ``"Species"``, ``"Chain"``); values are lists of permissible
        values.  Only rows matching all filters are included.
        Default is ``None`` (no filtering).

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

    sequences = _collect_parquet_sequences(files, sequence_column, filters, num_workers=num_workers)

    if not sequences:
        logger.warning("No sequences found after filtering. Nothing to optimize.")
        return

    _split_and_optimize(sequences, output_dir, val_fraction, seed, chunk_bytes, num_workers)


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
        Parsed CLI arguments with species, vaccine, disease, chain, isotype.

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
            "Train/val splits are performed iid across individual sequences."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing input files (S3 or local).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path for the optimized LitData output (S3 or local).",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Input file format: 'csv' (OAS CSV with JSON header) or 'parquet' (default: 'csv').",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.0,
        help="Fraction of sequences to hold out for validation, iid (default: 0.0).",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=str,
        default="64MB",
        help="Target chunk size (default: '64MB').",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for iid train/val splitting (default: 42).",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Glob pattern(s) for input files. "
            "Defaults to '*.csv *.csv.gz' for csv format (ignored for parquet)."
        ),
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="sequence_alignment_aa",
        help="Name of the column containing sequences (default: 'sequence_alignment_aa').",
    )

    # Metadata filter arguments (used by both csv and parquet modes)
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Comma-separated permissible Species values (e.g. 'human,mouse').",
    )
    parser.add_argument(
        "--vaccine",
        type=str,
        default=None,
        help="Comma-separated permissible Vaccine values (e.g. 'None,Influenza').",
    )
    parser.add_argument(
        "--disease",
        type=str,
        default=None,
        help="Comma-separated permissible Disease values (e.g. 'None,HIV').",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Comma-separated permissible Chain values (e.g. 'Heavy,Light').",
    )
    parser.add_argument(
        "--isotype",
        type=str,
        default=None,
        help="Comma-separated permissible Isotype values (e.g. 'IGHG,IGHA').",
    )

    args = parser.parse_args()
    filters = _build_filters(args)

    if args.input_format == "parquet":
        optimize_parquet_sequences(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            chunk_bytes=args.chunk_bytes,
            num_workers=args.num_workers,
            seed=args.seed,
            sequence_column=args.sequence_column,
            filters=filters,
        )
    else:
        file_glob = args.file_glob or ["*.csv", "*.csv.gz"]
        optimize_sequences(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            chunk_bytes=args.chunk_bytes,
            num_workers=args.num_workers,
            seed=args.seed,
            file_glob=file_glob,
            sequence_column=args.sequence_column,
            filters=filters,
        )


if __name__ == "__main__":
    main()
