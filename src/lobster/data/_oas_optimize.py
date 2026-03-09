from __future__ import annotations

import csv
import gzip
import io
import logging
import os
from collections.abc import Iterator
from functools import partial

import pandas as pd
from upath import UPath

from ._streaming_optimize import (
    CollectionProgress,
    PassthroughOptimizerConverter,
    SplitOptimizerConverter,
    run_optimize,
    sort_files_by_size,
)

logger = logging.getLogger(__name__)

FILTERABLE_METADATA_COLUMNS = ("Species", "Vaccine", "Disease", "Chain", "Isotype")


def parse_oas_metadata(line: str) -> dict[str, str]:
    """Parse the OAS JSON metadata line."""
    import json

    stripped = line.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1].replace('""', '"')

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse OAS metadata line as JSON: {line!r}") from exc


def file_passes_filters(metadata: dict[str, str], filters: dict[str, list[str]]) -> bool:
    """Check whether a file's metadata passes all specified filters."""
    for column, allowed_values in filters.items():
        file_value = metadata.get(column)
        if file_value is None or file_value not in allowed_values:
            return False
    return True


def read_oas_file(filepath: str) -> str:
    """Read an OAS CSV or CSV.GZ file."""
    raw_bytes = UPath(filepath).read_bytes()
    if filepath.endswith(".gz"):
        raw_bytes = gzip.decompress(raw_bytes)
    return raw_bytes.decode("utf-8")


def convert_oas_csv(
    filepath: str,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
) -> Iterator[dict[str, str]]:
    """Yield sequence items from one OAS CSV file."""
    text = read_oas_file(filepath)
    lines = text.splitlines()

    if len(lines) < 2:
        logger.warning(f"File {filepath} has fewer than 2 lines, skipping.")
        return

    try:
        metadata = parse_oas_metadata(lines[0])
    except ValueError:
        logger.warning(f"Could not parse metadata for {filepath}, skipping.")
        return

    if filters and not file_passes_filters(metadata, filters):
        logger.info(f"File {filepath} excluded by metadata filters: {metadata}")
        return

    reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
    if sequence_column not in (reader.fieldnames or []):
        logger.warning(
            f"Column '{sequence_column}' not found in {filepath} "
            f"(columns: {reader.fieldnames}). Skipping file."
        )
        return

    for row in reader:
        sequence = (row.get(sequence_column) or "").strip()
        if sequence:
            yield {"sequence": sequence}


def convert_oas_parquet(
    filepath: str,
    sequence_column: str = "sequence_alignment_aa",
    filters: dict[str, list[str]] | None = None,
) -> Iterator[dict[str, str]]:
    """Yield sequence items from one OAS parquet file."""
    try:
        dataframe = pd.read_parquet(filepath)
    except Exception:
        logger.warning(f"Could not read parquet file {filepath}, skipping.")
        return

    if sequence_column not in dataframe.columns:
        logger.warning(
            f"Column '{sequence_column}' not found in {filepath} "
            f"(columns: {list(dataframe.columns)}). Skipping file."
        )
        return

    if filters:
        for column, allowed_values in filters.items():
            if column in dataframe.columns:
                dataframe = dataframe[dataframe[column].isin(allowed_values)]
            else:
                logger.info(
                    f"Filter column '{column}' not found in {filepath}, "
                    f"skipping this filter for this file."
                )

    for sequence in dataframe[sequence_column].dropna():
        normalized = str(sequence).strip()
        if normalized:
            yield {"sequence": normalized}


def optimize_oas_csv_sequences(
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
    """Optimize a directory of OAS CSV files into LitData streaming format."""
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    if isinstance(file_glob, str):
        file_glob = [file_glob]

    files: list[str] = []
    input_path = UPath(input_dir)
    for pattern in file_glob:
        files.extend(str(filepath) for filepath in input_path.rglob(pattern))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No files matching {file_glob} found in {input_dir}")

    logger.info(f"Found {len(files)} files in {input_dir}")
    if filters:
        logger.info(f"Metadata filters: {filters}")

    num_workers = num_workers or os.cpu_count() or 1
    files = sort_files_by_size(files, num_workers=num_workers)
    reader = partial(convert_oas_csv, sequence_column=sequence_column, filters=filters)

    if val_fraction > 0.0:
        for split in ("train", "val"):
            progress = CollectionProgress(progress_dir, split_name=split) if progress_dir else None
            run_optimize(
                SplitOptimizerConverter(reader, val_fraction=val_fraction, seed=seed, emit_val=split == "val"),
                files,
                str(UPath(output_dir) / split),
                chunk_bytes,
                num_workers,
                progress,
                split,
            )
            if progress is not None:
                progress.clear()
        return

    progress = CollectionProgress(progress_dir, split_name="all") if progress_dir else None
    run_optimize(
        PassthroughOptimizerConverter(reader),
        files,
        output_dir,
        chunk_bytes,
        num_workers,
        progress,
        "all",
    )
    if progress is not None:
        progress.clear()


def optimize_oas_parquet_sequences(
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
    """Optimize a directory of OAS parquet files into LitData streaming format."""
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    files = sorted(str(filepath) for filepath in UPath(input_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    logger.info(f"Found {len(files)} parquet files in {input_dir}")
    if filters:
        logger.info(f"Row-level filters: {filters}")

    num_workers = num_workers or os.cpu_count() or 1
    files = sort_files_by_size(files, num_workers=num_workers)
    reader = partial(convert_oas_parquet, sequence_column=sequence_column, filters=filters)

    if val_fraction > 0.0:
        for split in ("train", "val"):
            progress = CollectionProgress(progress_dir, split_name=split) if progress_dir else None
            run_optimize(
                SplitOptimizerConverter(reader, val_fraction=val_fraction, seed=seed, emit_val=split == "val"),
                files,
                str(UPath(output_dir) / split),
                chunk_bytes,
                num_workers,
                progress,
                split,
            )
            if progress is not None:
                progress.clear()
        return

    progress = CollectionProgress(progress_dir, split_name="all") if progress_dir else None
    run_optimize(
        PassthroughOptimizerConverter(reader),
        files,
        output_dir,
        chunk_bytes,
        num_workers,
        progress,
        "all",
    )
    if progress is not None:
        progress.clear()
