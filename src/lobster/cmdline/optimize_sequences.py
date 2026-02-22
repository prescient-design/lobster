"""CLI script to optimize OAS antibody sequence CSV files for LitData streaming.

Reads CSV files from an S3 (or local) directory in the Observed Antibody Space
(OAS) format:

- **Line 0**: JSON metadata dict describing the file (species, chain, isotype, etc.)
- **Line 1**: CSV header row
- **Lines 2+**: CSV data rows

Each file can be included or excluded based on metadata filters (e.g. only
human heavy-chain sequences).  Sequences are read from the
``sequence_alignment_aa`` column and written to an optimized LitData chunked
dataset suitable for streaming with
``StreamingSequenceLightningDataModule``.

Usage
-----
.. code-block:: bash

    lobster_optimize_sequences \\
        --input_dir s3://my-bucket/oas/raw/ \\
        --output_dir s3://my-bucket/oas/optimized/ \\
        --val_fraction 0.05 \\
        --species human \\
        --chain Heavy \\
        --chunk_bytes 64MB \\
        --num_workers 8
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

from litdata import StreamingDataset, optimize
from upath import UPath

logger = logging.getLogger(__name__)

# Metadata columns that can be used for filtering OAS files.
FILTERABLE_METADATA_COLUMNS = ("Species", "Vaccine", "Disease", "Chain", "Isotype")


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


class _OASConverter:
    """Picklable callable for ``litdata.optimize`` workers.

    ``litdata.optimize`` serialises the worker function via pickle, so a
    plain closure won't work.  This class stores the filter config and
    sequence column name as instance attributes.

    Parameters
    ----------
    sequence_column : str
        Name of the CSV column containing sequences.
    filters : dict[str, list[str]] or None
        Metadata filters.
    """

    def __init__(self, sequence_column: str, filters: dict[str, list[str]] | None):
        self.sequence_column = sequence_column
        self.filters = filters

    def __call__(self, filepath: str):
        yield from _convert_oas_csv(
            filepath,
            sequence_column=self.sequence_column,
            filters=self.filters,
        )


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
        Fraction of *files* to hold out for validation.  If 0, all files
        go to a single output directory with no subdirectories.
        Default is 0.0.
    chunk_bytes : str, optional
        Target size for each output chunk.  Default is ``"64MB"``.
    num_workers : int or None, optional
        Number of parallel workers for optimization.
        Defaults to ``min(os.cpu_count(), num_files)``.
    seed : int, optional
        Random seed for train/val file splitting.  Default is 42.
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
        num_workers = min(os.cpu_count() or 1, len(files))

    convert_fn = _OASConverter(sequence_column, filters)

    if val_fraction > 0.0:
        rng = random.Random(seed)
        shuffled = list(files)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_files = shuffled[:n_val]
        train_files = shuffled[n_val:]

        logger.info(f"Splitting: {len(train_files)} train files, {len(val_files)} val files")

        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            split_output = str(UPath(output_dir) / split_name)
            logger.info(f"Optimizing {split_name} split → {split_output} ({len(split_files)} files)")

            optimize(
                convert_fn,
                split_files,
                split_output,
                num_workers=min(num_workers, len(split_files)),
                chunk_bytes=chunk_bytes,
                mode="overwrite",
            )

            ds = StreamingDataset(split_output)
            logger.info(f"{split_name} split: {len(ds)} sequences")
    else:
        logger.info(f"Optimizing all files → {output_dir}")

        optimize(
            convert_fn,
            files,
            output_dir,
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
            mode="overwrite",
        )

        ds = StreamingDataset(output_dir)
        logger.info(f"Total: {len(ds)} sequences")


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


def main():
    """Entry point for the ``lobster_optimize_sequences`` CLI command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Optimize OAS antibody sequence CSV files into LitData streaming format. "
            "Each CSV has a JSON metadata line (line 0), a header row (line 1), and "
            "data rows (lines 2+).  Sequences are read from the specified column and "
            "files can be filtered by metadata columns."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing OAS CSV files (S3 or local).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path for the optimized LitData output (S3 or local).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.0,
        help="Fraction of files to hold out for validation (default: 0.0).",
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
        help="Number of parallel workers (default: min(cpu_count, num_files)).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splitting (default: 42).",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        nargs="+",
        default=["*.csv", "*.csv.gz"],
        help="Glob pattern(s) for input files (default: '*.csv' '*.csv.gz').",
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="sequence_alignment_aa",
        help="Name of the CSV column containing sequences (default: 'sequence_alignment_aa').",
    )

    # Metadata filter arguments
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

    # Build filters dict from CLI arguments
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

    optimize_sequences(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        chunk_bytes=args.chunk_bytes,
        num_workers=args.num_workers,
        seed=args.seed,
        file_glob=args.file_glob,
        sequence_column=args.sequence_column,
        filters=filters if filters else None,
    )


if __name__ == "__main__":
    main()
