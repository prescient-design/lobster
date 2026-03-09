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
import logging

from lobster.data._oas_optimize import (
    FILTERABLE_METADATA_COLUMNS,
    convert_oas_csv as _convert_oas_csv,
    convert_oas_parquet as _convert_oas_parquet,
    file_passes_filters as _file_passes_filters,
    optimize_oas_csv_sequences,
    optimize_oas_parquet_sequences,
    parse_oas_metadata as _parse_oas_metadata,
    read_oas_file as _read_oas_file,
)
from lobster.data._streaming_optimize import (
    CollectionProgress,
    sequence_is_val as _sequence_is_val,
    sort_files_by_size as _sort_files_by_size,
)

logger = logging.getLogger(__name__)

optimize_sequences = optimize_oas_csv_sequences
optimize_parquet_sequences = optimize_oas_parquet_sequences

__all__ = [
    "CollectionProgress",
    "FILTERABLE_METADATA_COLUMNS",
    "_build_filters",
    "_convert_oas_csv",
    "_convert_oas_parquet",
    "_file_passes_filters",
    "_parse_filter_arg",
    "_parse_oas_metadata",
    "_read_oas_file",
    "_sequence_is_val",
    "_sort_files_by_size",
    "main",
    "optimize_parquet_sequences",
    "optimize_sequences",
]


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
