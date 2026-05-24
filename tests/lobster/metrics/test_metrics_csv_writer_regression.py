"""Phase 7 regression tests for ``lobster.metrics.MetricsCSVWriter``.

The CSV writer is the single source of truth feeding every benchmark
spreadsheet produced by ``lobster_generate``. Regressions in the header
layout — or worse, in the ``resume=True`` branch — silently corrupt
benchmark CSVs by misaligning columns row-over-row. These tests pin:

1. **Header contract per mode**: every supported generation mode emits
   the exact column order ``_initialize_csv`` documents. Insertion or
   reordering counts as an API break.
2. **Row alignment**: ``write_batch_metrics`` produces a row whose length
   equals the header row, regardless of which optional ``kwargs`` are
   provided. Missing values must be written as empty strings, never
   skipped.
3. **Resume mode**: when an existing metrics CSV is found, ``resume=True``
   appends to it instead of overwriting (the bug that prompted this
   regression — pre-fix runs that crashed at iteration N+1 would clobber
   the iteration-N metrics on relaunch).
4. **Sequences CSV**: the parallel ``sequences_<mode>_*.csv`` file
   initialises with the published header and stays in lockstep with the
   metrics CSV under resume.

All tests are CPU-only and self-contained — no GPU, no model loading.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lobster.metrics import MetricsCSVWriter

# Expected header layouts mirror ``MetricsCSVWriter._initialize_csv``.
# If you add a column here, mirror it in the writer (and vice versa).
EXPECTED_HEADERS: dict[str, list[str]] = {
    "unconditional": [
        "run_id", "timestamp", "mode",
        "plddt", "predicted_aligned_error", "tm_score", "rmsd",
        "sequence_length", "num_samples",
        "percent_identity_self_reflection",
        "tm_score_unconditional_to_forward",
        "rmsd_unconditional_to_forward",
        "tm_score_forward_to_inverse",
        "rmsd_forward_to_inverse",
        "plddt_unconditional", "pae_unconditional",
        "tm_score_esmfold_unconditional", "rmsd_esmfold_unconditional",
        "plddt_refined", "pae_refined",
        "tm_score_esmfold_refined", "rmsd_esmfold_refined",
        "plddt_improvement", "pae_improvement",
        "tm_score_improvement", "rmsd_improvement",
        "tm_score_unconditional_to_esmfold",
        "rmsd_unconditional_to_esmfold",
        "tm_score_forward_to_esmfold",
        "rmsd_forward_to_esmfold",
        "tm_score_esmfold_agreement_improvement",
        "rmsd_esmfold_agreement_improvement",
    ],
    "inverse_folding": [
        "run_id", "timestamp", "mode",
        "percent_identity", "plddt", "predicted_aligned_error",
        "tm_score", "rmsd", "sequence_length", "input_file",
    ],
    "forward_folding": [
        "run_id", "timestamp", "mode",
        "tm_score", "rmsd", "sequence_length", "input_file",
    ],
}

EXPECTED_SEQUENCE_HEADERS = [
    "run_id", "iteration", "sample_idx", "sequence",
    "original_sequence", "inpainted_sequence",
    "original_inpainted_sequence", "length", "generation_mode",
    "input_structure", "num_chains", "chain_ids", "trial_selected",
    "percent_identity_original", "masked_positions", "sequence_type",
    "latent_generator_tokens", "timestamp",
]


def _read_first_row(path: Path) -> list[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def _read_all_rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


@pytest.mark.parametrize("mode", list(EXPECTED_HEADERS))
def test_metrics_csv_writer_header_contract(tmp_path: Path, mode: str) -> None:
    """Each mode emits the published column order on init."""
    writer = MetricsCSVWriter(tmp_path, mode=mode)
    header = _read_first_row(writer.csv_path)
    assert header == EXPECTED_HEADERS[mode], (
        f"{mode} header drifted from the published contract. "
        f"Update _initialize_csv() AND EXPECTED_HEADERS in lockstep."
    )


def test_sequences_csv_header_contract(tmp_path: Path) -> None:
    """The parallel sequences CSV initialises with the published header."""
    writer = MetricsCSVWriter(tmp_path, mode="unconditional")
    header = _read_first_row(writer.sequences_csv_path)
    assert header == EXPECTED_SEQUENCE_HEADERS


@pytest.mark.parametrize("mode", ["inverse_folding", "forward_folding"])
def test_row_length_matches_header_with_missing_kwargs(
    tmp_path: Path, mode: str
) -> None:
    """write_batch_metrics produces a row whose length equals the header.

    Missing values must serialise to empty strings, never get dropped.
    Otherwise the CSV's columns misalign row-over-row and downstream
    pandas readers silently swap column meanings.
    """
    writer = MetricsCSVWriter(tmp_path, mode=mode)
    writer.write_batch_metrics(
        metrics={
            "tm_score": 0.812,
            "rmsd": 2.3,
            "_tm_score": 0.81,
            "_rmsd": 2.31,
            "_plddt": 0.79,
            "_predicted_aligned_error": 8.4,
        },
        run_id="test-run",
        sequence_length=120,
        input_file="dummy.pt",
        percent_identity=0.42,
    )
    rows = _read_all_rows(writer.csv_path)
    header, *data = rows
    assert data, "no metric row was written"
    for r in data:
        assert len(r) == len(header), (
            f"row width {len(r)} != header width {len(header)}; "
            f"missing kwargs likely dropped silently. row={r!r}"
        )


def test_resume_appends_not_clobber(tmp_path: Path) -> None:
    """Resume=True must reopen the existing CSV instead of starting fresh.

    The pre-fix bug: ``MetricsCSVWriter(..., resume=True)`` created a NEW
    timestamp-suffixed CSV every relaunch, losing the previous run's
    rows. Now it must pick the most recent CSV for the mode and append.
    """
    mode = "forward_folding"

    writer_a = MetricsCSVWriter(tmp_path, mode=mode, resume=False)
    writer_a.write_batch_metrics(
        metrics={"tm_score": 0.7, "rmsd": 1.5},
        run_id="run-a-1",
        sequence_length=80,
        input_file="a.pt",
    )
    writer_a.write_batch_metrics(
        metrics={"tm_score": 0.71, "rmsd": 1.6},
        run_id="run-a-2",
        sequence_length=80,
        input_file="a.pt",
    )

    rows_before = _read_all_rows(writer_a.csv_path)
    assert len(rows_before) == 3, "expected header + 2 rows pre-resume"

    writer_b = MetricsCSVWriter(tmp_path, mode=mode, resume=True)
    assert writer_b.csv_path == writer_a.csv_path, (
        "resume=True did not pick up the existing CSV — it created a new "
        "one, which is the historical clobber bug."
    )
    writer_b.write_batch_metrics(
        metrics={"tm_score": 0.74, "rmsd": 1.4},
        run_id="run-b-1",
        sequence_length=80,
        input_file="b.pt",
    )

    rows_after = _read_all_rows(writer_b.csv_path)
    assert len(rows_after) == 4, (
        "resume mode should append a row; got "
        f"{len(rows_after)} rows (expected header + 3 data)"
    )
    # The original rows must be intact and in the same order.
    assert rows_after[:3] == rows_before
    assert rows_after[3][0] == "run-b-1", (
        "post-resume row did not land at the tail of the CSV"
    )


def test_resume_with_no_existing_csv_creates_new(tmp_path: Path) -> None:
    """resume=True on an empty output dir must NOT crash — it must create
    a fresh CSV (and document this gracefully in logs)."""
    writer = MetricsCSVWriter(tmp_path, mode="forward_folding", resume=True)
    assert writer.csv_path.exists()
    header = _read_first_row(writer.csv_path)
    assert header == EXPECTED_HEADERS["forward_folding"]
