"""Per-sample paired analysis of self-reflection ESMFold agreement.

CRITICAL: The unconditional CSV writer in lobster.metrics._generation_utils.MetricsCSVWriter
has a column-mismatch bug. The header declares 32 columns including
``rmsd_unconditional_to_forward_kabsch`` and ``rmsd_forward_to_inverse_kabsch`` columns
that the *writer* never writes, while the writer writes those two as values that the
header doesn't have positions for. Net result: every SR-diagnostic value beyond column
``rmsd_unconditional_to_forward`` (col 12) is shifted 2 columns to the LEFT in the
``self_reflection_*`` rows. The ``unconditional_*`` (final) rows are unaffected.

This script de-scrambles the SR-diagnostic rows to recover the true per-sample
initial-vs-ESMFold and refined-vs-ESMFold agreement metrics. Use it on eval dirs
generated with ``generation.self_reflection.use_esmfold_validation=true``.

Usage:
  uv run python scripts/analyze_selfreflection_paired.py \\
      --eval-dir /cv/scratch/u/lisanzas/evaluations/<DIR> \\
      [--label "GenUME-TED-val25-base"]

Compare two SR-paired runs side-by-side:
  uv run python scripts/analyze_selfreflection_paired.py \\
      --eval-dir <DIR_A> --label "TED" \\
      --compare-eval-dir <DIR_B> --compare-label "base"
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

LENGTHS = [100, 200, 300, 400, 500]

# CSV header label -> the value that the writer ACTUALLY puts in this column for SR-diagnostic rows.
# Because the CSV header has 6 extra labels and the writer has 2 extra `_kabsch` cells, the SR-diagnostic
# block is offset by 2 columns to the LEFT. Names below describe the TRUE quantity stored in each labeled column.
TRUE_COLUMN_MEANINGS = {
    "tm_score_esmfold_unconditional": "plddt_init_seq_esmfold",   # ESMFold pLDDT of initial seq
    "rmsd_esmfold_unconditional":     "pae_init_seq_esmfold",     # ESMFold PAE of initial seq
    "plddt_refined":                  "tm_init_vs_esmfold",       # TM(initial backbone, ESMFold of initial seq)
    "pae_refined":                    "rmsd_init_vs_esmfold",     # RMSD(initial backbone, ESMFold of initial seq)
    "tm_score_esmfold_refined":       "plddt_refn_seq_esmfold",   # ESMFold pLDDT of refined seq
    "rmsd_esmfold_refined":           "pae_refn_seq_esmfold",     # ESMFold PAE of refined seq
    "plddt_improvement":              "tm_refn_vs_esmfold",       # TM(refined backbone, ESMFold of refined seq)
    "pae_improvement":                "rmsd_refn_vs_esmfold",     # RMSD(refined backbone, ESMFold of refined seq)
    "tm_score_improvement":           "plddt_improvement",        # ESMFold pLDDT improvement (refined - initial)
    "rmsd_improvement":               "pae_improvement",          # ESMFold PAE improvement (initial - refined)
    "tm_score_unconditional_to_esmfold": "tm_improvement",        # TM improvement (refined - initial)
    "rmsd_unconditional_to_esmfold":  "rmsd_improvement",         # RMSD improvement (initial - refined)
}


def load_paired(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (final_rows, sr_diag_rows). The SR-diagnostic rows are renamed to expose
    the TRUE values (de-scrambling the writer-bug column shift).
    """
    paths = sorted(glob.glob(str(eval_dir / "unconditional_metrics_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {eval_dir}")
    df = pd.read_csv(paths[-1])
    final = df[df["run_id"].str.startswith("unconditional_")].reset_index(drop=True)
    diag = df[df["run_id"].str.startswith("self_reflection_")].rename(columns=TRUE_COLUMN_MEANINGS).reset_index(drop=True)
    return final, diag


def per_length_agreement(diag: pd.DataFrame, label: str) -> None:
    """Print per-length paired agreement of generated backbone with ESMFold refold.
    True initial vs refined comparison; both use the same alignment methodology
    (model backbone vs ESMFold of corresponding sequence).
    """
    print(f"\n=== {label}: per-length ESMFold agreement (paired, n=100/length) ===")
    print(
        f"{'L':>5} {'N':>4} | "
        f"{'TM_init':>8} {'TM_refn':>8} {'dTM':>7} | "
        f"{'RMSD_init':>10} {'RMSD_refn':>10} {'dRMSD':>7}"
    )
    for L in LENGTHS + ["all"]:
        d = diag if L == "all" else diag[diag["sequence_length"] == L]
        d = d.dropna(subset=["tm_init_vs_esmfold", "tm_refn_vs_esmfold"])
        if len(d) == 0:
            continue
        tm_i = d["tm_init_vs_esmfold"].mean()
        tm_r = d["tm_refn_vs_esmfold"].mean()
        rm_i = d["rmsd_init_vs_esmfold"].mean()
        rm_r = d["rmsd_refn_vs_esmfold"].mean()
        print(
            f"{str(L):>5} {len(d):>4} | "
            f"{tm_i:>8.3f} {tm_r:>8.3f} {tm_r - tm_i:>+7.3f} | "
            f"{rm_i:>10.2f} {rm_r:>10.2f} {rm_r - rm_i:>+7.2f}"
        )


def passing_table(diag: pd.DataFrame, label: str) -> None:
    """Per-length pass-rate (TM>0.9, RMSD<2A) of initial vs refined."""
    print(f"\n=== {label}: pass-rate (initial vs refined, paired) ===")
    print(
        f"{'L':>5} {'N':>4} | "
        f"{'%TM>0.9 i':>10} {'%TM>0.9 r':>10} {'d pp':>6} | "
        f"{'%RMSD<2 i':>10} {'%RMSD<2 r':>10} {'d pp':>6}"
    )
    for L in LENGTHS + ["all"]:
        d = diag if L == "all" else diag[diag["sequence_length"] == L]
        d = d.dropna(subset=["tm_init_vs_esmfold", "tm_refn_vs_esmfold"])
        if len(d) == 0:
            continue
        tmi = (d["tm_init_vs_esmfold"] > 0.9).mean() * 100
        tmr = (d["tm_refn_vs_esmfold"] > 0.9).mean() * 100
        rmi = (d["rmsd_init_vs_esmfold"] < 2.0).mean() * 100
        rmr = (d["rmsd_refn_vs_esmfold"] < 2.0).mean() * 100
        print(
            f"{str(L):>5} {len(d):>4} | "
            f"{tmi:>10.1f} {tmr:>10.1f} {tmr - tmi:>+6.1f} | "
            f"{rmi:>10.1f} {rmr:>10.1f} {rmr - rmi:>+6.1f}"
        )


def per_sample_improvement_distribution(diag: pd.DataFrame, label: str) -> None:
    """Distribution of per-sample improvements (refined - initial)."""
    d = diag.dropna(subset=["tm_init_vs_esmfold", "tm_refn_vs_esmfold"])
    if len(d) == 0:
        return
    d_tm = d["tm_refn_vs_esmfold"] - d["tm_init_vs_esmfold"]
    d_rm = d["rmsd_refn_vs_esmfold"] - d["rmsd_init_vs_esmfold"]
    print(f"\n=== {label}: per-sample improvement distribution (refined - initial), n={len(d)} ===")
    print(
        f"  TM   : mean={d_tm.mean():+.4f}  median={d_tm.median():+.4f}  "
        f"pct improved = {(d_tm > 0).mean() * 100:.1f}%  pct degraded = {(d_tm < 0).mean() * 100:.1f}%"
    )
    print(
        f"  RMSD : mean={d_rm.mean():+.3f}  median={d_rm.median():+.3f}  "
        f"pct improved = {(d_rm < 0).mean() * 100:.1f}%  pct degraded = {(d_rm > 0).mean() * 100:.1f}%"
    )


def report(eval_dir: Path, label: str) -> pd.DataFrame:
    final, diag = load_paired(eval_dir)
    print(f"\n# {label}")
    print(f"  eval dir: {eval_dir}")
    print(f"  rows: final={len(final)}, sr_diag={len(diag)}")

    needed = ["tm_init_vs_esmfold", "rmsd_init_vs_esmfold", "tm_refn_vs_esmfold", "rmsd_refn_vs_esmfold"]
    missing = [c for c in needed if c not in diag.columns or diag[c].isna().all()]
    if missing:
        print(f"  WARNING: missing/NaN de-scrambled columns: {missing}")
        print("  Was this run launched with generation.self_reflection.use_esmfold_validation=true?")
        return diag

    per_length_agreement(diag, label)
    passing_table(diag, label)
    per_sample_improvement_distribution(diag, label)
    return diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare-eval-dir", type=Path, default=None)
    ap.add_argument("--compare-label", default="run B")
    args = ap.parse_args()

    diag_a = report(args.eval_dir, args.label)
    if args.compare_eval_dir is not None:
        diag_b = report(args.compare_eval_dir, args.compare_label)
        # Aggregate side-by-side
        def agg(d):
            d = d.dropna(subset=["tm_init_vs_esmfold", "tm_refn_vs_esmfold"])
            return dict(
                n=len(d),
                tm_init=d["tm_init_vs_esmfold"].mean(),
                tm_refn=d["tm_refn_vs_esmfold"].mean(),
                rm_init=d["rmsd_init_vs_esmfold"].mean(),
                rm_refn=d["rmsd_refn_vs_esmfold"].mean(),
                pct_tm_improved=((d["tm_refn_vs_esmfold"] - d["tm_init_vs_esmfold"]) > 0).mean() * 100,
                pct_rmsd_improved=((d["rmsd_refn_vs_esmfold"] - d["rmsd_init_vs_esmfold"]) < 0).mean() * 100,
                pct_tm_above_0p9_init=(d["tm_init_vs_esmfold"] > 0.9).mean() * 100,
                pct_tm_above_0p9_refn=(d["tm_refn_vs_esmfold"] > 0.9).mean() * 100,
                pct_rmsd_below_2_init=(d["rmsd_init_vs_esmfold"] < 2.0).mean() * 100,
                pct_rmsd_below_2_refn=(d["rmsd_refn_vs_esmfold"] < 2.0).mean() * 100,
            )

        a, b = agg(diag_a), agg(diag_b)
        print(f"\n=== Two-run comparison (aggregate, paired) ===")
        print(f"{'metric':>26}  {args.label:>14}  {args.compare_label:>14}")
        print("-" * 64)
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float):
                print(f"{key:>26}  {va:>14.3f}  {vb:>14.3f}")
            else:
                print(f"{key:>26}  {va:>14}  {vb:>14}")


if __name__ == "__main__":
    main()
