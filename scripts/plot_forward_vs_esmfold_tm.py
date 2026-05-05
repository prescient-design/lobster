"""Per-sample correlation: Lobster forward-fold TM vs ESMFold TM.

For every sample in an SR-paired unconditional-generation eval (run with
``generation.self_reflection.use_esmfold_validation=true``), plots:

    x: TM(initial backbone, lobster forward-folded structure of initial seq)
       (= ``tm_score_unconditional_to_forward``)
    y: TM(initial backbone, ESMFold structure of initial seq)
       (= raw value in the column labeled ``plddt_refined`` due to a
        CSV-writer/header off-by-2 bug; see analyze_selfreflection_paired.py)

Optionally overlays a second eval directory and reports Pearson + Spearman
correlations per checkpoint and per length.

Usage:
  uv run python scripts/plot_forward_vs_esmfold_tm.py \\
      --eval-dir /cv/scratch/u/lisanzas/evaluations/<TED_dir> \\
      --label "TED-val25-base" \\
      --compare-eval-dir /cv/scratch/u/lisanzas/evaluations/<base_dir> \\
      --compare-label "val25-base" \\
      --out /cv/home/lisanzas/lobster/forward_vs_esmfold_tm.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Header label -> the value the writer ACTUALLY stored there (de-scrambling).
TRUE_COLUMN_MEANINGS = {
    "tm_score_esmfold_unconditional": "plddt_init_seq_esmfold",
    "rmsd_esmfold_unconditional":     "pae_init_seq_esmfold",
    "plddt_refined":                  "tm_init_vs_esmfold",       # << y for our plot
    "pae_refined":                    "rmsd_init_vs_esmfold",
    "tm_score_esmfold_refined":       "plddt_refn_seq_esmfold",
    "rmsd_esmfold_refined":           "pae_refn_seq_esmfold",
    "plddt_improvement":              "tm_refn_vs_esmfold",
    "pae_improvement":                "rmsd_refn_vs_esmfold",
    "tm_score_improvement":           "plddt_improvement",
    "rmsd_improvement":               "pae_improvement",
    "tm_score_unconditional_to_esmfold": "tm_improvement",
    "rmsd_unconditional_to_esmfold":  "rmsd_improvement",
}


def load_paired(eval_dir: Path) -> pd.DataFrame:
    """Return the SR-diagnostic rows with de-scrambled columns + the
    `tm_score_unconditional_to_forward` column already present and correct.
    """
    paths = sorted(glob.glob(str(eval_dir / "unconditional_metrics_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {eval_dir}")
    df = pd.read_csv(paths[-1])
    diag = df[df["run_id"].str.startswith("self_reflection_")].copy()
    diag = diag.rename(columns=TRUE_COLUMN_MEANINGS).reset_index(drop=True)

    keep = ["run_id", "sequence_length",
            "tm_score_unconditional_to_forward",  # x (already correct)
            "tm_init_vs_esmfold"]                 # y (de-scrambled)
    missing = [c for c in keep if c not in diag.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {paths[-1]}: {missing}")
    out = diag[keep].rename(columns={
        "tm_score_unconditional_to_forward": "tm_lobster_fwd",
        "tm_init_vs_esmfold": "tm_esmfold",
    })
    out = out.dropna(subset=["tm_lobster_fwd", "tm_esmfold"])
    return out


def report_correlation(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: lobster forward TM vs ESMFold TM (n={len(df)}) ===")
    pr, ppv = pearsonr(df["tm_lobster_fwd"], df["tm_esmfold"])
    sr, spv = spearmanr(df["tm_lobster_fwd"], df["tm_esmfold"])
    print(f"  Pearson  r = {pr:.3f} (p={ppv:.2e})")
    print(f"  Spearman r = {sr:.3f} (p={spv:.2e})")
    print(f"  {'L':>5} {'N':>4} {'mean_fwd':>10} {'mean_esm':>10} {'r':>6}")
    for L in [100, 200, 300, 400, 500]:
        sub = df[df["sequence_length"] == L]
        if len(sub) < 3:
            continue
        r, _ = pearsonr(sub["tm_lobster_fwd"], sub["tm_esmfold"])
        print(
            f"  {L:>5} {len(sub):>4} "
            f"{sub['tm_lobster_fwd'].mean():>10.3f} "
            f"{sub['tm_esmfold'].mean():>10.3f} "
            f"{r:>6.3f}"
        )


def plot(runs: list[tuple[str, pd.DataFrame]], out_path: Path,
         qc_threshold: float = 0.8334) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 6),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    length_to_color = {L: cmap(i / 4) for i, L in enumerate([100, 200, 300, 400, 500])}

    for ax, (label, df) in zip(axes[0], runs):
        for L in [100, 200, 300, 400, 500]:
            sub = df[df["sequence_length"] == L]
            ax.scatter(
                sub["tm_lobster_fwd"], sub["tm_esmfold"],
                s=18, alpha=0.55, edgecolor="none",
                color=length_to_color[L], label=f"L={L} (n={len(sub)})",
            )
        # Identity line
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="y = x")
        # SR forward-TM threshold (samples to the LEFT of this line are SR-rejected)
        ax.axvline(qc_threshold, color="red", ls=":", lw=1, alpha=0.6,
                   label=f"SR threshold = {qc_threshold:.3f}")
        # ESMFold designability TM threshold
        ax.axhline(0.9, color="purple", ls=":", lw=1, alpha=0.6,
                   label="ESMFold TM > 0.9")
        # Pearson r
        pr, _ = pearsonr(df["tm_lobster_fwd"], df["tm_esmfold"])
        sr, _ = spearmanr(df["tm_lobster_fwd"], df["tm_esmfold"])
        ax.text(0.04, 0.96,
                f"n = {len(df)}\nPearson  r = {pr:.3f}\nSpearman = {sr:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.85))
        ax.set_xlabel("Leflur forward-fold TM\n(initial backbone vs Leflur forward-fold)")
        ax.set_ylabel("ESMFold TM\n(initial backbone vs ESMFold)")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Per-sample TM agreement: Leflur forward-fold vs ESMFold (initial design)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--label", default="run A")
    ap.add_argument("--compare-eval-dir", type=Path, default=None)
    ap.add_argument("--compare-label", default="run B")
    ap.add_argument("--out", type=Path,
                    default=Path("forward_vs_esmfold_tm.png"))
    ap.add_argument("--qc-threshold", type=float, default=0.8334123066155882,
                    help="SR forward-TM rejection threshold to draw on plot")
    args = ap.parse_args()

    runs: list[tuple[str, pd.DataFrame]] = []
    df_a = load_paired(args.eval_dir)
    report_correlation(df_a, args.label)
    runs.append((args.label, df_a))

    if args.compare_eval_dir is not None:
        df_b = load_paired(args.compare_eval_dir)
        report_correlation(df_b, args.compare_label)
        runs.append((args.compare_label, df_b))

    plot(runs, args.out, qc_threshold=args.qc_threshold)


if __name__ == "__main__":
    main()
