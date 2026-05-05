"""Per-target NLL ↔ quality scatter + within-target rank-correlation analysis.

For each `target` (CAMEO id for FF/IF; or a (length, slot) tuple for UC) we
have N candidates with both a PLL score and an external quality metric (TM
vs GT for FF/IF, or self-consistency sc-TM for UC). This script produces:

1. Scatter / swarm: x = target index (sorted by median quality), y = quality,
   colour = PLL NLL. Reveals whether PLL signal is cross-target only or
   also within-target.

2. Per-target rank correlation: for each target compute Spearman ρ between
   the PLL score and the quality metric across its N candidates. Histogram
   + summary stats (fraction with ρ ≤ -0.3 / ≤ -0.5 / > 0).

Inputs: `bestofN_*_candidates_*.csv`.

Outputs (to --output-dir, default = candidates CSV's parent):
  - <tag>_<score_col>_scatter_<ts>.png
  - <tag>_<score_col>_per_target_corr_<ts>.png
  - <tag>_<score_col>_per_target_corr_<ts>.csv

Usage examples:
    # FF, struc_score_unif (already done):
    uv run python scripts/plot_ff_struc_pll_per_target.py \\
        --candidates /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025401.csv \\
        --score-col struc_score_unif --quality-col tm_score --tag ff

    # IF, joint_true_score_unif:
    uv run python scripts/plot_ff_struc_pll_per_target.py \\
        --candidates /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse/bestofN_if_candidates_20260503T010309.csv \\
        --score-col joint_true_score_unif --quality-col esmfold_tm_score --tag if
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_ff_struc_pll")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025401.csv"
        ),
    )
    p.add_argument("--score-col", default="struc_score_unif")
    p.add_argument("--quality-col", default="tm_score")
    p.add_argument("--tag", default="ff", help="Filename prefix tag (ff / if / uc)")
    p.add_argument("--group-cols", default="target", help="Comma-separated group key columns")
    p.add_argument("--quality-label", default=None, help="Y-axis label override")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    quality_label = args.quality_label or args.quality_col

    if args.output_dir is None:
        args.output_dir = args.candidates.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    logger.info("Loading %s", args.candidates)
    df = pd.read_csv(args.candidates)
    logger.info("Loaded %d candidates over %d targets", len(df), df.target.nunique())

    df = df.dropna(subset=[args.score_col, args.quality_col]).copy()

    # Build a single 'target' column from the grouping keys (works for IF/FF
    # where it's just `target`, and UC where it's `length`+`slot`).
    if len(group_cols) == 1:
        df["_group"] = df[group_cols[0]].astype(str)
    else:
        df["_group"] = df[group_cols].astype(str).agg("_".join, axis=1)

    target_order = (
        df.groupby("_group")[args.quality_col]
        .median()
        .sort_values()
        .index.tolist()
    )
    target_to_x = {t: i for i, t in enumerate(target_order)}
    df["target_x"] = df["_group"].map(target_to_x)

    rows = []
    for tgt, sub in df.groupby("_group"):
        if len(sub) < 5:
            continue
        try:
            rho, prho = stats.spearmanr(sub[args.score_col], sub[args.quality_col])
        except Exception:
            rho, prho = float("nan"), float("nan")
        try:
            r, pr = stats.pearsonr(sub[args.score_col], sub[args.quality_col])
        except Exception:
            r, pr = float("nan"), float("nan")
        rows.append(
            {
                "target": tgt,
                "length": int(sub.length.iloc[0]) if "length" in sub.columns else -1,
                "n_candidates": len(sub),
                "median_q": float(sub[args.quality_col].median()),
                "max_q": float(sub[args.quality_col].max()),
                "min_q": float(sub[args.quality_col].min()),
                "mean_score": float(sub[args.score_col].mean()),
                "spearman_rho": float(rho),
                "spearman_p": float(prho),
                "pearson_r": float(r),
                "pearson_p": float(pr),
            }
        )
    pt = pd.DataFrame(rows).sort_values("spearman_rho")

    file_prefix = f"{args.tag}_{args.score_col}"
    csv_out = args.output_dir / f"{file_prefix}_per_target_corr_{ts}.csv"
    pt.to_csv(csv_out, index=False)
    logger.info("Wrote per-target correlations: %s", csv_out)

    print(f"\n=== Per-target correlation of {args.score_col} vs {args.quality_col} ===")
    print(f"  n_targets evaluated      : {len(pt)}")
    print(f"  Spearman ρ  mean          : {pt.spearman_rho.mean():+.3f}")
    print(f"  Spearman ρ  median        : {pt.spearman_rho.median():+.3f}")
    print(f"  Spearman ρ  fraction ≤ -0.3: {(pt.spearman_rho <= -0.3).mean()*100:.1f}%")
    print(f"  Spearman ρ  fraction ≤ -0.5: {(pt.spearman_rho <= -0.5).mean()*100:.1f}%")
    print(f"  Spearman ρ  fraction  >  0: {(pt.spearman_rho > 0).mean()*100:.1f}%")
    print(f"  Pearson r   mean          : {pt.pearson_r.mean():+.3f}")
    print(f"  Pearson r   median        : {pt.pearson_r.median():+.3f}")

    # ---------------- Plot 1: scatter swarm coloured by NLL ----------------
    fig, ax = plt.subplots(figsize=(14, 5))
    norm = Normalize(vmin=df[args.score_col].quantile(0.02), vmax=df[args.score_col].quantile(0.98))
    sc = ax.scatter(
        df.target_x + (np.random.default_rng(0).uniform(-0.25, 0.25, size=len(df))),
        df[args.quality_col],
        c=df[args.score_col],
        cmap="viridis_r",
        norm=norm,
        s=10,
        alpha=0.65,
        edgecolors="none",
    )
    medtm = df.groupby("target_x")[args.quality_col].median()
    ax.plot(medtm.index, medtm.values, color="black", lw=0.5, label="median quality")
    ax.set_xlabel(f"target (sorted by median {quality_label}, n={len(target_order)})")
    ax.set_ylabel(quality_label)
    ax.set_title(
        f"{args.tag.upper()} best-of-30 (TED ckpt, n={len(pt)} targets × candidates)\n"
        f"colour = {args.score_col} NLL (lower = better)"
    )
    ax.set_ylim(max(0, df[args.quality_col].min() - 0.05), min(1.05, df[args.quality_col].max() + 0.05))
    ax.set_xlim(-1, len(target_order))
    ax.grid(alpha=0.2)
    cb = plt.colorbar(sc, ax=ax, label=f"{args.score_col} NLL")
    fig.tight_layout()
    scatter_out = args.output_dir / f"{file_prefix}_scatter_{ts}.png"
    fig.savefig(scatter_out, dpi=140, bbox_inches="tight")
    logger.info("Wrote scatter: %s", scatter_out)
    plt.close(fig)

    # ---------------- Plot 2: per-target correlation histogram + length scatter ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.hist(pt.spearman_rho, bins=30, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(pt.spearman_rho.mean(), color="red", lw=1.5, ls="--",
               label=f"mean = {pt.spearman_rho.mean():+.3f}")
    ax.axvline(pt.spearman_rho.median(), color="orange", lw=1.5, ls="--",
               label=f"median = {pt.spearman_rho.median():+.3f}")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"Per-target Spearman ρ  ({args.score_col} vs {args.quality_col})")
    ax.set_ylabel("# targets")
    ax.set_title(
        f"Within-target rank correlation across the 30 candidates\n"
        f"frac ρ ≤ -0.3: {(pt.spearman_rho <= -0.3).mean()*100:.0f}%   |   "
        f"frac ρ ≤ -0.5: {(pt.spearman_rho <= -0.5).mean()*100:.0f}%   |   "
        f"frac ρ > 0:   {(pt.spearman_rho > 0).mean()*100:.0f}%"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.scatter(pt.length, pt.spearman_rho, c=pt.median_q, cmap="viridis", s=22, alpha=0.85,
               edgecolors="black", linewidths=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(-0.3, color="grey", lw=0.5, ls=":", label="ρ = -0.3")
    ax.axhline(-0.5, color="grey", lw=0.5, ls="--", label="ρ = -0.5")
    ax.set_xlabel("target length")
    ax.set_ylabel("Per-target Spearman ρ")
    ax.set_title(f"Per-target ρ vs target length, coloured by median {quality_label}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    plt.colorbar(ax.collections[0], ax=ax, label=f"median {quality_label}")

    fig.tight_layout()
    corr_out = args.output_dir / f"{file_prefix}_per_target_corr_{ts}.png"
    fig.savefig(corr_out, dpi=140, bbox_inches="tight")
    logger.info("Wrote correlation plot: %s", corr_out)
    plt.close(fig)


if __name__ == "__main__":
    main()
