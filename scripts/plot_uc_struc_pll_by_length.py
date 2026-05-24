"""Unconditional best-of-N: scatter of sc-TM vs length, coloured by NLL.

Companion to scripts/plot_ff_struc_pll_per_target.py but with **length on the
x-axis** (5 buckets: 100/200/300/400/500), since UC has many candidates per
length and only 5 distinct lengths. Shows whether the model's NLL signal
generalises uniformly across length bins or whether long sequences
particularly suffer.

Usage:
    uv run python scripts/plot_uc_struc_pll_by_length.py \\
        --candidates /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional/bestofN_uc_candidates_20260503T020756.csv \\
        --score-col struc_score_unif --quality-col esmfold_tm_score
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
logger = logging.getLogger("plot_uc_length")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional/bestofN_uc_candidates_20260503T020756.csv"
        ),
    )
    p.add_argument("--score-col", default="struc_score_unif")
    p.add_argument("--quality-col", default="esmfold_tm_score")
    p.add_argument("--quality-label", default="ESMFold sc-TM")
    p.add_argument("--tag", default="uc")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = args.candidates.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    logger.info("Loading %s", args.candidates)
    df = pd.read_csv(args.candidates)
    logger.info("Loaded %d candidates", len(df))

    df = df.dropna(subset=[args.score_col, args.quality_col, "length"]).copy()
    df = df[df.length > 0]  # filter junk rows
    df["length"] = df["length"].astype(int)
    lengths = sorted(df.length.unique())
    logger.info("Lengths present: %s", lengths)

    rng = np.random.default_rng(0)
    df["x_jitter"] = df["length"] + rng.uniform(-25, 25, size=len(df))

    # Per-length pooled stats
    print(f"\n=== UC: pooled per-length correlations ({args.score_col} vs {args.quality_col}) ===")
    print(f"  {'length':>6s}  {'n':>5s}  {'pearson r':>10s}  {'spearman ρ':>11s}  {'design (RMSD<2)':>16s}  {'TM mean':>8s}")
    per_len_rows = []
    for L in lengths:
        sub = df[df.length == L]
        if len(sub) < 5:
            continue
        try:
            r, _ = stats.pearsonr(sub[args.score_col], sub[args.quality_col])
        except Exception:
            r = float("nan")
        try:
            rho, _ = stats.spearmanr(sub[args.score_col], sub[args.quality_col])
        except Exception:
            rho = float("nan")
        rmsd_col = "esmfold_rmsd" if "esmfold_rmsd" in sub.columns else None
        des = float((sub[rmsd_col] < 2).mean()) * 100 if rmsd_col is not None else float("nan")
        print(
            f"  {L:>6d}  {len(sub):>5d}  {r:>+10.3f}  {rho:>+11.3f}  {des:>15.1f}%  {sub[args.quality_col].mean():>8.3f}"
        )
        per_len_rows.append({"length": L, "n": len(sub), "pearson_r": r, "spearman_rho": rho,
                             "designable_pct": des, "mean_quality": sub[args.quality_col].mean()})
    pd.DataFrame(per_len_rows).to_csv(
        args.output_dir / f"{args.tag}_{args.score_col}_per_length_corr_{ts}.csv", index=False
    )

    # ---------------- Plot 1: scatter coloured by NLL ----------------
    fig, ax = plt.subplots(figsize=(11, 5))
    norm = Normalize(
        vmin=df[args.score_col].quantile(0.02),
        vmax=df[args.score_col].quantile(0.98),
    )
    sc = ax.scatter(
        df.x_jitter,
        df[args.quality_col],
        c=df[args.score_col],
        cmap="viridis_r",
        norm=norm,
        s=10,
        alpha=0.5,
        edgecolors="none",
    )
    # Median ± IQR per length
    for L in lengths:
        sub = df[df.length == L]
        if len(sub) == 0:
            continue
        med = sub[args.quality_col].median()
        q1 = sub[args.quality_col].quantile(0.25)
        q3 = sub[args.quality_col].quantile(0.75)
        ax.plot([L - 30, L + 30], [med, med], color="black", lw=2, solid_capstyle="round")
        ax.plot([L, L], [q1, q3], color="black", lw=1, alpha=0.6)
    # Designability threshold line is at sc-RMSD<2 not on TM, so we don't draw it on TM.
    ax.set_xlabel("sequence length")
    ax.set_ylabel(args.quality_label)
    ax.set_title(
        f"UC best-of-30 (LEFLUR-P-VAL, n={len(df)} candidates over {len(lengths)} lengths × ≤10 slots × 30 draws)\n"
        f"colour = {args.score_col} NLL (lower = better)"
    )
    ax.set_xticks(lengths)
    ax.set_xlim(min(lengths) - 60, max(lengths) + 60)
    ax.set_ylim(max(0.0, df[args.quality_col].min() - 0.05), min(1.05, df[args.quality_col].max() + 0.05))
    ax.grid(alpha=0.2)
    cb = plt.colorbar(sc, ax=ax, label=f"{args.score_col} NLL")
    fig.tight_layout()
    out = args.output_dir / f"{args.tag}_{args.score_col}_scatter_by_length_{ts}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Wrote scatter: %s", out)
    plt.close(fig)

    # ---------------- Plot 2: per-(length,slot) Spearman ρ ----------------
    # Each (length, slot) defines a "target" with 30 candidates.
    pt_rows = []
    for (L, slot), sub in df.groupby(["length", "slot"]):
        if len(sub) < 5:
            continue
        try:
            rho, _ = stats.spearmanr(sub[args.score_col], sub[args.quality_col])
        except Exception:
            rho = float("nan")
        pt_rows.append({"length": int(L), "slot": int(slot), "spearman_rho": float(rho),
                        "median_q": float(sub[args.quality_col].median())})
    pt = pd.DataFrame(pt_rows)
    if not pt.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        ax.hist(pt.spearman_rho, bins=25, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(pt.spearman_rho.mean(), color="red", lw=1.5, ls="--",
                   label=f"mean = {pt.spearman_rho.mean():+.3f}")
        ax.axvline(pt.spearman_rho.median(), color="orange", lw=1.5, ls="--",
                   label=f"median = {pt.spearman_rho.median():+.3f}")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(f"Per-(length,slot) Spearman ρ  ({args.score_col} vs {args.quality_col})")
        ax.set_ylabel("# (length, slot) cells")
        ax.set_title(
            f"Within-target rank correlation across 30 candidates per cell\n"
            f"frac ρ ≤ -0.3: {(pt.spearman_rho <= -0.3).mean()*100:.0f}%   |   "
            f"frac ρ > 0:   {(pt.spearman_rho > 0).mean()*100:.0f}%"
        )
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)

        ax = axes[1]
        ax.scatter(pt.length, pt.spearman_rho, c=pt.median_q, cmap="viridis", s=22, alpha=0.85,
                   edgecolors="black", linewidths=0.3)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(-0.3, color="grey", lw=0.5, ls=":", label="ρ = -0.3")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("Per-(length,slot) Spearman ρ")
        ax.set_title("Per-target ρ vs length, coloured by median quality")
        ax.set_xticks(lengths)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        plt.colorbar(ax.collections[0], ax=ax, label="median quality")

        fig.tight_layout()
        out2 = args.output_dir / f"{args.tag}_{args.score_col}_per_cell_corr_{ts}.png"
        fig.savefig(out2, dpi=140, bbox_inches="tight")
        logger.info("Wrote per-cell correlation plot: %s", out2)
        plt.close(fig)

        print(f"\n=== UC: per-(length,slot) within-target Spearman ρ (n_cells={len(pt)}) ===")
        print(f"  mean    : {pt.spearman_rho.mean():+.3f}")
        print(f"  median  : {pt.spearman_rho.median():+.3f}")
        print(f"  frac ≤ -0.3: {(pt.spearman_rho <= -0.3).mean()*100:.1f}%")
        print(f"  frac >  0  : {(pt.spearman_rho > 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
