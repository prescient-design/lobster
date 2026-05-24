"""Across-target scatter: PLL vs quality (one dot per target).

For the protein-ligand E0 study (N=1 per target), per-target rank
correlations are undefined — but the across-target axis is informative.
This script makes 2-panel (ALL | PLINDER) scatter plots, one per
(task, quality) pair, with one dot per target, a least-squares line,
and Spearman ρ + Pearson r in each panel title.

By default the PLL variant plotted is the one chosen as best for each
(ckpt, quality) by `correlate_pll_quality_ligand.py`, but you can override
via `--variant`.

Usage:
    uv run python scripts/plot_pll_vs_quality_scatter_ligand.py \\
        --report-root /cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand \\
        --tasks ff if cg
"""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Headline quality columns (matches correlate_pll_quality_ligand.py).
TASK_QUALITY = {
    "ff": ["tm_score", "rmsd", "ligand_rmsd", "ligand_centroid_distance"],
    "if": ["aar"],
    "cg": ["cofold_iptm", "cofold_ligand_iptm", "cofold_iPDE",
           "cofold_complex_pde", "cofold_ptm", "cofold_complex_plddt",
           "cofold_both_pass", "tm_to_gt", "rmsd_to_gt", "pseudo_aar",
           "ligand_pocket_min_dist"],
}

PLL_VARIANTS = [
    "seq_score_unif", "struc_score_unif",
    "lig_atom_score_unif", "lig_struc_score_unif",
    "joint_protein_score_unif", "joint_ligand_score_unif",
    "joint_all_score_unif", "joint_true_4_score_unif",
]


def _latest_csv(d: Path, glob_pat: str) -> Path | None:
    matches = sorted(d.glob(glob_pat))
    return matches[-1] if matches else None


def _best_variant(df: pd.DataFrame, quality: str) -> str | None:
    """Pick the PLL variant with the largest |Spearman r| on this CSV."""
    best, best_abs = None, -1.0
    y = df[quality].to_numpy(dtype=float)
    for v in PLL_VARIANTS:
        if v not in df.columns:
            continue
        x = df[v].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            continue
        try:
            r, _ = stats.spearmanr(x[m], y[m])
        except Exception:
            continue
        if not np.isfinite(r):
            continue
        if abs(r) > best_abs:
            best_abs, best = abs(r), v
    return best


def _scatter_panel(ax, x: np.ndarray, y: np.ndarray, *, xlab: str, ylab: str,
                   title_prefix: str) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        ax.text(0.5, 0.5, "n<5", ha="center", va="center")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"{title_prefix} (n={int(m.sum())})")
        return
    x, y = x[m], y[m]
    ax.scatter(x, y, s=22, alpha=0.7, edgecolors="black", linewidths=0.3,
               color="steelblue")
    try:
        slope, intercept, *_ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.2, alpha=0.85)
    except Exception:
        pass
    rho, prho = stats.spearmanr(x, y)
    r, pr = stats.pearsonr(x, y)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(
        f"{title_prefix}  (n={len(x)})\n"
        f"Spearman ρ = {rho:+.3f}  (p={prho:.1e})    "
        f"Pearson r = {r:+.3f}  (p={pr:.1e})",
        fontsize=10,
    )
    ax.grid(alpha=0.25)


def _process(task: str, quality: str, ckpt_csvs: dict[str, Path],
             variant_override: str | None, out_root: Path, ts: str) -> None:
    panels = {}
    chosen_variants = {}
    for ckpt, csv in ckpt_csvs.items():
        if csv is None or not csv.exists():
            continue
        df = pd.read_csv(csv)
        if quality not in df.columns:
            continue
        v = variant_override or _best_variant(df, quality)
        if v is None or v not in df.columns:
            continue
        chosen_variants[ckpt] = v
        panels[ckpt] = (df[v].to_numpy(dtype=float),
                        df[quality].to_numpy(dtype=float))
    if not panels:
        print(f"[skip] {task}/{quality}: no usable data")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.0), squeeze=False)
    for ax, (ckpt, (x, y)) in zip(axes[0], panels.items()):
        _scatter_panel(
            ax, x, y,
            xlab=f"{chosen_variants[ckpt]} (lower = more confident)",
            ylab=quality,
            title_prefix=f"{ckpt.upper()} / {task.upper()}",
        )
    fig.suptitle(f"E0 {task.upper()} — {quality} vs PLL  (1 dot / target, N=1)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = out_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scatter_{task}_{quality}_{ts}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}  ({list(chosen_variants.values())})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report-root", required=True, type=Path)
    p.add_argument("--ckpts", nargs="+", default=["all", "plinder"])
    p.add_argument("--tasks", nargs="+", default=["ff", "if", "cg"])
    p.add_argument("--subdir", default="full")
    p.add_argument("--variant", default=None,
                   help="Force a specific PLL variant column (default: per-quality best)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to <report-root>/scatter")
    args = p.parse_args()

    out_root = args.output_dir or (args.report_root / "scatter")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    for task in args.tasks:
        ckpt_csvs: dict[str, Path | None] = {}
        for ckpt in args.ckpts:
            d = args.report_root / ckpt / task / args.subdir
            cand = (_latest_csv(d, "bestofN_cg_lig_candidates_*_with_cofold.csv")
                    if task == "cg" else None) \
                or _latest_csv(d, f"bestofN_{task}_lig_candidates_*.csv")
            ckpt_csvs[ckpt] = cand
            if cand:
                print(f"[load] {ckpt}/{task} <- {cand.name}")
        for q in TASK_QUALITY[task]:
            _process(task, q, ckpt_csvs, args.variant, out_root, ts)


if __name__ == "__main__":
    main()
