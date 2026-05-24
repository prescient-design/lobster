"""Correlate Gen-UME pseudo-likelihood scores with eval-dir quality metrics.

For each eval directory, joins `pll_scores_<ts>.csv` (produced by
`scripts/score_gen_ume_pll.py`) with the existing per-sample metrics CSV
(forward_folding_metrics_*, inverse_folding_metrics_*, unconditional_metrics_*),
computes Pearson + Spearman of every PLL variant against every numeric quality
column, writes a summary CSV + a markdown report, and saves a panel of
scatterplots per task.

Quality target columns (auto-detected per task):
    forward_folding:   tm_score, rmsd
    inverse_folding:   tm_score, rmsd, percent_identity, plddt
    unconditional:     tm_score_esmfold_unconditional, rmsd_esmfold_unconditional,
                       tm_score_esmfold_refined, rmsd_esmfold_refined,
                       plddt_unconditional, plddt_refined,
                       tm_score_unconditional_to_esmfold

Usage:
    uv run python scripts/correlate_pll_with_quality.py \\
        --eval-dirs <dir1> [<dir2> ...] \\
        --output-dir /cv/home/lisanzas/lobster/pll_correlation_report
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("correlate_pll_with_quality")


_PLL_COLS = [
    "seq_score_unif",
    "seq_score_arllh",
    "struc_score_unif",
    "struc_score_arllh",
    "joint_score_unif",
    "joint_score_arllh",
    "joint_true_score_unif",
    "joint_true_score_arllh",
    "seq_score_t0.25",
    "seq_score_t0.5",
    "seq_score_t0.75",
    "struc_score_t0.25",
    "struc_score_t0.5",
    "struc_score_t0.75",
]


_METRIC_PREFIX = {
    "forward_folding": "forward_folding_metrics_",
    "inverse_folding": "inverse_folding_metrics_",
    "unconditional": "unconditional_metrics_",
}

_QUALITY_TARGETS = {
    "forward_folding": ["tm_score", "rmsd"],
    "inverse_folding": ["tm_score", "rmsd", "percent_identity", "plddt"],
    # On unconditional rows: tm_score / rmsd / plddt / pae are the ESMFold-vs-unconditional
    # consistency metrics for the *initial* sample (= what we scored).
    # We also lift the SR-refined metrics from the matching self_reflection row.
    "unconditional": [
        "tm_score",
        "rmsd",
        "plddt",
        "predicted_aligned_error",
        "tm_score_esmfold_refined",
        "rmsd_esmfold_refined",
        "plddt_refined",
        "pae_refined",
        "tm_score_improvement",
        "rmsd_improvement",
    ],
}


def _augment_unconditional(metrics: pd.DataFrame) -> pd.DataFrame:
    """Lift selected SR-refined metric columns from `self_reflection_length_X_iter_Y`
    rows onto the matching `unconditional_length_X_iter_Y` rows so they can be joined
    with PLL scores in one place.
    """
    sr = metrics[metrics["run_id"].astype(str).str.startswith("self_reflection_")].copy()
    if sr.empty:
        return metrics
    # Strip prefix to get the matching key
    sr["_key"] = sr["run_id"].str.replace("^self_reflection_", "unconditional_", regex=True)
    lift_cols = [
        c
        for c in [
            "tm_score_esmfold_refined",
            "rmsd_esmfold_refined",
            "plddt_refined",
            "pae_refined",
            "tm_score_improvement",
            "rmsd_improvement",
            "percent_identity_self_reflection",
            "tm_score_unconditional_to_forward",
            "rmsd_unconditional_to_forward",
        ]
        if c in sr.columns
    ]
    sr_to_join = sr[["_key"] + lift_cols].rename(columns={"_key": "run_id"})
    base = metrics[metrics["run_id"].astype(str).str.startswith("unconditional_")].copy()
    out = base.merge(sr_to_join, on="run_id", how="left", suffixes=("", "_sr"))
    # Prefer SR-side values if base columns existed but were empty
    for c in lift_cols:
        if c in base.columns and (c + "_sr") in out.columns:
            out[c] = out[c].where(out[c].notna(), out[c + "_sr"])
            out = out.drop(columns=[c + "_sr"])
    return out


def _detect_task(eval_dir: Path) -> str:
    for task, prefix in _METRIC_PREFIX.items():
        if list(eval_dir.glob(f"{prefix}*.csv")):
            return task
    raise FileNotFoundError(f"No (forward|inverse|unconditional)_metrics_*.csv under {eval_dir}")


def _newest(globbed: list[Path]) -> Path:
    return sorted(globbed)[-1]


def _load_pll(eval_dir: Path) -> pd.DataFrame:
    matches = sorted(eval_dir.glob("pll_scores_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No pll_scores_*.csv under {eval_dir}")
    path = _newest(matches)
    df = pd.read_csv(path)
    df.attrs["source"] = str(path)
    return df


def _load_metrics(eval_dir: Path, task: str) -> pd.DataFrame:
    matches = sorted(eval_dir.glob(f"{_METRIC_PREFIX[task]}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {_METRIC_PREFIX[task]}*.csv under {eval_dir}")
    path = _newest(matches)
    df = pd.read_csv(path)
    df.attrs["source"] = str(path)
    return df


def _join(pll: pd.DataFrame, metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    """Left-join PLL onto metrics by run_id (and length sanity check)."""
    if task == "unconditional":
        m = _augment_unconditional(metrics)
    else:
        m = metrics.copy()

    pll_keep = pll[["run_id"] + [c for c in _PLL_COLS if c in pll.columns]].copy()
    pll_keep["run_id"] = pll_keep["run_id"].astype(str)
    m["run_id"] = m["run_id"].astype(str)

    joined = m.merge(pll_keep, on="run_id", how="inner")
    return joined


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    xv, yv = x[valid], y[valid]
    if np.allclose(xv.std(), 0) or np.allclose(yv.std(), 0):
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    p_r, p_p = pearsonr(xv, yv)
    s_r, s_p = spearmanr(xv, yv)
    return float(p_r), float(p_p), float(s_r), float(s_p), n


def _make_scatter(joined: pd.DataFrame, task: str, eval_name: str, out_path: Path) -> None:
    """Per-task panel: rows = quality targets, cols = key PLL variants."""
    quality_cols = [c for c in _QUALITY_TARGETS[task] if c in joined.columns]
    pll_cols = [c for c in ["seq_score_unif", "struc_score_unif", "joint_score_unif"] if c in joined.columns]
    if not quality_cols or not pll_cols:
        logger.warning("Nothing to plot for %s (quality=%s, pll=%s)", task, quality_cols, pll_cols)
        return

    nrows, ncols = len(quality_cols), len(pll_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)

    for i, qc in enumerate(quality_cols):
        for j, pc in enumerate(pll_cols):
            ax = axes[i][j]
            x = pd.to_numeric(joined[pc], errors="coerce").to_numpy()
            y = pd.to_numeric(joined[qc], errors="coerce").to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 5:
                ax.text(0.5, 0.5, "n<5", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            xv, yv = x[valid], y[valid]
            ax.scatter(xv, yv, s=8, alpha=0.5, edgecolors="none")
            try:
                p_r, _ = pearsonr(xv, yv)
                s_r, _ = spearmanr(xv, yv)
                ax.set_title(f"r={p_r:+.2f}  rho={s_r:+.2f}  n={int(valid.sum())}", fontsize=9)
            except Exception:
                pass
            if i == nrows - 1:
                ax.set_xlabel(pc, fontsize=9)
            if j == 0:
                ax.set_ylabel(qc, fontsize=9)

    fig.suptitle(f"{task}  --  {eval_name}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _per_dir_correlations(joined: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    for qc in _QUALITY_TARGETS[task]:
        if qc not in joined.columns:
            continue
        for pc in _PLL_COLS:
            if pc not in joined.columns:
                continue
            x = pd.to_numeric(joined[pc], errors="coerce").to_numpy()
            y = pd.to_numeric(joined[qc], errors="coerce").to_numpy()
            p_r, p_p, s_r, s_p, n = _safe_corr(x, y)
            rows.append(
                {
                    "task": task,
                    "quality": qc,
                    "pll": pc,
                    "n": n,
                    "pearson_r": p_r,
                    "pearson_p": p_p,
                    "spearman_r": s_r,
                    "spearman_p": s_p,
                }
            )
    return pd.DataFrame(rows)


_CAMEO_BUCKETS: list[tuple[str, int, int]] = [
    # (label, low_inclusive, high_exclusive)
    ("L<=150", 0, 151),
    ("150<L<=300", 151, 301),
    ("300<L<=450", 301, 451),
    ("L>450", 451, 10**9),
]


def _length_column(joined: pd.DataFrame) -> str | None:
    for col in ("sequence_length", "length"):
        if col in joined.columns:
            return col
    return None


def _bucket_assignments(joined: pd.DataFrame, task: str, mode: str) -> pd.Series | None:
    """Return a Series of per-row bucket labels (str), or None if no length column.

    mode = 'auto' picks 'exact' for unconditional (5 fixed lengths) and 'cameo'
    bins for the FF/IF tasks (variable per-target lengths).
    """
    length_col = _length_column(joined)
    if length_col is None:
        return None

    lengths = pd.to_numeric(joined[length_col], errors="coerce")

    chosen = mode
    if chosen == "auto":
        chosen = "exact" if task == "unconditional" else "cameo"

    if chosen == "exact":
        return lengths.apply(lambda v: f"L={int(v)}" if pd.notna(v) else None)

    if chosen == "cameo":
        def _label(v):
            if pd.isna(v):
                return None
            iv = int(v)
            for lbl, lo, hi in _CAMEO_BUCKETS:
                if lo <= iv < hi:
                    return lbl
            return None
        return lengths.apply(_label)

    raise ValueError(f"Unknown length-bucket mode: {mode}")


def _per_length_correlations(
    joined: pd.DataFrame,
    task: str,
    bucket_labels: pd.Series,
) -> pd.DataFrame:
    """One row per (bucket, quality, pll) with Pearson + Spearman + n."""
    rows = []
    df = joined.copy()
    df["__bucket"] = bucket_labels.values
    for bucket, sub in df.groupby("__bucket", sort=False, dropna=True):
        if bucket is None:
            continue
        for qc in _QUALITY_TARGETS[task]:
            if qc not in sub.columns:
                continue
            for pc in _PLL_COLS:
                if pc not in sub.columns:
                    continue
                x = pd.to_numeric(sub[pc], errors="coerce").to_numpy()
                y = pd.to_numeric(sub[qc], errors="coerce").to_numpy()
                p_r, p_p, s_r, s_p, n = _safe_corr(x, y)
                rows.append(
                    {
                        "task": task,
                        "bucket": str(bucket),
                        "quality": qc,
                        "pll": pc,
                        "n": n,
                        "pearson_r": p_r,
                        "pearson_p": p_p,
                        "spearman_r": s_r,
                        "spearman_p": s_p,
                    }
                )
    return pd.DataFrame(rows)


def _bucket_sort_key(label: str) -> tuple:
    if label.startswith("L="):
        try:
            return (0, int(label[2:]))
        except ValueError:
            return (0, 10**9)
    for i, (lbl, _lo, _hi) in enumerate(_CAMEO_BUCKETS):
        if lbl == label:
            return (1, i)
    return (2, label)


def _per_length_markdown_section(
    eval_dir: Path,
    task: str,
    per_length_corr: pd.DataFrame,
) -> str:
    """Per-length section: for each quality target, a table with one column per
    bucket and one row per PLL variant (Pearson r and Spearman ρ stacked).
    """
    name = eval_dir.name
    if per_length_corr.empty:
        return f"### Per-length splits ({name})\n\n_No length column found; skipping._\n"

    buckets = sorted(per_length_corr["bucket"].unique(), key=_bucket_sort_key)
    n_per_bucket = {
        b: int(per_length_corr.loc[per_length_corr["bucket"] == b, "n"].max())
        for b in buckets
    }

    lines = [f"### Per-length splits ({name})", ""]
    lines.append("Bucket sizes (max across rows): "
                 + ", ".join(f"`{b}`={n_per_bucket[b]}" for b in buckets))
    lines.append("")

    pll_order = [c for c in _PLL_COLS if c in per_length_corr["pll"].unique()]

    for qc in [q for q in _QUALITY_TARGETS[task] if q in per_length_corr["quality"].unique()]:
        lines.append(f"#### vs `{qc}` (Pearson r / Spearman ρ)")
        lines.append("")
        header = "| pll variant | " + " | ".join(buckets) + " |"
        sep = "|---" * (1 + len(buckets)) + "|"
        lines.append(header)
        lines.append(sep)
        sub = per_length_corr[per_length_corr["quality"] == qc]
        for pc in pll_order:
            cells = []
            for b in buckets:
                hit = sub[(sub["bucket"] == b) & (sub["pll"] == pc)]
                if hit.empty or pd.isna(hit["pearson_r"].iloc[0]):
                    cells.append("—")
                else:
                    pr = hit["pearson_r"].iloc[0]
                    sr = hit["spearman_r"].iloc[0]
                    n = int(hit["n"].iloc[0])
                    cells.append(f"{pr:+.2f}/{sr:+.2f} (n={n})")
            lines.append(f"| `{pc}` | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _markdown_section(eval_dir: Path, task: str, joined: pd.DataFrame, corr: pd.DataFrame) -> str:
    name = eval_dir.name
    lines = [f"## {name} (task={task}, n={len(joined)})", ""]
    for qc in [q for q in _QUALITY_TARGETS[task] if q in joined.columns]:
        sub = corr[corr["quality"] == qc].sort_values("pearson_r", key=lambda s: s.abs(), ascending=False).head(8)
        lines.append(f"### vs `{qc}`")
        lines.append("")
        lines.append("| pll variant | n | pearson r | pearson p | spearman r | spearman p |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| `{r['pll']}` | {r['n']} | {r['pearson_r']:+.3f} | {r['pearson_p']:.2e} | {r['spearman_r']:+.3f} | {r['spearman_p']:.2e} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dirs", nargs="+", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--no-scatter",
        action="store_true",
        help="Skip scatter plot panels (just write CSV + markdown).",
    )
    p.add_argument(
        "--per-length",
        action="store_true",
        help=(
            "Also compute per-length-bucket Pearson/Spearman tables. "
            "For unconditional, buckets = exact sequence_length values "
            "(typically {100,200,300,400,500}); for FF/IF, buckets = "
            "L<=150 / 150<L<=300 / 300<L<=450 / L>450."
        ),
    )
    p.add_argument(
        "--length-bucket-mode",
        choices=("auto", "exact", "cameo"),
        default="auto",
        help=(
            "How to bucket lengths for --per-length. 'auto' = 'exact' for "
            "unconditional, 'cameo' bins for FF/IF. 'exact' groups by unique "
            "sequence_length; 'cameo' uses fixed length buckets."
        ),
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", args.output_dir)

    all_corr_frames: list[pd.DataFrame] = []
    all_per_length_frames: list[pd.DataFrame] = []
    md_blocks: list[str] = []
    per_length_md_blocks: list[str] = []
    summary_meta: list[dict] = []

    for eval_dir in args.eval_dirs:
        if not eval_dir.is_dir():
            logger.error("Skipping missing dir: %s", eval_dir)
            continue
        try:
            task = _detect_task(eval_dir)
            pll = _load_pll(eval_dir)
            metrics = _load_metrics(eval_dir, task)
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", eval_dir, e)
            continue

        joined = _join(pll, metrics, task)
        n_joined = len(joined)
        if n_joined == 0:
            logger.warning("Empty join for %s (task=%s); skipping", eval_dir, task)
            continue

        logger.info(
            "%s [task=%s]: joined %d rows (pll=%d, metrics=%d)",
            eval_dir.name,
            task,
            n_joined,
            len(pll),
            len(metrics),
        )

        corr = _per_dir_correlations(joined, task)
        corr["eval_dir"] = eval_dir.name
        all_corr_frames.append(corr)

        if not args.no_scatter:
            scatter_path = args.output_dir / f"scatter_{task}_{eval_dir.name}.png"
            _make_scatter(joined, task, eval_dir.name, scatter_path)
            logger.info("  wrote %s", scatter_path.name)

        joined.to_csv(args.output_dir / f"joined_{task}_{eval_dir.name}.csv", index=False)
        md_blocks.append(_markdown_section(eval_dir, task, joined, corr))

        if args.per_length:
            buckets = _bucket_assignments(joined, task, args.length_bucket_mode)
            if buckets is None:
                logger.warning("No length column on joined CSV for %s; skipping per-length", eval_dir.name)
            else:
                pl_corr = _per_length_correlations(joined, task, buckets)
                pl_corr["eval_dir"] = eval_dir.name
                all_per_length_frames.append(pl_corr)
                per_length_md_blocks.append(_per_length_markdown_section(eval_dir, task, pl_corr))

        summary_meta.append(
            {
                "eval_dir": str(eval_dir),
                "task": task,
                "n_joined": n_joined,
                "n_pll": len(pll),
                "n_metrics": len(metrics),
                "pll_source": pll.attrs.get("source", ""),
                "metrics_source": metrics.attrs.get("source", ""),
            }
        )

    if not all_corr_frames:
        logger.error("No data collected. Exiting.")
        return

    full_corr = pd.concat(all_corr_frames, ignore_index=True)
    full_corr.to_csv(args.output_dir / "pll_quality_correlations.csv", index=False)
    logger.info("Wrote correlation table: %d rows", len(full_corr))

    md_path = args.output_dir / "pll_quality_correlation.md"
    with md_path.open("w") as fh:
        fh.write(f"# Gen-UME PLL vs quality correlations\n\n")
        fh.write(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n\n")
        fh.write("## Index\n\n")
        for meta in summary_meta:
            fh.write(f"- **{Path(meta['eval_dir']).name}** (task={meta['task']}, n={meta['n_joined']})\n")
        fh.write("\n")
        for block in md_blocks:
            fh.write(block)
            fh.write("\n")
    logger.info("Wrote markdown report: %s", md_path)

    if args.per_length and all_per_length_frames:
        pl_full = pd.concat(all_per_length_frames, ignore_index=True)
        pl_csv_path = args.output_dir / "pll_quality_correlations_per_length.csv"
        pl_full.to_csv(pl_csv_path, index=False)
        logger.info("Wrote per-length correlation table: %d rows -> %s", len(pl_full), pl_csv_path)

        pl_md_path = args.output_dir / "pll_quality_correlation_per_length.md"
        with pl_md_path.open("w") as fh:
            fh.write("# Gen-UME PLL vs quality correlations -- per-length splits\n\n")
            fh.write(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n\n")
            fh.write(
                "Each cell shows `Pearson r / Spearman \u03c1 (n=...)`. Bucketing mode = "
                f"`{args.length_bucket_mode}` (auto = exact for unconditional, "
                "L<=150 / 150-300 / 300-450 / >450 for FF/IF).\n\n"
            )
            for block in per_length_md_blocks:
                fh.write(block)
                fh.write("\n")
        logger.info("Wrote per-length markdown report: %s", pl_md_path)

    with (args.output_dir / "summary.json").open("w") as fh:
        json.dump(summary_meta, fh, indent=2)


if __name__ == "__main__":
    main()
