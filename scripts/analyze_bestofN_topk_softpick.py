"""Top-K soft-pick analysis for best-of-N PLL runs.

Question: does picking the absolute argmin/argmax by NLL over-select for the
model's pathologies? If so, sampling uniformly from the top-K should be more
robust at the cost of mean quality.

For each task (IF / FF / UC) and each PLL picker variant, compares:
  - hard pick   = current argmin/argmax
  - top-K soft  = take best K candidates by score, then uniform random
                  among them (averaged over `--repeats` random seeds)

Prints per-picker tables: hard mean, top-K mean (Δ vs hard), single-shot
baseline, oracle. Also runs paired McNemar (binary: designable) and Wilcoxon
(continuous: rmsd / tm) of top-K-soft vs the current hard pick.

Usage:
    uv run python scripts/analyze_bestofN_topk_softpick.py \\
        --top-k 5 --repeats 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("topk_softpick")


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

_TASKS = {
    "IF": {
        "csv": "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse/bestofN_if_candidates_20260503T010309.csv",
        "group_cols": ["target"],
        "pickers": {
            "seq_pll": ("seq_score_unif", "argmin"),
            "struc_pll": ("struc_score_unif", "argmin"),
            "joint_pll_sum": ("joint_score_unif", "argmin"),
            "joint_true_pll": ("joint_true_score_unif", "argmin"),
            "oracle_tm": ("esmfold_tm_score", "argmax"),
        },
        "quality_metrics": {
            "tm": "esmfold_tm_score",
            "rmsd": "esmfold_rmsd",
            "plddt": "esmfold_plddt",
            "aar": "aar",
        },
        "designable_col": "esmfold_rmsd",
        "designable_op": lambda v: v < 2.0,
    },
    "FF": {
        "csv": "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025401.csv",
        "group_cols": ["target"],
        "pickers": {
            "seq_pll": ("seq_score_unif", "argmin"),
            "struc_pll": ("struc_score_unif", "argmin"),
            "joint_pll_sum": ("joint_score_unif", "argmin"),
            "oracle_tm": ("tm_score", "argmax"),
        },
        "quality_metrics": {
            "tm": "tm_score",
            "rmsd": "rmsd",
        },
        # FF "designable" = TM > 0.5 (foldable)
        "designable_col": "tm_score",
        "designable_op": lambda v: v > 0.5,
        "extra_thresholds": {"tm_gt_0.8": ("tm_score", lambda v: v > 0.8)},
    },
    "UC": {
        "csv": "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional/bestofN_uc_candidates_20260503T020756.csv",
        "group_cols": ["length", "slot"],
        "pickers": {
            "seq_pll": ("seq_score_unif", "argmin"),
            "struc_pll": ("struc_score_unif", "argmin"),
            "joint_pll_sum": ("joint_score_unif", "argmin"),
            "joint_true_pll": ("joint_true_score_unif", "argmin"),
            "oracle_tm": ("esmfold_tm_score", "argmax"),
        },
        "quality_metrics": {
            "tm": "esmfold_tm_score",
            "rmsd": "esmfold_rmsd",
            "plddt": "esmfold_plddt",
        },
        "designable_col": "esmfold_rmsd",
        "designable_op": lambda v: v < 2.0,
    },
}


# ---------------------------------------------------------------------------
# Picker implementations
# ---------------------------------------------------------------------------


def _hard_pick(group: pd.DataFrame, score_col: str, mode: str) -> pd.Series:
    """Return the row chosen by argmin/argmax."""
    valid = group[score_col].notna()
    sub = group[valid]
    if len(sub) == 0:
        return group.iloc[0]
    if mode == "argmin":
        return sub.loc[sub[score_col].idxmin()]
    elif mode == "argmax":
        return sub.loc[sub[score_col].idxmax()]
    else:
        raise ValueError(mode)


def _topk_indices(group: pd.DataFrame, score_col: str, mode: str, k: int) -> np.ndarray:
    """Return positional indices (within the group) of the top-K by score."""
    valid = group[score_col].notna().values
    if not valid.any():
        return np.array([0])
    sorted_idx = np.argsort(group[score_col].values)
    if mode == "argmax":
        sorted_idx = sorted_idx[::-1]
    # Filter NaN positions to the back
    sorted_idx = [i for i in sorted_idx if valid[i]]
    return np.array(sorted_idx[: max(1, k)])


def _topk_soft_pick_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
    score_col: str,
    mode: str,
    k: int,
    metric_cols: list[str],
    repeats: int,
    seed: int = 42,
) -> dict:
    """For each group, sample one of top-K candidates per repeat; return per-group
    mean across repeats for each metric (so we get a stable per-group estimate),
    plus the per-repeat all-group mean (for SE).
    """
    rng = np.random.default_rng(seed)

    # Per-group mean over repeats (one row per group per metric)
    per_group_mean = {m: [] for m in metric_cols}
    group_keys = []

    # Per-repeat global mean (for SE across repeats)
    per_repeat_global = {m: np.zeros(repeats) for m in metric_cols}

    for gid, g in df.groupby(list(group_cols), sort=False):
        topk = _topk_indices(g, score_col, mode, k)
        chosen = rng.choice(topk, size=repeats, replace=True)

        # Stack metrics for the chosen rows
        for m in metric_cols:
            vals = g[m].values
            picked = vals[chosen]  # [repeats]
            per_group_mean[m].append(np.nanmean(picked))
        group_keys.append(gid)

    # Compute the global means across groups for each repeat — for SE estimate
    # We re-iterate to compute per-repeat means (to estimate variance across repeats).
    rng2 = np.random.default_rng(seed)
    metrics_per_repeat = {m: [] for m in metric_cols}
    for gid, g in df.groupby(list(group_cols), sort=False):
        topk = _topk_indices(g, score_col, mode, k)
        chosen = rng2.choice(topk, size=repeats, replace=True)
        for m in metric_cols:
            metrics_per_repeat[m].append(g[m].values[chosen])

    for m in metric_cols:
        arr = np.array(metrics_per_repeat[m])  # [n_groups, repeats]
        per_repeat_global[m] = np.nanmean(arr, axis=0)  # [repeats]

    return {
        "per_group_mean": per_group_mean,
        "group_keys": group_keys,
        "per_repeat_global": per_repeat_global,
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _mcnemar_pair(a_pass: np.ndarray, b_pass: np.ndarray) -> tuple[float, int, int]:
    """McNemar exact: H1 a != b. b/c discordance pair counts."""
    a_pass = np.asarray(a_pass, dtype=bool)
    b_pass = np.asarray(b_pass, dtype=bool)
    b = int(((~a_pass) & b_pass).sum())  # a fail, b pass
    c = int((a_pass & (~b_pass)).sum())  # a pass, b fail
    n_disc = b + c
    if n_disc == 0:
        return float("nan"), b, c
    p = stats.binomtest(c, n_disc, 0.5, alternative="two-sided").pvalue
    return p, b, c


def _wilcoxon_pair(a: np.ndarray, b: np.ndarray, alt: str = "two-sided") -> float:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5:
        return float("nan")
    try:
        return stats.wilcoxon(s.a, s.b, alternative=alt, zero_method="wilcox").pvalue
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Per-task analysis
# ---------------------------------------------------------------------------


def analyze_task(name: str, cfg: dict, k: int, repeats: int, seed: int) -> dict:
    df = pd.read_csv(cfg["csv"])
    group_cols = cfg["group_cols"]
    metrics = cfg["quality_metrics"]
    metric_cols = list(metrics.values())
    desig_col = cfg["designable_col"]
    desig_op = cfg["designable_op"]
    extra_thresholds = cfg.get("extra_thresholds", {})

    n_groups = df.groupby(list(group_cols), sort=False).ngroups
    print(f"\n{'='*100}")
    print(f"TASK = {name}    n_groups={n_groups}    n_candidates={len(df)}    top-K = {k}    repeats = {repeats}")
    print(f"{'='*100}")

    # ----- baseline + oracle reference -----
    # Single-shot baseline = candidate_idx=0 per group
    baseline_rows = (
        df.sort_values("candidate_idx").groupby(list(group_cols), sort=False).head(1)
    )
    base_metrics = {m: baseline_rows[c].mean() for m, c in metrics.items()}
    base_pass = desig_op(baseline_rows[desig_col]).mean() * 100
    base_extra = {
        name_t: op(baseline_rows[col]).mean() * 100 for name_t, (col, op) in extra_thresholds.items()
    }

    print("\nBaseline (single-shot, candidate_idx=0):")
    for m, v in base_metrics.items():
        print(f"  {m:12s} = {v:.4f}")
    print(f"  designable    = {base_pass:.1f}%   (threshold on {desig_col})")
    for nm, v in base_extra.items():
        print(f"  {nm:12s} = {v:.1f}%")

    # ----- per-picker comparison -----
    print("\nPer-picker hard vs top-K soft (mean over groups):")
    header = f"  {'picker':18s} {'mode':6s} {'metric':12s} {'hard':>9s} {'top-K':>9s} {'delta':>9s} {'McN p':>9s} {'b/c':>9s} {'Wilcox p':>10s}"
    print(header)

    results = {}

    for picker, (score_col, mode) in cfg["pickers"].items():
        # ---- hard pick (argmin/argmax) ----
        hard_rows = df.groupby(list(group_cols), sort=False).apply(
            lambda g: _hard_pick(g, score_col, mode)
        )
        # ---- top-K soft (averaged over repeats per group) ----
        soft = _topk_soft_pick_metrics(
            df, group_cols, score_col, mode, k, metric_cols, repeats, seed=seed
        )

        # Per-group hard metrics
        hard_metric_arr = {m: hard_rows[c].values for m, c in metrics.items()}
        hard_pass = desig_op(hard_rows[desig_col]).values
        hard_extra = {
            nm: op(hard_rows[col]).values for nm, (col, op) in extra_thresholds.items()
        }

        # Soft per-group means
        soft_metric_arr = {m: np.array(soft["per_group_mean"][c]) for m, c in metrics.items()}

        # Soft "designable" rate per group = expected pass-rate over repeats.
        # Compute it explicitly via a parallel run on the boolean column.
        soft_pass_per_group = []
        rng = np.random.default_rng(seed + 1)
        for gid, g in df.groupby(list(group_cols), sort=False):
            topk = _topk_indices(g, score_col, mode, k)
            cand_pass = desig_op(g[desig_col]).values[topk]
            chosen = rng.choice(topk, size=repeats, replace=True)
            cand_pass_full = desig_op(g[desig_col]).values
            soft_pass_per_group.append(np.nanmean(cand_pass_full[chosen]))
        soft_pass_per_group = np.array(soft_pass_per_group)

        # Print per-metric rows
        for mname, mcol in metrics.items():
            hard_mean = np.nanmean(hard_metric_arr[mname])
            soft_mean = np.nanmean(soft_metric_arr[mname])
            delta = soft_mean - hard_mean
            # Wilcoxon: paired soft vs hard (per-group means)
            wp = _wilcoxon_pair(soft_metric_arr[mname], hard_metric_arr[mname], alt="two-sided")
            print(
                f"  {picker:18s} {mode:6s} {mname:12s} {hard_mean:9.4f} {soft_mean:9.4f} "
                f"{delta:+9.4f} {'':>9s} {'':>9s} {wp:>10.4f}"
            )

        # Designable row (binary)
        hard_pass_pct = hard_pass.mean() * 100
        soft_pass_pct = soft_pass_per_group.mean() * 100
        delta_pass = soft_pass_pct - hard_pass_pct
        # McNemar: compare hard pass vs soft "expected pass" — but McNemar needs
        # binary on each group. Use soft majority (>=0.5) for the binary table.
        soft_pass_binary = soft_pass_per_group >= 0.5
        mp, b, c = _mcnemar_pair(hard_pass, soft_pass_binary)
        print(
            f"  {picker:18s} {mode:6s} {'desig (%)':12s} {hard_pass_pct:9.1f} {soft_pass_pct:9.1f} "
            f"{delta_pass:+9.1f} {mp:>9.4f} {b:>4d}/{c:<4d}"
        )

        for nm, (col, op) in extra_thresholds.items():
            hard_pct = hard_extra[nm].mean() * 100
            # Soft "expected pct" — re-do with op on col
            soft_pct_per_group = []
            rng = np.random.default_rng(seed + 7)
            for gid, g in df.groupby(list(group_cols), sort=False):
                topk = _topk_indices(g, score_col, mode, k)
                chosen = rng.choice(topk, size=repeats, replace=True)
                soft_pct_per_group.append(np.nanmean(op(g[col]).values[chosen]))
            soft_pct_per_group = np.array(soft_pct_per_group)
            soft_pct = soft_pct_per_group.mean() * 100
            delta_pct = soft_pct - hard_pct
            soft_pct_binary = soft_pct_per_group >= 0.5
            mp, b, c = _mcnemar_pair(hard_extra[nm], soft_pct_binary)
            print(
                f"  {picker:18s} {mode:6s} {nm:12s} {hard_pct:9.1f} {soft_pct:9.1f} "
                f"{delta_pct:+9.1f} {mp:>9.4f} {b:>4d}/{c:<4d}"
            )

        results[picker] = {
            "hard_metric": {m: float(np.nanmean(hard_metric_arr[m])) for m in metrics},
            "soft_metric": {m: float(np.nanmean(soft_metric_arr[m])) for m in metrics},
            "hard_pass": float(hard_pass_pct),
            "soft_pass": float(soft_pass_pct),
        }

    return {
        "baseline": base_metrics | {"designable_pct": float(base_pass)},
        "pickers": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", default="IF,FF,UC", help="Comma-separated subset of {IF,FF,UC}")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary = {}
    for t in tasks:
        if t not in _TASKS:
            logger.warning("Unknown task %s; skipping", t)
            continue
        summary[t] = analyze_task(t, _TASKS[t], args.top_k, args.repeats, args.seed)


if __name__ == "__main__":
    main()
