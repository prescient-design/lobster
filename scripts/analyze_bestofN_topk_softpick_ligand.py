"""Top-K soft-pick analysis for the protein-ligand best-of-N runs (FF / IF / CG).

Mirrors the protein-only `scripts/analyze_bestofN_topk_softpick.py`. For each
PLL ranker, compares:
  - hard pick   = argmin (PLL) or argmax (oracle)
  - top-K soft  = uniform random over the top-K candidates by score, averaged
                  over `--repeats` random seeds for a stable per-target estimate.

Question: does ligand-aware argmin amplify pathologies (analogous to the
V-bias degeneracy that protein-only `seq_pll`/`joint_true_pll` showed under
LEFLUR-P-VAL unconditional decoding, where top-5 soft-pick recovered +8 to
+11 pp designability)? Run this on the IF and CG candidates CSVs once the
N=30 jobs land; re-run with `--candidates <…>_with_cofold.csv` to switch
the headline metric to ipTM / RF3-pass.

For each picker, prints per-quality hard / top-K / Δ + paired Wilcoxon, and
binary McNemar on each pass-threshold. Also prints the single-shot baseline
and oracle for reference.

Usage:
    uv run python scripts/analyze_bestofN_topk_softpick_ligand.py \\
        --task ff --top-k 5 --repeats 1000 \\
        --candidates /cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand/all/ff/full_N30/bestofN_ff_lig_candidates_*.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("topk_softpick_ligand")


# 4-modality picker set. PLL = argmin, oracle = argmax.
_PLL_PICKERS = {
    "seq_pll":           ("seq_score_unif",            "argmin"),
    "struc_pll":         ("struc_score_unif",          "argmin"),
    "lig_atom_pll":      ("lig_atom_score_unif",       "argmin"),
    "lig_struc_pll":     ("lig_struc_score_unif",      "argmin"),
    "joint_protein_pll": ("joint_protein_score_unif",  "argmin"),
    "joint_ligand_pll":  ("joint_ligand_score_unif",   "argmin"),
    "joint_all_pll":     ("joint_all_score_unif",      "argmin"),
    "joint_true_4_pll":  ("joint_true_4_score_unif",   "argmin"),
}


_TASKS = {
    "ff": {
        "group_cols": ["pdb_id"],
        "metrics": {"tm": "tm_score", "rmsd": "rmsd",
                    "lig_rmsd": "ligand_rmsd",
                    "lig_centroid": "ligand_centroid_distance"},
        "designable_col": "tm_score",
        "designable_op": lambda v: v > 0.5,
        "extra_thresholds": {"tm_gt_0.7": ("tm_score", lambda v: v > 0.7),
                             "tm_gt_0.8": ("tm_score", lambda v: v > 0.8)},
        "oracle_pick": ("oracle_tm", "tm_score", "argmax"),
    },
    "if": {
        "group_cols": ["pdb_id"],
        "metrics": {"aar": "aar"},
        "designable_col": None,
        "designable_op": None,
        "extra_thresholds": {"aar_gt_0.30": ("aar", lambda v: v > 0.30),
                             "aar_gt_0.40": ("aar", lambda v: v > 0.40)},
        "oracle_pick": ("oracle_aar", "aar", "argmax"),
    },
    "cg": {
        "group_cols": ["pdb_id"],
        "metrics": {"tm_to_gt": "tm_to_gt", "rmsd_to_gt": "rmsd_to_gt",
                    "pseudo_aar": "pseudo_aar",
                    "lig_pocket_min_dist": "ligand_pocket_min_dist"},
        "designable_col": "tm_to_gt",
        "designable_op": lambda v: v > 0.5,
        "extra_thresholds": {"tm_gt_0.7": ("tm_to_gt", lambda v: v > 0.7)},
        "oracle_pick": ("oracle_tm", "tm_to_gt", "argmax"),
    },
}


# Cofold-augmented metrics added when the candidates CSV has been merged with
# Boltz2 cofold output via merge_{cg,if}_cofold_into_candidates.py.
_COFOLD_METRICS = {"iptm": "cofold_iptm",
                   "ligand_iptm": "cofold_ligand_iptm",
                   "complex_ipde": "cofold_complex_ipde",
                   "complex_plddt": "cofold_complex_plddt"}
_COFOLD_THRESHOLDS = {"ligand_iptm_ge_0.9": ("cofold_ligand_iptm", lambda v: v >= 0.9),
                      "ligand_iptm_ge_0.7": ("cofold_ligand_iptm", lambda v: v >= 0.7),
                      "complex_ipde_le_1.0": ("cofold_complex_ipde", lambda v: v <= 1.0),
                      "both_pass": ("cofold_both_pass", lambda v: v >= 0.5)}


def _resolve(arg: str) -> Path:
    paths = sorted(glob.glob(arg)) if "*" in arg else [arg]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError(arg)
    return Path(paths[-1])


def _hard_pick(g: pd.DataFrame, score_col: str, mode: str) -> pd.Series:
    valid = g[score_col].notna()
    sub = g[valid]
    if len(sub) == 0:
        return g.iloc[0]
    return sub.loc[sub[score_col].idxmin()] if mode == "argmin" else sub.loc[sub[score_col].idxmax()]


def _topk_indices(g: pd.DataFrame, score_col: str, mode: str, k: int) -> np.ndarray:
    vals = g[score_col].values
    valid = ~pd.isna(vals)
    if not valid.any():
        return np.array([0])
    order = np.argsort(vals)
    if mode == "argmax":
        order = order[::-1]
    order = [i for i in order if valid[i]]
    return np.array(order[: max(1, k)])


def _mcnemar(a_pass: np.ndarray, b_pass: np.ndarray) -> tuple[float, int, int]:
    a = np.asarray(a_pass, dtype=bool); b = np.asarray(b_pass, dtype=bool)
    bb = int(((~a) & b).sum()); cc = int((a & (~b)).sum())
    n = bb + cc
    if n == 0:
        return float("nan"), bb, cc
    return stats.binomtest(cc, n, 0.5, alternative="two-sided").pvalue, bb, cc


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5:
        return float("nan")
    try:
        return stats.wilcoxon(s.a, s.b, alternative="two-sided", zero_method="wilcox").pvalue
    except ValueError:
        return float("nan")


def analyze(task: str, candidates_path: Path, k: int, repeats: int, seed: int) -> None:
    cfg = _TASKS[task]
    df = pd.read_csv(candidates_path)
    if "pdb_id" not in df.columns or "candidate_idx" not in df.columns:
        raise ValueError(f"{candidates_path}: missing pdb_id / candidate_idx")

    metrics = dict(cfg["metrics"])
    extra_thr = dict(cfg["extra_thresholds"])
    # Augment with cofold metrics if present
    cofold_present = any(c.startswith("cofold_") for c in df.columns)
    if cofold_present:
        for name, col in _COFOLD_METRICS.items():
            if col in df.columns:
                metrics[name] = col
        for name, (col, op) in _COFOLD_THRESHOLDS.items():
            if col in df.columns:
                extra_thr[name] = (col, op)

    pickers = {n: (col, m) for n, (col, m) in _PLL_PICKERS.items() if col in df.columns}
    oracle_name, oracle_col, oracle_mode = cfg["oracle_pick"]
    if oracle_col in df.columns:
        pickers[oracle_name] = (oracle_col, oracle_mode)
    if cofold_present and "cofold_iptm" in df.columns:
        pickers["oracle_iptm"] = ("cofold_iptm", "argmax")
    if cofold_present and "cofold_both_pass" in df.columns:
        pickers["oracle_both_pass"] = ("cofold_both_pass", "argmax")

    n_groups = df.groupby(cfg["group_cols"], sort=False).ngroups
    print(f"\n{'='*100}")
    print(f"TASK = {task.upper()}    n_groups={n_groups}    n_candidates={len(df)}    "
          f"top-K={k}    repeats={repeats}    cofold={'yes' if cofold_present else 'no'}")
    print(f"{'='*100}")

    # Single-shot baseline = candidate_idx=0 per group
    base = df.sort_values("candidate_idx").groupby(cfg["group_cols"], sort=False).head(1)
    print("\nBaseline (single-shot, candidate_idx=0):")
    for mname, mcol in metrics.items():
        if mcol in base.columns:
            print(f"  {mname:24s} = {base[mcol].mean():.4f}")
    if cfg["designable_op"] is not None and cfg["designable_col"] in base.columns:
        bp = cfg["designable_op"](base[cfg["designable_col"]]).mean() * 100
        print(f"  {'designable':24s} = {bp:.1f}%   (rule on {cfg['designable_col']})")
    for nm, (col, op) in extra_thr.items():
        if col in base.columns:
            print(f"  {nm:24s} = {op(base[col]).mean()*100:.1f}%")

    # Per picker
    print("\nPer-picker hard vs top-K soft (mean over groups):")
    print(f"  {'picker':22s} {'mode':6s} {'metric':24s} {'hard':>9s} {'top-K':>9s} "
          f"{'delta':>9s} {'McN p':>9s} {'b/c':>10s} {'Wilcox p':>10s}")

    for picker, (score_col, mode) in pickers.items():
        # hard pick
        hard_rows = df.groupby(cfg["group_cols"], sort=False).apply(
            lambda g: _hard_pick(g, score_col, mode)
        )
        # top-K per group, averaged metric per group
        rng_seed = seed
        per_group_topk = {m: [] for m in metrics}
        for _, g in df.groupby(cfg["group_cols"], sort=False):
            topk = _topk_indices(g, score_col, mode, k)
            rng = np.random.default_rng(rng_seed)
            chosen = rng.choice(topk, size=repeats, replace=True)
            for mname, mcol in metrics.items():
                if mcol in g.columns:
                    vals = g[mcol].values[chosen]
                    per_group_topk[mname].append(np.nanmean(vals))
                else:
                    per_group_topk[mname].append(np.nan)
            rng_seed += 1

        for mname, mcol in metrics.items():
            if mcol not in df.columns:
                continue
            hard_arr = pd.to_numeric(hard_rows[mcol], errors="coerce").values
            soft_arr = np.array(per_group_topk[mname])
            wp = _wilcoxon(soft_arr, hard_arr)
            dh, ds = np.nanmean(hard_arr), np.nanmean(soft_arr)
            print(f"  {picker:22s} {mode:6s} {mname:24s} {dh:9.4f} {ds:9.4f} "
                  f"{ds-dh:+9.4f} {'':>9s} {'':>10s} {wp:>10.4f}")

        # binary thresholds (McNemar)
        rng_seed = seed + 1
        all_thresholds = list(extra_thr.items())
        if cfg["designable_op"] is not None and cfg["designable_col"] in df.columns:
            all_thresholds = [("designable", (cfg["designable_col"], cfg["designable_op"]))] + all_thresholds
        for nm, (col, op) in all_thresholds:
            if col not in df.columns:
                continue
            hard_pass = op(pd.to_numeric(hard_rows[col], errors="coerce")).fillna(False).values
            soft_pass_per_group = []
            for _, g in df.groupby(cfg["group_cols"], sort=False):
                topk = _topk_indices(g, score_col, mode, k)
                rng = np.random.default_rng(rng_seed)
                chosen = rng.choice(topk, size=repeats, replace=True)
                soft_pass_per_group.append(np.nanmean(op(g[col]).fillna(False).values[chosen]))
                rng_seed += 1
            soft_pass_per_group = np.array(soft_pass_per_group)
            soft_bin = soft_pass_per_group >= 0.5
            mp, bb, cc = _mcnemar(hard_pass, soft_bin)
            dh = hard_pass.mean() * 100
            ds = soft_pass_per_group.mean() * 100
            print(f"  {picker:22s} {mode:6s} {nm:24s} {dh:9.1f} {ds:9.1f} "
                  f"{ds-dh:+9.1f} {mp:>9.4f} {bb:>4d}/{cc:<4d}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=("ff", "if", "cg"))
    p.add_argument("--candidates", required=True,
                   help="Path or glob to bestofN_<task>_lig_candidates_*.csv "
                        "(plain or _with_cofold.csv)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cand_path = _resolve(args.candidates)
    logger.info("Reading %s", cand_path)
    analyze(args.task, cand_path, args.top_k, args.repeats, args.seed)


if __name__ == "__main__":
    main()
