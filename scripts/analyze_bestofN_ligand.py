"""Analyze a protein-ligand best-of-N PLL run (FF / IF / CG).

Mirrors the protein-only `analyze_bestofN_{ff,if,uc}.py` analyzers but for the
4-modality protein-ligand drivers (`forward_fold_bestofN_pll_ligand.py`,
`inverse_fold_bestofN_pll_ligand.py`, `conditioned_gen_bestofN_pll_ligand.py`).

Reads the candidate-level CSV emitted by the chosen task driver, replays each
PLL picker per target, and reports:

  - Per-picker overall + per-length aggregate stats (mean / median quality,
    pass-rates, pass rate where applicable).
  - Paired McNemar (binary) and Wilcoxon signed-rank (continuous) of each
    picker vs `random_pick` (= candidate 0 = single-shot baseline), pairing
    per target / per slot.
  - Gap-to-oracle stats per picker.
  - Markdown report + paper-ready LaTeX table + per-picker metrics CSV.

Picker name map vs the protein-only paper:
  - random_pick           = candidate 0 (single-shot)         [identical]
  - seq_pll_pick          = argmin seq_score_unif             [identical]
  - struc_pll_pick        = argmin struc_score_unif           [identical]
  - joint_protein_pll     = argmin (seq + struc)              ~ protein-only joint_pll
  - joint_true_4_pll      = argmin joint_true_4_score_unif    ~ extension of joint_true_pll
  - lig_atom_pll          = argmin lig_atom_score_unif        [new]
  - lig_struc_pll         = argmin lig_struc_score_unif       [new]
  - joint_ligand_pll      = argmin (lig_atom + lig_struc)     [new]
  - joint_all_pll         = argmin sum of 4                   [new]
  - oracle_*              = argmax over the headline quality  [identical]

Cofold-augmented mode: pass `--candidates *_with_cofold.csv` (after running
`merge_{cg,if}_cofold_into_candidates.py`) to add `cofold_iptm`,
`cofold_ligand_iptm`, `cofold_both_pass` (= iptm>=0.9 & iPDE<=1.0) as quality
metrics, plus an `oracle_iptm_pick` and an `oracle_both_pass_pick`.

Usage:
    uv run python scripts/analyze_bestofN_ligand.py \\
        --task ff \\
        --candidates /cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand/all/ff/full_N30/bestofN_ff_lig_candidates_*.csv \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand/all/ff/full_N30
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
logger = logging.getLogger("analyze_bestofN_ligand")


# ---------------------------------------------------------------------------
# Shared picker dictionary (PLL = argmin; oracle = argmax)
# ---------------------------------------------------------------------------

_PLL_PICKERS = {
    "seq_pll_pick":           ("seq_score_unif",          "argmin"),
    "struc_pll_pick":         ("struc_score_unif",        "argmin"),
    "lig_atom_pll_pick":      ("lig_atom_score_unif",     "argmin"),
    "lig_struc_pll_pick":     ("lig_struc_score_unif",    "argmin"),
    "joint_protein_pll_pick": ("joint_protein_score_unif", "argmin"),
    "joint_ligand_pll_pick":  ("joint_ligand_score_unif",  "argmin"),
    "joint_all_pll_pick":     ("joint_all_score_unif",     "argmin"),
    "joint_true_4_pll_pick":  ("joint_true_4_score_unif",  "argmin"),
}


# Per-task config: headline quality, secondary metrics, designability rule,
# pass thresholds, oracle keys, and length-bucket binning.
_TASKS: dict[str, dict] = {
    "ff": {
        # Forward folding = predict the protein-ligand co-complex from
        # (GT seq + ligand atom/bond context); compare to GT crystal.
        # Headline = backbone TM-score; designability = GF+IP (good fold + in
        # pocket, the standard PoseBusters-style metric used in the production
        # `evaluate_protein_ligand_forward_folding.py`).
        "headline":         "tm_score",
        "headline_dir":     "argmax",
        "secondary":        ["rmsd_overall", "rmsd_pocket",
                             "ligand_rmsd_aligned", "ligand_centroid_dist",
                             "n_pocket_residues", "n_pocket_contacts"],
        "pass_col":   "good_fold_and_in_pocket",
        "pass_thr":   0.5,
        "pass_op":    "ge",
        "pass_label": "GF+IP (TM>0.5 & ligand in pocket)",
        "pass_thresholds":  [("tm>0.5", "tm_score", 0.5, "gt"),
                             ("tm>0.7", "tm_score", 0.7, "gt"),
                             ("tm>0.8", "tm_score", 0.8, "gt"),
                             ("in_pocket", "ligand_in_pocket", 0.5, "ge"),
                             ("gf_ip", "good_fold_and_in_pocket", 0.5, "ge"),
                             ("pocket_rmsd<5A", "rmsd_pocket", 5.0, "lt")],
        "oracle_picks":     [("oracle_tm_pick", "tm_score", "argmax"),
                             ("oracle_gfip_pick", "good_fold_and_in_pocket", "argmax"),
                             ("oracle_pocket_rmsd_pick", "rmsd_pocket", "argmin")],
        "group_cols":       ["pdb_id"],
        "bucket_col":       "L",
        "buckets":          [("L<=150", 0, 151), ("150<L<=300", 151, 301),
                             ("300<L<=450", 301, 451), ("L>450", 451, 10**9)],
    },
    "if": {
        # Inverse folding = predict sequence from (GT backbone + ligand context).
        # Inline metrics mirror the production `protein_ligand_inverse_folding.py`
        # benchmark: overall, pocket, and non-pocket AAR (pocket = CA within 5A of
        # GT ligand). The full IF designability metric is sc-TM via Boltz2 cofold,
        # added by `merge_if_cofold_into_candidates.py` (cofold_* columns).
        "headline":         "aar_pocket",
        "headline_dir":     "argmax",
        "secondary":        ["aar", "aar_nonpocket",
                             "n_pocket_residues", "n_nonpocket_residues"],
        "pass_col":   None,
        "pass_thr":   None,
        "pass_op":    None,
        "pass_label": None,
        "pass_thresholds":  [("aar>0.30", "aar", 0.30, "gt"),
                             ("aar>0.40", "aar", 0.40, "gt"),
                             ("aar>0.50", "aar", 0.50, "gt"),
                             ("aar_pkt>0.40", "aar_pocket", 0.40, "gt"),
                             ("aar_pkt>0.50", "aar_pocket", 0.50, "gt"),
                             ("aar_pkt>0.60", "aar_pocket", 0.60, "gt")],
        "oracle_picks":     [("oracle_aar_pick", "aar", "argmax"),
                             ("oracle_aar_pocket_pick", "aar_pocket", "argmax")],
        "group_cols":       ["pdb_id"],
        "bucket_col":       "L",
        "buckets":          [("L<=150", 0, 151), ("150<L<=300", 151, 301),
                             ("300<L<=450", 301, 451), ("L>450", 451, 10**9)],
    },
    "cg": {
        # CG = ligand-conditioned de novo protein generation. There is no GT
        # protein to compare to. PRODUCTION HEADLINE = Boltz2 cofold ipTM /
        # both_pass, added by `merge_cg_cofold_into_candidates.py` and consumed
        # via the _COFOLD_* lists below. Inline metrics are production-parity
        # contact metrics (decoded protein vs decoded ligand) — they match
        # `compute_contact_metrics` / `compute_binding_pocket` in
        # src/lobster/metrics/ligand_conditioned_protein_generation.py.
        # Pre-cofold headline = `n_pocket_residues` (did the design produce a
        # pocket-coupled ligand?). Once cofold is merged, the cofold pass-rate
        # thresholds are auto-promoted (see _COFOLD_PASS_THRESHOLDS).
        "headline":         "n_pocket_residues",
        "headline_dir":     "argmax",
        "secondary":        [
            "n_contacts",
            "n_residues_in_contact", "frac_residues_in_contact",
            "n_ligand_atoms_in_contact", "frac_ligand_atoms_in_contact",
            "min_protein_ligand_dist", "mean_min_dist_per_residue",
        ],
        # No inline `pass_col` — true designability comes from cofold.
        "pass_col":   None,
        "pass_thr":   None,
        "pass_op":    None,
        "pass_label": None,
        "pass_thresholds":  [("contact",   "n_contacts", 1.0, "ge"),
                             ("frac_res_in_contact>5%", "frac_residues_in_contact", 0.05, "gt"),
                             ("min_pl_dist<4.5A", "min_protein_ligand_dist", 4.5, "lt")],
        "oracle_picks":     [("oracle_contact_pick", "n_contacts", "argmax"),
                             ("oracle_min_pl_dist_pick", "min_protein_ligand_dist", "argmin")],
        "group_cols":       ["pdb_id"],
        "bucket_col":       "L",
        "buckets":          [("L<=150", 0, 151), ("150<L<=300", 151, 301),
                             ("300<L<=450", 301, 451), ("L>450", 451, 10**9)],
    },
}


# Cofold-augmented columns are added when *_with_cofold.csv is supplied.
# Field names match the Boltz2 confidence dict (see scripts/eval_cg_boltz_checkpoint.py:90-91
# for the canonical both-pass definition: ligand_iptm>=0.9 & complex_ipde<=1.0).
# Structural cofold metrics (cofold_tm_score, cofold_rmsd_*, cofold_ligand_in_pocket,
# cofold_good_fold_and_in_pocket) come from
# `lobster.cmdline.merge_cofold_results --parse_structures` (TM-align + Kabsch pocket
# RMSD + atom-level ligand placement vs the GT crystal).
_COFOLD_QUALITY_KEYS = {
    "cofold_iptm":                   "argmax",
    "cofold_ligand_iptm":            "argmax",
    "cofold_ptm":                    "argmax",
    "cofold_complex_plddt":          "argmax",
    "cofold_complex_ipde":           "argmin",
    "cofold_complex_pde":            "argmin",
    "cofold_both_pass":              "argmax",
    "cofold_tm_score":               "argmax",
    "cofold_rmsd_overall":           "argmin",
    "cofold_rmsd_pocket":            "argmin",
    "cofold_ligand_rmsd":            "argmin",
    "cofold_ligand_centroid_dist":   "argmin",
    "cofold_n_protein_ligand_contacts": "argmax",
    "cofold_ligand_in_pocket":       "argmax",
    "cofold_good_fold_and_in_pocket": "argmax",
}
_COFOLD_PASS_THRESHOLDS = [("ligand_iptm>=0.9",  "cofold_ligand_iptm", 0.9, "ge"),
                           ("ligand_iptm>=0.7",  "cofold_ligand_iptm", 0.7, "ge"),
                           ("complex_ipde<=1.0", "cofold_complex_ipde", 1.0, "le"),
                           ("both_pass",         "cofold_both_pass",   0.5, "ge"),
                           ("TM>=0.5",           "cofold_tm_score",    0.5, "ge"),
                           ("ligand_in_pocket",  "cofold_ligand_in_pocket", 0.5, "ge"),
                           ("GF+IP",             "cofold_good_fold_and_in_pocket", 0.5, "ge")]


# ---------------------------------------------------------------------------
# Picker / stat helpers
# ---------------------------------------------------------------------------


def _hard_pick(group: pd.DataFrame, score_col: str, mode: str) -> int:
    """Return the integer position within `group` chosen by argmin/argmax."""
    if score_col not in group.columns:
        return 0
    valid = group[score_col].notna().values
    if not valid.any():
        return 0
    arr = np.asarray(group[score_col].values, dtype=float).copy()
    if mode == "argmin":
        arr[~valid] = np.inf
        return int(np.argmin(arr))
    arr[~valid] = -np.inf
    return int(np.argmax(arr))


def _apply_threshold(series: pd.Series, thr: float, op: str) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce")
    if op == "gt":
        return (s > thr).fillna(False).values
    if op == "ge":
        return (s >= thr).fillna(False).values
    if op == "lt":
        return (s < thr).fillna(False).values
    if op == "le":
        return (s <= thr).fillna(False).values
    raise ValueError(op)


def _mcnemar(a_pass: np.ndarray, b_pass: np.ndarray) -> tuple[float, int, int]:
    """Exact McNemar two-sided. Returns (p, b=worse_to_better, c=better_to_worse)."""
    a = np.asarray(a_pass, dtype=bool)
    b = np.asarray(b_pass, dtype=bool)
    bb = int(((~a) & b).sum())
    cc = int((a & (~b)).sum())
    n = bb + cc
    if n == 0:
        return float("nan"), bb, cc
    p = stats.binomtest(cc, n, 0.5, alternative="two-sided").pvalue
    return p, bb, cc


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    # Cast to float so boolean columns (e.g. cofold_ligand_in_pocket) work
    # with scipy's wilcoxon which does `x - y` internally.
    s = pd.DataFrame({"a": pd.to_numeric(pd.Series(a), errors="coerce"),
                      "b": pd.to_numeric(pd.Series(b), errors="coerce")}).dropna()
    if len(s) < 5:
        return float("nan")
    try:
        return stats.wilcoxon(s.a.astype(float).values, s.b.astype(float).values,
                              alternative="two-sided", zero_method="wilcox").pvalue
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _resolve_candidates(arg: str) -> Path:
    paths = sorted(glob.glob(arg)) if "*" in arg else [arg]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError(arg)
    return Path(paths[-1])


def _load_candidates(path: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pdb_id" not in df.columns or "candidate_idx" not in df.columns:
        raise ValueError(f"{path}: missing pdb_id / candidate_idx")
    df = df.sort_values(["pdb_id", "candidate_idx"]).reset_index(drop=True)
    cfg = _TASKS[task]
    if cfg["headline"] not in df.columns:
        raise ValueError(f"{path}: missing headline column {cfg['headline']!r}")
    return df


def _build_pickers(df: pd.DataFrame, task: str) -> list[tuple[str, str, str]]:
    """Return [(picker_name, score_col, direction), ...] filtered to columns we have."""
    cfg = _TASKS[task]
    out: list[tuple[str, str, str]] = [("random_pick", "_candidate_idx", "candidate0")]
    for name, (col, mode) in _PLL_PICKERS.items():
        if col in df.columns:
            out.append((name, col, mode))
    for name, col, mode in cfg["oracle_picks"]:
        if col in df.columns:
            out.append((name, col, mode))
    # Cofold-augmented oracles
    for col, mode in _COFOLD_QUALITY_KEYS.items():
        if col in df.columns:
            out.append((f"oracle_{col.removeprefix('cofold_')}_pick", col, mode))
    return out


def _picks_per_target(df: pd.DataFrame, pickers: list, group_cols: list[str]) -> dict[str, pd.DataFrame]:
    """For each picker, return a DataFrame indexed by group key with the chosen row."""
    out: dict[str, pd.DataFrame] = {}
    for name, col, mode in pickers:
        chosen = []
        for _, g in df.groupby(group_cols, sort=False):
            if name == "random_pick":
                row = g.sort_values("candidate_idx").iloc[0]
            else:
                idx = _hard_pick(g, col, mode)
                row = g.iloc[idx]
            chosen.append(row)
        out[name] = pd.DataFrame(chosen).reset_index(drop=True)
    return out


def _picker_stats(picks: pd.DataFrame, task: str, df_columns: list[str]) -> dict:
    cfg = _TASKS[task]
    headline = cfg["headline"]
    out: dict[str, float] = {"n": int(len(picks))}
    out[f"{headline}_mean"] = float(pd.to_numeric(picks[headline], errors="coerce").mean())
    out[f"{headline}_median"] = float(pd.to_numeric(picks[headline], errors="coerce").median())
    for col in cfg["secondary"]:
        if col in picks.columns:
            out[f"{col}_mean"] = float(pd.to_numeric(picks[col], errors="coerce").mean())
    if cfg["pass_op"] is not None and cfg["pass_col"] in picks.columns:
        ok = _apply_threshold(picks[cfg["pass_col"]],
                              cfg["pass_thr"], cfg["pass_op"])
        out["pass_pct"] = float(ok.mean()) * 100.0
    for label, col, thr, op in cfg["pass_thresholds"]:
        if col in picks.columns:
            out[f"pass_{label}"] = float(_apply_threshold(picks[col], thr, op).mean()) * 100.0
    # Cofold pass-rates
    for label, col, thr, op in _COFOLD_PASS_THRESHOLDS:
        if col in df_columns and col in picks.columns:
            out[f"pass_{label}"] = float(_apply_threshold(picks[col], thr, op).mean()) * 100.0
    for col in [k for k in _COFOLD_QUALITY_KEYS if k in df_columns and k in picks.columns]:
        out[f"{col}_mean"] = float(pd.to_numeric(picks[col], errors="coerce").mean())
    return out


def _paired_stats(
    picks: pd.DataFrame,
    base_picks: pd.DataFrame,
    task: str,
    df_columns: list[str],
) -> dict:
    """Paired McNemar (binary) and Wilcoxon (continuous) of picks vs base_picks."""
    cfg = _TASKS[task]
    out: dict[str, float] = {}

    # Continuous: headline + each cofold quality column we have
    cont_cols = [cfg["headline"], *cfg["secondary"]]
    cont_cols += [k for k in _COFOLD_QUALITY_KEYS if k in df_columns]
    for col in cont_cols:
        if col not in picks.columns or col not in base_picks.columns:
            continue
        a = pd.to_numeric(picks[col], errors="coerce").values
        b = pd.to_numeric(base_picks[col], errors="coerce").values
        out[f"wilcoxon_{col}_p"] = _wilcoxon(a, b)

    # Binary: pass + each pass threshold
    binary_specs: list[tuple[str, str, float, str]] = list(cfg["pass_thresholds"])
    for label, col, thr, op in _COFOLD_PASS_THRESHOLDS:
        if col in df_columns:
            binary_specs.append((label, col, thr, op))
    if cfg["pass_op"] is not None and cfg["pass_col"] in picks.columns:
        binary_specs.append(("pass", cfg["pass_col"],
                             cfg["pass_thr"], cfg["pass_op"]))
    for label, col, thr, op in binary_specs:
        if col not in picks.columns or col not in base_picks.columns:
            continue
        a_pass = _apply_threshold(picks[col], thr, op)
        b_pass = _apply_threshold(base_picks[col], thr, op)
        p, bb, cc = _mcnemar(b_pass, a_pass)  # base = a, picker = b in McNemar convention
        out[f"mcnemar_{label}_p"] = p
        out[f"mcnemar_{label}_b"] = bb  # base fail, picker pass = "improvements"
        out[f"mcnemar_{label}_c"] = cc  # base pass, picker fail = "regressions"
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_picker_table(stats_rows: list[dict], pickers_order: list[str], task: str, label: str) -> str:
    cfg = _TASKS[task]
    cols = [f"{cfg['headline']}_mean", f"{cfg['headline']}_median"]
    cols += [f"{c}_mean" for c in cfg["secondary"]]
    if "pass_pct" in stats_rows[0]:
        cols.append("pass_pct")
    cols += [f"pass_{label_t}" for label_t, _, _, _ in cfg["pass_thresholds"]]
    # Cofold cols if present
    extras = [c for c in stats_rows[0].keys()
              if c.startswith("cofold_") or c.startswith("pass_iptm") or c.startswith("pass_ligand_iptm")
              or c == "pass_both_pass"]
    cols.extend([c for c in extras if c not in cols])

    n = stats_rows[0]["n"]
    out = [f"### {label} (n={n})", ""]
    header = "| picker | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    out += [header, sep]
    pmap = {s["picker"]: s for s in stats_rows}
    for p in pickers_order:
        if p not in pmap:
            continue
        s = pmap[p]
        cells = [p] + [
            (f"{s[c]:.3f}" if "mean" in c or "median" in c or "iptm" in c or "plddt" in c or "pde" in c.lower()
             else f"{s[c]:.1f}%" if c.endswith("_pct") or c.startswith("pass_") else f"{s[c]:.3f}")
            if c in s and pd.notna(s[c]) else "-"
            for c in cols
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def _fmt_stats_table(paired_rows: list[dict], pickers_order: list[str]) -> str:
    if not paired_rows:
        return ""
    cols = [k for k in paired_rows[0].keys() if k != "picker"]
    out = ["### Paired stats vs random_pick", ""]
    out.append("| picker | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    pmap = {r["picker"]: r for r in paired_rows}
    for p in pickers_order:
        if p == "random_pick" or p not in pmap:
            continue
        r = pmap[p]
        cells = [p]
        for c in cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            elif "_b" == c[-2:] or "_c" == c[-2:]:
                cells.append(f"{int(v)}")
            else:
                cells.append(f"{v:.4f}")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def _fmt_latex(stats_rows: list[dict], pickers_order: list[str], task: str, n: int) -> str:
    cfg = _TASKS[task]
    headline = cfg["headline"]
    pmap = {s["picker"]: s for s in stats_rows}
    pretty = {
        "random_pick": r"\textsc{Random}",
        "seq_pll_pick": r"\textsc{Seq PLL}",
        "struc_pll_pick": r"\textsc{Struc PLL}",
        "lig_atom_pll_pick": r"\textsc{Lig-atom PLL}",
        "lig_struc_pll_pick": r"\textsc{Lig-struc PLL}",
        "joint_protein_pll_pick": r"\textsc{Joint-protein PLL}",
        "joint_ligand_pll_pick": r"\textsc{Joint-ligand PLL}",
        "joint_all_pll_pick": r"\textsc{Joint-all PLL}",
        "joint_true_4_pll_pick": r"\textsc{Joint-true(4) PLL}",
        "oracle_tm_pick": r"\textsc{Oracle (TM)}",
        "oracle_aar_pick": r"\textsc{Oracle (AAR)}",
        "oracle_iptm_pick": r"\textsc{Oracle (ipTM)}",
        "oracle_both_pass_pick": r"\textsc{Oracle (both-pass)}",
    }
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{Best-of-30 PLL picker comparison on PoseBusters ($n{=}" + f"{n}" + r"$, "
             + ("FF" if task == "ff" else "IF" if task == "if" else "CG") + r" task).}",
             r"\label{tab:bestofN_pl_" + task + r"}",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"Ranker & " + headline.replace("_", r"\_") + r" mean & Pass>0.5 \\",
             r"\midrule"]
    for p in pickers_order:
        if p not in pmap:
            continue
        s = pmap[p]
        h = s.get(f"{headline}_mean", float("nan"))
        pass_key = next((k for k in s if k.startswith("pass_") and ">0.5" in k or k == "pass_both_pass"), None)
        pp = s.get(pass_key, float("nan")) if pass_key else float("nan")
        lines.append(
            f"{pretty.get(p, p)} & "
            f"{('-' if not np.isfinite(h) else f'{h:.3f}')} & "
            f"{('-' if not np.isfinite(pp) else f'{pp:.1f}\\%')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=("ff", "if", "cg"))
    p.add_argument("--candidates", required=True, help="Path or glob to bestofN_<task>_lig_candidates_*.csv")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--report-name", default=None,
                   help="Default: bestofN_<task>_lig_report.md (or _with_cofold.md when applicable)")
    p.add_argument("--latex-name", default=None,
                   help="Default: bestofN_<task>_lig_table.tex")
    args = p.parse_args()

    cand_path = _resolve_candidates(args.candidates)
    df = _load_candidates(cand_path, args.task)
    cofold_present = any(c.startswith("cofold_") for c in df.columns)
    logger.info("Loaded %d candidates from %s (cofold cols: %s)",
                len(df), cand_path.name, "yes" if cofold_present else "no")

    pickers = _build_pickers(df, args.task)
    pickers_order = [name for name, _, _ in pickers]
    cfg = _TASKS[args.task]

    # ---- per-target picks ----
    picks_by_picker = _picks_per_target(df, pickers, cfg["group_cols"])
    base_picks = picks_by_picker["random_pick"]

    # ---- aggregate stats per picker (overall) ----
    overall_rows = []
    paired_rows = []
    for name, _, _ in pickers:
        st = _picker_stats(picks_by_picker[name], args.task, list(df.columns))
        st["picker"] = name
        overall_rows.append(st)
        if name == "random_pick":
            continue
        ps = _paired_stats(picks_by_picker[name], base_picks, args.task, list(df.columns))
        ps["picker"] = name
        paired_rows.append(ps)

    # ---- per-length-bucket aggregate (skip if bucket col missing) ----
    per_bucket_md_blocks = []
    if cfg["bucket_col"] in df.columns:
        for label, lo, hi in cfg["buckets"]:
            mask = (df[cfg["bucket_col"]].astype(int) >= lo) & (df[cfg["bucket_col"]].astype(int) < hi)
            sub_df = df[mask]
            if sub_df.empty:
                continue
            sub_picks = _picks_per_target(sub_df, pickers, cfg["group_cols"])
            sub_rows = []
            for name, _, _ in pickers:
                st = _picker_stats(sub_picks[name], args.task, list(df.columns))
                st["picker"] = name
                sub_rows.append(st)
            per_bucket_md_blocks.append(_fmt_picker_table(sub_rows, pickers_order, args.task, f"Length {label}"))

    # ---- write outputs ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_with_cofold" if cofold_present else ""

    md_chunks = [
        f"# Protein-ligand best-of-N picker comparison — {args.task.upper()}",
        "",
        f"Source: `{cand_path}`",
        f"Candidates: {len(df)}; targets/groups: {len(base_picks)}",
        f"Pickers: {', '.join(pickers_order)}",
        "",
        _fmt_picker_table(overall_rows, pickers_order, args.task, "Overall"),
        _fmt_stats_table(paired_rows, pickers_order),
    ]
    if per_bucket_md_blocks:
        md_chunks.append("\n## Per-length-bucket breakdown\n")
        md_chunks.extend(per_bucket_md_blocks)

    report_name = args.report_name or f"bestofN_{args.task}_lig_report{suffix}.md"
    latex_name = args.latex_name or f"bestofN_{args.task}_lig_table{suffix}.tex"
    metrics_csv = args.output_dir / f"bestofN_{args.task}_lig_picker_metrics{suffix}.csv"
    paired_csv = args.output_dir / f"bestofN_{args.task}_lig_paired_stats{suffix}.csv"

    pd.DataFrame(overall_rows).to_csv(metrics_csv, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)
    (args.output_dir / report_name).write_text("\n".join(md_chunks))
    (args.output_dir / latex_name).write_text(_fmt_latex(overall_rows, pickers_order, args.task, len(base_picks)))

    logger.info("Wrote %s, %s, %s, %s",
                metrics_csv.name, paired_csv.name, report_name, latex_name)


if __name__ == "__main__":
    main()
