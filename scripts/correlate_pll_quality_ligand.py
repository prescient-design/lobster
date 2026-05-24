"""Correlate per-target 4-modality PLL with quality for protein-ligand E0.

Reads `bestofN_{ff,if,cg}_lig_candidates_*.csv` from the
`pll_correlation_report_protein_ligand/{ckpt}/{task}/full*/` tree, and
emits per (ckpt, task) Pearson + Spearman tables of every PLL variant
against the in-loop quality metric:

  - FF: tm_score, rmsd, ligand_rmsd, ligand_centroid_distance
  - IF: aar
  - CG: tm_to_gt, rmsd_to_gt, pseudo_aar, ligand_pocket_min_dist

For each (ckpt, task, quality_metric) writes:
  - `<task>_corr_<quality>_<ts>.csv` — rows = PLL variants, cols = pearson_r,
    pearson_p, spearman_r, spearman_p, n
  - `<task>_corr_<quality>_<ts>.png` — bar chart of |r| per variant

A combined markdown report `pll_correlation_report_protein_ligand/REPORT_<ts>.md`
summarises winners.

Usage:
    uv run python scripts/correlate_pll_quality_ligand.py \\
        --report-root /cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand
"""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PLL_VARIANTS = [
    "seq_score_unif", "struc_score_unif", "lig_atom_score_unif", "lig_struc_score_unif",
    "joint_protein_score_unif", "joint_ligand_score_unif",
    "joint_all_score_unif", "joint_true_4_score_unif",
]
TASK_QUALITY = {
    "ff": ["tm_score", "rmsd", "ligand_rmsd", "ligand_centroid_distance"],
    "if": ["aar"],
    # CG headline metrics are Boltz2 cofold quantities (model confidence) per
    # `enchanted-churning-shell.md` lines 116–144. The in-loop (decoded)
    # tm_to_gt / pseudo_aar / pocket-distance metrics are kept as secondary.
    "cg": ["cofold_iptm", "cofold_ligand_iptm", "cofold_complex_ipde",
           "cofold_complex_pde", "cofold_ptm", "cofold_complex_plddt",
           "cofold_both_pass"],
}


_CG_COFOLD_GLOB = "bestofN_cg_lig_candidates_*_with_cofold.csv"


def _latest_csv(dir_: Path, pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(dir_ / pattern)))
    return Path(matches[-1]) if matches else None


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    xf, yf = x[mask], y[mask]
    if kind == "pearson":
        r, p = stats.pearsonr(xf, yf)
    else:
        r, p = stats.spearmanr(xf, yf)
    return float(r), float(p), n


def _correlate_dir(task: str, ckpt_label: str, candidates_csv: Path,
                   out_dir: Path, ts: str) -> dict:
    df = pd.read_csv(candidates_csv)
    summary_rows = []
    for q in TASK_QUALITY[task]:
        if q not in df.columns:
            continue
        rows = []
        for v in PLL_VARIANTS:
            if v not in df.columns:
                rows.append({"variant": v, "n": 0,
                             "pearson_r": np.nan, "pearson_p": np.nan,
                             "spearman_r": np.nan, "spearman_p": np.nan})
                continue
            x = df[v].to_numpy(dtype=float)
            y = df[q].to_numpy(dtype=float)
            pr, pp, n_p = _safe_corr(x, y, "pearson")
            sr, sp, n_s = _safe_corr(x, y, "spearman")
            rows.append({
                "variant": v, "n": n_p,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
            })
        out_csv = out_dir / f"{task}_corr_{q}_{ts}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [r["variant"] for r in rows]
        vals = [abs(r["spearman_r"]) if r["n"] >= 5 else 0 for r in rows]
        bar_colors = ["tab:blue" if v >= 0.3 else "lightgray" for v in vals]
        ax.bar(names, vals, color=bar_colors)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("|Spearman r|")
        ax.set_title(f"{ckpt_label} / {task.upper()} — |Spearman r| of PLL variants vs {q}\n(n={rows[0]['n'] if rows else 0})")
        ax.axhline(0.3, ls="--", color="k", lw=0.8, alpha=0.6)
        ax.set_ylim(0, max(0.6, (max(vals) if vals else 0) * 1.15))
        fig.tight_layout()
        fig.savefig(out_dir / f"{task}_corr_{q}_{ts}.png", dpi=150)
        plt.close(fig)

        # Pick best by |Spearman| within rows.
        finite = [r for r in rows if r["n"] >= 5]
        if finite:
            best = max(finite, key=lambda r: abs(r["spearman_r"]))
            summary_rows.append({
                "ckpt": ckpt_label, "task": task, "quality": q,
                "n": best["n"], "best_variant": best["variant"],
                "best_spearman_r": best["spearman_r"],
                "best_spearman_p": best["spearman_p"],
                "best_pearson_r": best["pearson_r"],
                "best_pearson_p": best["pearson_p"],
            })
    return {"summary_rows": summary_rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report-root", required=True, type=Path)
    p.add_argument("--ckpts", nargs="+", default=["all", "plinder"])
    p.add_argument("--tasks", nargs="+", default=["ff", "if", "cg"])
    p.add_argument("--subdir", default="full",
                   help="Subdirectory under <ckpt>/<task> to scan (default 'full' for E0).")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    all_summary_rows: list[dict] = []

    for ckpt in args.ckpts:
        for task in args.tasks:
            d = args.report_root / ckpt / task / args.subdir
            if not d.exists():
                print(f"[skip] missing dir {d}")
                continue
            # Prefer the cofold-augmented CSV for CG when present.
            cand = (_latest_csv(d, _CG_COFOLD_GLOB) if task == "cg" else None) \
                or _latest_csv(d, f"bestofN_{task}_lig_candidates_*.csv")
            if cand is None:
                print(f"[skip] no candidates CSV in {d}")
                continue
            print(f"[corr] {ckpt}/{task} -> {cand.name}")
            res = _correlate_dir(task, ckpt, cand, d, ts)
            all_summary_rows.extend(res["summary_rows"])

    if not all_summary_rows:
        print("No correlations computed.")
        return

    summary_df = pd.DataFrame(all_summary_rows).sort_values(
        ["task", "quality", "ckpt"]).reset_index(drop=True)
    summary_path = args.report_root / f"pll_correlation_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}")

    md = [f"# Protein-ligand PLL correlation report ({ts})\n"]
    md.append(f"Source: `{args.report_root}`\n")
    md.append("\n## Best PLL variant per (ckpt, task, quality)\n")
    md.append(summary_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Detailed CSVs / plots\n")
    for ckpt in args.ckpts:
        for task in args.tasks:
            d = args.report_root / ckpt / task / args.subdir
            if not d.exists():
                continue
            md.append(f"- `{ckpt}/{task}/{args.subdir}/{task}_corr_<quality>_{ts}.csv`")
    md_path = args.report_root / f"REPORT_{ts}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[md] {md_path}")


if __name__ == "__main__":
    main()
