"""Analyze inverse-folding best-of-N results from `inverse_fold_bestofN_pll.py`.

Reads `bestofN_if_summary_*.csv` (and optionally `bestofN_if_candidates_*.csv`) and
produces per-picker aggregate metrics (mean AAR / ESMFold TM / RMSD / pLDDT and
TM pass-rates) overall and per-length-bucket. Also writes a LaTeX-ready table.

Usage:
    uv run python scripts/analyze_bestofN_if.py \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_bestofN_if")


_PICKERS = [
    ("random_pick", "random (single-shot)"),
    ("seq_pll_pick", "seq PLL"),
    ("struc_pll_pick", "struc PLL"),
    ("joint_pll_pick", "joint PLL (sum)"),
    ("joint_true_pll_pick", "joint PLL (true)"),
    ("oracle_tm_pick", "oracle (TM)"),
    ("oracle_aar_pick", "oracle (AAR)"),
]


_LEN_BUCKETS = [
    ("L<=150", 0, 151),
    ("150<L<=300", 151, 301),
    ("300<L<=450", 301, 451),
    ("L>450", 451, 10**9),
]


def _bucket_label(L: int) -> str:
    for label, lo, hi in _LEN_BUCKETS:
        if lo <= L < hi:
            return label
    return "?"


def _picker_stats(df: pd.DataFrame, picker: str, tm_thresholds=(0.5, 0.7, 0.8, 0.9)) -> dict:
    aar = df[f"{picker}_aar"]
    tm = df[f"{picker}_esmfold_tm_score"]
    rmsd = df[f"{picker}_esmfold_rmsd"]
    plddt = df[f"{picker}_esmfold_plddt"]
    n = len(df)
    out = {
        "n": n,
        "aar_mean": float(aar.mean()),
        "tm_mean": float(tm.mean()),
        "tm_median": float(tm.median()),
        "rmsd_mean": float(rmsd.mean()),
        "plddt_mean": float(plddt.mean()),
        "designable": float((rmsd < 2.0).mean()) * 100.0,
    }
    for thr in tm_thresholds:
        out[f"pass_tm>{thr}"] = float((tm > thr).mean()) * 100.0
    return out


def _aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for picker, _ in _PICKERS:
        s = _picker_stats(df, picker)
        s["picker"] = picker
        s["bucket"] = label
        rows.append(s)
    return pd.DataFrame(rows)


def _format_table_md(df_agg: pd.DataFrame, label: str) -> str:
    sub = df_agg[df_agg.bucket == label].set_index("picker").loc[[p for p, _ in _PICKERS]]
    lines = [f"### {label} (n={int(sub.n.iloc[0])})", ""]
    lines.append(
        "| picker | AAR (%) | sc-TM mean | sc-TM med | RMSD (Å) | pLDDT | Designable (RMSD<2) | pass>0.7 | pass>0.8 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for picker, friendly in _PICKERS:
        r = sub.loc[picker]
        lines.append(
            "| {} | {:.1f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.1f}% | {:.1f}% | {:.1f}% |".format(
                friendly,
                100.0 * r.aar_mean,
                r.tm_mean,
                r.tm_median,
                r.rmsd_mean,
                r.plddt_mean,
                r["designable"],
                r["pass_tm>0.7"],
                r["pass_tm>0.8"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_latex(df_agg: pd.DataFrame, label: str) -> str:
    sub = df_agg[df_agg.bucket == label].set_index("picker").loc[[p for p, _ in _PICKERS]]
    n = int(sub.n.iloc[0])
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Inverse folding best-of-30 on CAMEO ($n{=}" + f"{n}" + r"$, TED ckpt). "
        r"For each backbone, generate 30 candidate sequences and select one by the indicated "
        r"ranker. Report mean amino-acid recovery (AAR), ESMFold TM-score (mean and median), "
        r"ESMFold RMSD, ESMFold pLDDT, and TM-pass rates.}"
    )
    lines.append(r"\label{tab:if_bestofN}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Ranker & AAR (\%) & sc-TM & RMSD (\AA) & pLDDT & Desig. (\%) & Pass$>$0.7 & Pass$>$0.8 \\"
    )
    lines.append(r"\midrule")
    for picker, friendly in _PICKERS:
        r = sub.loc[picker]
        lines.append(
            r"\textsc{{{}}} & {:.1f} & {:.3f} & {:.2f} & {:.3f} & {:.1f}\% & {:.1f}\% & {:.1f}\% \\".format(
                friendly.replace(" ", r"\_"),
                100.0 * r.aar_mean,
                r.tm_mean,
                r.rmsd_mean,
                r.plddt_mean,
                r["designable"],
                r["pass_tm>0.7"],
                r["pass_tm>0.8"],
            )
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, type=Path, help="Dir containing bestofN_if_summary_*.csv")
    p.add_argument("--summary-csv", type=Path, default=None, help="Override path to summary CSV")
    p.add_argument("--report-name", default="bestofN_if_report.md")
    p.add_argument("--latex-name", default="bestofN_if_table.tex")
    args = p.parse_args()

    if args.summary_csv is not None:
        summ_path = args.summary_csv
    else:
        cands = sorted(args.output_dir.glob("bestofN_if_summary_*.csv"))
        if not cands:
            raise FileNotFoundError(f"No bestofN_if_summary_*.csv under {args.output_dir}")
        summ_path = cands[-1]
    logger.info("Reading %s", summ_path)
    df = pd.read_csv(summ_path)
    logger.info("Loaded %d targets", len(df))

    df["bucket"] = df["length"].apply(_bucket_label)

    overall = _aggregate(df, "overall")
    per_bucket = []
    for label, _, _ in _LEN_BUCKETS:
        sub = df[df.bucket == label]
        if len(sub) == 0:
            continue
        per_bucket.append(_aggregate(sub, label))

    agg_all = pd.concat([overall, *per_bucket], ignore_index=True)
    agg_csv = args.output_dir / "bestofN_if_picker_metrics.csv"
    agg_all.to_csv(agg_csv, index=False)
    logger.info("Wrote per-picker metrics: %s", agg_csv)

    md_chunks = ["# Inverse folding best-of-N — picker comparison", "",
                 f"Source summary: `{summ_path.name}` ({len(df)} targets).", ""]
    md_chunks.append(_format_table_md(agg_all, "overall"))
    for label, _, _ in _LEN_BUCKETS:
        if not (agg_all.bucket == label).any():
            continue
        md_chunks.append(_format_table_md(agg_all, label))

    report_path = args.output_dir / args.report_name
    report_path.write_text("\n".join(md_chunks))
    logger.info("Wrote report: %s", report_path)

    latex_path = args.output_dir / args.latex_name
    latex_path.write_text(_format_latex(agg_all, "overall"))
    logger.info("Wrote LaTeX table: %s", latex_path)


if __name__ == "__main__":
    main()
