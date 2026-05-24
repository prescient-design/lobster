"""Analyze unconditional best-of-N results from `unconditional_bestofN_pll.py`.

Reads `bestofN_uc_summary_*.csv` and produces per-picker aggregate metrics
(mean ESMFold sc-TM / sc-RMSD / pLDDT and TM pass-rates) overall and
per-length. Also writes a LaTeX-ready table.

Usage:
    uv run python scripts/analyze_bestofN_uc.py \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_bestofN_uc")


_PICKERS = [
    ("random_pick", "random (single-shot)"),
    ("seq_pll_pick", "seq PLL"),
    ("struc_pll_pick", "struc PLL"),
    ("joint_pll_pick", "joint PLL (sum)"),
    ("joint_true_pll_pick", "joint PLL (true)"),
    ("oracle_tm_pick", "oracle (sc-TM)"),
]


def _picker_stats(df: pd.DataFrame, picker: str, tm_thresholds=(0.5, 0.7, 0.8, 0.9)) -> dict:
    tm = df[f"{picker}_esmfold_tm_score"]
    rmsd = df[f"{picker}_esmfold_rmsd"]
    plddt = df[f"{picker}_esmfold_plddt"]
    out = {
        "n": len(df),
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
    lines = [f"### {label} (n_slots={int(sub.n.iloc[0])})", ""]
    lines.append(
        "| picker | sc-TM mean | sc-TM med | RMSD (Å) | pLDDT | Designable (RMSD<2) | pass>0.7 | pass>0.8 | pass>0.9 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for picker, friendly in _PICKERS:
        r = sub.loc[picker]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% |".format(
                friendly,
                r.tm_mean,
                r.tm_median,
                r.rmsd_mean,
                r.plddt_mean,
                r["designable"],
                r["pass_tm>0.7"],
                r["pass_tm>0.8"],
                r["pass_tm>0.9"],
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
        r"\caption{Unconditional best-of-30 on LEFLUR-P-VAL (TED ckpt, $n_\mathrm{slots}{=}"
        + f"{n}"
        + r"$). For each design slot, generate 30 candidate (sequence, structure) pairs and "
        r"select one by the indicated ranker. Report mean self-consistency TM (sc-TM) between "
        r"ESMFold(seq) and the model's own decoded backbone, mean sc-RMSD, mean ESMFold pLDDT, "
        r"and TM-pass rates.}"
    )
    lines.append(r"\label{tab:uc_bestofN}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Ranker & sc-TM & RMSD (\AA) & pLDDT & Desig. (\%) & Pass$>$0.7 & Pass$>$0.8 \\"
    )
    lines.append(r"\midrule")
    for picker, friendly in _PICKERS:
        r = sub.loc[picker]
        lines.append(
            r"\textsc{{{}}} & {:.3f} & {:.2f} & {:.3f} & {:.1f}\% & {:.1f}\% & {:.1f}\% \\".format(
                friendly.replace(" ", r"\_"),
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
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--report-name", default="bestofN_uc_report.md")
    p.add_argument("--latex-name", default="bestofN_uc_table.tex")
    args = p.parse_args()

    if args.summary_csv is not None:
        summ_path = args.summary_csv
    else:
        cands = sorted(args.output_dir.glob("bestofN_uc_summary_*.csv"))
        if not cands:
            raise FileNotFoundError(f"No bestofN_uc_summary_*.csv under {args.output_dir}")
        summ_path = cands[-1]
    logger.info("Reading %s", summ_path)
    df = pd.read_csv(summ_path)
    logger.info("Loaded %d slots across lengths %s", len(df), sorted(df.length.unique()))

    overall = _aggregate(df, "overall")
    per_length = []
    for L in sorted(df.length.unique()):
        sub = df[df.length == L]
        per_length.append(_aggregate(sub, f"L={int(L)}"))

    agg_all = pd.concat([overall, *per_length], ignore_index=True)
    agg_csv = args.output_dir / "bestofN_uc_picker_metrics.csv"
    agg_all.to_csv(agg_csv, index=False)
    logger.info("Wrote per-picker metrics: %s", agg_csv)

    md_chunks = ["# Unconditional best-of-N — picker comparison (LEFLUR-P-VAL formulation)", "",
                 f"Source summary: `{summ_path.name}` ({len(df)} slots).", ""]
    md_chunks.append(_format_table_md(agg_all, "overall"))
    for L in sorted(df.length.unique()):
        md_chunks.append(_format_table_md(agg_all, f"L={int(L)}"))

    report_path = args.output_dir / args.report_name
    report_path.write_text("\n".join(md_chunks))
    logger.info("Wrote report: %s", report_path)

    latex_path = args.output_dir / args.latex_name
    latex_path.write_text(_format_latex(agg_all, "overall"))
    logger.info("Wrote LaTeX table: %s", latex_path)


if __name__ == "__main__":
    main()
