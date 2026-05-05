"""Compare IF self-reflection vs single-shot vs best-of-30 PLL pickers (CAMEO TED).

Reads:
  - if_sr_summary_*.csv             (from inverse_fold_self_reflection.py)
  - bestofN_if_summary_*.csv        (from inverse_fold_bestofN_pll.py)

For each ranker (random_pick, struc_pll_pick, joint_true_pll_pick, oracle_tm_pick,
SR-accept) computes overall and per-length: AAR, ESMFold sc-TM/RMSD/pLDDT, and
designability (RMSD<2). Pairs all rankers by target and runs McNemar on
designability and Wilcoxon on continuous metrics vs random_pick (= single-shot).

Outputs:
  - if_sr_vs_bestofN_report.md
  - if_sr_vs_bestofN_table.tex
  - if_sr_vs_bestofN_picker_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_if_sr")


def _mcnemar(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    bb = int(((~a) & b).sum()); cc = int((a & (~b)).sum())
    n = bb + cc
    if n == 0: return float("nan"), bb, cc
    return stats.binomtest(cc, n, 0.5, alternative="two-sided").pvalue, bb, cc


def _wilcoxon(a, b, alt="two-sided"):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5: return float("nan")
    try:
        return stats.wilcoxon(s.a, s.b, alternative=alt, zero_method="wilcox").pvalue
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sr-summary",
        required=True,
        type=Path,
        help="if_sr_summary_*.csv from inverse_fold_self_reflection.py",
    )
    p.add_argument(
        "--bestofN-summary",
        type=Path,
        default=Path(
            "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse/bestofN_if_summary_20260503T010309.csv"
        ),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = args.sr_summary.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sr = pd.read_csv(args.sr_summary)
    bn = pd.read_csv(args.bestofN_summary)
    logger.info("Loaded SR: %d rows  best-of-N: %d rows", len(sr), len(bn))

    # Build a unified per-target DataFrame
    sr_red = sr[
        [
            "target", "length", "attempts_used", "accepted", "fallback_used",
            "accepted_aar", "accepted_esmfold_tm", "accepted_esmfold_rmsd",
            "accepted_esmfold_plddt",
        ]
    ].rename(
        columns={
            "accepted_aar": "sr_aar",
            "accepted_esmfold_tm": "sr_esmfold_tm_score",
            "accepted_esmfold_rmsd": "sr_esmfold_rmsd",
            "accepted_esmfold_plddt": "sr_esmfold_plddt",
        }
    )

    pickers_bn = ["random_pick", "struc_pll_pick", "joint_pll_pick", "joint_true_pll_pick", "oracle_tm_pick"]
    bn_red = bn[["target", "length"] + [
        f"{pk}_{m}" for pk in pickers_bn for m in ("aar", "esmfold_tm_score", "esmfold_rmsd", "esmfold_plddt")
    ]]

    df = bn_red.merge(sr_red, on=["target", "length"], how="inner")
    logger.info("Joined targets: %d", len(df))

    # Define each picker as a uniform set of columns for downstream loops
    PICKERS = {
        "single_shot (= random_pick)": "random_pick",
        "best-of-30 struc_pll": "struc_pll_pick",
        "best-of-30 joint_pll(sum)": "joint_pll_pick",
        "best-of-30 joint_true_pll": "joint_true_pll_pick",
        "best-of-30 oracle (TM)": "oracle_tm_pick",
        "IF self-reflection": "sr",
    }

    def _cols(pk_id):
        return {
            "aar": f"{pk_id}_aar" if pk_id != "sr" else "sr_aar",
            "tm": f"{pk_id}_esmfold_tm_score" if pk_id != "sr" else "sr_esmfold_tm_score",
            "rmsd": f"{pk_id}_esmfold_rmsd" if pk_id != "sr" else "sr_esmfold_rmsd",
            "plddt": f"{pk_id}_esmfold_plddt" if pk_id != "sr" else "sr_esmfold_plddt",
        }

    # --------- per-picker overall + per-length stats ---------
    rows = []
    for pk_name, pk_id in PICKERS.items():
        c = _cols(pk_id)
        for bucket, sub in [("overall", df)] + [
            (f"L<={L_max}", df[df.length <= L_max])
            for L_max in [150, 300, 450]
        ]:
            if len(sub) == 0:
                continue
            tm = sub[c["tm"]]; rmsd = sub[c["rmsd"]]; aar = sub[c["aar"]]; plddt = sub[c["plddt"]]
            rows.append(
                {
                    "picker": pk_name,
                    "bucket": bucket,
                    "n": len(sub),
                    "aar_mean": float(aar.mean()) * 100,
                    "tm_mean": float(tm.mean()),
                    "tm_median": float(tm.median()),
                    "rmsd_mean": float(rmsd.mean()),
                    "plddt_mean": float(plddt.mean()),
                    "designable_pct": float((rmsd < 2.0).mean()) * 100,
                    "pass_tm_gt_0.7": float((tm > 0.7).mean()) * 100,
                    "pass_tm_gt_0.8": float((tm > 0.8).mean()) * 100,
                }
            )
    metrics_df = pd.DataFrame(rows)
    csv_out = args.output_dir / "if_sr_vs_bestofN_picker_metrics.csv"
    metrics_df.to_csv(csv_out, index=False)
    logger.info("Wrote %s", csv_out)

    # --------- paired stats vs single_shot ---------
    base = df[_cols("random_pick")["rmsd"]]
    base_pass = (base < 2.0)
    base_tm = df[_cols("random_pick")["tm"]]

    sig_rows = []
    print("\n=== IF: paired stats vs single_shot baseline (n=%d) ===" % len(df))
    print(f"  baseline: designable = {base_pass.mean()*100:.1f}%   sc-TM mean = {base_tm.mean():.3f}   sc-RMSD = {base.mean():.2f}")
    print()
    print(f"  {'picker':30s} {'desig (%)':>10s} {'McN p':>9s} {'b/c':>9s} {'ΔTM':>8s} {'Wilcox p':>10s} {'ΔRMSD':>9s} {'Wilcox p':>10s}")

    for pk_name, pk_id in PICKERS.items():
        if pk_id == "random_pick":
            continue
        c = _cols(pk_id)
        rmsd = df[c["rmsd"]]; tm = df[c["tm"]]
        ppass = (rmsd < 2.0)
        mp, bb, cc = _mcnemar(base_pass, ppass)
        wp_tm = _wilcoxon(tm, base_tm, alt="greater")
        wp_rmsd = _wilcoxon(rmsd, base, alt="less")
        d_pass = (ppass.mean() - base_pass.mean()) * 100
        d_tm = tm.mean() - base_tm.mean()
        d_rmsd = rmsd.mean() - base.mean()
        print(
            f"  {pk_name:30s} {d_pass:+10.1f} {mp:>9.4f} {bb:>4d}/{cc:<4d} {d_tm:+8.4f} {wp_tm:>10.4f} {d_rmsd:+9.3f} {wp_rmsd:>10.4f}"
        )
        sig_rows.append(
            {
                "picker": pk_name, "delta_designable_pp": d_pass, "mcnemar_p": mp, "b_worse": bb, "c_better": cc,
                "delta_tm": d_tm, "wilcox_p_tm": wp_tm, "delta_rmsd": d_rmsd, "wilcox_p_rmsd": wp_rmsd,
            }
        )
    pd.DataFrame(sig_rows).to_csv(args.output_dir / "if_sr_vs_bestofN_paired_stats.csv", index=False)

    # --------- markdown report ---------
    md = ["# Inverse-folding Self-Reflection vs best-of-30 PLL pickers", ""]
    md.append(f"n_targets = {len(df)}; SR cap = {sr.max_attempts.iloc[0]}, threshold = {sr.min_tm_score.iloc[0]}")
    sr_pass = sr.accepted.mean() * 100
    sr_attempts = sr.attempts_used.mean()
    md.append(f"SR acceptance rate (within cap): {sr_pass:.1f}% ; mean attempts/target: {sr_attempts:.1f}")
    md.append("")
    md.append("## Overall metrics")
    md.append("")
    md.append(
        "| Picker | n | AAR (%) | sc-TM mean | sc-TM med | RMSD (Å) | pLDDT | Designable (RMSD<2) | Pass>0.7 | Pass>0.8 |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    overall = metrics_df[metrics_df.bucket == "overall"]
    for _, r in overall.iterrows():
        md.append(
            f"| {r.picker} | {int(r.n)} | {r.aar_mean:.1f} | {r.tm_mean:.3f} | {r.tm_median:.3f} | {r.rmsd_mean:.2f} | {r.plddt_mean:.3f} | {r['designable_pct']:.1f}% | {r['pass_tm_gt_0.7']:.1f}% | {r['pass_tm_gt_0.8']:.1f}% |"
        )
    md.append("")
    md.append("## Paired stats vs single_shot baseline")
    md.append("")
    md.append(
        "| Picker | Δ designable (pp) | McNemar p | b/c | ΔTM | Wilcoxon p | ΔRMSD | Wilcoxon p |"
    )
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sig_rows:
        md.append(
            f"| {r['picker']} | {r['delta_designable_pp']:+.1f} | {r['mcnemar_p']:.4f} | {r['b_worse']}/{r['c_better']} | {r['delta_tm']:+.4f} | {r['wilcox_p_tm']:.4f} | {r['delta_rmsd']:+.3f} | {r['wilcox_p_rmsd']:.4f} |"
        )
    md.append("")
    md.append("## Per-length designability (%)")
    md.append("")
    pivot = metrics_df[metrics_df.bucket != "overall"].pivot_table(
        index="picker", columns="bucket", values="designable_pct"
    ).round(1)
    md.append(pivot.to_markdown())
    (args.output_dir / "if_sr_vs_bestofN_report.md").write_text("\n".join(md))
    logger.info("Wrote %s", args.output_dir / "if_sr_vs_bestofN_report.md")


if __name__ == "__main__":
    main()
