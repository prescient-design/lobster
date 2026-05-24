"""Compare the SR forward-fold-TM gate against an NLL/PLL-based gate.

Inputs (all from a SR concordance run that had `save_failed_attempts=true`):
  1. SR-accepted: PLL scores from `pll_scores_<ts>.csv` in the SR-paired eval
     dir (where the existing scoring already ran), joined to
     `unconditional_metrics_<ts>.csv` for `tm_score_unconditional_to_forward`
     (= lobster forward-fold TM, the SR gate quantity) and ESMFold metrics
     (tm_score / rmsd / plddt vs the initial backbone).
  2. SR-rejected: PLL scores from `pll_scores_failed_attempts_<ts>.csv` in
     the concordance dir, already merged with
     `failed_self_reflection.csv` + `esmfold_failed_attempts_<ts>.csv` so each
     row has lobster forward-fold TM + ESMFold metrics.

Questions answered:
  (Q1) How well does PLL predict the lobster forward-fold TM (the SR gate)?
       => correlations overall and per-length, on the full attempt pool.
  (Q2) If we used PLL as the gate at the same accept rate as SR's current
       T = 0.833 on forward-fold-TM, would we accept the same designs?
       => 2x2 contingency, Cohen's kappa, raw agreement %.
  (Q3) Would a PLL gate accept BETTER designs?
       => for each gate, compute ESMFold-pass rate of the accepted set
          (precision per accept) and of the rejected set (regret per reject).

Outputs (markdown + CSV) to --output-dir.

Usage:
  uv run python scripts/compare_pll_vs_sr_gate.py \\
    --paired-eval-dir <SR_paired_dir>          # has pll_scores_*.csv + unconditional_metrics_*.csv
    --concordance-dir <SR_concordance_dir>     # has pll_scores_failed_attempts_*.csv
    --pll-variant struc_score_unif             # which PLL column to use as the gate
    --sr-threshold 0.833                       # the SR forward-fold-TM gate
    --esmfold-pass-tm 0.9                      # ESMFold "designable" criterion
    --esmfold-pass-rmsd 2.0                    # alternative ESMFold criterion
    --output-dir <OUT_DIR>
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_pll_vs_sr_gate")


def _newest(globbed):
    return sorted(globbed)[-1]


def _load_accepted(paired_dir: Path) -> pd.DataFrame:
    pll_path = _newest(list(paired_dir.glob("pll_scores_*.csv")))
    metrics_path = _newest(list(paired_dir.glob("unconditional_metrics_*.csv")))
    pll = pd.read_csv(pll_path)
    metrics = pd.read_csv(metrics_path)
    metrics["run_id"] = metrics["run_id"].astype(str)
    pll["run_id"] = pll["run_id"].astype(str)

    base = metrics[metrics["run_id"].str.startswith("unconditional_")].copy()

    sr_rows = metrics[metrics["run_id"].str.startswith("self_reflection_")].copy()
    sr_rows["_key"] = sr_rows["run_id"].str.replace("^self_reflection_", "unconditional_", regex=True)
    sr_lift = [c for c in (
        "tm_score_unconditional_to_forward",
        "rmsd_unconditional_to_forward",
        "percent_identity_self_reflection",
    ) if c in sr_rows.columns]
    sr_to_join = sr_rows[["_key"] + sr_lift].rename(columns={"_key": "run_id"})

    base = base.merge(sr_to_join, on="run_id", how="left", suffixes=("", "_sr"))
    for c in sr_lift:
        sr_alt = c + "_sr"
        if sr_alt in base.columns:
            base[c] = base[c].where(base[c].notna(), base[sr_alt])
            base = base.drop(columns=[sr_alt])

    out = base.merge(pll, on="run_id", how="inner")
    out["sr_decision"] = "accept"
    out["lobster_forward_tm"] = pd.to_numeric(out.get("tm_score_unconditional_to_forward"), errors="coerce")
    out["esmfold_tm"] = pd.to_numeric(out.get("tm_score"), errors="coerce")
    out["esmfold_rmsd"] = pd.to_numeric(out.get("rmsd"), errors="coerce")
    out["esmfold_plddt"] = pd.to_numeric(out.get("plddt"), errors="coerce")
    if "sequence_length" in out.columns:
        out["length"] = pd.to_numeric(out["sequence_length"], errors="coerce")
    elif "length" in out.columns:
        out["length"] = pd.to_numeric(out["length"], errors="coerce")
    logger.info("Accepted set: loaded %d rows from %s", len(out), paired_dir.name)
    return out


def _load_rejected(concordance_dir: Path) -> pd.DataFrame:
    matches = sorted(concordance_dir.glob("pll_scores_failed_attempts_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No pll_scores_failed_attempts_*.csv under {concordance_dir}")
    df = pd.read_csv(matches[-1])
    df["sr_decision"] = "reject"
    df["lobster_forward_tm"] = pd.to_numeric(df.get("tm_score_unconditional_to_forward"), errors="coerce")
    df["esmfold_tm"] = pd.to_numeric(df.get("esmfold_tm"), errors="coerce")
    df["esmfold_rmsd"] = pd.to_numeric(df.get("esmfold_rmsd"), errors="coerce")
    df["esmfold_plddt"] = pd.to_numeric(df.get("esmfold_plddt"), errors="coerce")
    if "sequence_length" in df.columns:
        df["length"] = pd.to_numeric(df["sequence_length"], errors="coerce")
    elif "length" in df.columns:
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
    logger.info("Rejected set: loaded %d rows from %s", len(df), matches[-1].name)
    return df


def _safe_corr(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    if np.allclose(x[valid].std(), 0) or np.allclose(y[valid].std(), 0):
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    pr, pp = pearsonr(x[valid], y[valid])
    sr, sp = spearmanr(x[valid], y[valid])
    return float(pr), float(pp), float(sr), float(sp), n


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary classifiers (0/1)."""
    a = a.astype(int)
    b = b.astype(int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1 = float(a.mean())
    pb1 = float(b.mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def _gate_quality(label: str, accepted_mask: np.ndarray, esm_pass: np.ndarray) -> dict:
    n_attempts = int(len(accepted_mask))
    n_acc = int(accepted_mask.sum())
    n_rej = n_attempts - n_acc
    valid_esm = np.isfinite(esm_pass.astype(float)) if esm_pass.dtype != bool else np.ones_like(esm_pass, dtype=bool)
    esm_pass_b = esm_pass.astype(bool)

    acc_valid = accepted_mask & valid_esm
    rej_valid = (~accepted_mask) & valid_esm

    n_acc_valid = int(acc_valid.sum())
    n_rej_valid = int(rej_valid.sum())

    acc_pass = int((acc_valid & esm_pass_b).sum())
    rej_pass = int((rej_valid & esm_pass_b).sum())

    acc_precision = (acc_pass / n_acc_valid) if n_acc_valid > 0 else float("nan")
    rej_precision = (rej_pass / n_rej_valid) if n_rej_valid > 0 else float("nan")

    return {
        "gate": label,
        "n_attempts": n_attempts,
        "n_accepted": n_acc,
        "n_rejected": n_rej,
        "n_acc_with_esm": n_acc_valid,
        "acc_precision": acc_precision,
        "n_acc_designable": acc_pass,
        "n_rej_with_esm": n_rej_valid,
        "rej_precision": rej_precision,
        "n_rej_designable_lost": rej_pass,
    }


def _per_length_table(df: pd.DataFrame, pll_variant: str) -> pd.DataFrame:
    rows = []
    for L, sub in df.groupby("length", dropna=True):
        x = pd.to_numeric(sub[pll_variant], errors="coerce").to_numpy()
        y = pd.to_numeric(sub["lobster_forward_tm"], errors="coerce").to_numpy()
        pr, pp, sr, sp, n = _safe_corr(x, y)
        rows.append({"length": int(L), "n": n, "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})
    return pd.DataFrame(rows).sort_values("length").reset_index(drop=True)


def _gate_decisions_at_threshold(pll_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Lower PLL = better; gate accepts if PLL <= threshold."""
    return pll_scores <= threshold


def _pll_threshold_matching_rate(pll_scores: np.ndarray, target_accept_rate: float) -> float:
    """Return the PLL threshold that yields ~target_accept_rate (lowest = best)."""
    finite = pll_scores[np.isfinite(pll_scores)]
    if len(finite) == 0:
        return float("nan")
    return float(np.quantile(finite, target_accept_rate))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paired-eval-dir", required=True, type=Path)
    p.add_argument("--concordance-dir", required=True, type=Path)
    p.add_argument(
        "--pll-variants", nargs="+",
        default=["struc_score_unif", "joint_score_unif", "seq_score_unif"],
        help="PLL columns to consider as gate signals.",
    )
    p.add_argument("--sr-threshold", type=float, default=0.833)
    p.add_argument("--esmfold-pass-tm", type=float, default=0.9)
    p.add_argument("--esmfold-pass-rmsd", type=float, default=2.0)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted = _load_accepted(args.paired_eval_dir)
    rejected = _load_rejected(args.concordance_dir)

    common_cols = [
        "length", "lobster_forward_tm", "esmfold_tm", "esmfold_rmsd", "esmfold_plddt", "sr_decision",
        *args.pll_variants,
    ]
    for c in args.pll_variants:
        if c not in accepted.columns:
            raise SystemExit(f"PLL variant {c} not in accepted CSV; available: "
                             f"{[c2 for c2 in accepted.columns if 'score' in c2]}")
        if c not in rejected.columns:
            raise SystemExit(f"PLL variant {c} not in rejected CSV; available: "
                             f"{[c2 for c2 in rejected.columns if 'score' in c2]}")

    a = accepted[common_cols].copy()
    r = rejected[common_cols].copy()
    pool = pd.concat([a, r], ignore_index=True)
    pool = pool.dropna(subset=["lobster_forward_tm"])
    logger.info("Combined attempt pool: %d (%d accepted + %d rejected)",
                len(pool), int((pool["sr_decision"] == "accept").sum()), int((pool["sr_decision"] == "reject").sum()))

    pool.to_csv(args.output_dir / "combined_attempt_pool.csv", index=False)

    # ---- Q1: PLL ↔ lobster forward-fold TM ----
    md = ["# PLL vs SR forward-fold-TM gate -- comparison report", ""]
    md.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    md.append("")
    md.append(f"- Combined attempt pool: **n = {len(pool)}** "
              f"({int((pool['sr_decision'] == 'accept').sum())} SR-accepted "
              f"+ {int((pool['sr_decision'] == 'reject').sum())} SR-rejected)")
    md.append(f"- Paired eval dir: `{args.paired_eval_dir}`")
    md.append(f"- Concordance dir: `{args.concordance_dir}`")
    md.append(f"- SR forward-fold-TM threshold: T = {args.sr_threshold}")
    md.append(f"- ESMFold-pass criteria: TM \u2265 {args.esmfold_pass_tm}, RMSD < {args.esmfold_pass_rmsd}")
    md.append("")

    md.append("## (Q1) Does PLL predict the lobster forward-fold TM (the SR gate quantity)?")
    md.append("")
    md.append("Pearson r (PLL vs lobster forward-fold TM); negative = lower-PLL goes with higher-TM, the desired direction.")
    md.append("")
    md.append("### Aggregate (full attempt pool)")
    md.append("")
    md.append("| pll variant | n | pearson r | pearson p | spearman r | spearman p |")
    md.append("|---|---|---|---|---|---|")
    aggr_rows = []
    for v in args.pll_variants:
        x = pd.to_numeric(pool[v], errors="coerce").to_numpy()
        y = pd.to_numeric(pool["lobster_forward_tm"], errors="coerce").to_numpy()
        pr, pp, sr, sp, n = _safe_corr(x, y)
        md.append(f"| `{v}` | {n} | {pr:+.3f} | {pp:.2e} | {sr:+.3f} | {sp:.2e} |")
        aggr_rows.append({"variant": v, "n": n, "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})
    pd.DataFrame(aggr_rows).to_csv(args.output_dir / "q1_pll_vs_forward_tm_aggregate.csv", index=False)
    md.append("")

    md.append("### Per-length (full attempt pool)")
    for v in args.pll_variants:
        md.append("")
        md.append(f"#### `{v}`")
        md.append("")
        pl = _per_length_table(pool, v)
        md.append("| L | n | pearson r | spearman r |")
        md.append("|---|---|---|---|")
        for _, row in pl.iterrows():
            md.append(f"| {int(row['length'])} | {int(row['n'])} | {row['pearson_r']:+.3f} | {row['spearman_r']:+.3f} |")
        pl.to_csv(args.output_dir / f"q1_per_length_{v}.csv", index=False)
    md.append("")

    # ---- Q2 + Q3: gate concordance + quality ----
    md.append("## (Q2) Would a PLL-based gate accept the same designs as SR's forward-fold-TM gate?")
    md.append("")
    md.append("For each PLL variant we compute the threshold that matches SR's empirical retain rate "
              "(so the two gates accept the same number of attempts), then score agreement.")
    md.append("")

    sr_accept = (pool["lobster_forward_tm"] >= args.sr_threshold).to_numpy()
    sr_accept_rate = float(sr_accept.mean())
    n_sr_accept = int(sr_accept.sum())
    n_sr_reject = int((~sr_accept).sum())
    md.append(f"- SR retain rate at T = {args.sr_threshold}: **{sr_accept_rate:.1%}** "
              f"({n_sr_accept}/{len(pool)} accepted, {n_sr_reject} rejected)")
    md.append("")
    md.append("ESMFold-pass = (ESMFold TM \u2265 " f"{args.esmfold_pass_tm}" ") AND (ESMFold RMSD < "
              f"{args.esmfold_pass_rmsd}). Computed only on rows that have ESMFold metrics.")
    md.append("")

    esm_pass_tm_only = (pool["esmfold_tm"] >= args.esmfold_pass_tm)
    esm_pass_rmsd_only = (pool["esmfold_rmsd"] < args.esmfold_pass_rmsd)
    esm_pass_joint = esm_pass_tm_only & esm_pass_rmsd_only
    esm_valid = pool["esmfold_tm"].notna() & pool["esmfold_rmsd"].notna()
    md.append(
        f"- Rows with valid ESMFold metrics: {int(esm_valid.sum())} / {len(pool)} "
        f"(SR-accepted: {int((esm_valid & (pool['sr_decision'] == 'accept')).sum())}, "
        f"SR-rejected: {int((esm_valid & (pool['sr_decision'] == 'reject')).sum())})"
    )
    md.append("")

    gate_rows = []
    sr_acc_quality_tm = _gate_quality("SR forward-fold-TM gate", sr_accept, esm_pass_tm_only.to_numpy())
    sr_acc_quality_joint = _gate_quality("SR forward-fold-TM gate", sr_accept, esm_pass_joint.to_numpy())

    for v in args.pll_variants:
        pll_scores = pd.to_numeric(pool[v], errors="coerce").to_numpy()
        # Set gate threshold so the PLL-accept rate matches SR's accept rate (lower PLL = accept).
        threshold = _pll_threshold_matching_rate(pll_scores, sr_accept_rate)
        pll_accept = _gate_decisions_at_threshold(pll_scores, threshold)
        valid = np.isfinite(pll_scores)
        agree = int(((pll_accept == sr_accept) & valid).sum())
        n_v = int(valid.sum())
        kappa = _cohens_kappa(sr_accept[valid], pll_accept[valid])

        # 2x2 contingency on the valid set
        a11 = int((sr_accept & pll_accept & valid).sum())   # both accept
        a10 = int((sr_accept & ~pll_accept & valid).sum())   # SR-accept, PLL-reject
        a01 = int((~sr_accept & pll_accept & valid).sum())   # SR-reject, PLL-accept
        a00 = int((~sr_accept & ~pll_accept & valid).sum())  # both reject

        md.append(f"### `{v}` (PLL threshold = {threshold:.4f}, matches SR's "
                  f"{sr_accept_rate:.1%} retain rate)")
        md.append("")
        md.append(f"- Raw agreement: **{agree/n_v:.1%}** ({agree} / {n_v})")
        md.append(f"- Cohen's kappa: **{kappa:+.3f}**")
        md.append("")
        md.append("Confusion matrix (rows = SR decision, cols = PLL decision; valid only):")
        md.append("")
        md.append("|  | PLL accept | PLL reject |")
        md.append("|---|---|---|")
        md.append(f"| **SR accept** | {a11} | {a10} |")
        md.append(f"| **SR reject** | {a01} | {a00} |")
        md.append("")

        # Gate quality at matched accept rate (Q3)
        pll_q_tm = _gate_quality(f"PLL-as-gate ({v})", pll_accept, esm_pass_tm_only.to_numpy())
        pll_q_joint = _gate_quality(f"PLL-as-gate ({v})", pll_accept, esm_pass_joint.to_numpy())
        gate_rows.append({"criterion": "ESM TM>=t", **pll_q_tm, "threshold": threshold})
        gate_rows.append({"criterion": "ESM TM>=t AND RMSD<t", **pll_q_joint, "threshold": threshold})

    gate_rows.append({"criterion": "ESM TM>=t", **sr_acc_quality_tm, "threshold": args.sr_threshold})
    gate_rows.append({"criterion": "ESM TM>=t AND RMSD<t", **sr_acc_quality_joint, "threshold": args.sr_threshold})

    md.append("## (Q3) Which gate accepts BETTER designs?")
    md.append("")
    md.append("For each gate (matched accept rate), what fraction of accepted designs ESMFold deems designable, "
              "and what fraction of rejected designs ESMFold *would* have deemed designable (= regret).")
    md.append("")

    for crit in ("ESM TM>=t", "ESM TM>=t AND RMSD<t"):
        sub = [r for r in gate_rows if r["criterion"] == crit]
        crit_label = (
            f"ESMFold TM \u2265 {args.esmfold_pass_tm}"
            if crit == "ESM TM>=t"
            else f"ESMFold TM \u2265 {args.esmfold_pass_tm} AND ESMFold RMSD < {args.esmfold_pass_rmsd}"
        )
        md.append(f"### Criterion: {crit_label}")
        md.append("")
        md.append("| gate | # accepted | acc precision (ESM-pass per accept) | # rejected | rej precision (ESM-pass per reject) |")
        md.append("|---|---|---|---|---|")
        for row in sub:
            md.append(
                f"| {row['gate']} | "
                f"{row['n_accepted']} (ESM={row['n_acc_with_esm']}) | "
                f"{row['acc_precision']:.1%} ({row['n_acc_designable']}/{row['n_acc_with_esm']}) | "
                f"{row['n_rejected']} (ESM={row['n_rej_with_esm']}) | "
                f"{row['rej_precision']:.1%} ({row['n_rej_designable_lost']}/{row['n_rej_with_esm']}) |"
            )
        md.append("")

    pd.DataFrame(gate_rows).to_csv(args.output_dir / "q3_gate_quality.csv", index=False)

    # ---- Q3b: PLL threshold sweep (does any threshold beat SR's 76.6%?) ----
    md.append("### PLL threshold sweep (gate quality vs PLL acceptance percentile)")
    md.append("")
    md.append("Each row = a PLL accept rate; PLL threshold = that quantile of PLL scores. "
              "We then report acceptance precision under both ESMFold criteria, "
              "for the strongest single PLL variant (`struc_score_unif` by default).")
    md.append("")
    sweep_rates = [0.6, 0.7, 0.75, 0.80, 0.83, 0.85, 0.90, 0.95, 1.00]
    primary = args.pll_variants[0]
    pll_scores = pd.to_numeric(pool[primary], errors="coerce").to_numpy()
    md.append(f"Primary PLL variant: `{primary}`. SR-accept rate at T={args.sr_threshold} is {sr_accept_rate:.1%} "
              "(also shown for reference).")
    md.append("")
    md.append("| accept rate | PLL threshold | n acc (ESM valid) | acc precision (TM\u2265t) | acc precision (TM\u2265t AND RMSD<t) |")
    md.append("|---|---|---|---|---|")
    for rate in sweep_rates:
        thr = _pll_threshold_matching_rate(pll_scores, rate)
        accept = _gate_decisions_at_threshold(pll_scores, thr) & np.isfinite(pll_scores)
        q_tm = _gate_quality(f"PLL@{rate:.2f}", accept, esm_pass_tm_only.to_numpy())
        q_joint = _gate_quality(f"PLL@{rate:.2f}", accept, esm_pass_joint.to_numpy())
        md.append(
            f"| {rate:.0%} | {thr:.4f} | {q_tm['n_accepted']} ({q_tm['n_acc_with_esm']}) | "
            f"{q_tm['acc_precision']:.1%} | {q_joint['acc_precision']:.1%} |"
        )
    md.append("")
    md.append(f"For comparison, **SR gate** at T={args.sr_threshold}: "
              f"acc precision (TM\u2265{args.esmfold_pass_tm}) = "
              f"{sr_acc_quality_tm['acc_precision']:.1%}, "
              f"acc precision (TM\u2265{args.esmfold_pass_tm} AND RMSD<{args.esmfold_pass_rmsd}) = "
              f"{sr_acc_quality_joint['acc_precision']:.1%}.")
    md.append("")

    out_md = args.output_dir / "pll_vs_sr_gate_report.md"
    out_md.write_text("\n".join(md))
    logger.info("Wrote report: %s", out_md)


if __name__ == "__main__":
    main()
