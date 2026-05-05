"""3-way SR-threshold comparison: no-SR vs SR(TM>=0.833) vs SR(TM>=0.9).

For each Gen-UME unconditional ablation config (TED-stoch, TED-val25-base, val25-base),
compares the three SR variants on:

  1. Designability per length  (from per-sample ESMFold metrics on the saved design)
  2. Diversity per length       (from unconditional_diversity_length_*.csv)
  3. Novelty per length         (from novelty_vs_{denovo,pdb}_summary.csv)

Designability is reported under both the lenient (TM>0.5 AND RMSD<2.0)
and the strict (TM>0.9 AND RMSD<2.0) definitions. The lenient one matches
the threshold used by the diversity-cluster filter in the eval pipeline.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

LENGTHS = [100, 200, 300, 400, 500]

EVAL_ROOT = Path("/cv/scratch/u/lisanzas/evaluations")

# rows are config, columns are SR variants
CONFIGS = ["val25base", "ted_val25base", "ted_stoch"]
VARIANTS = ["nosr", "sr0833", "sr0900"]

DIRS = {
    ("val25base", "nosr"):
        EVAL_ROOT / "gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25",
    ("val25base", "sr0833"):
        EVAL_ROOT / "gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25_selfreflect_paired",
    ("val25base", "sr0900"):
        EVAL_ROOT / "gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25_sr_tm0p9",
    ("ted_val25base", "nosr"):
        EVAL_ROOT / "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25",
    ("ted_val25base", "sr0833"):
        EVAL_ROOT / "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_selfreflect_paired",
    ("ted_val25base", "sr0900"):
        EVAL_ROOT / "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_sr_tm0p9",
    ("ted_stoch", "nosr"):
        EVAL_ROOT / "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10",
    ("ted_stoch", "sr0833"):
        EVAL_ROOT / "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10_selfreflect_paired",
    ("ted_stoch", "sr0900"):
        EVAL_ROOT / "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10_sr_tm0p9",
}

VARIANT_LABEL = {"nosr": "no-SR", "sr0833": "SR≥0.833", "sr0900": "SR≥0.900"}


def load_unconditional_rows(eval_dir: Path) -> pd.DataFrame:
    """Load *only* the `unconditional_*` rows (the saved design per sample),
    skipping any `self_reflection_*` diagnostic rows. Prefer the most recent
    metrics CSV without the `_old_rmsd` suffix.
    """
    paths = sorted(p for p in glob.glob(str(eval_dir / "unconditional_metrics_*.csv")) if "_old_rmsd" not in p)
    if not paths:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {eval_dir}")
    df = pd.read_csv(paths[-1])
    df = df[df["run_id"].astype(str).str.startswith("unconditional_")].reset_index(drop=True)
    return df


def designability_per_length(df: pd.DataFrame) -> pd.DataFrame:
    """For each length, compute pass-rates under several common designability defs.

    On `unconditional_*` rows, `tm_score` and `rmsd` are the ESMFold-vs-uncond
    self-consistency metrics for the saved (initial) design. `plddt` is the
    ESMFold pLDDT of the generated sequence.
    """
    out = []
    for L in LENGTHS:
        sub = df[df["sequence_length"] == L]
        n = len(sub)
        if n == 0:
            continue
        tm = pd.to_numeric(sub["tm_score"], errors="coerce")
        rm = pd.to_numeric(sub["rmsd"], errors="coerce")
        pl = pd.to_numeric(sub["plddt"], errors="coerce")
        out.append(
            {
                "length": L,
                "n": n,
                "pass_TM>0.5_RMSD<2.0": ((tm > 0.5) & (rm < 2.0)).mean() * 100,
                "pass_TM>0.9_RMSD<2.0": ((tm > 0.9) & (rm < 2.0)).mean() * 100,
                "pass_TM>0.5": (tm > 0.5).mean() * 100,
                "pass_RMSD<2.0": (rm < 2.0).mean() * 100,
                "mean_pLDDT": pl.mean(),
                "mean_TM": tm.mean(),
                "mean_RMSD": rm.mean(),
            }
        )
    overall = {
        "length": "all",
        "n": int(sum(r["n"] for r in out)),
    }
    if out:
        for k in [
            "pass_TM>0.5_RMSD<2.0", "pass_TM>0.9_RMSD<2.0",
            "pass_TM>0.5", "pass_RMSD<2.0",
            "mean_pLDDT", "mean_TM", "mean_RMSD",
        ]:
            # weight by n to avoid bias when one length has fewer rows
            overall[k] = sum(r[k] * r["n"] for r in out) / overall["n"]
        out.append(overall)
    return pd.DataFrame(out)


def diversity_per_length(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for L in LENGTHS:
        files = glob.glob(str(eval_dir / f"unconditional_diversity_length_{L}_*.csv"))
        if not files:
            continue
        d = pd.read_csv(files[-1])
        if d.empty:
            continue
        r = d.iloc[-1]
        rows.append({
            "length": L,
            "n_designable": int(r["total_structures"]),
            "n_clusters": int(r["num_clusters"]),
            "diversity_pct": float(r["diversity_percentage"]),
        })
    return pd.DataFrame(rows)


def novelty_per_length(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for tag in ("denovo", "pdb"):
        f = eval_dir / f"novelty_vs_{tag}_summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            rows.append({
                "length": int(r["length"]),
                "novelty_target": tag,
                "n_queries": int(r["total_queries"]),
                "mean_maxTM": float(r["mean_max_tmscore"]),
                "median_maxTM": float(r["median_max_tmscore"]),
                "min_maxTM": float(r["min_max_tmscore"]),
                "pct_novel_TM<0.5": float(r["pct_highly_novel_tmscore_lt_0.5"]),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["novelty_target", "length"]).reset_index(drop=True)


def emit_designability(report: list[str], all_des: dict[tuple[str, str], pd.DataFrame]) -> None:
    """Per-config block: per-length pass-rate table comparing 3 SR variants."""
    report.append("# Designability comparison (per length, 3 SR variants)\n")
    report.append("Designability defined as **ESMFold scTM > 0.5 AND scRMSD < 2.0 Å** (the standard\n"
                  "lenient threshold; matches the gate used for the diversity cluster filter).\n"
                  "Stricter cutoff `TM > 0.9 AND RMSD < 2.0` is reported for context.\n")
    for cfg in CONFIGS:
        report.append(f"\n## {cfg}\n")
        report.append("### Designable (TM>0.5 AND RMSD<2.0)\n")
        report.append("| L | no-SR | SR≥0.833 | SR≥0.900 | Δ(0.9 − no-SR) | Δ(0.9 − 0.833) |")
        report.append("|---|---:|---:|---:|---:|---:|")
        for L in LENGTHS + ["all"]:
            row = [str(L)]
            vals = {}
            for v in VARIANTS:
                d = all_des.get((cfg, v))
                if d is None or d.empty:
                    row.append("-"); vals[v] = None; continue
                if L == "all":
                    sub = d[d["length"] == "all"]
                else:
                    sub = d[d["length"] == L]
                if sub.empty:
                    row.append("-"); vals[v] = None; continue
                v0 = float(sub["pass_TM>0.5_RMSD<2.0"].iloc[0])
                vals[v] = v0
                row.append(f"{v0:.1f}%")
            d_no = (vals["sr0900"] - vals["nosr"]) if (vals["sr0900"] is not None and vals["nosr"] is not None) else None
            d_83 = (vals["sr0900"] - vals["sr0833"]) if (vals["sr0900"] is not None and vals["sr0833"] is not None) else None
            row.append(f"{d_no:+.1f}pp" if d_no is not None else "-")
            row.append(f"{d_83:+.1f}pp" if d_83 is not None else "-")
            report.append("| " + " | ".join(row) + " |")

        report.append("\n### Designable (TM>0.9 AND RMSD<2.0) — strict\n")
        report.append("| L | no-SR | SR≥0.833 | SR≥0.900 | Δ(0.9 − no-SR) | Δ(0.9 − 0.833) |")
        report.append("|---|---:|---:|---:|---:|---:|")
        for L in LENGTHS + ["all"]:
            row = [str(L)]
            vals = {}
            for v in VARIANTS:
                d = all_des.get((cfg, v))
                if d is None or d.empty:
                    row.append("-"); vals[v] = None; continue
                if L == "all":
                    sub = d[d["length"] == "all"]
                else:
                    sub = d[d["length"] == L]
                if sub.empty:
                    row.append("-"); vals[v] = None; continue
                v0 = float(sub["pass_TM>0.9_RMSD<2.0"].iloc[0])
                vals[v] = v0
                row.append(f"{v0:.1f}%")
            d_no = (vals["sr0900"] - vals["nosr"]) if (vals["sr0900"] is not None and vals["nosr"] is not None) else None
            d_83 = (vals["sr0900"] - vals["sr0833"]) if (vals["sr0900"] is not None and vals["sr0833"] is not None) else None
            row.append(f"{d_no:+.1f}pp" if d_no is not None else "-")
            row.append(f"{d_83:+.1f}pp" if d_83 is not None else "-")
            report.append("| " + " | ".join(row) + " |")


def emit_diversity(report: list[str], all_div: dict[tuple[str, str], pd.DataFrame]) -> None:
    report.append("\n\n# Diversity comparison (per length, 3 SR variants)\n")
    report.append("Diversity = `n_clusters / n_designable`, where designable uses the eval pipeline's\n"
                  "internal gate (TM>0.5, RMSD<2.0). `n_designable` shown alongside since SR\n"
                  "obviously changes the denominator (smaller designable pool → noisier diversity).\n")
    for cfg in CONFIGS:
        report.append(f"\n## {cfg}\n")
        report.append("### Diversity %\n")
        report.append("| L | no-SR | SR≥0.833 | SR≥0.900 | Δ(0.9 − no-SR) |")
        report.append("|---|---:|---:|---:|---:|")
        for L in LENGTHS:
            row = [str(L)]
            vals = {}
            for v in VARIANTS:
                d = all_div.get((cfg, v))
                if d is None or d.empty:
                    row.append("-"); vals[v] = None; continue
                sub = d[d["length"] == L]
                if sub.empty:
                    row.append("-"); vals[v] = None; continue
                v0 = float(sub["diversity_pct"].iloc[0])
                vals[v] = v0
                row.append(f"{v0:.1f}%")
            d_no = (vals["sr0900"] - vals["nosr"]) if (vals["sr0900"] is not None and vals["nosr"] is not None) else None
            row.append(f"{d_no:+.1f}pp" if d_no is not None else "-")
            report.append("| " + " | ".join(row) + " |")

        report.append("\n### Designable count per length (denominator)\n")
        report.append("| L | no-SR | SR≥0.833 | SR≥0.900 |")
        report.append("|---|---:|---:|---:|")
        for L in LENGTHS:
            row = [str(L)]
            for v in VARIANTS:
                d = all_div.get((cfg, v))
                if d is None or d.empty:
                    row.append("-"); continue
                sub = d[d["length"] == L]
                row.append(str(int(sub["n_designable"].iloc[0])) if not sub.empty else "-")
            report.append("| " + " | ".join(row) + " |")


def emit_novelty(report: list[str], all_nov: dict[tuple[str, str], pd.DataFrame]) -> None:
    report.append("\n\n# Novelty comparison (per length, 3 SR variants)\n")
    report.append("`mean_maxTM` of each designed cluster representative against either the DeNovo\n"
                  "training set or the PDB. **Lower = more novel.** Computed only on cluster reps,\n"
                  "so SR variants with fewer designables are queried with fewer reps (`n_queries`).\n")
    for cfg in CONFIGS:
        report.append(f"\n## {cfg}\n")
        for tag, title in (("denovo", "Novelty vs DeNovo (mean max TM)"), ("pdb", "Novelty vs PDB (mean max TM)")):
            report.append(f"\n### {title}\n")
            report.append("| L | no-SR | SR≥0.833 | SR≥0.900 | Δ(0.9 − no-SR) |")
            report.append("|---|---:|---:|---:|---:|")
            for L in LENGTHS:
                row = [str(L)]
                vals = {}
                for v in VARIANTS:
                    d = all_nov.get((cfg, v))
                    if d is None or d.empty:
                        row.append("-"); vals[v] = None; continue
                    sub = d[(d["length"] == L) & (d["novelty_target"] == tag)]
                    if sub.empty:
                        row.append("-"); vals[v] = None; continue
                    v0 = float(sub["mean_maxTM"].iloc[0])
                    vals[v] = v0
                    row.append(f"{v0:.3f}")
                d_no = (vals["sr0900"] - vals["nosr"]) if (vals["sr0900"] is not None and vals["nosr"] is not None) else None
                row.append(f"{d_no:+.3f}" if d_no is not None else "-")
                report.append("| " + " | ".join(row) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("/cv/home/lisanzas/lobster/sr_threshold_sweep_report.md"))
    args = ap.parse_args()

    all_des: dict[tuple[str, str], pd.DataFrame] = {}
    all_div: dict[tuple[str, str], pd.DataFrame] = {}
    all_nov: dict[tuple[str, str], pd.DataFrame] = {}

    for cfg in CONFIGS:
        for v in VARIANTS:
            d = DIRS[(cfg, v)]
            if not d.is_dir():
                print(f"WARN: missing dir {d}")
                continue
            try:
                df = load_unconditional_rows(d)
                all_des[(cfg, v)] = designability_per_length(df)
            except Exception as e:
                print(f"WARN: designability failed for {cfg}/{v}: {e}")
            try:
                all_div[(cfg, v)] = diversity_per_length(d)
            except Exception as e:
                print(f"WARN: diversity failed for {cfg}/{v}: {e}")
            try:
                all_nov[(cfg, v)] = novelty_per_length(d)
            except Exception as e:
                print(f"WARN: novelty failed for {cfg}/{v}: {e}")

    report: list[str] = []
    report.append("# SR-threshold sweep: no-SR vs SR(TM≥0.833) vs SR(TM≥0.900)\n")
    report.append("100 samples generated per length × 5 lengths × 3 sampling configs × 3 SR variants.\n")
    emit_designability(report, all_des)
    emit_diversity(report, all_div)
    emit_novelty(report, all_nov)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report))
    print(f"\nWrote: {args.output}")
    print("=" * 60)
    print("\n".join(report))


if __name__ == "__main__":
    main()
