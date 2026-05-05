#!/usr/bin/env python3
"""Compile benchmark comparison tables for GenUME vs competitor models.

Reads evaluation directories for GenUME configs and competitor models (LaProteina, DPLM2),
collects all metrics, and generates a markdown file with per-length and aggregate tables.

Usage:
    cd /cv/home/lisanzas/lobster
    uv run python scripts/compile_benchmark_table.py \
        --output /cv/home/lisanzas/gen_ume_benchmark_comparison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

LENGTHS = [100, 200, 300, 400, 500]

GENUME_CONFIGS = {
    "GenUME-base": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-11T12-11-53_unconditional",
        "desc": "stoch=20/60, default temps",
    },
    "GenUME-stoch": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-12T19-31-50_unconditional_seq10_struc10",
        "desc": "stoch=10/10, default temps",
    },
    "GenUME-temp": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-16T13-19-41_unconditional_seq10_struc10_tseq0.5_tstruc0.4",
        "desc": "stoch=10/10, tseq=0.5, tstruc=0.4",
    },
    "GenUME-val": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-16T22-46-28_unconditional_seq10_struc10_tseq0.5_tstruc0.4_biasV1.0_steps10",
        "desc": "stoch=10/10, tseq=0.5, tstruc=0.4, V=1.0 10steps",
    },
    "GenUME-val25": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-16T22-46-28_unconditional_seq10_struc10_tseq0.5_tstruc0.4_biasV1.0_steps25",
        "desc": "stoch=10/10, tseq=0.5, tstruc=0.4, V=1.0 25steps",
    },
    "GenUME-val25-base": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25",
        "desc": "stoch=20/60, default temps, V=1.0 25steps",
    },
}

COMPETITOR_CONFIGS = {
    "LaProteina": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/benchmark_laproteina_unconditional",
        "metrics_key": "laproteina",
    },
    "DPLM2": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/benchmark_dplm2_unconditional",
        "metrics_key": "dplm2",
    },
    "Proteina+ProteinMPNN-CA": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/benchmark_proteina_pmpnn_unconditional",
        "metrics_key": "proteina",
    },
    "Genie2+ProteinMPNN-CA": {
        "dir": "/cv/scratch/u/lisanzas/evaluations/benchmark_genie2_pmpnn_unconditional",
        "metrics_key": "genie2",
    },
}


def load_genume_metrics(eval_dir: Path) -> pd.DataFrame | None:
    """Load GenUME metrics from unconditional_metrics_*.csv files (excluding backups)."""
    csv_files = sorted(eval_dir.glob("unconditional_metrics_*.csv"))
    csv_files = [f for f in csv_files if "_old_rmsd" not in f.name]
    if not csv_files:
        return None
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    rows = []
    for _, r in df.iterrows():
        length = int(r.get("sequence_length", 0))
        plddt = r.get("plddt", r.get("plddt_unconditional", 0))
        tm = r.get("tm_score", r.get("tm_score_esmfold_unconditional", 0))
        rmsd = r.get("rmsd", r.get("rmsd_esmfold_unconditional", 0))
        rows.append({"length": length, "plddt": plddt, "tm_score": tm, "rmsd": rmsd})
    return pd.DataFrame(rows)


def load_competitor_metrics(eval_dir: Path, metrics_key: str) -> pd.DataFrame | None:
    """Load competitor metrics from unconditional_metrics_{metrics_key}.csv."""
    csv_path = eval_dir / f"unconditional_metrics_{metrics_key.lower()}.csv"
    if not csv_path.exists():
        csv_files = sorted(eval_dir.glob("unconditional_metrics_*.csv"))
        if csv_files:
            csv_path = csv_files[0]
        else:
            return None
    return pd.read_csv(csv_path)


def load_clusters(eval_dir: Path) -> dict[int, int]:
    """Load per-length cluster counts from foldseek_results."""
    clusters = {}
    for length in LENGTHS:
        rep_fasta = eval_dir / "foldseek_results" / f"length_{length}" / "res_rep_seq.fasta"
        if rep_fasta.exists():
            clusters[length] = sum(1 for line in open(rep_fasta) if line.startswith(">"))
        else:
            clusters[length] = 0
    return clusters


def load_sse(eval_dir: Path) -> pd.DataFrame | None:
    """Load SSE parquet."""
    pq = eval_dir / "uncond_sse_index.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return None


def load_novelty(eval_dir: Path, ref_label: str) -> dict[int | str, float]:
    """Load per-length mean max TM-score from novelty summary CSV.

    Returns per-length values plus a weighted "All" aggregate.
    """
    csv_path = eval_dir / f"novelty_vs_{ref_label}_summary.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    result = {}
    for _, r in df.iterrows():
        result[int(r["length"])] = float(r["mean_max_tmscore"])
    if "total_queries" in df.columns and df["total_queries"].sum() > 0:
        wavg = (df["mean_max_tmscore"] * df["total_queries"]).sum() / df["total_queries"].sum()
        result["All"] = float(wavg)
    return result


def fmt(val, decimals=3):
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


def fmt_pct(val):
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/cv/home/lisanzas/gen_ume_benchmark_comparison.md")
    args = parser.parse_args()

    all_models = {}

    # Load GenUME configs
    for name, cfg in GENUME_CONFIGS.items():
        eval_dir = Path(cfg["dir"])
        if not eval_dir.exists():
            logger.warning(f"GenUME config {name} dir not found: {eval_dir}")
            continue
        metrics = load_genume_metrics(eval_dir)
        clusters = load_clusters(eval_dir)
        sse = load_sse(eval_dir)
        novelty_pdb = load_novelty(eval_dir, "pdb")
        novelty_afdb = load_novelty(eval_dir, "afdb")
        novelty_denovo = load_novelty(eval_dir, "denovo")
        all_models[name] = {
            "metrics": metrics, "clusters": clusters, "sse": sse,
            "novelty_pdb": novelty_pdb, "novelty_afdb": novelty_afdb,
            "novelty_denovo": novelty_denovo, "is_genume": True,
        }

    # Load competitor configs
    for name, cfg in COMPETITOR_CONFIGS.items():
        eval_dir = Path(cfg["dir"])
        if not eval_dir.exists():
            logger.warning(f"Competitor {name} dir not found: {eval_dir}")
            continue
        metrics = load_competitor_metrics(eval_dir, cfg.get("metrics_key", name.lower()))
        clusters = load_clusters(eval_dir)
        sse = load_sse(eval_dir)
        novelty_pdb = load_novelty(eval_dir, "pdb")
        novelty_afdb = load_novelty(eval_dir, "afdb")
        all_models[name] = {
            "metrics": metrics, "clusters": clusters, "sse": sse,
            "novelty_pdb": novelty_pdb, "novelty_afdb": novelty_afdb,
            "novelty_denovo": {}, "is_genume": False,
        }

    lines = []
    lines.append("# GenUME Benchmark Comparison\n")
    lines.append("## Model Configurations\n")
    for name, cfg in GENUME_CONFIGS.items():
        lines.append(f"- **{name}**: {cfg['desc']}")
    lines.append(f"- **LaProteina**: co-design model (from de novo dataset)")
    lines.append(f"- **DPLM2**: co-design model (from de novo dataset)")
    lines.append(f"- **Proteina+ProteinMPNN-CA**: backbone model + CA-only inverse folder (from de novo dataset)")
    lines.append(f"- **Genie2+ProteinMPNN-CA**: backbone model + CA-only inverse folder (from de novo dataset)")
    lines.append("")

    model_order = list(GENUME_CONFIGS.keys()) + list(COMPETITOR_CONFIGS.keys())
    model_order = [m for m in model_order if m in all_models]

    # ── Table 1: Designability ──
    lines.append("## Table 1: Designability and Quality\n")
    lines.append("| Length | Model | N | Pass Rate (%) | pLDDT | TM-score | RMSD (A) |")
    lines.append("|--------|-------|---|---------------|-------|----------|----------|")

    for length in LENGTHS + ["All"]:
        for model_name in model_order:
            data = all_models[model_name]
            metrics = data["metrics"]
            if metrics is None or metrics.empty:
                bold = "**" if model_name.startswith("GenUME") else ""
                lines.append(f"| {length} | {bold}{model_name}{bold} | — | — | — | — | — |")
                continue

            if length == "All":
                mdf = metrics
            else:
                mdf = metrics[metrics["length"] == length]

            if mdf.empty:
                bold = "**" if model_name.startswith("GenUME") else ""
                lines.append(f"| {length} | {bold}{model_name}{bold} | — | — | — | — | — |")
                continue

            n = len(mdf)
            pass_col = "pass" if "pass" in mdf.columns else None
            if pass_col:
                pass_rate = mdf[pass_col].mean() * 100
            else:
                pass_rate = (mdf["rmsd"] < 2.0).mean() * 100
            plddt = mdf["plddt"].mean()
            tm = mdf["tm_score"].mean()
            rmsd = mdf["rmsd"].mean()

            bold = "**" if model_name.startswith("GenUME") else ""
            length_str = f"**{length}**" if length == "All" else str(length)
            if length == "All":
                lines.append(
                    f"| {length_str} | {bold}{model_name}{bold} | **{n}** | "
                    f"**{fmt_pct(pass_rate)}** | **{fmt(plddt)}** | **{fmt(tm)}** | **{fmt(rmsd, 2)}** |"
                )
            else:
                lines.append(
                    f"| {length_str} | {bold}{model_name}{bold} | {n} | "
                    f"{fmt_pct(pass_rate)} | {fmt(plddt)} | {fmt(tm)} | {fmt(rmsd, 2)} |"
                )
    lines.append("")

    # ── Table 2: Clusters ──
    lines.append("## Table 2: Structural Diversity (Clusters, TM > 0.5)\n")
    header = "| Length |"
    sep = "|--------|"
    for m in model_order:
        header += f" {m} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for length in LENGTHS:
        row = f"| {length} |"
        for m in model_order:
            c = all_models[m]["clusters"].get(length, 0)
            row += f" {c if c else '—'} |"
        lines.append(row)

    total_row = "| **Total** |"
    for m in model_order:
        t = sum(all_models[m]["clusters"].get(l, 0) for l in LENGTHS)
        total_row += f" **{t if t else '—'}** |"
    lines.append(total_row)
    lines.append("")

    # ── Table 3: SSE ──
    lines.append("## Table 3: Secondary Structure Composition (P-SEA)\n")
    lines.append("| Length | Model | Helix (%) | Strand (%) | Coil (%) | All-helical | Mixed | All-beta |")
    lines.append("|--------|-------|-----------|------------|----------|-------------|-------|----------|")

    for length in LENGTHS + ["All"]:
        for model_name in model_order:
            data = all_models[model_name]
            sse = data["sse"]
            if sse is None or sse.empty:
                bold = "**" if model_name.startswith("GenUME") else ""
                lines.append(f"| {length} | {bold}{model_name}{bold} | — | — | — | — | — | — |")
                continue

            if length == "All":
                sdf = sse
            else:
                sdf = sse[sse["length"] == length]

            if sdf.empty:
                bold = "**" if model_name.startswith("GenUME") else ""
                lines.append(f"| {length} | {bold}{model_name}{bold} | — | — | — | — | — | — |")
                continue

            helix = sdf["helix"].mean() * 100
            strand = sdf["strand"].mean() * 100
            coil = sdf["coil"].mean() * 100
            all_h = int((sdf["ss_category"] == "all_helical").sum())
            mixed = int((sdf["ss_category"] == "mixed").sum())
            all_b = int((sdf["ss_category"] == "all_beta").sum())

            bold = "**" if model_name.startswith("GenUME") else ""
            length_str = f"**{length}**" if length == "All" else str(length)
            if length == "All":
                lines.append(
                    f"| {length_str} | {bold}{model_name}{bold} | "
                    f"**{fmt_pct(helix)}** | **{fmt_pct(strand)}** | **{fmt_pct(coil)}** | "
                    f"**{all_h}** | **{mixed}** | **{all_b}** |"
                )
            else:
                lines.append(
                    f"| {length_str} | {bold}{model_name}{bold} | "
                    f"{fmt_pct(helix)} | {fmt_pct(strand)} | {fmt_pct(coil)} | "
                    f"{all_h} | {mixed} | {all_b} |"
                )
    lines.append("")

    # ── Table 4: Novelty ──
    lines.append("## Table 4: Novelty (Mean Max TM-score to Reference Set)\n")
    lines.append("Lower = more novel.\n")
    lines.append("| Length | Model | vs PDB | vs AFDB | vs DeNovo |")
    lines.append("|--------|-------|--------|---------|-----------|")

    for length in LENGTHS + ["All"]:
        for model_name in model_order:
            data = all_models[model_name]
            is_genume = data["is_genume"]

            pdb_v = data["novelty_pdb"].get(length)
            afdb_v = data["novelty_afdb"].get(length)
            denovo_v = data["novelty_denovo"].get(length)

            denovo_str = fmt(denovo_v) if is_genume else "n/a"

            bold = "**" if model_name.startswith("GenUME") else ""
            length_str = f"**{length}**" if length == "All" else str(length)
            if length == "All":
                lines.append(
                    f"| {length_str} | {bold}{model_name}{bold} | "
                    f"**{fmt(pdb_v)}** | **{fmt(afdb_v)}** | **{denovo_str}** |"
                )
            else:
                lines.append(
                    f"| {length_str} | {bold}{model_name}{bold} | "
                    f"{fmt(pdb_v)} | {fmt(afdb_v)} | {denovo_str} |"
                )
    lines.append("")
    lines.append("DeNovo novelty only computed for GenUME configs (LaProteina/DPLM2 outputs are part of the de novo training set).\n")

    output = "\n".join(lines)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)
    logger.info(f"Wrote benchmark comparison to {output_path}")


if __name__ == "__main__":
    main()
