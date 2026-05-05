#!/usr/bin/env python3
"""TM-score novelty analysis for unconditional generation.

Compares ESMFold-validated generated structures against cluster representatives
from (a) denovo and (b) PDB seqid40, using Foldseek search. Low max TM-score
to any representative = high novelty. Runs separate analyses for each reference.

Prerequisites:
  - Run convert_pdb_cluster_reps_to_pdb.py to populate --pdb-reps-pdb-dir
  - Denovo cluster reps at --denovo-reps-dir/length_{L}/cluster_representatives/

Options:
  --use-existing-clusters: Use cluster reps from prior diversity step (foldseek_results/,
    foldseek_temp_dir/). More efficient. If not set, uses all designs per length.

Usage:
    cd /cv/home/lisanzas/lobster
    uv run python scripts/analyze_tm_score_novelty.py \
        --uncond-dir /path/to/unconditional_eval \
        --denovo-reps-dir /cv/scratch/u/lisanzas/denovo_dataset/clustered \
        --pdb-reps-pdb-dir /cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb \
        --foldseek-bin /cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from biotite.sequence.io import fasta

from lobster.metrics.cal_foldseek_clusters import copy_structures_by_rmsd, setup_foldseek_path


def run_novelty_search(
    query_dir: Path,
    ref_dir: Path,
    result_dir: Path,
    foldseek_bin_path: str | None = None,
    alignment_type: int = 1,
) -> Path | None:
    """Run Foldseek search: query vs reference, return path to alignment TSV."""
    if foldseek_bin_path:
        setup_foldseek_path(foldseek_bin_path)
    foldseek = "foldseek"

    query_db = result_dir / "query_db"
    ref_db = result_dir / "ref_db"
    align_db = result_dir / "align_db"
    tmp_dir = result_dir / "tmp"
    result_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # createdb
    for name, inp in [("query", query_dir), ("ref", ref_dir)]:
        db = query_db if name == "query" else ref_db
        cmd = [foldseek, "createdb", str(inp), str(db)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            logger.error(f"Foldseek createdb ({name}) failed: {p.stderr}")
            return None

    # search (TM alignment)
    cmd = [
        foldseek,
        "search",
        str(query_db),
        str(ref_db),
        str(align_db),
        str(tmp_dir),
        "--alignment-type",
        str(alignment_type),
        "-a",
        "1",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error(f"Foldseek search failed: {p.stderr}")
        return None

    # convertalis to get TM-scores
    tsv_path = result_dir / "alignments.tsv"
    cmd = [
        foldseek,
        "convertalis",
        str(query_db),
        str(ref_db),
        str(align_db),
        str(tsv_path),
        "--format-output",
        "query,target,alntmscore",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error(f"Foldseek convertalis failed: {p.stderr}")
        return None

    return tsv_path


def compute_novelty_metrics(tsv_path: Path) -> dict | None:
    """Parse alignment TSV and compute max TM per query, then aggregate metrics."""
    if not tsv_path.exists():
        return None
    df = pd.read_csv(tsv_path, sep="\t", names=["query", "target", "alntmscore"])
    if df.empty:
        return {"total_queries": 0}
    df["alntmscore"] = pd.to_numeric(df["alntmscore"], errors="coerce")
    df = df.dropna(subset=["alntmscore"])
    if df.empty:
        return {"total_queries": 0}
    max_tm = df.groupby("query")["alntmscore"].max()
    return {
        "total_queries": len(max_tm),
        "mean_max_tmscore": float(max_tm.mean()),
        "median_max_tmscore": float(max_tm.median()),
        "min_max_tmscore": float(max_tm.min()),
        "pct_highly_novel_tmscore_lt_0.5": float((max_tm < 0.5).mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="TM-score novelty analysis for unconditional generation")
    parser.add_argument("--uncond-dir", type=str, required=True, help="Unconditional generation output directory")
    parser.add_argument(
        "--denovo-reps-dir",
        type=str,
        default="/cv/scratch/u/lisanzas/denovo_dataset/clustered",
        help="Denovo cluster_representatives base (length_{L}/cluster_representatives/)",
    )
    parser.add_argument(
        "--pdb-reps-pdb-dir",
        type=str,
        default="/cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb",
        help="PDB cluster rep PDB files (from convert_pdb_cluster_reps_to_pdb.py)",
    )
    parser.add_argument(
        "--foldseek-bin",
        type=str,
        default="/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin",
    )
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--lengths", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    parser.add_argument(
        "--use-existing-clusters",
        action="store_true",
        help="Use cluster reps from prior diversity step (foldseek_results/). If not set, use all designs per length.",
    )
    parser.add_argument(
        "--ref-label",
        type=str,
        default="pdb",
        help="Label for the PDB-reps reference set (affects output filenames: novelty_vs_{label}_summary.csv)",
    )
    args = parser.parse_args()

    uncond_dir = Path(args.uncond_dir)
    denovo_reps_dir = Path(args.denovo_reps_dir)
    pdb_reps_dir = Path(args.pdb_reps_pdb_dir)

    if not uncond_dir.exists():
        logger.error(f"Unconditional dir not found: {uncond_dir}")
        return 1
    if not pdb_reps_dir.exists() or not list(pdb_reps_dir.glob("*.pdb")):
        logger.error(f"PDB reps dir empty or missing. Run: uv run python scripts/convert_pdb_cluster_reps_to_pdb.py")
        return 1

    novelty_dir = uncond_dir / "novelty_analysis"
    novelty_dir.mkdir(parents=True, exist_ok=True)

    denovo_rows = []
    pdb_rows = []

    for length in args.lengths:
        logger.info(f"Processing length {length}")

        # Get query structures: either existing cluster reps or all designs
        num_reps_queried = 0
        num_queries = 0
        query_reps_dir = None

        if args.use_existing_clusters:
            fs_results = uncond_dir / "foldseek_results" / f"length_{length}"
            fs_temp = uncond_dir / "foldseek_temp_dir" / f"length_{length}"
            rep_fasta = fs_results / "res_rep_seq.fasta"
            if rep_fasta.exists() and fs_temp.exists():
                rep_file = fasta.FastaFile.read(str(rep_fasta))
                rep_names = [k.strip() for k in rep_file.keys()]
                query_reps_dir = novelty_dir / f"query_reps_length_{length}"
                query_reps_dir.mkdir(parents=True, exist_ok=True)
                for name in rep_names:
                    src = fs_temp / f"{name}.pdb"
                    if src.exists():
                        (query_reps_dir / f"{name}.pdb").write_bytes(src.read_bytes())
                num_reps_queried = len(list(query_reps_dir.glob("*.pdb")))
                num_queries = len(list(fs_temp.glob("*.pdb")))
                logger.info(f"  Using {num_reps_queried} cluster reps from prior diversity (of {num_queries} total)")
            else:
                logger.warning(f"Existing clusters not found for length {length}, falling back to all designs")

        if query_reps_dir is None or num_reps_queried == 0:
            query_temp, num_queries = copy_structures_by_rmsd(
                uncond_dir, length, rmsd_threshold=args.rmsd_threshold
            )
            if query_temp is None or num_queries == 0:
                logger.warning(f"No query structures for length {length}, skipping")
                continue
            query_reps_dir = query_temp
            num_reps_queried = num_queries
            logger.info(f"  Using all {num_queries} designs as queries")

        # Denovo: length-specific reps
        denovo_ref = denovo_reps_dir / f"length_{length}" / "cluster_representatives"
        tsv = None
        if denovo_ref.exists() and list(denovo_ref.glob("*.pdb")):
            result_dir = novelty_dir / f"denovo_length_{length}"
            tsv = run_novelty_search(
                query_reps_dir, denovo_ref, result_dir, args.foldseek_bin
            )
        if tsv:
            m = compute_novelty_metrics(tsv)
            if m:
                m["length"] = length
                m["total_structures"] = num_queries
                m["cluster_reps_queried"] = num_reps_queried
                denovo_rows.append(m)
                logger.info(f"  Denovo: {m}")
        else:
            if not (denovo_ref.exists() and list(denovo_ref.glob("*.pdb"))):
                logger.warning(f"  Denovo reps not found at {denovo_ref}")

        # PDB/AFDB reps: all reps (same ref for all lengths)
        result_dir = novelty_dir / f"{args.ref_label}_length_{length}"
        tsv = run_novelty_search(query_reps_dir, pdb_reps_dir, result_dir, args.foldseek_bin)
        if tsv:
            m = compute_novelty_metrics(tsv)
            if m:
                m["length"] = length
                m["total_structures"] = num_queries
                m["cluster_reps_queried"] = num_reps_queried
                pdb_rows.append(m)
                logger.info(f"  PDB: {m}")

    # Write summary CSVs
    if denovo_rows:
        denovo_df = pd.DataFrame(denovo_rows)
        denovo_path = uncond_dir / "novelty_vs_denovo_summary.csv"
        denovo_df.to_csv(denovo_path, index=False)
        logger.info(f"Wrote {denovo_path}")
    if pdb_rows:
        pdb_df = pd.DataFrame(pdb_rows)
        pdb_path = uncond_dir / f"novelty_vs_{args.ref_label}_summary.csv"
        pdb_df.to_csv(pdb_path, index=False)
        logger.info(f"Wrote {pdb_path}")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
