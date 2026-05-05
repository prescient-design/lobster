#!/usr/bin/env python3
"""Evaluate competitor co-design models (LaProteina, DPLM2) for unconditional benchmarking.

Mirrors the GenUME unconditional evaluation pipeline exactly:
1. Subsample 100 random backbone PDBs per length from the full 8k generation set
2. Extract codesign sequences from backbone PDBs
3. Fold each sequence with ESMFold
4. Compute designability metrics (RMSD, TM-score, pLDDT)
5. Run Foldseek clustering on backbone PDBs
6. Run P-SEA secondary structure analysis on backbone PDBs
7. Run Foldseek novelty search against PDB + AFDB reference sets

Usage:
    cd /cv/home/lisanzas/lobster
    uv run python scripts/eval_competitor_unconditional.py \
        --model laproteina \
        --output-dir /cv/scratch/u/lisanzas/evaluations/benchmark_laproteina_unconditional
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

SEED = 42
LENGTHS = [100, 200, 300, 400, 500]
N_SAMPLES = 100

GEN_BASE = Path("/cv/scratch/u/lisanzas/denovo_dataset/generation")
PDB_REPS_DIR = Path("/cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb")
AFDB_REPS_DIR = Path("/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb")
FOLDSEEK_BIN = "/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin"

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def extract_sequence_and_ca(pdb_path: Path) -> tuple[str, np.ndarray] | None:
    """Extract one-letter sequence and CA coordinates from a PDB file."""
    residues: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(dict)
    res_names: dict[tuple[str, int], str] = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM ") or line.startswith("HETATM")):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resseq = int(line[22:26].strip() or 0)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            key = (chain, resseq)
            residues[key][atom] = [x, y, z]
            res_names[key] = resname

    seq_chars = []
    ca_coords = []
    for key in sorted(residues.keys()):
        r = residues[key]
        if "CA" not in r:
            continue
        aa = THREE_TO_ONE.get(res_names.get(key, ""), "X")
        if aa == "X":
            continue
        seq_chars.append(aa)
        ca_coords.append(r["CA"])

    if not seq_chars:
        return None
    return "".join(seq_chars), np.array(ca_coords, dtype=np.float32)


def extract_backbone_coords(pdb_path: Path) -> np.ndarray | None:
    """Extract N, CA, C backbone coordinates for P-SEA. Returns (L, 3, 3)."""
    residues: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(dict)
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM ") or line.startswith("HETATM")):
                continue
            atom = line[12:16].strip()
            if atom not in ("N", "CA", "C"):
                continue
            chain = line[21]
            resseq = int(line[22:26].strip() or 0)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            residues[(chain, resseq)][atom] = [x, y, z]

    coords = []
    for key in sorted(residues.keys()):
        r = residues[key]
        if "N" in r and "CA" in r and "C" in r:
            coords.append([r["N"], r["CA"], r["C"]])
    if not coords:
        return None
    return np.array(coords, dtype=np.float32)


def subsample_backbones(model: str, length: int) -> list[Path]:
    """Randomly pick N_SAMPLES backbone PDBs from the full generation set."""
    gen_dir = GEN_BASE / model / f"length_{length}" / "pdbs"
    all_pdbs = sorted(gen_dir.glob("*.pdb"))
    if not all_pdbs:
        logger.warning(f"No PDBs found in {gen_dir}")
        return []
    rng = random.Random(SEED + length)
    return rng.sample(all_pdbs, min(N_SAMPLES, len(all_pdbs)))


def fold_with_esmfold_single(seq: str, plm_fold, device: torch.device) -> dict:
    """Fold a single sequence with ESMFold. Returns result dict."""
    try:
        tokenized_input = plm_fold.tokenizer.encode_plus(
            seq,
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)

        with torch.no_grad():
            outputs = plm_fold.model(tokenized_input)

        pred_coords = outputs["positions"][-1][0, :len(seq), :3, :]  # (L, 3, 3) N/CA/C
        plddt = outputs["plddt"][0, :len(seq)].mean().item()
        pae = outputs["predicted_aligned_error"][0, :len(seq), :len(seq)].mean().item() if "predicted_aligned_error" in outputs else 0.0

        return {
            "pred_coords": pred_coords.cpu(),
            "plddt": plddt,
            "pae": pae,
            "success": True,
        }
    except Exception as e:
        logger.warning(f"ESMFold failed for seq len {len(seq)}: {e}")
        return {"success": False}


def compute_ca_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Kabsch-aligned CA RMSD between two (L, 3) arrays."""
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    c1_aligned = (R @ c1.T).T
    return float(np.sqrt(np.mean(np.sum((c1_aligned - c2) ** 2, axis=1))))


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray, seq: str) -> float:
    """TM-score between two CA coordinate arrays."""
    from tmtools import tm_align
    result = tm_align(coords1, coords2, seq, seq)
    return float(result.tm_norm_chain1)


def save_pdb_from_esmfold(coords: torch.Tensor, seq: str, path: Path):
    """Write a minimal PDB with N, CA, C atoms from ESMFold prediction."""
    ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
    atom_names = ["N", "CA", "C"]
    with open(path, "w") as f:
        atom_idx = 1
        for i, aa in enumerate(seq):
            resname = ONE_TO_THREE.get(aa, "UNK")
            for j, aname in enumerate(atom_names):
                x, y, z = coords[i, j].tolist()
                f.write(
                    f"ATOM  {atom_idx:5d}  {aname:<3s} {resname:>3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {aname[0]:>1s}  \n"
                )
                atom_idx += 1
        f.write("END\n")


def run_foldseek_clustering(pdb_dir: Path, output_dir: Path, tmscore_threshold: float = 0.5) -> int:
    """Run Foldseek easy-cluster on PDBs. Returns number of clusters."""
    from lobster.metrics.cal_foldseek_clusters import setup_foldseek_path
    setup_foldseek_path(FOLDSEEK_BIN)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "foldseek", "easy-cluster",
        str(pdb_dir), str(output_dir / "res"), str(output_dir),
        "--alignment-type", "1",
        "--cov-mode", "0",
        "--min-seq-id", "0",
        "--tmscore-threshold", str(tmscore_threshold),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error(f"Foldseek clustering failed: {p.stderr}")
        return 0

    rep_fasta = output_dir / "res_rep_seq.fasta"
    if rep_fasta.exists():
        return sum(1 for line in open(rep_fasta) if line.startswith(">"))
    return 0


def run_sse_analysis(pdb_dir: Path, output_path: Path, model_name: str):
    """Run P-SEA secondary structure analysis on PDBs and save parquet."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_secondary_structure import load_pdb_backbone, ss_content_from_coords, assign_ss_category

    import pandas as pd

    results = []
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    pdb_files = [f for f in pdb_files if "esmfold" not in f.name.lower()]

    for fp in tqdm(pdb_files, desc="SSE analysis"):
        coords = load_pdb_backbone(fp)
        if coords is None:
            continue

        # Parse length from filename
        length = 0
        parts = fp.stem.split("_")
        for i, p in enumerate(parts):
            if p == "length" and i + 1 < len(parts) and parts[i + 1].isdigit():
                length = int(parts[i + 1])
                break

        ss = ss_content_from_coords(coords)
        ss_cat = assign_ss_category(ss["H"], ss["E"])
        results.append({
            "structure_id": fp.stem,
            "source": model_name,
            "length": length,
            "helix": ss["H"],
            "strand": ss["E"],
            "coil": ss["C"],
            "ss_category": ss_cat,
            "sse_per_residue": ss.get("sse_per_residue", ""),
        })

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)
    logger.info(f"SSE analysis: {len(results)} structures -> {output_path}")
    return df


def run_novelty_search(query_dir: Path, ref_dir: Path, result_dir: Path) -> Path | None:
    """Run Foldseek search for novelty (query cluster reps vs reference)."""
    from lobster.metrics.cal_foldseek_clusters import setup_foldseek_path
    setup_foldseek_path(FOLDSEEK_BIN)

    result_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = result_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_db = result_dir / "query_db"
    ref_db = result_dir / "ref_db"
    align_db = result_dir / "align_db"

    for name, inp, db in [("query", query_dir, query_db), ("ref", ref_dir, ref_db)]:
        cmd = ["foldseek", "createdb", str(inp), str(db)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            logger.error(f"Foldseek createdb ({name}) failed: {p.stderr}")
            return None

    cmd = [
        "foldseek", "search", str(query_db), str(ref_db), str(align_db), str(tmp_dir),
        "--alignment-type", "1", "-a", "1",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error(f"Foldseek search failed: {p.stderr}")
        return None

    tsv_path = result_dir / "alignments.tsv"
    cmd = [
        "foldseek", "convertalis", str(query_db), str(ref_db), str(align_db), str(tsv_path),
        "--format-output", "query,target,alntmscore",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error(f"Foldseek convertalis failed: {p.stderr}")
        return None

    return tsv_path


def compute_novelty_metrics(tsv_path: Path) -> dict | None:
    """Parse alignment TSV, compute per-query max TM, then aggregate."""
    import pandas as pd
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
    parser = argparse.ArgumentParser(description="Evaluate competitor co-design models")
    parser.add_argument("--model", required=True, choices=["laproteina", "dplm2"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--skip-esmfold", action="store_true", help="Skip ESMFold (use if already done)")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-sse", action="store_true")
    parser.add_argument("--skip-novelty", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import pandas as pd

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Step 1: Subsample backbone PDBs ──
    logger.info(f"=== Subsampling {args.n_samples} backbone PDBs per length for {args.model} ===")
    all_metrics = []

    for length in args.lengths:
        logger.info(f"\n--- Length {length} ---")
        backbone_pdbs = subsample_backbones(args.model, length)
        if not backbone_pdbs:
            continue

        # Copy backbone PDBs to output dir with standardized names
        backbone_dir = output_dir / "backbone_pdbs" / f"length_{length}"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(backbone_pdbs):
            dst = backbone_dir / f"generated_structure_length_{length}_{i:03d}.pdb"
            if not dst.exists():
                shutil.copy2(src, dst)
            # Also copy to output_dir root (for compatibility with existing scripts)
            root_dst = output_dir / f"generated_structure_length_{length}_{i:03d}.pdb"
            if not root_dst.exists():
                shutil.copy2(src, root_dst)

        logger.info(f"  Copied {len(backbone_pdbs)} backbone PDBs")

    # ── Step 2: ESMFold validation ──
    if not args.skip_esmfold:
        logger.info("\n=== Loading ESMFold ===")
        from lobster.model import LobsterPLMFold
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=512)
        plm_fold.to(device)

        for length in args.lengths:
            logger.info(f"\n--- ESMFold for length {length} ---")
            backbone_dir = output_dir / "backbone_pdbs" / f"length_{length}"
            pdb_files = sorted(backbone_dir.glob("*.pdb"))

            for pdb_path in tqdm(pdb_files, desc=f"L{length}"):
                esmfold_path = output_dir / f"{pdb_path.stem}_esmfold_000.pdb"
                if esmfold_path.exists():
                    continue

                result = extract_sequence_and_ca(pdb_path)
                if result is None:
                    logger.warning(f"  Failed to extract seq from {pdb_path.name}")
                    continue
                seq, ref_ca = result

                fr = fold_with_esmfold_single(seq, plm_fold, device)
                if not fr["success"]:
                    all_metrics.append({
                        "structure_id": pdb_path.stem,
                        "length": length,
                        "plddt": 0.0, "pae": 0.0, "tm_score": 0.0, "rmsd": 99.0,
                        "pass": False,
                    })
                    continue

                pred_ca = fr["pred_coords"][:, 1, :].numpy()  # CA atoms

                min_len = min(len(ref_ca), len(pred_ca))
                ref_ca_trim = ref_ca[:min_len]
                pred_ca_trim = pred_ca[:min_len]

                rmsd = compute_ca_rmsd(ref_ca_trim, pred_ca_trim)
                tm = compute_tm_score(ref_ca_trim, pred_ca_trim, seq[:min_len])

                save_pdb_from_esmfold(fr["pred_coords"][:len(seq)], seq, esmfold_path)

                all_metrics.append({
                    "structure_id": pdb_path.stem,
                    "length": length,
                    "plddt": fr["plddt"],
                    "pae": fr["pae"],
                    "tm_score": tm,
                    "rmsd": rmsd,
                    "pass": rmsd < 2.0,
                })

            # Save metrics incrementally after each length
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                metrics_path = output_dir / f"unconditional_metrics_{args.model}.csv"
                metrics_df.to_csv(metrics_path, index=False)

        # Free GPU memory
        del plm_fold
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Save metrics CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = output_dir / f"unconditional_metrics_{args.model}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics: {metrics_path}")

        # Per-length summary
        for length in args.lengths:
            ldf = metrics_df[metrics_df["length"] == length]
            if ldf.empty:
                continue
            logger.info(
                f"  L{length}: N={len(ldf)}, pass={ldf['pass'].mean()*100:.0f}%, "
                f"pLDDT={ldf['plddt'].mean():.3f}, TM={ldf['tm_score'].mean():.3f}, "
                f"RMSD={ldf['rmsd'].mean():.2f}"
            )

    # ── Step 3: Foldseek clustering (only passing designs, RMSD < 2) ──
    if not args.skip_clustering:
        logger.info("\n=== Foldseek clustering (passing designs only) ===")
        metrics_path = output_dir / f"unconditional_metrics_{args.model}.csv"
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path)
        else:
            mdf = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()

        total_clusters = 0
        for length in args.lengths:
            backbone_dir = output_dir / "backbone_pdbs" / f"length_{length}"
            cluster_dir = output_dir / "foldseek_results" / f"length_{length}"

            passing_ids = set()
            if not mdf.empty:
                ldf = mdf[(mdf["length"] == length) & (mdf["pass"] == True)]
                passing_ids = set(ldf["structure_id"].tolist())

            if not passing_ids:
                logger.warning(f"  L{length}: 0 passing designs, skipping clustering")
                continue

            temp_dir = output_dir / "foldseek_temp_dir" / f"length_{length}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for sid in passing_ids:
                src = backbone_dir / f"{sid}.pdb"
                if src.exists():
                    shutil.copy2(src, temp_dir / f"{sid}.pdb")

            n_copied = len(list(temp_dir.glob("*.pdb")))
            logger.info(f"  L{length}: {n_copied} passing designs (of {len(list(backbone_dir.glob('*.pdb')))} total)")

            n_clusters = run_foldseek_clustering(temp_dir, cluster_dir)
            total_clusters += n_clusters
            logger.info(f"  L{length}: {n_clusters} clusters")
        logger.info(f"  Total clusters: {total_clusters}")

    # ── Step 4: SSE analysis ──
    if not args.skip_sse:
        logger.info("\n=== SSE analysis ===")
        run_sse_analysis(output_dir, output_dir / "uncond_sse_index.parquet", args.model)

    # ── Step 5: Novelty analysis ──
    if not args.skip_novelty:
        logger.info("\n=== Novelty analysis ===")
        import pandas as pd
        from biotite.sequence.io import fasta as fasta_io

        novelty_dir = output_dir / "novelty_analysis"
        novelty_dir.mkdir(parents=True, exist_ok=True)

        pdb_rows = []
        afdb_rows = []

        for length in args.lengths:
            # Get cluster reps as query
            fs_results = output_dir / "foldseek_results" / f"length_{length}"
            rep_fasta = fs_results / "res_rep_seq.fasta"
            backbone_dir = output_dir / "backbone_pdbs" / f"length_{length}"

            if not rep_fasta.exists():
                logger.warning(f"  No cluster reps for L{length}, skipping novelty")
                continue

            rep_file = fasta_io.FastaFile.read(str(rep_fasta))
            rep_names = [k.strip() for k in rep_file.keys()]

            query_dir = novelty_dir / f"query_reps_length_{length}"
            query_dir.mkdir(parents=True, exist_ok=True)
            for name in rep_names:
                src = backbone_dir / f"{name}.pdb"
                if src.exists():
                    dst = query_dir / f"{name}.pdb"
                    if not dst.exists():
                        dst.write_bytes(src.read_bytes())

            num_reps = len(list(query_dir.glob("*.pdb")))
            num_total = len(list(backbone_dir.glob("*.pdb")))
            logger.info(f"  L{length}: {num_reps} cluster reps of {num_total} total")

            # vs PDB
            if PDB_REPS_DIR.exists() and list(PDB_REPS_DIR.glob("*.pdb")):
                result_dir = novelty_dir / f"pdb_length_{length}"
                tsv = run_novelty_search(query_dir, PDB_REPS_DIR, result_dir)
                if tsv:
                    m = compute_novelty_metrics(tsv)
                    if m:
                        m["length"] = length
                        m["total_structures"] = num_total
                        m["cluster_reps_queried"] = num_reps
                        pdb_rows.append(m)
                        logger.info(f"    PDB: mean_max_TM={m['mean_max_tmscore']:.3f}")

            # vs AFDB
            if AFDB_REPS_DIR.exists() and list(AFDB_REPS_DIR.glob("*.pdb")):
                result_dir = novelty_dir / f"afdb_length_{length}"
                tsv = run_novelty_search(query_dir, AFDB_REPS_DIR, result_dir)
                if tsv:
                    m = compute_novelty_metrics(tsv)
                    if m:
                        m["length"] = length
                        m["total_structures"] = num_total
                        m["cluster_reps_queried"] = num_reps
                        afdb_rows.append(m)
                        logger.info(f"    AFDB: mean_max_TM={m['mean_max_tmscore']:.3f}")

        if pdb_rows:
            pdb_df = pd.DataFrame(pdb_rows)
            pdb_path = output_dir / "novelty_vs_pdb_summary.csv"
            pdb_df.to_csv(pdb_path, index=False)
            logger.info(f"Wrote {pdb_path}")
        if afdb_rows:
            afdb_df = pd.DataFrame(afdb_rows)
            afdb_path = output_dir / "novelty_vs_afdb_summary.csv"
            afdb_df.to_csv(afdb_path, index=False)
            logger.info(f"Wrote {afdb_path}")

    logger.info("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
