"""Run RF3 co-folding on GT sequences + ligand SMILES from FF eval.

Only runs RF3 and saves structures + confidence metrics. No lobster dependency.
Contact analysis is done separately.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from glob import glob

import numpy as np
import pandas as pd
from loguru import logger

PROTEINA_DIR = "/cv/scratch/u/lisanzas/proteina-complexa"
RF3_CKPT = f"{PROTEINA_DIR}/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt"
RF3_BIN = f"{PROTEINA_DIR}/.venv/bin/rf3"


def run_rf3(name: str, sequence: str, smiles: str, work_dir: str) -> dict:
    """Run RF3 co-folding, save CIF, return confidence metrics."""
    rf3_work = os.path.join(work_dir, f"_rf3_{name}")
    os.makedirs(rf3_work, exist_ok=True)

    inp_json = os.path.join(rf3_work, "input.json")
    with open(inp_json, "w") as f:
        json.dump({"name": name, "components": [{"seq": sequence}, {"smiles": smiles}]}, f)

    try:
        result = subprocess.run(
            [RF3_BIN, "fold", f"inputs={inp_json}", f"ckpt_path={RF3_CKPT}",
             f"out_dir={rf3_work}", "early_stopping_plddt_threshold=0"],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(rf3_work, ignore_errors=True)
        return {"error": "timeout"}

    if result.returncode != 0:
        shutil.rmtree(rf3_work, ignore_errors=True)
        return {"error": f"rc{result.returncode}"}

    metrics = {}
    conf_files = glob(os.path.join(rf3_work, "**/*_summary_confidences.json"), recursive=True)
    if conf_files:
        with open(conf_files[0]) as f:
            conf = json.load(f)
        if isinstance(conf, list):
            conf = conf[0]
        metrics["rf3_plddt"] = float(conf.get("overall_plddt", 0))
        metrics["rf3_pTM"] = float(conf.get("ptm", 0))
        metrics["rf3_ipTM"] = float(conf.get("iptm", 0))
        metrics["rf3_ranking_score"] = float(conf.get("ranking_score", 0))

        chain_pair_pae_min = conf.get("chain_pair_pae_min", [])
        if chain_pair_pae_min and len(chain_pair_pae_min) > 1:
            arr = np.array(chain_pair_pae_min, dtype=float)
            last_col = arr[:-1, -1]
            metrics["rf3_min_ipAE"] = float(np.nanmin(last_col)) / 31.0 if not np.all(np.isnan(last_col)) else 100.0 / 31.0
        else:
            metrics["rf3_min_ipAE"] = 100.0 / 31.0

    # Save CIF to output dir
    for cif in glob(os.path.join(rf3_work, "**/*_model.cif"), recursive=True):
        shutil.copy2(cif, os.path.join(work_dir, f"{name}_rf3.cif"))
        break

    # Try CIF→PDB conversion
    try:
        from proteinfoundation.utils.rf3_utils import convert_cif_to_pdb_rf3
        for cif in glob(os.path.join(rf3_work, "**/*_model.cif"), recursive=True):
            pdb = convert_cif_to_pdb_rf3(cif)
            shutil.copy2(pdb, os.path.join(work_dir, f"{name}_rf3.pdb"))
            break
    except Exception:
        pass

    shutil.rmtree(rf3_work, ignore_errors=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff_csv", required=True, help="FF results CSV with pdb_id, sequence, smiles")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.ff_csv)
    end = args.end_idx or len(df)
    df = df.iloc[args.start_idx:end]
    logger.info(f"Running RF3 on {len(df)} samples (idx {args.start_idx}-{end})")

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        pdb_id = row["pdb_id"]
        seq = row["sequence"]
        smiles = row["smiles"]

        if not seq or not smiles or pd.isna(seq) or pd.isna(smiles):
            logger.warning(f"Skipping {pdb_id}: missing sequence or SMILES")
            results.append({"pdb_id": pdb_id, "error": "missing_data"})
            continue

        logger.info(f"[{i+1}/{len(df)}] RF3: {pdb_id} (len={len(seq)})")
        t0 = time.time()
        rf3_metrics = run_rf3(pdb_id, seq, smiles, args.output_dir)
        rf3_metrics["pdb_id"] = pdb_id
        rf3_metrics["sequence"] = seq
        rf3_metrics["smiles"] = smiles
        rf3_metrics["rf3_time_sec"] = time.time() - t0
        results.append(rf3_metrics)

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(args.output_dir, "rf3_results.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(out_df)} results to {out_csv}")


if __name__ == "__main__":
    main()
