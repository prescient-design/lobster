#!/usr/bin/env python3
"""Benchmark ligand-conditioned protein generation: Gen-UME vs Proteina-Complexa.

Head-to-head comparison on the same set of ligands. For each ligand, generates
N protein binders with each model, then scores all designs with RF3 (RosettaFold3)
for independent structural validation.

Architecture:
  - One SLURM job per (model, ligand) pair
  - Each job: generate N designs → run RF3 on each → save metrics + structures
  - After all jobs complete: merge results and produce comparison table

Usage:
    # Submit all jobs (30 ligands × 2 models = 60 SLURM jobs)
    python scripts/benchmark_conditioned_gen.py submit \
        --checkpoint /path/to/genume.ckpt \
        --num_designs 5 \
        --output_dir /scratch/benchmark_results

    # After jobs complete, merge and compare
    python scripts/benchmark_conditioned_gen.py merge \
        --output_dir /scratch/benchmark_results

    # Run a single ligand locally (for debugging)
    python scripts/benchmark_conditioned_gen.py run_genume \
        --checkpoint /path/to/genume.ckpt \
        --ligand_id 7DUA_HJ0 --num_designs 5 \
        --output_dir /scratch/benchmark_results/genume/7DUA_HJ0

    python scripts/benchmark_conditioned_gen.py run_proteina \
        --ligand_id 7DUA_HJ0 --num_designs 5 \
        --output_dir /scratch/benchmark_results/proteina/7DUA_HJ0
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

LOBSTER_DIR = "/cv/home/lisanzas/lobster"
PROTEINA_DIR = "/cv/scratch/u/lisanzas/proteina-complexa"
DEFAULT_DATA_DIR = f"{LOBSTER_DIR}/data/proteina_ligand_targets/processed/"
DEFAULT_RAW_DATA_DIR = f"{LOBSTER_DIR}/data/posebusters/posebusters_benchmark_set/"
RF3_CKPT = f"{PROTEINA_DIR}/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt"
RF3_BIN = f"{PROTEINA_DIR}/.venv/bin/rf3"
BOLTZ_VENV = "/cv/scratch/u/lisanzas/uv_envs/boltz/.venv"

# Gen-UME generation hyperparameters (best from sweep: low_temp_more_steps)
GENUME_PARAMS = {
    "temperature_seq": 0.153,
    "temperature_struc": 0.05,
    "temperature_ligand": 0.1,
    "stochasticity_seq": 20,
    "stochasticity_struc": 20,
    "stochasticity_ligand": 5,
    "nsteps": 200,
    "length": 100,
}

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_ligand_ids(data_dir: str = None):
    """Get sorted list of ligand IDs from the benchmark data directory."""
    d = data_dir or DEFAULT_DATA_DIR
    return sorted(f.stem.replace("_ligand", "") for f in Path(d).glob("*_ligand.pt"))


def get_smiles(ligand_id: str, data_dir: str = None, raw_data_dir: str = None) -> str:
    """Get canonical SMILES for a ligand.

    Checks (in order):
    1. SMILES stored in the ligand .pt file
    2. SDF file in raw_data_dir
    """
    import torch

    # Try .pt file first
    d = data_dir or DEFAULT_DATA_DIR
    pt_path = os.path.join(d, f"{ligand_id}_ligand.pt")
    if os.path.exists(pt_path):
        ligand_data = torch.load(pt_path, weights_only=False)
        if isinstance(ligand_data, dict) and "smiles" in ligand_data:
            return ligand_data["smiles"]

    # Fall back to SDF
    raw = raw_data_dir or DEFAULT_RAW_DATA_DIR
    sdf_path = os.path.join(raw, ligand_id, f"{ligand_id}_ligand.sdf")
    if os.path.exists(sdf_path):
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = next(iter(Chem.SDMolSupplier(sdf_path, removeHs=True)), None)
        return Chem.MolToSmiles(mol, canonical=True) if mol else ""

    return ""


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract single-letter protein sequence from PDB ATOM records (CA atoms)."""
    seq, last_res = [], None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = line[22:26].strip()
                if resnum != last_res:
                    seq.append(AA3TO1.get(line[17:20].strip(), "X"))
                    last_res = resnum
    return "".join(seq)


# ============================================================================
# RF3 prediction (shared by both models)
# ============================================================================

def run_rf3(name: str, sequence: str, smiles: str, work_dir: str) -> dict:
    """Run RF3 (RosettaFold3) co-folding on a (sequence, SMILES) pair.

    RF3 independently predicts the protein-ligand complex structure from
    sequence + SMILES. This provides an unbiased assessment of whether the
    designed protein would actually fold and bind the target ligand.

    The function creates a temporary RF3 work directory, runs prediction,
    extracts confidence metrics, copies the predicted PDB to work_dir,
    and cleans up the temporary files.

    Parameters
    ----------
    name : str
        Unique identifier for this design (used for file naming).
    sequence : str
        Protein sequence in single-letter amino acid codes.
    smiles : str
        Ligand SMILES string.
    work_dir : str
        Output directory. Predicted structure saved as {work_dir}/{name}_rf3.pdb.

    Returns
    -------
    dict
        On success: rf3_ipTM (interface pTM, 0-1, higher=better interaction),
        rf3_pTM (protein pTM), rf3_plddt (predicted LDDT), rf3_min_ipAE
        (min interface PAE, normalized by /31), rf3_ranking_score.
        On failure: {"error": reason_string}.
    """
    rf3_work = os.path.join(work_dir, f"_rf3_{name}")
    os.makedirs(rf3_work, exist_ok=True)

    inp_json = os.path.join(rf3_work, "input.json")
    with open(inp_json, "w") as f:
        json.dump({"name": name, "components": [{"seq": sequence, "chain_id": "B"}, {"smiles": smiles}]}, f)

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

    # Parse confidence metrics
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

    # Copy predicted structure to output dir
    for cif in glob(os.path.join(rf3_work, "**/*_model.cif"), recursive=True):
        shutil.copy2(cif, os.path.join(work_dir, f"{name}_rf3.cif"))
        # Try CIF→PDB conversion (requires proteina venv)
        try:
            from proteinfoundation.utils.rf3_utils import convert_cif_to_pdb_rf3
            pdb = convert_cif_to_pdb_rf3(cif)
            shutil.copy2(pdb, os.path.join(work_dir, f"{name}_rf3.pdb"))
        except Exception:
            pass  # CIF still saved
        break

    shutil.rmtree(rf3_work, ignore_errors=True)
    return metrics


def run_boltz(name: str, sequence: str, smiles: str, work_dir: str) -> dict:
    """Run Boltz2 co-folding on a (sequence, SMILES) pair.

    Parameters
    ----------
    name : str
        Unique identifier for this design.
    sequence : str
        Protein sequence.
    smiles : str
        Ligand SMILES string.
    work_dir : str
        Output directory.

    Returns
    -------
    dict
        Confidence metrics (cofold_ipTM, cofold_pTM, cofold_pLDDT, etc.)
        or {"error": reason}.
    """
    boltz_work = os.path.join(work_dir, f"_boltz_{name}")
    os.makedirs(boltz_work, exist_ok=True)

    inp_json = os.path.join(boltz_work, "input.json")
    out_json = os.path.join(boltz_work, "output.json")
    with open(inp_json, "w") as f:
        json.dump({"id": name, "sequence": sequence, "smiles": smiles}, f)

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "lobster.cmdline.run_cofold_local",
             "--input_json", inp_json, "--output_json", out_json,
             "--backend", "boltz", "--work_dir", boltz_work],
            capture_output=True, text=True, timeout=600,
            cwd=LOBSTER_DIR,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(boltz_work, ignore_errors=True)
        return {"error": "timeout"}

    if result.returncode != 0:
        shutil.rmtree(boltz_work, ignore_errors=True)
        return {"error": f"rc{result.returncode}"}

    if not os.path.exists(out_json):
        shutil.rmtree(boltz_work, ignore_errors=True)
        return {"error": "no_output"}

    with open(out_json) as f:
        out = json.load(f)

    if out.get("error"):
        shutil.rmtree(boltz_work, ignore_errors=True)
        return {"error": out["error"]}

    metrics = {}
    conf = out.get("confidence", {})
    metrics["cofold_ipTM"] = float(conf.get("iptm", 0))
    metrics["cofold_pTM"] = float(conf.get("ptm", 0))
    metrics["cofold_pLDDT"] = float(conf.get("plddt", conf.get("complex_plddt", 0)))
    metrics["cofold_ranking_score"] = float(conf.get("confidence_score", conf.get("ranking_score", 0)))
    metrics["cofold_iPDE"] = float(conf.get("complex_ipde", conf.get("ipde", float("nan"))))

    # Save predicted structure (CIF/PDB) to output dir before cleanup
    for pattern in ["**/*_model.cif", "**/*.cif", "**/*_model.pdb", "**/*.pdb"]:
        found = glob(os.path.join(boltz_work, pattern), recursive=True)
        if found:
            ext = os.path.splitext(found[0])[1]
            shutil.copy2(found[0], os.path.join(work_dir, f"{name}_boltz{ext}"))
            break

    shutil.rmtree(boltz_work, ignore_errors=True)
    return metrics


def run_structure_prediction(name: str, sequence: str, smiles: str, work_dir: str,
                             backend: str = "boltz") -> dict:
    """Run structure prediction (RF3 or Boltz2) on a (sequence, SMILES) pair."""
    if backend == "rf3":
        return run_rf3(name, sequence, smiles, work_dir)
    elif backend == "boltz":
        return run_boltz(name, sequence, smiles, work_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Gen-UME: generate + ESMFold + RF3
# ============================================================================

def run_genume(checkpoint: str, ligand_id: str, num_designs: int, output_dir: str,
               data_dir: str = None, backend: str = "boltz"):
    """Generate proteins with Gen-UME, validate with structure prediction.

    Pipeline per ligand:
      1. Load Gen-UME checkpoint
      2. Generate num_designs proteins via discrete flow matching, conditioned
         on ligand atom types + bond matrix (atom_bond_only mode, no 3D coords)
      3. Decode each design to 3D coordinates via the FSQ tokenizer decoder
      4. Run co-folding (Boltz2 or RF3) on (sequence, SMILES) for validation
      5. Save all structures and metrics

    Parameters
    ----------
    checkpoint : str
        Path to Gen-UME ProteinLigandEncoderLightningModule checkpoint.
    ligand_id : str
        Ligand identifier (e.g. '7DUA_HJ0' or '7V11_OQO').
    num_designs : int
        Number of independent designs to generate for this ligand.
    output_dir : str
        Output directory for structures and results.csv.
    data_dir : str, optional
        Data directory with *_ligand.pt files. Default: proteina_ligand_targets.
    backend : str
        Structure prediction backend: 'boltz' or 'rf3'. Default: 'boltz'.
    """
    import pandas as pd
    import torch
    from loguru import logger

    d = data_dir or DEFAULT_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    smiles = get_smiles(ligand_id, data_dir=d)
    if not smiles:
        logger.error(f"No SMILES for {ligand_id}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from lobster.model.gen_ume import ProteinLigandEncoderLightningModule
    from lobster.metrics.ligand_conditioned_protein_generation import LigandConditionedProteinGenerationEvaluator

    logger.info(f"Loading Gen-UME from {checkpoint}")
    model = ProteinLigandEncoderLightningModule.load_from_checkpoint(checkpoint, map_location=device)
    model.eval().to(device)

    evaluator = LigandConditionedProteinGenerationEvaluator(
        data_dir=d, length=GENUME_PARAMS["length"],
        pocket_distance_threshold=5.0, num_samples=None, num_designs=num_designs,
        nsteps=GENUME_PARAMS["nsteps"], device=device, max_length=512,
        temperature_seq=GENUME_PARAMS["temperature_seq"],
        temperature_struc=GENUME_PARAMS["temperature_struc"],
        stochasticity_seq=GENUME_PARAMS["stochasticity_seq"],
        stochasticity_struc=GENUME_PARAMS["stochasticity_struc"],
        temperature_ligand=GENUME_PARAMS["temperature_ligand"],
        stochasticity_ligand=GENUME_PARAMS["stochasticity_ligand"],
        ligand_context_mode="atom_bond_only",
        save_structures=True, minimize_ligand=True, plm_fold=None,
    )

    samples = [s for s in evaluator.load_test_set() if s["ligand_id"] == ligand_id]
    if not samples:
        logger.error(f"Ligand {ligand_id} not found")
        return

    logger.info(f"Generating {num_designs} designs for {ligand_id}")
    t_gen_start = time.time()
    results_df = evaluator.evaluate(model, samples, structure_path=output_dir)["results_df"]
    total_gen_time = time.time() - t_gen_start
    per_design_gen_time = total_gen_time / max(len(results_df), 1)
    logger.info(f"Generation took {total_gen_time:.1f}s total, {per_design_gen_time:.1f}s/design")

    # Structure prediction scoring
    logger.info(f"Running {backend} on {len(results_df)} designs")
    sp_rows = []
    for _, row in results_df.iterrows():
        name = f"{ligand_id}_d{int(row['design_idx'])}"
        seq = row.get("sequence", "")
        if not seq or not isinstance(seq, str):
            sp_rows.append({"error": "no_sequence"})
            continue
        logger.info(f"  {backend}: {name}")
        sp_rows.append(run_structure_prediction(name, seq, smiles, output_dir, backend=backend))

    combined = pd.concat([results_df.reset_index(drop=True), pd.DataFrame(sp_rows)], axis=1)
    combined["gen_time_sec"] = per_design_gen_time
    combined["model"] = "genume"
    combined["smiles"] = smiles
    combined.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    logger.info(f"Saved {len(combined)} results to {output_dir}/results.csv")


# ============================================================================
# Gen-UME filtered: generate many → ESMFold filter → RF3 top-K
# ============================================================================

def run_genume_filtered(checkpoint: str, ligand_id: str, num_designs: int,
                        rf3_top_k: int, output_dir: str):
    """Generate many designs, filter by ESMFold PAE, RF3 only top-K.

    Pipeline per ligand:
      1. Generate num_designs proteins (with ESMFold self-consistency)
      2. Rank by ESMFold PAE (ascending — lower is better)
      3. Run RF3 only on the top rf3_top_k designs
      4. Save all designs with ESMFold metrics; RF3 scores for top-K only

    Parameters
    ----------
    checkpoint : str
        Path to Gen-UME checkpoint.
    ligand_id : str
        PoseBusters ligand identifier.
    num_designs : int
        Total designs to generate (all scored by ESMFold).
    rf3_top_k : int
        Number of top designs (lowest ESMFold PAE) to send to RF3.
    output_dir : str
        Output directory.
    """
    import pandas as pd
    import torch
    from loguru import logger

    os.makedirs(output_dir, exist_ok=True)
    smiles = get_smiles(ligand_id)
    if not smiles:
        logger.error(f"No SMILES for {ligand_id}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from lobster.model.gen_ume import ProteinLigandEncoderLightningModule
    from lobster.metrics.ligand_conditioned_protein_generation import LigandConditionedProteinGenerationEvaluator
    from lobster.model import LobsterPLMFold

    logger.info(f"Loading Gen-UME from {checkpoint}")
    model = ProteinLigandEncoderLightningModule.load_from_checkpoint(checkpoint, map_location=device)
    model.eval().to(device)

    logger.info("Loading ESMFold for filtering...")
    plm_fold = LobsterPLMFold("esmfold_v1")
    plm_fold.eval().to(device)

    evaluator = LigandConditionedProteinGenerationEvaluator(
        data_dir=DATA_DIR, length=GENUME_PARAMS["length"],
        pocket_distance_threshold=5.0, num_samples=None, num_designs=num_designs,
        nsteps=GENUME_PARAMS["nsteps"], device=device, max_length=512,
        temperature_seq=GENUME_PARAMS["temperature_seq"],
        temperature_struc=GENUME_PARAMS["temperature_struc"],
        stochasticity_seq=GENUME_PARAMS["stochasticity_seq"],
        stochasticity_struc=GENUME_PARAMS["stochasticity_struc"],
        temperature_ligand=GENUME_PARAMS["temperature_ligand"],
        stochasticity_ligand=GENUME_PARAMS["stochasticity_ligand"],
        ligand_context_mode="atom_bond_only",
        save_structures=True, minimize_ligand=True, plm_fold=plm_fold,
    )

    samples = [s for s in evaluator.load_test_set() if s["ligand_id"] == ligand_id]
    if not samples:
        logger.error(f"Ligand {ligand_id} not found")
        return

    logger.info(f"Generating {num_designs} designs for {ligand_id} (with ESMFold)")
    t_gen_start = time.time()
    results_df = evaluator.evaluate(model, samples, structure_path=output_dir)["results_df"]
    gen_esm_time = time.time() - t_gen_start
    per_design_time = gen_esm_time / max(len(results_df), 1)
    logger.info(f"Generation + ESMFold took {gen_esm_time:.1f}s total, {per_design_time:.1f}s/design")

    # Rank by ESMFold PAE (lower is better) and select top-K for RF3
    results_df["pae"] = pd.to_numeric(results_df["pae"], errors="coerce")
    results_df = results_df.sort_values("pae", ascending=True, na_position="last")
    top_k_designs = results_df.head(rf3_top_k)

    n_low_pae = (results_df["pae"].dropna() < 5).sum()
    logger.info(f"ESMFold filter: {n_low_pae}/{len(results_df)} designs with PAE<5, "
                f"sending top {rf3_top_k} to RF3")
    logger.info(f"Top-{rf3_top_k} PAE values: {top_k_designs['pae'].tolist()}")

    # RF3 scoring on top-K only
    rf3_rows = {}
    for idx, row in top_k_designs.iterrows():
        name = f"{ligand_id}_d{int(row['design_idx'])}"
        seq = row.get("sequence", "")
        if not seq or not isinstance(seq, str):
            rf3_rows[idx] = {"rf3_error": "no_sequence"}
            continue
        logger.info(f"  RF3: {name} (PAE={row['pae']:.2f}, pLDDT={row.get('plddt', float('nan')):.3f})")
        rf3_rows[idx] = run_rf3(name, seq, smiles, output_dir)

    # Merge RF3 into full dataframe (non-RF3 designs get NaN for RF3 columns)
    rf3_df = pd.DataFrame.from_dict(rf3_rows, orient="index")
    combined = results_df.join(rf3_df)
    combined["gen_time_sec"] = per_design_time
    combined["model"] = "genume_filtered"
    combined["smiles"] = smiles
    combined["rf3_submitted"] = combined.index.isin(rf3_rows.keys())
    combined.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    logger.info(f"Saved {len(combined)} results ({len(rf3_rows)} with RF3) to {output_dir}/results.csv")


# ============================================================================
# Proteina-Complexa: generate + RF3
# ============================================================================

def run_proteina(ligand_id: str, num_designs: int, output_dir: str, backend: str = "boltz",
                 seed_offset: int = 0):
    """Generate binders with Proteina-Complexa, score with structure prediction.

    Pipeline per ligand:
      1. For each design, call `complexa generate` in single-pass mode
         with nsamples=1 (exactly 1 binder per call)
      2. Extract protein sequence from the generated binder PDB
      3. Run RF3 co-folding on (sequence, SMILES) → compute ipTM, pTM
      4. Save binder PDB, RF3 PDB, and metrics

    Proteina-Complexa generates binders using a ligand-conditioned partially
    latent flow matching model. The target ligand is defined in the main
    target YAML (ligand_targets_dict.yaml) as task PB_{ligand_id}, which
    must be populated by running prepare_proteina_posebusters.py first.

    Requires the proteina-complexa venv to be activated (for the complexa CLI
    and RF3 binary). The SLURM submission handles this automatically.

    Parameters
    ----------
    ligand_id : str
        PoseBusters ligand identifier (e.g. '7DUA_HJ0').
    num_designs : int
        Number of independent designs to generate for this ligand.
    output_dir : str
        Output directory for structures and results.csv.
    """
    import pandas as pd
    from loguru import logger

    os.makedirs(output_dir, exist_ok=True)
    smiles = get_smiles(ligand_id)
    if not smiles:
        logger.error(f"No SMILES for {ligand_id}")
        return

    task_name = f"PB_{ligand_id}"
    results = []

    for d in range(seed_offset, seed_offset + num_designs):
        run_name = f"bench_{ligand_id}_d{d}"
        logger.info(f"Proteina design {d - seed_offset + 1}/{num_designs}: {run_name}")

        t_gen_start = time.time()
        gen_result = subprocess.run(
            [f"{PROTEINA_DIR}/.venv/bin/complexa", "generate",
             f"configs/search_ligand_binder_local_pipeline.yaml",
             f"++run_name={run_name}", f"++generation.task_name={task_name}",
             f"++ckpt_path={PROTEINA_DIR}/ckpts", "++ckpt_name=complexa_ligand.ckpt",
             f"++autoencoder_ckpt_path={PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt",
             "++generation.search.algorithm=single-pass",
             "++generation.dataloader.dataset.nres.nsamples=1",
             f"++seed={d}"],  # Different seed per design for diversity
            capture_output=True, text=True, timeout=600, cwd=PROTEINA_DIR,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )

        gen_time = time.time() - t_gen_start

        if gen_result.returncode != 0:
            logger.warning(f"  Generation failed (rc={gen_result.returncode}): {gen_result.stderr[-500:] if gen_result.stderr else 'no stderr'}")
            results.append({"ligand_id": ligand_id, "design_idx": d, "error": "gen_failed", "model": "proteina", "gen_time_sec": gen_time})
            continue

        # Find generated binder PDB
        pattern = f"{PROTEINA_DIR}/inference/search_ligand_binder_local_pipeline_{task_name}_{run_name}"
        binder_pdbs = glob(f"{pattern}/*/*_binder.pdb")
        if not binder_pdbs:
            logger.warning(f"  No binder PDB found")
            results.append({"ligand_id": ligand_id, "design_idx": d, "error": "no_pdb", "model": "proteina"})
            continue

        binder_pdb = binder_pdbs[0]
        seq_str = extract_sequence_from_pdb(binder_pdb)

        # Copy binder PDB to output
        shutil.copy2(binder_pdb, os.path.join(output_dir, f"{ligand_id}_d{d}_proteina.pdb"))

        # RF3 scoring
        name = f"{ligand_id}_d{d}"
        logger.info(f"  {backend}: {name} (len={len(seq_str)})")
        sp_metrics = run_structure_prediction(name, seq_str, smiles, output_dir, backend=backend)

        logger.info(f"  Generation took {gen_time:.1f}s")

        row = {"ligand_id": ligand_id, "design_idx": d, "sequence": seq_str,
               "smiles": smiles, "protein_length": len(seq_str), "model": "proteina",
               "gen_time_sec": gen_time}
        row.update(sp_metrics)
        results.append(row)

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "results.csv"), index=False)
    logger.info(f"Saved {len(results)} results to {output_dir}/results.csv")


# ============================================================================
# SLURM submission
# ============================================================================

def submit_all(checkpoint: str, num_designs: int, output_dir: str, num_ligands: int = None,
               genume_designs: int | None = None):
    """Submit SLURM jobs for all ligands × both models.

    Creates one SLURM job per (model, ligand) pair. With 30 ligands and
    2 models, this submits 60 jobs that run in parallel. Each job generates
    num_designs binders and runs RF3 scoring, taking ~30-60 min per job.

    Output directory structure:
      {output_dir}/
        genume/{ligand_id}/results.csv        — Gen-UME metrics + RF3 scores
        genume/{ligand_id}/*_decoded.pdb       — Gen-UME decoded structures
        genume/{ligand_id}/*_rf3.pdb           — RF3 predicted structures
        proteina/{ligand_id}/results.csv       — Proteina metrics + RF3 scores
        proteina/{ligand_id}/*_proteina.pdb    — Proteina generated structures
        proteina/{ligand_id}/*_rf3.pdb         — RF3 predicted structures
        logs/                                  — SLURM stdout/stderr

    Parameters
    ----------
    checkpoint : str
        Path to Gen-UME checkpoint.
    num_designs : int
        Number of designs per ligand per model.
    output_dir : str
        Base output directory.
    num_ligands : int, optional
        Limit number of ligands (for testing). None = all available.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ligand_ids = get_ligand_ids()[:num_ligands]
    n_genume = genume_designs if genume_designs is not None else num_designs

    print(f"Submitting {len(ligand_ids)} ligands × 2 models = {len(ligand_ids)*2} jobs")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Gen-UME designs/ligand: {n_genume}")
    print(f"  Proteina designs/ligand: {num_designs}")
    print(f"  Output: {output_dir}")
    print()

    for lig_id in ligand_ids:
        for model, cmd_template in [
            ("genume", (
                f"cd {LOBSTER_DIR} && "
                f"uv run python scripts/benchmark_conditioned_gen.py run_genume "
                f"--checkpoint '{checkpoint}' --ligand_id {lig_id} "
                f"--num_designs {n_genume} --output_dir '{output_dir}/genume/{lig_id}'"
            )),
            ("proteina", (
                f"cd {LOBSTER_DIR} && "
                f"bash -c 'cd {PROTEINA_DIR} && source .venv/bin/activate && source env.sh && "
                f"cd {LOBSTER_DIR} && "
                f"python scripts/benchmark_conditioned_gen.py run_proteina "
                f"--ligand_id {lig_id} --num_designs {num_designs} "
                f"--output_dir {output_dir}/proteina/{lig_id}'"
            )),
        ]:
            short = "gu" if model == "genume" else "pc"
            result = subprocess.run(
                ["sbatch", "--parsable",
                 "--partition=ai4dd-b200", "--account=llm", "--qos=llm",
                 "--nodes=1", "--ntasks-per-node=1", "--gres=gpu:b200:1",
                 "--cpus-per-task=16", "--mem=128G", "-t", "04:00:00",
                 f"--job-name=bench-{short}-{lig_id[:8]}",
                 f"-o", f"{log_dir}/{model}_{lig_id}_%j.out",
                 f"-e", f"{log_dir}/{model}_{lig_id}_%j.err",
                 f"--wrap={cmd_template}"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  {model:10s} {lig_id}: Job {result.stdout.strip()}")

    print(f"\nWhen done: python scripts/benchmark_conditioned_gen.py merge --output_dir {output_dir}")


def submit_filtered(checkpoint: str, num_designs: int, rf3_top_k: int,
                    proteina_designs: int, output_dir: str, num_ligands: int | None = None):
    """Submit time-matched benchmark: Gen-UME (ESMFold-filtered) vs Proteina.

    Gen-UME generates num_designs per ligand, filters by ESMFold PAE, sends
    top rf3_top_k to RF3. Proteina generates proteina_designs per ligand, all
    go to RF3. Default config (111 designs, top-5 RF3) matches Proteina's
    ~900s wall time per ligand.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ligand_ids = get_ligand_ids()[:num_ligands]

    print(f"Submitting {len(ligand_ids)} ligands × 2 models = {len(ligand_ids)*2} jobs")
    print(f"  Gen-UME: {num_designs} designs → ESMFold → RF3 top {rf3_top_k}")
    print(f"  Proteina: {proteina_designs} designs → RF3 all")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output: {output_dir}")
    print()

    for lig_id in ligand_ids:
        for model, cmd_template in [
            ("genume_filtered", (
                f"cd {LOBSTER_DIR} && "
                f"uv run python scripts/benchmark_conditioned_gen.py run_genume_filtered "
                f"--checkpoint '{checkpoint}' --ligand_id {lig_id} "
                f"--num_designs {num_designs} --rf3_top_k {rf3_top_k} "
                f"--output_dir '{output_dir}/genume_filtered/{lig_id}'"
            )),
            ("proteina", (
                f"cd {LOBSTER_DIR} && "
                f"bash -c 'cd {PROTEINA_DIR} && source .venv/bin/activate && source env.sh && "
                f"cd {LOBSTER_DIR} && "
                f"python scripts/benchmark_conditioned_gen.py run_proteina "
                f"--ligand_id {lig_id} --num_designs {proteina_designs} "
                f"--output_dir {output_dir}/proteina/{lig_id}'"
            )),
        ]:
            short = "gf" if model == "genume_filtered" else "pc"
            result = subprocess.run(
                ["sbatch", "--parsable",
                 "--partition=ai4dd-b200", "--account=llm", "--qos=llm",
                 "--nodes=1", "--ntasks-per-node=1", "--gres=gpu:b200:1",
                 "--cpus-per-task=16", "--mem=128G", "-t", "04:00:00",
                 f"--job-name=bench-{short}-{lig_id[:8]}",
                 f"-o", f"{log_dir}/{model}_{lig_id}_%j.out",
                 f"-e", f"{log_dir}/{model}_{lig_id}_%j.err",
                 f"--wrap={cmd_template}"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  {model:18s} {lig_id}: Job {result.stdout.strip()}")

    print(f"\nWhen done: python scripts/benchmark_conditioned_gen.py merge --output_dir {output_dir}")


# ============================================================================
# Merge results
# ============================================================================

def merge_results(output_dir: str):
    """Merge per-ligand results and print head-to-head comparison table.

    Reads results.csv from each {output_dir}/{model}/{ligand_id}/ directory,
    concatenates into a single DataFrame, and reports:
      - All-designs averages for each model
      - Best-per-ligand comparison (highest RF3 ipTM per ligand)

    Parameters
    ----------
    output_dir : str
        Base output directory (same as passed to submit).
    """
    import pandas as pd

    all_dfs = []
    for model in ["genume", "genume_filtered", "proteina"]:
        model_dir = os.path.join(output_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for lig_dir in sorted(os.listdir(model_dir)):
            csv_path = os.path.join(model_dir, lig_dir, "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["model"] = model
                df["ligand_id"] = lig_dir
                all_dfs.append(df)

    if not all_dfs:
        print("No results found")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    print("=" * 70)
    print("BENCHMARK: Gen-UME vs Proteina-Complexa")
    print("=" * 70)

    for model in ["genume", "genume_filtered", "proteina"]:
        sub = combined[combined["model"] == model]
        if len(sub) == 0:
            continue
        iptm = pd.to_numeric(sub.get("rf3_ipTM", pd.Series()), errors="coerce").dropna()
        ptm = pd.to_numeric(sub.get("rf3_pTM", pd.Series()), errors="coerce").dropna()
        plddt = pd.to_numeric(sub.get("rf3_plddt", pd.Series()), errors="coerce").dropna()

        label = model.upper().replace("_", " ")
        n_total = len(sub)
        n_rf3 = len(iptm)
        extra = f", {n_rf3} with RF3" if n_rf3 < n_total else ""
        print(f"\n  {label} ({n_total} designs{extra}, {sub['ligand_id'].nunique()} ligands):")
        if len(iptm) > 0:
            print(f"    RF3 designs: ipTM={iptm.mean():.3f}, pTM={ptm.mean():.3f}, pLDDT={plddt.mean():.3f}")
            print(f"    ipTM > 0.5: {(iptm > 0.5).mean()*100:.0f}%")
            print(f"    ipTM > 0.7: {(iptm > 0.7).mean()*100:.0f}%")
        gen_times = pd.to_numeric(sub.get("gen_time_sec", pd.Series()), errors="coerce").dropna()
        if len(gen_times) > 0:
            print(f"    Gen time: {gen_times.mean():.1f}s/design (median={gen_times.median():.1f}s)")
            print(f"    Est. wall time: {gen_times.mean() * n_total + n_rf3 * 53:.0f}s")

    # Per-ligand best
    print("\n  BEST PER LIGAND:")
    for model in ["genume", "genume_filtered", "proteina"]:
        sub = combined[combined["model"] == model].copy()
        sub["_iptm"] = pd.to_numeric(sub.get("rf3_ipTM", pd.Series()), errors="coerce")
        sub = sub.dropna(subset=["_iptm"])
        if len(sub) == 0:
            continue
        best = sub.loc[sub.groupby("ligand_id")["_iptm"].idxmax()]
        label = model.upper().replace("_", " ")
        print(f"    {label:18s}: ipTM={best['_iptm'].mean():.3f}, "
              f">0.5={( best['_iptm'] > 0.5).mean()*100:.0f}%, "
              f">0.7={(best['_iptm'] > 0.7).mean()*100:.0f}%")

    print(f"\nFull results: {output_dir}/all_results.csv")

    # Generate comparison plots
    _plot_distributions(combined, output_dir)


def _plot_distributions(df: "pd.DataFrame", output_dir: str):
    """Generate distribution plots comparing Gen-UME vs Proteina-Complexa.

    Creates a 2×3 figure with histograms for each RF3 metric and a
    per-ligand scatter plot of best ipTM values.

    Saves to {output_dir}/benchmark_distributions.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col in ["rf3_ipTM", "rf3_pTM", "rf3_plddt", "rf3_min_ipAE", "rf3_ranking_score"]:
        df[col] = __import__("pandas").to_numeric(df[col], errors="coerce")
    df["rf3_ipAE_x31"] = df["rf3_min_ipAE"] * 31

    # Use genume_filtered if available, otherwise genume
    if "genume_filtered" in df["model"].values:
        genume = df[df["model"] == "genume_filtered"].dropna(subset=["rf3_ipTM"])
        genume_label = "Gen-UME (filtered)"
    else:
        genume = df[df["model"] == "genume"].dropna(subset=["rf3_ipTM"])
        genume_label = "Gen-UME"
    proteina = df[df["model"] == "proteina"].dropna(subset=["rf3_ipTM"])

    if len(genume) == 0 or len(proteina) == 0:
        print("  Skipping plots: insufficient data")
        return

    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_ligands = max(genume["ligand_id"].nunique(), proteina["ligand_id"].nunique())
    n_designs = max(
        genume.groupby("ligand_id").size().mode().iloc[0] if len(genume) > 0 else 0,
        proteina.groupby("ligand_id").size().mode().iloc[0] if len(proteina) > 0 else 0,
    )
    fig.suptitle(
        f"Gen-UME vs Proteina-Complexa — RF3 Confidence Distributions\n"
        f"({n_ligands} PoseBusters ligands × {n_designs} designs each)",
        fontsize=14, fontweight="bold",
    )

    metrics = [
        ("rf3_ipTM", "RF3 ipTM (interface quality)"),
        ("rf3_pTM", "RF3 pTM (protein fold)"),
        ("rf3_plddt", "RF3 pLDDT"),
        ("rf3_ipAE_x31", "RF3 ipAE × 31 (Å)"),
        ("rf3_ranking_score", "RF3 Ranking Score"),
    ]

    for i, (col, title) in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        vals_g = genume[col].dropna()
        vals_p = proteina[col].dropna()
        lo = min(vals_g.min(), vals_p.min())
        hi = max(vals_g.max(), vals_p.max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(vals_g, bins=bins, alpha=0.6, label=f"Gen-UME (mean={vals_g.mean():.3f})", color="steelblue")
        ax.hist(vals_p, bins=bins, alpha=0.6, label=f"Proteina (mean={vals_p.mean():.3f})", color="coral")
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.set_title(title)

    # Per-ligand best scatter
    ax = axes[1, 2]
    g_best = genume.loc[genume.groupby("ligand_id")["rf3_ipTM"].idxmax()].set_index("ligand_id")["rf3_ipTM"]
    p_best = proteina.loc[proteina.groupby("ligand_id")["rf3_ipTM"].idxmax()].set_index("ligand_id")["rf3_ipTM"]
    common = g_best.index.intersection(p_best.index)
    g_vals = g_best[common].values
    p_vals = p_best[common].values

    ax.scatter(g_vals, p_vals, s=40, alpha=0.7, color="purple", edgecolors="black", linewidth=0.5)
    lims = [0, max(g_vals.max(), p_vals.max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    g_wins = int((g_vals >= p_vals).sum())
    ax.set_xlabel("Gen-UME best ipTM")
    ax.set_ylabel("Proteina best ipTM")
    ax.set_title(f"Best per ligand (Gen-UME wins {g_wins}/{len(common)})")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_distributions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark: Gen-UME vs Proteina-Complexa")
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args for all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", type=str, default=None,
                        help=f"Data directory with *_ligand.pt files (default: {DEFAULT_DATA_DIR})")
    common.add_argument("--backend", type=str, default="boltz", choices=["boltz", "rf3"],
                        help="Structure prediction backend (default: boltz)")

    p = sub.add_parser("submit", help="Submit SLURM jobs for all ligands × both models", parents=[common])
    p.add_argument("--checkpoint", required=True, help="Gen-UME checkpoint path")
    p.add_argument("--num_designs", type=int, default=5, help="Designs per ligand (Proteina default)")
    p.add_argument("--genume_designs", type=int, default=None, help="Override designs for Gen-UME (default: same as --num_designs)")
    p.add_argument("--output_dir", required=True, help="Base output directory")
    p.add_argument("--num_ligands", type=int, default=None, help="Limit ligands (for testing)")

    p = sub.add_parser("merge", help="Merge results and print comparison")
    p.add_argument("--output_dir", required=True)

    p = sub.add_parser("run_genume", help="Run Gen-UME on a single ligand", parents=[common])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ligand_id", required=True)
    p.add_argument("--num_designs", type=int, default=5)
    p.add_argument("--output_dir", required=True)

    p = sub.add_parser("run_proteina", help="Run Proteina on a single ligand", parents=[common])
    p.add_argument("--ligand_id", required=True)
    p.add_argument("--num_designs", type=int, default=5)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed_offset", type=int, default=0, help="Seed offset for diversity (default: 0)")

    p = sub.add_parser("run_genume_filtered", help="Gen-UME: generate many → ESMFold filter → top-K", parents=[common])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ligand_id", required=True)
    p.add_argument("--num_designs", type=int, default=111, help="Total designs to generate")
    p.add_argument("--rf3_top_k", type=int, default=5, help="Number of top designs to send to structure prediction")
    p.add_argument("--output_dir", required=True)

    p = sub.add_parser("submit_filtered", help="Submit filtered benchmark: Gen-UME (ESMFold→cofold) vs Proteina", parents=[common])
    p.add_argument("--checkpoint", required=True, help="Gen-UME checkpoint path")
    p.add_argument("--num_designs", type=int, default=111, help="Gen-UME designs to generate per ligand")
    p.add_argument("--rf3_top_k", type=int, default=5, help="Top-K designs to send to structure prediction")
    p.add_argument("--proteina_designs", type=int, default=5, help="Proteina designs per ligand")
    p.add_argument("--output_dir", required=True, help="Base output directory")
    p.add_argument("--num_ligands", type=int, default=None, help="Limit ligands (for testing)")

    args = parser.parse_args()

    if args.command == "submit":
        submit_all(args.checkpoint, args.num_designs, args.output_dir, args.num_ligands,
                   genume_designs=args.genume_designs)
    elif args.command == "merge":
        merge_results(args.output_dir)
    elif args.command == "run_genume":
        run_genume(args.checkpoint, args.ligand_id, args.num_designs, args.output_dir,
                   data_dir=args.data_dir, backend=args.backend)
    elif args.command == "run_proteina":
        run_proteina(args.ligand_id, args.num_designs, args.output_dir,
                     backend=args.backend, seed_offset=args.seed_offset)
    elif args.command == "run_genume_filtered":
        run_genume_filtered(args.checkpoint, args.ligand_id, args.num_designs,
                            args.rf3_top_k, args.output_dir)
    elif args.command == "submit_filtered":
        submit_filtered(args.checkpoint, args.num_designs, args.rf3_top_k,
                        args.proteina_designs, args.output_dir, args.num_ligands)


if __name__ == "__main__":
    main()
