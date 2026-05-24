#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/smiles_test_%j.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/smiles_test_%j.err
#SBATCH --job-name=smiles-test
#SBATCH -t 01:00:00

set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
CONFIG_NAME="search_ligand_binder_local_pipeline"
TASK_NAME="PLINDER_11as_C_145"
RUN_NAME="plinder_${TASK_NAME}"

cd "$PROTEINA_DIR"
source .venv/bin/activate
source env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL_DIR="${PROTEINA_DIR}/evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
BACKUP_DIR="${EVAL_DIR}_backup_before_smiles_fix"

if [ -d "$EVAL_DIR" ] && [ ! -d "$BACKUP_DIR" ]; then
    echo "[BACKUP] Moving old eval results to ${BACKUP_DIR}"
    mv "$EVAL_DIR" "$BACKUP_DIR"
fi

echo "[TEST] Running evaluate only for ${TASK_NAME} with SMILES fix"
echo "[TEST] SMILES should be: NC(=O)C[C@H](N)C(=O)O (9 heavy atoms)"

complexa evaluate "configs/${CONFIG_NAME}.yaml" \
    "++run_name=${RUN_NAME}" \
    "++generation.task_name=${TASK_NAME}" \
    "++ckpt_path=${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    "++autoencoder_ckpt_path=${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
    --verbose

echo "[DONE] Evaluate finished. Checking RF3 output atom counts..."

python3 - << 'PYEOF'
import glob, os

eval_base = "/cv/scratch/u/lisanzas/proteina-complexa/evaluation_results"
task = "PLINDER_11as_C_145"
run = "plinder_PLINDER_11as_C_145"
d = f"{eval_base}/search_ligand_binder_local_pipeline_{task}_{run}"

if not os.path.isdir(d):
    print(f"[ERROR] Eval dir not found: {d}")
    exit(1)

for job_dir in sorted(glob.glob(f"{d}/job_*")):
    job_name = os.path.basename(job_dir)
    gen_pdb = f"{job_dir}/{job_name}.pdb"

    # Count gen ligand atoms
    gen_lig = sum(1 for line in open(gen_pdb) if line.startswith("HETATM")) if os.path.exists(gen_pdb) else -1

    # Find RF3 output PDBs
    rf3_pdbs = sorted(glob.glob(f"{job_dir}/rf3_outputs/*/complex_*_model.pdb"))
    for rf3_pdb in rf3_pdbs:
        rf3_lig = sum(1 for line in open(rf3_pdb) if line.startswith("HETATM"))
        match = "MATCH" if gen_lig == rf3_lig else "MISMATCH"
        print(f"  {job_name} | gen={gen_lig} rf3={rf3_lig} [{match}] | {os.path.basename(rf3_pdb)}")

# Also check the CSV for ligand_scRMSD values
import csv
csv_files = sorted(glob.glob(f"{d}/binder_results_*.csv"))
if csv_files:
    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scrmsd = row.get("ligand_scRMSD_aligned_allatom", "N/A")
            seq_type = row.get("sequence_type", "?")
            print(f"  seq_type={seq_type} ligand_scRMSD_aligned_allatom={scrmsd}")
PYEOF
