#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/smiles_batch_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/smiles_batch_%A_%a.err
#SBATCH --job-name=smiles-batch
#SBATCH -t 01:30:00
#SBATCH --array=0-4

set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
CONFIG_NAME="search_ligand_binder_local_pipeline"

# 5 ligands with different heavy atom counts:
#   1a14_G_0    = 14 atoms (NAG sugar)
#   10mh_D_5    = 26 atoms (SAH)
#   1avn_E_186  =  8 atoms (tiny)
#   1acz_B_120  = 77 atoms (large)
#   1a3g_F_6    = 15 atoms (PLP)
TASKS=(
    "PLINDER_1a14_G_0"
    "PLINDER_10mh_D_5"
    "PLINDER_1avn_E_186"
    "PLINDER_1acz_B_120"
    "PLINDER_1a3g_F_6"
)

TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
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

echo "[TEST] Running evaluate for ${TASK_NAME} with SMILES fix"

complexa evaluate "configs/${CONFIG_NAME}.yaml" \
    "++run_name=${RUN_NAME}" \
    "++generation.task_name=${TASK_NAME}" \
    "++ckpt_path=${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    "++autoencoder_ckpt_path=${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
    --verbose

echo "[DONE] Evaluate finished for ${TASK_NAME}. Checking atom counts..."

python3 - << 'PYEOF'
import glob, os, csv, sys

task = os.environ["TASK_NAME"]
run = f"plinder_{task}"
eval_base = "/cv/scratch/u/lisanzas/proteina-complexa/evaluation_results"
d = f"{eval_base}/search_ligand_binder_local_pipeline_{task}_{run}"

if not os.path.isdir(d):
    print(f"[ERROR] Eval dir not found: {d}")
    sys.exit(1)

match_count = 0
mismatch_count = 0
for job_dir in sorted(glob.glob(f"{d}/job_*")):
    job_name = os.path.basename(job_dir)
    gen_pdb = f"{job_dir}/{job_name}.pdb"
    gen_lig = sum(1 for line in open(gen_pdb) if line.startswith("HETATM")) if os.path.exists(gen_pdb) else -1
    rf3_pdbs = sorted(glob.glob(f"{job_dir}/rf3_outputs/*/complex_*_model.pdb"))
    for rf3_pdb in rf3_pdbs:
        rf3_lig = sum(1 for line in open(rf3_pdb) if line.startswith("HETATM"))
        if gen_lig == rf3_lig:
            match_count += 1
        else:
            mismatch_count += 1
            print(f"  MISMATCH: {job_name} gen={gen_lig} rf3={rf3_lig}")

print(f"[{task}] Atom count: {match_count} MATCH, {mismatch_count} MISMATCH")

csv_files = sorted(glob.glob(f"{d}/binder_results_*.csv"))
if csv_files:
    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        inf_count = 0
        finite_count = 0
        for row in reader:
            for key in ("self_ligand_scRMSD_aligned_allatom", "mpnn_ligand_scRMSD_aligned_allatom"):
                val = row.get(key)
                if val and val != "N/A":
                    if "inf" in val.lower():
                        inf_count += 1
                    else:
                        finite_count += 1
        print(f"[{task}] ligand_scRMSD: {finite_count} finite, {inf_count} inf")
PYEOF
