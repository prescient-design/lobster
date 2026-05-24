#!/usr/bin/env bash
# CG best-of-N for the 4-ligand Proteina-Complexa benchmark, sharded across
# many GPUs via a SLURM array job so total wall-clock fits well inside the
# 24h limit and the resulting candidate set can be cofolded with Boltz2
# afterwards.
#
# Layout:
#   4 ligands x 25 chunks x N=120 candidates per chunk = 12,000 total
#   = 100 array tasks, each task = (one ligand, one slot-chunk).
#
# Per array task wall-clock estimate (B200): 120 cands * ~8s + 12s model load
# ≈ 17 min (well under the 24h job time-limit; we set 4h to be safe).
#
# Outputs:
#   ${OUTPUT_ROOT}/all/cg_proteina_4lig/full_N3000/${PDB_ID}/chunk_${CC}/
#     bestofN_cg_lig_candidates_<ts>.csv
#     bestofN_cg_lig_summary_<ts>.csv
#
# After all array tasks finish, concatenate the per-chunk candidate CSVs
# into a single per-ligand candidates.csv before submitting Boltz2 cofold.
#
# Usage:
#   bash slurm/scripts/run_cg_proteina_4lig_array.sh                # submit array
#   bash slurm/scripts/run_cg_proteina_4lig_array.sh --worker       # (sbatch internal)
#
# Env overrides:
#   N_PER_CHUNK     (default 120) candidates per array task
#   N_CHUNKS        (default 25)  chunks per ligand  (N_PER_CHUNK*N_CHUNKS = N per ligand)
#   NSTEPS          (default 200)
#   CG_LENGTH       (default 100)
#   ARRAY_CONCURRENCY (default 100) max simultaneous array tasks
#   TIME_LIMIT      (default 04:00:00) per-task time limit

set -euo pipefail

CKPT_ALL="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_gen_ume_all_latest/last.ckpt"

SOURCE_DIR="${SOURCE_DIR:-/cv/home/lisanzas/lobster/data/proteina_ligand_targets/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand}"
SUBSET_TAG="${SUBSET_TAG:-proteina_4lig}"

N_PER_CHUNK="${N_PER_CHUNK:-120}"
N_CHUNKS="${N_CHUNKS:-25}"
K_DRAWS="${K_DRAWS:-32}"
NSTEPS="${NSTEPS:-200}"
CG_LENGTH="${CG_LENGTH:-100}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-100}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

# Hard-coded ligand list (order matters: index used for array-task->ligand map).
LIGANDS=(
    "5SDV_IAI"
    "7BKC_FAD"
    "7C7M_SAM"
    "7V11_OQO"
)
N_LIGANDS=${#LIGANDS[@]}

###############################################################################
# Worker
###############################################################################
if [ "${1:-}" = "--worker" ]; then
    : "${SOURCE_DIR:?SOURCE_DIR not set}"
    : "${CKPT:?CKPT not set}"
    : "${OUTPUT_ROOT_TASK:?OUTPUT_ROOT_TASK not set}"
    : "${N_PER_CHUNK:?N_PER_CHUNK not set}"
    : "${N_CHUNKS:?N_CHUNKS not set}"
    : "${NSTEPS:?NSTEPS not set}"
    : "${CG_LENGTH:?CG_LENGTH not set}"

    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
    K=${SLURM_ARRAY_TASK_ID}
    LIG_IDX=$(( K / N_CHUNKS ))
    CHUNK_IDX=$(( K % N_CHUNKS ))
    PDB_ID="${LIGANDS[${LIG_IDX}]}"
    CANDIDATE_OFFSET=$(( CHUNK_IDX * N_PER_CHUNK ))

    OUTPUT_DIR="${OUTPUT_ROOT_TASK}/${PDB_ID}/chunk_$(printf '%02d' ${CHUNK_IDX})"
    mkdir -p "${OUTPUT_DIR}"

    cd /cv/home/lisanzas/lobster
    echo "[worker] array_task=${K}  lig_idx=${LIG_IDX}  chunk_idx=${CHUNK_IDX}  pdb_id=${PDB_ID}"
    echo "[worker] N=${N_PER_CHUNK}  candidate_offset=${CANDIDATE_OFFSET}"
    echo "[worker] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[worker] node=$(hostname)  job=${SLURM_JOB_ID:-NA}.${SLURM_ARRAY_TASK_ID}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    uv run python scripts/conditioned_gen_bestofN_pll_ligand.py \
        --source-data-dir "${SOURCE_DIR}" \
        --ckpt "${CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --target-id "${PDB_ID}" \
        --candidate-offset "${CANDIDATE_OFFSET}" \
        --N "${N_PER_CHUNK}" \
        --K "${K_DRAWS}" \
        --nsteps "${NSTEPS}" \
        --length "${CG_LENGTH}"
    exit $?
fi

###############################################################################
# Submit
###############################################################################
TOTAL_TASKS=$(( N_LIGANDS * N_CHUNKS ))
LAST_IDX=$(( TOTAL_TASKS - 1 ))
N_PER_LIGAND=$(( N_PER_CHUNK * N_CHUNKS ))
N_TOTAL=$(( N_PER_LIGAND * N_LIGANDS ))

OUTPUT_ROOT_TASK="${OUTPUT_ROOT}/all/cg_${SUBSET_TAG}/full_N${N_PER_LIGAND}"
mkdir -p "${OUTPUT_ROOT_TASK}"
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/bestofN_pl_pll/cg_${SUBSET_TAG}

echo "Sharded CG best-of-N submission"
echo "  ligands             : ${LIGANDS[*]}"
echo "  N_PER_LIGAND        : ${N_PER_LIGAND} (= ${N_PER_CHUNK} per chunk x ${N_CHUNKS} chunks)"
echo "  N_TOTAL (candidates): ${N_TOTAL}"
echo "  array indices       : 0-${LAST_IDX}  (${TOTAL_TASKS} tasks)"
echo "  array concurrency   : ${ARRAY_CONCURRENCY}"
echo "  per-task time limit : ${TIME_LIMIT}"
echo "  CG length           : ${CG_LENGTH}"
echo "  CG nsteps           : ${NSTEPS}"
echo "  CKPT                : ${CKPT_ALL}"
echo "  SOURCE_DIR          : ${SOURCE_DIR}"
echo "  OUTPUT_ROOT_TASK    : ${OUTPUT_ROOT_TASK}"

if [ ! -f "${CKPT_ALL}" ]; then
    echo "ERROR: ckpt not found: ${CKPT_ALL}" >&2
    exit 1
fi

sbatch \
    --partition=ai4dd-b200 \
    --account=llm \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:b200:1 \
    --cpus-per-task=16 \
    --mem=128G \
    --time="${TIME_LIMIT}" \
    --qos=llm \
    --array="0-${LAST_IDX}%${ARRAY_CONCURRENCY}" \
    --job-name="cg_${SUBSET_TAG}_N${N_PER_LIGAND}" \
    --output="/cv/scratch/u/lisanzas/slurm_logs/bestofN_pl_pll/cg_${SUBSET_TAG}/%A_%a.out" \
    --error="/cv/scratch/u/lisanzas/slurm_logs/bestofN_pl_pll/cg_${SUBSET_TAG}/%A_%a.err" \
    --export="ALL,CKPT=${CKPT_ALL},SOURCE_DIR=${SOURCE_DIR},OUTPUT_ROOT_TASK=${OUTPUT_ROOT_TASK},N_PER_CHUNK=${N_PER_CHUNK},N_CHUNKS=${N_CHUNKS},K_DRAWS=${K_DRAWS},NSTEPS=${NSTEPS},CG_LENGTH=${CG_LENGTH}" \
    "$0" --worker

echo "Submitted."
