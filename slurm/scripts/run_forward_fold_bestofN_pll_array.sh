#!/usr/bin/env bash
# Forward-folding best-of-N on CAMEO, sharded as one SLURM-array task per target.
#
# Each array task generates N_CANDIDATES designs for a single CAMEO target,
# scores with PLL (seq / struc / joint / joint_true), and writes per-target CSVs
# under ${OUTPUT_DIR}/shards/<target>/.
#
# After all tasks finish, merge with:
#   uv run python scripts/concat_bestofN_ff_array.py --run-dir "${OUTPUT_DIR}"
#
# Usage:
#   bash slurm/scripts/run_forward_fold_bestofN_pll_array.sh              # submit TED N=100
#   bash slurm/scripts/run_forward_fold_bestofN_pll_array.sh ted        # same
#   bash slurm/scripts/run_forward_fold_bestofN_pll_array.sh denovo     # denovo ckpt
#   bash slurm/scripts/run_forward_fold_bestofN_pll_array.sh --worker   # (sbatch internal)
#
# Env overrides:
#   N_CANDIDATES       (default 100)
#   K_DRAWS            (default 32)
#   ARRAY_CONCURRENCY  (default: unset = no cap; set e.g. 64 to throttle)
#   TIME_LIMIT         (default 02:00:00) per-target wall time (~8-15 min typical)

set -euo pipefail

CKPT_DENOVO="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt"
CKPT_TED="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"

INPUTS_DIR="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed"
INPUTS_GLOB="${INPUTS_DIR}/*.pt"
OUTPUT_ROOT="/cv/scratch/u/lisanzas/evaluations"

N_CANDIDATES="${N_CANDIDATES:-100}"
K_DRAWS="${K_DRAWS:-32}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

# SLURM array throttle: omit %N suffix when unset/0/empty → all tasks eligible at once.
if [ -n "${ARRAY_CONCURRENCY:-}" ] && [ "${ARRAY_CONCURRENCY}" != "0" ]; then
    ARRAY_SPEC_SUFFIX="%${ARRAY_CONCURRENCY}"
else
    ARRAY_SPEC_SUFFIX=""
fi

###############################################################################
# Worker
###############################################################################
if [ "${1:-}" = "--worker" ]; then
    : "${CKPT:?CKPT not set}"
    : "${INPUTS_DIR:?INPUTS_DIR not set}"
    : "${OUTPUT_DIR:?OUTPUT_DIR not set}"
    : "${VARIANT:?VARIANT not set}"
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

    mapfile -t TARGET_PATHS < <(ls -1 "${INPUTS_DIR}"/*.pt 2>/dev/null | sort)
    N_TARGETS=${#TARGET_PATHS[@]}
    if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TARGETS}" ]; then
        echo "[worker] array id ${SLURM_ARRAY_TASK_ID} >= n_targets ${N_TARGETS}; nothing to do"
        exit 0
    fi

    TARGET_PATH="${TARGET_PATHS[${SLURM_ARRAY_TASK_ID}]}"
    TARGET_ID="$(basename "${TARGET_PATH}" .pt)"
    SHARD_DIR="${OUTPUT_DIR}/shards/${TARGET_ID}"
    mkdir -p "${SHARD_DIR}"

    cd /cv/home/lisanzas/lobster

    echo "[worker] VARIANT=${VARIANT}  target=${TARGET_ID}  (${SLURM_ARRAY_TASK_ID}/${N_TARGETS})"
    echo "[worker] N=${N_CANDIDATES}  K=${K_DRAWS}"
    echo "[worker] SHARD_DIR=${SHARD_DIR}"
    echo "[worker] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}.${SLURM_ARRAY_TASK_ID}  node=$(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    uv run python scripts/forward_fold_bestofN_pll.py \
        --inputs "${INPUTS_DIR}/*.pt" \
        --ckpt "${CKPT}" \
        --output-dir "${SHARD_DIR}" \
        --target-id "${TARGET_ID}" \
        --N "${N_CANDIDATES}" \
        --K "${K_DRAWS}"
    exit $?
fi

###############################################################################
# Submit
###############################################################################
VARIANT="${1:-ted}"

case "${VARIANT}" in
    ted|GenUME-TED|genume-ted)
        CKPT="${CKPT_TED}"
        VARIANT="ted"
        ;;
    denovo)
        CKPT="${CKPT_DENOVO}"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}" >&2
        echo "Usage: $0 [ted|denovo]" >&2
        exit 2
        ;;
esac

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: CKPT not found: ${CKPT}" >&2
    exit 1
fi

mapfile -t TARGET_PATHS < <(ls -1 "${INPUTS_DIR}"/*.pt 2>/dev/null | sort)
N_TARGETS=${#TARGET_PATHS[@]}
if [ "${N_TARGETS}" -eq 0 ]; then
    echo "ERROR: no targets at ${INPUTS_GLOB}" >&2
    exit 1
fi
LAST_IDX=$(( N_TARGETS - 1 ))

OUTPUT_DIR="${OUTPUT_ROOT}/gen_ume_${VARIANT}_cameo_bestofN_pll_N${N_CANDIDATES}"
mkdir -p "${OUTPUT_DIR}/shards"
mkdir -p "/cv/scratch/u/lisanzas/slurm_logs/bestofN_ff_pll_array/${VARIANT}_N${N_CANDIDATES}"

echo "FF best-of-N array submission"
echo "  variant             : ${VARIANT}"
echo "  n_targets           : ${N_TARGETS}"
echo "  N_CANDIDATES/target : ${N_CANDIDATES}"
echo "  total candidates    : $(( N_TARGETS * N_CANDIDATES ))"
if [ -n "${ARRAY_SPEC_SUFFIX}" ]; then
    echo "  array indices       : 0-${LAST_IDX}${ARRAY_SPEC_SUFFIX} (throttled)"
else
    echo "  array indices       : 0-${LAST_IDX} (no concurrency cap)"
fi
echo "  per-task time limit : ${TIME_LIMIT}"
echo "  ckpt                : ${CKPT}"
echo "  output              : ${OUTPUT_DIR}"

sbatch \
    --partition=ai4dd-b200 \
    --account=llm \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:b200:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time="${TIME_LIMIT}" \
    --qos=llm \
    --array="0-${LAST_IDX}${ARRAY_SPEC_SUFFIX}" \
    --job-name="bestofN_ff_${VARIANT}_N${N_CANDIDATES}" \
    --output="/cv/scratch/u/lisanzas/slurm_logs/bestofN_ff_pll_array/${VARIANT}_N${N_CANDIDATES}/%A_%a.out" \
    --error="/cv/scratch/u/lisanzas/slurm_logs/bestofN_ff_pll_array/${VARIANT}_N${N_CANDIDATES}/%A_%a.err" \
    --export="ALL,VARIANT=${VARIANT},CKPT=${CKPT},INPUTS_DIR=${INPUTS_DIR},OUTPUT_DIR=${OUTPUT_DIR},N_CANDIDATES=${N_CANDIDATES},K_DRAWS=${K_DRAWS}" \
    "$0" --worker

echo "Submitted. After completion, merge:"
echo "  cd /cv/home/lisanzas/lobster && uv run python scripts/concat_bestofN_ff_array.py --run-dir ${OUTPUT_DIR}"
