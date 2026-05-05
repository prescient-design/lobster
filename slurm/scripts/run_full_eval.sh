#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Full Evaluation Pipeline for Gen-UME Protein-Ligand
#
# Phase 1: Model evaluation (IF, FF, CG, LigandMPNN baseline)
# Phase 2: Co-folding validation (Boltz2 and/or RF3 on designed sequences)
# Phase 3: Merge co-fold results into eval CSVs
#
# Usage:
#   # Phase 1: run all model evals
#   CKPT=/path/to/checkpoint.ckpt EVAL_TAG=my_exp bash slurm/scripts/run_full_eval.sh
#
#   # Phase 2: run co-folding on Phase 1 results
#   SKIP_PHASE1=1 EVAL_TAG=my_exp CKPT=... \
#     COFOLD_BACKEND=rf3 COFOLD_TASKS=if,cg \
#     bash slurm/scripts/run_full_eval.sh
#
#   # Phase 3: merge co-fold results
#   SKIP_PHASE1=1 SKIP_PHASE2=1 EVAL_TAG=my_exp CKPT=... \
#     bash slurm/scripts/run_full_eval.sh
#
# Optional overrides:
#   EVAL_TAG=my_experiment        # Label for output dirs (default: auto from ckpt)
#   SKIP_PHASE1=1                 # Skip Phase 1 if already run
#   SKIP_PHASE2=1                 # Skip Phase 2
#   COFOLD_BACKEND=rf3            # Co-folding backend: boltz, rf3, or both (default: rf3)
#   COFOLD_TASKS=if,ff,cg,lmpnn  # Which eval tasks to co-fold (default: if,cg)
#   RF3_N_CHUNKS=4                # Number of parallel RF3 GPU jobs (default: 4)
#   CG_NUM_LIGANDS=4              # Number of ligands for conditioned gen (default: 4 = Proteina paper)
#   CG_NUM_DESIGNS=10             # Designs per ligand for conditioned gen (default: 10)
#   CG_DATA_DIR=...               # Override data dir for CG (default: proteina_ligand_targets)
# ==============================================================================

LOBSTER_DIR="/cv/home/lisanzas/lobster"
PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
BASE="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks"
DATA_DIR="${LOBSTER_DIR}/data/posebusters/processed/posebusters_benchmark_no_overlap/"
RAW_DATA_DIR="${LOBSTER_DIR}/data/posebusters/posebusters_benchmark_set/"

CKPT="${CKPT:?Set CKPT=/path/to/checkpoint.ckpt}"
EVAL_TAG="${EVAL_TAG:-$(date +%Y-%m-%dT%H-%M-%S)}"
COFOLD_BACKEND="${COFOLD_BACKEND:-boltz}"
COFOLD_TASKS="${COFOLD_TASKS:-if,cg}"
RF3_N_CHUNKS="${RF3_N_CHUNKS:-4}"
CG_NUM_LIGANDS="${CG_NUM_LIGANDS:-4}"
CG_NUM_DESIGNS="${CG_NUM_DESIGNS:-10}"
CG_DATA_DIR="${CG_DATA_DIR:-${LOBSTER_DIR}/data/proteina_ligand_targets/processed}"

# Copy checkpoint for reproducibility (training may overwrite last.ckpt)
EVAL_CKPT_DIR="${BASE}/checkpoints_${EVAL_TAG}"
mkdir -p "${EVAL_CKPT_DIR}"
CKPT_COPY="${EVAL_CKPT_DIR}/$(basename ${CKPT})"
if [ ! -f "${CKPT_COPY}" ]; then
    echo "Copying checkpoint for reproducibility..."
    cp "${CKPT}" "${CKPT_COPY}"
    echo "  ${CKPT} -> ${CKPT_COPY}"
fi
CKPT="${CKPT_COPY}"

echo "============================================================"
echo " Gen-UME Full Evaluation Pipeline"
echo " Checkpoint: ${CKPT}"
echo " Tag:        ${EVAL_TAG}"
echo " Base dir:   ${BASE}"
echo "============================================================"
echo ""

# Output directories
IF_DIR="${BASE}/gen_ume_pl_inverse_folding_${EVAL_TAG}"
FF_DIR="${BASE}/gen_ume_pl_forward_folding_${EVAL_TAG}"
CG_DIR="${BASE}/gen_ume_pl_conditioned_gen_${EVAL_TAG}"
LMPNN_DIR="${BASE}/ligandmpnn_baseline_${EVAL_TAG}"
LOG_DIR="/cv/scratch/u/lisanzas/slurm_logs/eval_full_${EVAL_TAG}"

# ==============================================================================
# Phase 1: Model evaluation (4 parallel GPU jobs)
# ==============================================================================
if [[ "${SKIP_PHASE1:-0}" != "1" ]]; then

echo "=== Phase 1: Submitting model evaluation jobs ==="

mkdir -p "${LOG_DIR}"

# Gen-UME Inverse Folding
JOB_IF=$(sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=16 --mem=128G \
    -t 1-00:00:00 --job-name="eval-if-${EVAL_TAG}" \
    -o "${LOG_DIR}/inv_fold_%j.out" \
    -e "${LOG_DIR}/inv_fold_%j.err" \
    --wrap="
cd ${LOBSTER_DIR}
mkdir -p ${IF_DIR}
uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
    --checkpoint '${CKPT}' \
    --data_dir '${DATA_DIR}' \
    --raw_data_dir '${RAW_DATA_DIR}' \
    --output inverse_folding_results.csv \
    --structure_path '${IF_DIR}' \
    --nsteps 100 --decode_structure --save_gt_structure --num_samples -1
echo 'DONE: inverse folding'
")
echo "  [1/4] Inverse Folding      -> Job ${JOB_IF}"

# Gen-UME Forward Folding
JOB_FF=$(sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=16 --mem=128G \
    -t 1-00:00:00 --job-name="eval-ff-${EVAL_TAG}" \
    -o "${LOG_DIR}/fwd_fold_%j.out" \
    -e "${LOG_DIR}/fwd_fold_%j.err" \
    --wrap="
cd ${LOBSTER_DIR}
mkdir -p ${FF_DIR}
uv run python -m lobster.cmdline.evaluate_protein_ligand_forward_folding \
    --checkpoint '${CKPT}' \
    --data_dir '${DATA_DIR}' \
    --raw_data_dir '${RAW_DATA_DIR}' \
    --output forward_folding_results.csv \
    --structure_path '${FF_DIR}' \
    --nsteps 100 --save_structures --save_gt_structure --num_samples -1 --minimize_ligand
echo 'DONE: forward folding'
")
echo "  [2/4] Forward Folding      -> Job ${JOB_FF}"

# Gen-UME Conditioned Generation
JOB_CG=$(sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=16 --mem=128G \
    -t 1-00:00:00 --job-name="eval-cg-${EVAL_TAG}" \
    -o "${LOG_DIR}/cond_gen_%j.out" \
    -e "${LOG_DIR}/cond_gen_%j.err" \
    --wrap="
cd ${LOBSTER_DIR}
mkdir -p ${CG_DIR}
uv run python -m lobster.cmdline.evaluate_ligand_conditioned_protein_generation \
    --checkpoint '${CKPT}' \
    --data_dir '${CG_DATA_DIR}' \
    --raw_data_dir '${RAW_DATA_DIR}' \
    --output conditioned_gen_results.csv \
    --structure_path '${CG_DIR}' \
    --nsteps 200 --num_samples ${CG_NUM_LIGANDS} \
    --num_designs ${CG_NUM_DESIGNS} \
    --save_structures --minimize_ligand
echo 'DONE: conditioned generation'
")
echo "  [3/4] Conditioned Gen      -> Job ${JOB_CG}"

# LigandMPNN Baseline (no checkpoint needed)
JOB_LMPNN=$(sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=16 --mem=128G \
    -t 1-00:00:00 --job-name="eval-lmpnn-${EVAL_TAG}" \
    -o "${LOG_DIR}/lmpnn_%j.out" \
    -e "${LOG_DIR}/lmpnn_%j.err" \
    --wrap="
cd ${LOBSTER_DIR}
mkdir -p ${LMPNN_DIR}
uv run python -m lobster.cmdline.evaluate_ligandmpnn_baseline \
    --data_dir '${DATA_DIR}' \
    --raw_data_dir '${RAW_DATA_DIR}' \
    --output ligandmpnn_baseline_results.csv \
    --structure_path '${LMPNN_DIR}' \
    --num_samples -1
echo 'DONE: ligandmpnn baseline'
")
echo "  [4/4] LigandMPNN Baseline  -> Job ${JOB_LMPNN}"

echo ""
echo "Phase 1 submitted. Wait for all 4 jobs to complete, then run:"
echo "  SKIP_PHASE1=1 EVAL_TAG=${EVAL_TAG} CKPT='${CKPT}' bash slurm/scripts/run_full_eval.sh"

fi  # Phase 1

# ==============================================================================
# Phase 2: Co-folding validation (Boltz2 and/or RF3)
# ==============================================================================
if [[ "${SKIP_PHASE1:-0}" == "1" && "${SKIP_PHASE2:-0}" != "1" ]]; then

echo "=== Phase 2: Co-folding validation ==="
echo "  Backend: ${COFOLD_BACKEND}"
echo "  Tasks:   ${COFOLD_TASKS}"
echo ""

mkdir -p "${LOG_DIR}"

# Build task list: label:csv_path:id_col
declare -A TASK_MAP
TASK_MAP[if]="gen_ume_inv_fold:${IF_DIR}/inverse_folding_results.csv:pdb_id"
TASK_MAP[ff]="gen_ume_fwd_fold:${FF_DIR}/forward_folding_results.csv:pdb_id"
TASK_MAP[cg]="gen_ume_cond_gen:${CG_DIR}/conditioned_gen_results.csv:ligand_id"
TASK_MAP[lmpnn]="ligandmpnn_baseline:${LMPNN_DIR}/ligandmpnn_baseline_results.csv:pdb_id"

# Parse COFOLD_TASKS
IFS=',' read -ra SELECTED_TASKS <<< "${COFOLD_TASKS}"

for TASK in "${SELECTED_TASKS[@]}"; do
    TASK=$(echo "${TASK}" | tr -d ' ')
    if [[ -z "${TASK_MAP[$TASK]+_}" ]]; then
        echo "  WARNING: Unknown task '${TASK}', skipping (valid: if, ff, cg, lmpnn)"
        continue
    fi

    IFS=':' read -r LABEL CSV ID_COL <<< "${TASK_MAP[$TASK]}"

    if [ ! -f "${CSV}" ]; then
        echo "  SKIP ${LABEL}: ${CSV} not found"
        continue
    fi

    N_SAMPLES=$(cd ${LOBSTER_DIR} && uv run python -c "import pandas as pd; print(len(pd.read_csv('${CSV}')))")
    echo "  ${LABEL}: ${N_SAMPLES} samples from ${CSV}"

    # --- Boltz2 ---
    if [[ "${COFOLD_BACKEND}" == "boltz" || "${COFOLD_BACKEND}" == "both" ]]; then
        COFOLD_DIR="${BASE}/boltz2_${LABEL}_${EVAL_TAG}"

        cd ${LOBSTER_DIR}
        uv run python -m lobster.cmdline.submit_cofold_batch \
            --eval_csv "${CSV}" \
            --output_dir "${COFOLD_DIR}" \
            --backend boltz \
            --id_col "${ID_COL}" \
            --submit

        echo "    Boltz2 submitted -> ${COFOLD_DIR}"
    fi

    # --- RF3 ---
    if [[ "${COFOLD_BACKEND}" == "rf3" || "${COFOLD_BACKEND}" == "both" ]]; then
        RF3_OUT="${BASE}/rf3_${LABEL}_${EVAL_TAG}"
        mkdir -p "${RF3_OUT}"

        CHUNK_SIZE=$(( (N_SAMPLES + RF3_N_CHUNKS - 1) / RF3_N_CHUNKS ))

        for i in $(seq 0 $((RF3_N_CHUNKS - 1))); do
            START=$((i * CHUNK_SIZE))
            END=$((START + CHUNK_SIZE))
            if [ ${END} -gt ${N_SAMPLES} ]; then END=${N_SAMPLES}; fi
            if [ ${START} -ge ${N_SAMPLES} ]; then continue; fi

            sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
                --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 \
                --cpus-per-task=16 --mem=128G -t 1-00:00:00 \
                --job-name="rf3-${TASK}-${i}" \
                -o "${LOG_DIR}/rf3_${LABEL}_chunk${i}_%j.out" \
                -e "${LOG_DIR}/rf3_${LABEL}_chunk${i}_%j.err" \
                --wrap="bash -c 'cd ${PROTEINA_DIR} && source .venv/bin/activate && source env.sh && python ${LOBSTER_DIR}/scripts/run_rf3_ff_baseline.py --ff_csv ${CSV} --output_dir ${RF3_OUT}/chunk_${i} --start_idx ${START} --end_idx ${END}'"
        done
        echo "    RF3 submitted (${RF3_N_CHUNKS} chunks) -> ${RF3_OUT}"
    fi
done

echo ""
echo "Phase 2 submitted. Wait for all co-fold jobs to complete, then run:"
echo "  SKIP_PHASE1=1 SKIP_PHASE2=1 EVAL_TAG=${EVAL_TAG} CKPT='${CKPT}' bash slurm/scripts/run_full_eval.sh"

fi  # Phase 2

# ==============================================================================
# Phase 3: Merge co-fold results into eval CSVs
# ==============================================================================
if [[ "${SKIP_PHASE1:-0}" == "1" && "${SKIP_PHASE2:-0}" == "1" ]]; then

echo "=== Phase 3: Merging co-fold results ==="

# Merge Boltz2 results (if they exist)
for LABEL_CSV in \
    "gen_ume_inv_fold:${IF_DIR}/inverse_folding_results.csv:pdb_id" \
    "ligandmpnn_baseline:${LMPNN_DIR}/ligandmpnn_baseline_results.csv:pdb_id" \
    "gen_ume_fwd_fold:${FF_DIR}/forward_folding_results.csv:pdb_id" \
    "gen_ume_cond_gen:${CG_DIR}/conditioned_gen_results.csv:ligand_id"; do

    LABEL="${LABEL_CSV%%:*}"
    REST="${LABEL_CSV#*:}"
    CSV="${REST%%:*}"
    ID_COL="${REST#*:}"
    COFOLD_DIR="${BASE}/boltz2_${LABEL}_${EVAL_TAG}"

    if [ ! -d "${COFOLD_DIR}" ] || [ ! -f "${CSV}" ]; then
        continue
    fi

    MERGED="${CSV%.csv}_with_boltz2.csv"
    cd ${LOBSTER_DIR}
    uv run python -m lobster.cmdline.merge_cofold_results \
        --results_dir "${COFOLD_DIR}/results" \
        --eval_csv "${CSV}" \
        --id_col "${ID_COL}" \
        --output "${MERGED}" \
        --parse_structures \
        --data_dir "${DATA_DIR}"

    echo "  Merged Boltz2: ${MERGED}"
done

# Merge RF3 results (if they exist)
for LABEL_CSV in \
    "gen_ume_inv_fold:${IF_DIR}/inverse_folding_results.csv:pdb_id" \
    "ligandmpnn_baseline:${LMPNN_DIR}/ligandmpnn_baseline_results.csv:pdb_id" \
    "gen_ume_fwd_fold:${FF_DIR}/forward_folding_results.csv:pdb_id" \
    "gen_ume_cond_gen:${CG_DIR}/conditioned_gen_results.csv:ligand_id"; do

    LABEL="${LABEL_CSV%%:*}"
    REST="${LABEL_CSV#*:}"
    CSV="${REST%%:*}"
    RF3_DIR="${BASE}/rf3_${LABEL}_${EVAL_TAG}"

    if [ ! -d "${RF3_DIR}" ] || [ ! -f "${CSV}" ]; then
        continue
    fi

    # Concatenate RF3 chunk results
    RF3_COMBINED="${RF3_DIR}/rf3_results_all.csv"
    cd ${LOBSTER_DIR}
    uv run python -c "
import pandas as pd, glob, os
dfs = [pd.read_csv(f) for f in sorted(glob.glob('${RF3_DIR}/chunk_*/rf3_results.csv'))]
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('${RF3_COMBINED}', index=False)
    print(f'  Merged RF3: {len(combined)} results -> ${RF3_COMBINED}')
else:
    print('  No RF3 chunk results found')
"
done

echo ""
echo "============================================================"
echo " Evaluation pipeline complete for tag: ${EVAL_TAG}"
echo " Results at: ${BASE}/*_${EVAL_TAG}/"
echo "============================================================"

fi  # Phase 3
