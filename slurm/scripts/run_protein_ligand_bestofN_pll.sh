#!/usr/bin/env bash
# Best-of-N PLL selection on PoseBusters for the 4-modality Gen-UME protein-
# ligand model. Wraps the three task drivers
# (forward_fold / inverse_fold / conditioned_gen) and submits one SLURM job
# per (task, checkpoint) combo. Hyperparameters mirror the production eval
# scripts (eval_gen_ume_protein_ligand_{forward,inverse}_folding.sh and
# eval_gen_ume_protein_ligand_conditioned_generation.sh) so that the
# generated samples being NLL-scored are produced under the same regime
# as the existing benchmark.
#
# Usage:
#   bash slurm/scripts/run_protein_ligand_bestofN_pll.sh                 # all 6 (3 tasks x 2 ckpts)
#   bash slurm/scripts/run_protein_ligand_bestofN_pll.sh ff              # all 2 ckpts on FF
#   bash slurm/scripts/run_protein_ligand_bestofN_pll.sh if all          # IF on ALL ckpt
#   bash slurm/scripts/run_protein_ligand_bestofN_pll.sh cg plinder      # CG on PLINDER ckpt
#   bash slurm/scripts/run_protein_ligand_bestofN_pll.sh --worker        # (sbatch internal)
#
# Env overrides:
#   N_CANDIDATES (default 1   for E0; set to 10 for E1 FF, 30 for E2 IF / E3 CG)
#   K_DRAWS      (default 32)
#   MAX_TARGETS  (default unlimited)
#   NSTEPS       (default 100, matches production eval scripts)

set -euo pipefail

CKPT_ALL="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_gen_ume_all_latest/last.ckpt"
CKPT_PLINDER="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_gen_ume_plinder_latest/last.ckpt"

SOURCE_DIR="${SOURCE_DIR:-/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cv/scratch/u/lisanzas/evaluations/pll_correlation_report_protein_ligand}"
SUBSET_TAG="${SUBSET_TAG:-}"

N_CANDIDATES="${N_CANDIDATES:-1}"
K_DRAWS="${K_DRAWS:-32}"
MAX_TARGETS="${MAX_TARGETS:-}"
# Per-task default nsteps mirror the production eval scripts:
#   FF, IF: nsteps=100   (eval_gen_ume_protein_ligand_{forward,inverse}_folding.sh)
#   CG    : nsteps=200   (ReST i1 "optimized params" run, scripts/eval_cg_boltz_checkpoint.py)
# Override via env: NSTEPS_FF, NSTEPS_IF, NSTEPS_CG, or NSTEPS (applies to all).
NSTEPS_FF="${NSTEPS_FF:-${NSTEPS:-100}}"
NSTEPS_IF="${NSTEPS_IF:-${NSTEPS:-100}}"
NSTEPS_CG="${NSTEPS_CG:-${NSTEPS:-200}}"
# CG length: 'gt' to match per-target GT length (allows in-loop TM-to-GT correlation),
# or an integer (e.g. 100) to mirror the ReST i1 CG benchmark — TM/RMSD/AAR vs GT
# are then skipped (downstream Boltz2 cofold is required for quality metrics).
CG_LENGTH="${CG_LENGTH:-gt}"

###############################################################################
# Worker
###############################################################################
if [ "${1:-}" = "--worker" ]; then
    : "${TASK:?TASK not set}"
    : "${CKPT:?CKPT not set}"
    : "${SOURCE_DIR:?SOURCE_DIR not set}"
    : "${OUTPUT_DIR:?OUTPUT_DIR not set}"
    : "${VARIANT:?VARIANT not set}"

    cd /cv/home/lisanzas/lobster
    : "${TASK_NSTEPS:?TASK_NSTEPS not set}"
    : "${CG_LENGTH:?CG_LENGTH not set}"
    echo "[worker] TASK=${TASK} VARIANT=${VARIANT}"
    echo "[worker] CKPT=${CKPT}"
    echo "[worker] SOURCE_DIR=${SOURCE_DIR}"
    echo "[worker] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[worker] N=${N_CANDIDATES}  K=${K_DRAWS}  NSTEPS=${TASK_NSTEPS}  MAX_TARGETS=${MAX_TARGETS:-(all)}  CG_LENGTH=${CG_LENGTH}"
    echo "[worker] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  node=$(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    extra=()
    if [ -n "${MAX_TARGETS}" ]; then
        extra+=(--max-targets "${MAX_TARGETS}")
    fi

    case "${TASK}" in
        ff)
            # Production FF defaults (matches evaluate_protein_ligand_forward_folding.py):
            # temperature_seq=0.5, temperature_struc=0.5, temperature_ligand=0.5,
            # stochasticity_seq=20, stochasticity_struc=20, stochasticity_ligand=20,
            # schedule_seq=Log, schedule_struc=Linear, ligand_context_mode=structure_tokens,
            # nsteps=100.
            uv run python scripts/forward_fold_bestofN_pll_ligand.py \
                --source-data-dir "${SOURCE_DIR}" \
                --ckpt "${CKPT}" \
                --output-dir "${OUTPUT_DIR}" \
                --N "${N_CANDIDATES}" --K "${K_DRAWS}" --nsteps "${TASK_NSTEPS}" \
                "${extra[@]}"
            ;;
        if)
            # Production IF defaults (matches evaluate_protein_ligand_inverse_folding.py): same as FF.
            uv run python scripts/inverse_fold_bestofN_pll_ligand.py \
                --source-data-dir "${SOURCE_DIR}" \
                --ckpt "${CKPT}" \
                --output-dir "${OUTPUT_DIR}" \
                --N "${N_CANDIDATES}" --K "${K_DRAWS}" --nsteps "${TASK_NSTEPS}" \
                "${extra[@]}"
            ;;
        cg)
            # Production CG defaults (matches scripts/eval_cg_boltz_checkpoint.py — the ReST
            # i1 "optimized params" run, nsteps=200):
            # temperature_seq=0.153, temperature_struc=0.05, temperature_ligand=0.1,
            # stochasticity_seq=20, stochasticity_struc=20, stochasticity_ligand=5,
            # schedule_seq=Linear, schedule_struc=Power,
            # schedule_ligand_atom=Power, schedule_ligand_struc=Linear,
            # ligand_context_mode=atom_bond_only, nsteps=200.
            # All set as the script's argparse defaults.
            uv run python scripts/conditioned_gen_bestofN_pll_ligand.py \
                --source-data-dir "${SOURCE_DIR}" \
                --ckpt "${CKPT}" \
                --output-dir "${OUTPUT_DIR}" \
                --N "${N_CANDIDATES}" --K "${K_DRAWS}" --nsteps "${TASK_NSTEPS}" \
                --length "${CG_LENGTH}" \
                "${extra[@]}"
            ;;
        *)
            echo "Unknown TASK: ${TASK}" >&2
            exit 2
            ;;
    esac
    exit $?
fi

###############################################################################
# Submit
###############################################################################
TASK_TARGET="${1:-all}"
CKPT_TARGET="${2:-all}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/bestofN_pl_pll

submit_one() {
    local task="$1"
    local variant="$2"
    local ckpt="$3"

    if [ ! -f "${ckpt}" ]; then
        echo "ERROR: CKPT not found for variant='${variant}': ${ckpt}" >&2
        return 1
    fi

    local task_nsteps
    case "${task}" in
        ff) task_nsteps="${NSTEPS_FF}" ;;
        if) task_nsteps="${NSTEPS_IF}" ;;
        cg) task_nsteps="${NSTEPS_CG}" ;;
    esac

    local suffix=""
    if [ "${N_CANDIDATES}" != "1" ]; then
        suffix="_N${N_CANDIDATES}"
    fi
    local task_dir="${task}${SUBSET_TAG:+_${SUBSET_TAG}}"
    local output_dir="${OUTPUT_ROOT}/${variant}/${task_dir}/full${suffix}"
    mkdir -p "${output_dir}"

    echo "Submitting task='${task}' variant='${variant}' N=${N_CANDIDATES} K=${K_DRAWS} nsteps=${task_nsteps}"
    if [ "${task}" = "cg" ]; then
        echo "  CG length: ${CG_LENGTH}"
    fi
    echo "  ckpt:   ${ckpt}"
    echo "  output: ${output_dir}"

    sbatch \
        --partition=ai4dd-b200 \
        --account=llm \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres=gpu:b200:1 \
        --cpus-per-task=16 \
        --mem=128G \
        --time=1-00:00:00 \
        --qos=llm \
        --job-name="bestofN_pl_${task}_${variant}_N${N_CANDIDATES}" \
        --output="/cv/scratch/u/lisanzas/slurm_logs/bestofN_pl_pll/%J_${task}_${variant}.out" \
        --error="/cv/scratch/u/lisanzas/slurm_logs/bestofN_pl_pll/%J_${task}_${variant}.err" \
        --export="ALL,TASK=${task},VARIANT=${variant},CKPT=${ckpt},SOURCE_DIR=${SOURCE_DIR},OUTPUT_DIR=${output_dir},N_CANDIDATES=${N_CANDIDATES},K_DRAWS=${K_DRAWS},TASK_NSTEPS=${task_nsteps},CG_LENGTH=${CG_LENGTH},MAX_TARGETS=${MAX_TARGETS}" \
        "$0" --worker
}

ckpts_to_run=()
case "${CKPT_TARGET}" in
    all|both)
        ckpts_to_run+=("all" "plinder")
        ;;
    all_only|gen_ume_all|allckpt)
        ckpts_to_run+=("all")
        ;;
    plinder|plinder_only)
        ckpts_to_run+=("plinder")
        ;;
    *)
        echo "Unknown CKPT_TARGET: ${CKPT_TARGET}" >&2
        exit 2
        ;;
esac

tasks_to_run=()
case "${TASK_TARGET}" in
    all)
        tasks_to_run+=("ff" "if" "cg")
        ;;
    ff|if|cg)
        tasks_to_run+=("${TASK_TARGET}")
        ;;
    *)
        echo "Unknown TASK_TARGET: ${TASK_TARGET}" >&2
        echo "Usage: $0 [all|ff|if|cg] [all|allckpt|plinder]" >&2
        exit 2
        ;;
esac

for task in "${tasks_to_run[@]}"; do
    for variant in "${ckpts_to_run[@]}"; do
        if [ "${variant}" = "all" ]; then ckpt="${CKPT_ALL}"; else ckpt="${CKPT_PLINDER}"; fi
        submit_one "${task}" "${variant}" "${ckpt}"
    done
done

echo "Done."
