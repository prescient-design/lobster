#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/sweep_conditioned_gen/%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/sweep_conditioned_gen/%A_%a.err
#SBATCH --mem=128G
#SBATCH --job-name=sweep_cg
#SBATCH -t 4:00:00
#SBATCH -q llm
#SBATCH --array=0-7

# Ligand-conditioned generation hyperparameter sweep
# Each array task tests a different configuration.
# Uses 3 ligands x 5 designs = 15 sequences per config for quick turnaround.

CKPT="${CKPT:-/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs//2026-03-11T13-22-20/last.ckpt}"
DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap/"
RAW_DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/posebusters_benchmark_set/"
BASE_OUT="/cv/home/lisanzas/conditioned_gen_sweep"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/sweep_conditioned_gen

# Sweep configurations: name|ligand_context_mode|temp_seq|temp_struc|stoch_seq|stoch_struc|temp_lig|stoch_lig|nsteps|sched_seq|sched_struc|sched_lig_atom|sched_lig_struc
CONFIGS=(
    # 0: baseline (current defaults)
    "baseline|atom_bond_only|0.15279667854390633|0.18605909386731256|10|10|0.5819150856331732|20|100|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 1: structure_tokens mode (ligand spatial info as context)
    "struc_tokens|structure_tokens|0.15279667854390633|0.18605909386731256|10|10|0.5819150856331732|20|100|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 2: lower ligand temperature (tighter ligand placement)
    "low_lig_temp|atom_bond_only|0.15279667854390633|0.18605909386731256|10|10|0.1|5|100|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 3: lower ligand temp + structure_tokens
    "struc_low_lig|structure_tokens|0.15279667854390633|0.18605909386731256|10|10|0.1|5|100|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 4: low everything (tight generation all around)
    "low_all_temp|atom_bond_only|0.05|0.05|5|5|0.1|5|100|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 5: structure_tokens + low all temps
    "struc_low_all|structure_tokens|0.05|0.05|5|5|0.1|5|100|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 6: more diffusion steps (200) with structure_tokens
    "struc_200steps|structure_tokens|0.15279667854390633|0.18605909386731256|10|10|0.5819150856331732|20|200|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
    # 7: structure_tokens + low ligand temp + 200 steps
    "struc_low_lig_200|structure_tokens|0.15279667854390633|0.18605909386731256|10|10|0.1|5|200|LinearInferenceSchedule|PowerInferenceSchedule|PowerInferenceSchedule|LinearInferenceSchedule"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r NAME CTX_MODE T_SEQ T_STRUC S_SEQ S_STRUC T_LIG S_LIG NSTEPS SCHED_SEQ SCHED_STRUC SCHED_LIG_ATOM SCHED_LIG_STRUC <<< "$CFG"

OUT_DIR="${BASE_OUT}/${NAME}"
mkdir -p "${OUT_DIR}"

echo "=== Config ${SLURM_ARRAY_TASK_ID}: ${NAME} ==="
echo "  ligand_context_mode: ${CTX_MODE}"
echo "  temperature_seq: ${T_SEQ}"
echo "  temperature_struc: ${T_STRUC}"
echo "  stochasticity_seq: ${S_SEQ}"
echo "  stochasticity_struc: ${S_STRUC}"
echo "  temperature_ligand: ${T_LIG}"
echo "  stochasticity_ligand: ${S_LIG}"
echo "  nsteps: ${NSTEPS}"
echo "  schedules: ${SCHED_SEQ} / ${SCHED_STRUC} / ${SCHED_LIG_ATOM} / ${SCHED_LIG_STRUC}"

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.evaluate_ligand_conditioned_protein_generation \
    --checkpoint "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --output "conditioned_gen_results.csv" \
    --structure_path "${OUT_DIR}" \
    --length 100 \
    --num_designs 5 \
    --num_samples 5 \
    --nsteps "${NSTEPS}" \
    --temperature_seq "${T_SEQ}" \
    --temperature_struc "${T_STRUC}" \
    --stochasticity_seq "${S_SEQ}" \
    --stochasticity_struc "${S_STRUC}" \
    --temperature_ligand "${T_LIG}" \
    --stochasticity_ligand "${S_LIG}" \
    --ligand_context_mode "${CTX_MODE}" \
    --inference_schedule_seq "${SCHED_SEQ}" \
    --inference_schedule_struc "${SCHED_STRUC}" \
    --inference_schedule_ligand_atom "${SCHED_LIG_ATOM}" \
    --inference_schedule_ligand_struc "${SCHED_LIG_STRUC}" \
    --save_structures

echo "=== Done: ${NAME} ==="
echo "Results: ${OUT_DIR}"
