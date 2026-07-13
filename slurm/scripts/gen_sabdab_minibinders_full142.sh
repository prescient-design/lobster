#!/usr/bin/env bash
# Full 142-target mini-binder generation (whole sabdab_ff validation set). All sampling
# hyperparameters are env-overridable for sweeps (defaults = the baseline binder recipe:
# nsteps=200, temp_seq=0.2732, temp_struc=0.3164, stoch_seq=20, stoch_struc=60, epitope ON,
# 10 designs/target, Linear structure schedule). Pass MODEL, CKPT via --export.
#
# Sweep knobs (env): NSTEPS, TEMP_SEQ, TEMP_STRUC, STOCH_SEQ, STOCH_STRUC, N_DESIGNS,
#   CFG_WEIGHT, SCHED_STRUC (e.g. PowerInferenceSchedule), SCHED_EXP (e.g. 1.5),
#   USE_EPI, TEMPLATE_TARGET, HOTSPOT_FRAC (+HOTSPOT_MODE=spatial), REPO_DIR.
# Run the distogram (or any complex_infra) checkpoint by passing REPO_DIR=<this worktree>.
#SBATCH --partition gpu
#SBATCH --account ai4dd
#SBATCH -q ai4dd_normal
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/gen_sabdab_minibinders_full142/%x_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/gen_sabdab_minibinders_full142/%x_%A_%a.err
#SBATCH -t 0-02:00:00
#SBATCH --array=0-141%24
set -uo pipefail

export HF_HOME=/cv/scratch/u/lisanzas/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/cv/scratch/u/lisanzas/.cache/huggingface/hub
export TORCH_HOME=/cv/scratch/u/lisanzas/.cache/torch
export TMPDIR=/cv/scratch/u/lisanzas/tmp
mkdir -p "$TMPDIR" /cv/scratch/u/lisanzas/slurm_logs/gen_sabdab_minibinders_full142
cd "${REPO_DIR:-/cv/home/lisanzas/lobster}"

CSV=/cv/scratch/u/lisanzas/denovo_dataset/binder/denovo/sabdab_ff/targets/sabdab_ff_targets.csv
MODEL="${MODEL:?set MODEL}"
CKPT="${CKPT:?set CKPT}"
OUTBASE=/cv/scratch/u/lisanzas/denovo_dataset/binder/denovo/sabdab_ff/minibinder/$MODEL/gen

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$CSV")
[ -z "$ROW" ] && { echo "no row for task $SLURM_ARRAY_TASK_ID"; exit 0; }
TARGET_ID=$(echo "$ROW" | cut -d, -f1)
ANTIGEN_PDB=$(echo "$ROW" | cut -d, -f2)
HOTSPOTS=$(echo "$ROW" | cut -d, -f6)      # curated 3 hotspots (col6)
FULL_EPI=$(echo "$ROW" | cut -d, -f7)      # full interface epitope residues (col7)
# HOTSPOT_FRAC (percent) titrates how much of the FULL epitope to condition on. Unset -> curated 3.
if [ -n "${HOTSPOT_FRAC:-}" ] && [ "${HOTSPOT_MODE:-spread}" = "spatial" ]; then
  EPI=$(python3 scripts/_select_hotspots_spatial.py --pdb "$ANTIGEN_PDB" --chain A \
        --epitope "$FULL_EPI" --frac "$HOTSPOT_FRAC")
elif [ -n "${HOTSPOT_FRAC:-}" ]; then
  EPI=$(FULL_EPI="$FULL_EPI" FRAC="$HOTSPOT_FRAC" python3 -c '
import os
xs=[int(v) for v in os.environ["FULL_EPI"].split()]
k=max(1,round(len(xs)*float(os.environ["FRAC"])/100.0))
if k>=len(xs): sel=xs
elif k==1: sel=[xs[len(xs)//2]]
else: sel=[xs[round(i*(len(xs)-1)/(k-1))] for i in range(k)]
print("["+",".join(map(str,sorted(set(sel))))+"]")')
else
  EPI="[$(echo "$HOTSPOTS" | tr ' ' ',')]"
fi

# Optional front-loaded structure schedule (Power exponent>1 concentrates steps at low t).
SCHED_ARGS=""
[ -n "${SCHED_STRUC:-}" ] && SCHED_ARGS="+generation.inference_schedule_struc=${SCHED_STRUC}"
[ -n "${SCHED_EXP:-}" ] && SCHED_ARGS="$SCHED_ARGS +generation.schedule_exponent=${SCHED_EXP}"

echo "=== MODEL=$MODEL $TARGET_ID nsteps=${NSTEPS:-200} cfg=${CFG_WEIGHT:-1.0} tstruc=${TEMP_STRUC:-0.3164} sstruc=${STOCH_STRUC:-60} sched=${SCHED_STRUC:-Linear}/${SCHED_EXP:-} hotspots=$EPI ==="
uv run python -m lobster.cmdline.generate \
  --config-name experiment/research/generate_binder_design \
  model.ckpt_path="$CKPT" \
  generation.input_structures="$ANTIGEN_PDB" \
  generation.target_chain=A \
  generation.epitope_indices="$EPI" \
  generation.use_epitope_conditioning="${USE_EPI:-true}" \
  +generation.template_target="${TEMPLATE_TARGET:-false}" \
  +generation.cfg_weight="${CFG_WEIGHT:-1.0}" \
  generation.n_designs_per_structure="${N_DESIGNS:-10}" \
  generation.binder_length=100 \
  generation.nsteps="${NSTEPS:-200}" \
  generation.temperature_seq="${TEMP_SEQ:-0.27315634404739075}" \
  generation.temperature_struc="${TEMP_STRUC:-0.31640411575109995}" \
  generation.stochasticity_seq="${STOCH_SEQ:-20}" \
  generation.stochasticity_struc="${STOCH_STRUC:-60}" \
  generation.use_esmfold=false \
  ${SCHED_ARGS} \
  seed=101 \
  output_dir="$OUTBASE/$TARGET_ID"
echo "DONE: $TARGET_ID ($?)"
