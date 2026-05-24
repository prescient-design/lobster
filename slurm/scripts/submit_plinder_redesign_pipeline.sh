#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PLINDer LigandMPNN Redesign → RF3 Evaluation → Filter Pipeline
#
# Phase 1: LigandMPNN redesign of ~46,944 PLINDer complexes (≤512 res, 10 designs each)
# Phase 2: RF3 refolding + evaluation of all ~469K designs (SLURM array)
# Phase 3: Merge RF3 results
# Phase 4: Filter by Proteina-Complexa RF3 success criteria
# Phase 5: Convert passing designs to Gen-UME .pt with alternative_sequences
#
# Usage:
#   bash slurm/scripts/submit_plinder_redesign_pipeline.sh
#
# To run a specific phase:
#   PHASE=2 bash slurm/scripts/submit_plinder_redesign_pipeline.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOBSTER_DIR="/cv/home/lisanzas/lobster"
PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"

DATA_DIR="/cv/scratch/u/lisanzas/plinder_processed/train/"
BASE_DIR="/cv/scratch/u/lisanzas/evaluations/plinder_redesign_${EVAL_TIMESTAMP}"
CHUNK_DIR="${BASE_DIR}/chunks"
MERGED_CSV="${BASE_DIR}/plinder_ligandmpnn_redesigns.csv"
RF3_CHUNK_DIR="${BASE_DIR}/rf3_chunks"
RF3_MERGED_CSV="${BASE_DIR}/plinder_redesigns_with_rf3.csv"
PASSING_CSV="${BASE_DIR}/plinder_redesigns_passing.csv"
PT_OUTPUT_DIR="/cv/scratch/u/lisanzas/plinder_redesigned/train/"

NUM_DESIGNS=10
MAX_PROTEIN_LENGTH=512
TEMPERATURE=0.1
COMPLEXES_PER_TASK=500
REDESIGNS_PER_RF3_TASK=100

# RF3 filter thresholds (Proteina-Complexa standard)
MAX_MIN_IPAE=2.0            # min_ipAE * 31 < 2.0 (normalized value, threshold in Å)
MAX_BINDER_SCRMSD=2.0       # binder_scRMSD_ca < 2.0 Å
MAX_LIGAND_SCRMSD=5.0       # ligand_scRMSD_aligned_allatom < 5.0 Å

PHASE="${PHASE:-all}"

echo "============================================================"
echo " PLINDer LigandMPNN Redesign Pipeline (RF3 Filtering)"
echo " Timestamp: ${EVAL_TIMESTAMP}"
echo " Base dir:  ${BASE_DIR}"
echo " Phase:     ${PHASE}"
echo "============================================================"
echo ""

mkdir -p "${BASE_DIR}" "${CHUNK_DIR}" "${RF3_CHUNK_DIR}"

# ---- Count total complexes ----
N_TOTAL=$(ls "${DATA_DIR}" | grep '_protein\.pt$' | wc -l)
echo "Total protein files: ${N_TOTAL}"
N_TASKS_P1=$(( (N_TOTAL + COMPLEXES_PER_TASK - 1) / COMPLEXES_PER_TASK ))
echo "Phase 1 SLURM tasks: ${N_TASKS_P1} (${COMPLEXES_PER_TASK} complexes/task)"
echo ""

# ============================================================================
# Phase 1: LigandMPNN Redesign (SLURM array, CPU-only)
# ============================================================================
if [[ "${PHASE}" == "all" || "${PHASE}" == "1" ]]; then

PHASE1_SCRIPT="${BASE_DIR}/run_redesign_array.sh"

cat > "${PHASE1_SCRIPT}" <<'SLURM_EOF'
#!/usr/bin/env bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=32G
SLURM_EOF

cat >> "${PHASE1_SCRIPT}" <<SLURM_ARGS
#SBATCH --array=0-$((N_TASKS_P1 - 1))
#SBATCH -o ${BASE_DIR}/logs/phase1_%A_%a.out
#SBATCH -e ${BASE_DIR}/logs/phase1_%A_%a.err
#SBATCH -t 4:00:00
#SBATCH -q llm
#SBATCH --job-name=plinder_lmpnn
SLURM_ARGS

cat >> "${PHASE1_SCRIPT}" <<SLURM_BODY

TASK_ID=\${SLURM_ARRAY_TASK_ID}
START_IDX=\$(( TASK_ID * ${COMPLEXES_PER_TASK} ))
END_IDX=\$(( START_IDX + ${COMPLEXES_PER_TASK} ))
OUTPUT_CSV="${CHUNK_DIR}/chunk_\${TASK_ID}.csv"

cd ${LOBSTER_DIR}

uv run python scripts/plinder_ligandmpnn_redesign.py \\
    --data_dir "${DATA_DIR}" \\
    --output_csv "\${OUTPUT_CSV}" \\
    --num_designs ${NUM_DESIGNS} \\
    --temperature ${TEMPERATURE} \\
    --max_protein_length ${MAX_PROTEIN_LENGTH} \\
    --start_idx \${START_IDX} \\
    --end_idx \${END_IDX}

echo "Task \${TASK_ID} completed: \${OUTPUT_CSV}"
SLURM_BODY

chmod +x "${PHASE1_SCRIPT}"
mkdir -p "${BASE_DIR}/logs"

JOB_P1=$(sbatch --parsable "${PHASE1_SCRIPT}")
echo "[Phase 1] LigandMPNN redesign submitted: Job ${JOB_P1}"
echo "          Array: 0-$((N_TASKS_P1 - 1)) (${COMPLEXES_PER_TASK} complexes/task)"
echo "          Output: ${CHUNK_DIR}/chunk_*.csv"
echo ""

fi  # Phase 1

# ============================================================================
# Phase 1b: Merge chunk CSVs (run after Phase 1 completes)
# ============================================================================
if [[ "${PHASE}" == "1b" ]]; then

echo "[Phase 1b] Merging chunk CSVs..."
cd "${LOBSTER_DIR}"
uv run python -c "
import pandas as pd
from glob import glob
chunks = sorted(glob('${CHUNK_DIR}/chunk_*.csv'))
print(f'Merging {len(chunks)} chunk files...')
dfs = [pd.read_csv(c) for c in chunks]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('${MERGED_CSV}', index=False)
n_complexes = merged['system_id'].nunique()
print(f'Merged: {len(merged)} designs from {n_complexes} complexes -> ${MERGED_CSV}')
"
echo ""

fi  # Phase 1b

# ============================================================================
# Phase 2: RF3 Evaluation (SLURM array, 1 GPU per task)
# ============================================================================
if [[ "${PHASE}" == "all" || "${PHASE}" == "2" ]]; then

if [ ! -f "${MERGED_CSV}" ]; then
    echo "ERROR: ${MERGED_CSV} not found. Run Phase 1b first."
    exit 1
fi

N_REDESIGNS=$(cd "${LOBSTER_DIR}" && uv run python -c "import pandas as pd; print(len(pd.read_csv('${MERGED_CSV}')))")
N_TASKS_P2=$(( (N_REDESIGNS + REDESIGNS_PER_RF3_TASK - 1) / REDESIGNS_PER_RF3_TASK ))
echo "Total redesigns: ${N_REDESIGNS}"
echo "Phase 2 SLURM tasks: ${N_TASKS_P2} (${REDESIGNS_PER_RF3_TASK} redesigns/task)"

PHASE2_SCRIPT="${BASE_DIR}/run_rf3_array.sh"

cat > "${PHASE2_SCRIPT}" <<'SLURM_EOF'
#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 8
#SBATCH --mem=64G
SLURM_EOF

cat >> "${PHASE2_SCRIPT}" <<SLURM_ARGS
#SBATCH --array=0-$((N_TASKS_P2 - 1))
#SBATCH -o ${BASE_DIR}/logs/phase2_%A_%a.out
#SBATCH -e ${BASE_DIR}/logs/phase2_%A_%a.err
#SBATCH -t 8:00:00
#SBATCH -q llm
#SBATCH --job-name=plinder_rf3
SLURM_ARGS

cat >> "${PHASE2_SCRIPT}" <<SLURM_BODY

TASK_ID=\${SLURM_ARRAY_TASK_ID}
START_IDX=\$(( TASK_ID * ${REDESIGNS_PER_RF3_TASK} ))
END_IDX=\$(( START_IDX + ${REDESIGNS_PER_RF3_TASK} ))
OUTPUT_CSV="${RF3_CHUNK_DIR}/rf3_chunk_\${TASK_ID}.csv"
RF3_OUT="${BASE_DIR}/rf3_outputs/task_\${TASK_ID}"

export RF3_CKPT_PATH="${PROTEINA_DIR}/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt"
export RF3_PATH="${PROTEINA_DIR}/.venv/bin/rf3"

cd ${PROTEINA_DIR}
source .venv/bin/activate

python ${LOBSTER_DIR}/scripts/run_rf3_eval_plinder.py \\
    --redesign_csv "${MERGED_CSV}" \\
    --data_dir "${DATA_DIR}" \\
    --output_csv "\${OUTPUT_CSV}" \\
    --rf3_out_dir "\${RF3_OUT}" \\
    --start_idx \${START_IDX} \\
    --end_idx \${END_IDX}

echo "Task \${TASK_ID} completed: \${OUTPUT_CSV}"
SLURM_BODY

chmod +x "${PHASE2_SCRIPT}"

JOB_P2=$(sbatch --parsable "${PHASE2_SCRIPT}")
echo "[Phase 2] RF3 evaluation submitted: Job ${JOB_P2}"
echo "          Array: 0-$((N_TASKS_P2 - 1)) (${REDESIGNS_PER_RF3_TASK} redesigns/task)"
echo "          Output: ${RF3_CHUNK_DIR}/rf3_chunk_*.csv"
echo ""

fi  # Phase 2

# ============================================================================
# Phase 3: Merge RF3 results
# ============================================================================
if [[ "${PHASE}" == "3" ]]; then

echo "[Phase 3] Merging RF3 result chunks..."
cd "${LOBSTER_DIR}"
uv run python -c "
import pandas as pd
from glob import glob

chunks = sorted(glob('${RF3_CHUNK_DIR}/rf3_chunk_*.csv'))
print(f'Merging {len(chunks)} RF3 chunk files...')
dfs = [pd.read_csv(c) for c in chunks]
rf3_results = pd.concat(dfs, ignore_index=True)

redesigns = pd.read_csv('${MERGED_CSV}')
redesigns['name'] = redesigns['system_id'] + '__d' + redesigns['design_idx'].astype(str)
merged = redesigns.merge(rf3_results, on='name', how='left', suffixes=('', '_rf3'))

# Clean up duplicate columns
for col in ['system_id_rf3', 'design_idx_rf3']:
    if col in merged.columns:
        merged.drop(columns=[col], inplace=True)

merged.to_csv('${RF3_MERGED_CSV}', index=False)
n_with_rf3 = merged.filter(like='rf3_').notna().any(axis=1).sum()
print(f'Merged: {len(merged)} rows ({n_with_rf3} with RF3 results) -> ${RF3_MERGED_CSV}')

for col in ['rf3_min_ipAE', 'rf3_plddt', 'rf3_binder_scRMSD_ca', 'rf3_ligand_scRMSD_aligned_allatom']:
    if col in merged.columns:
        vals = pd.to_numeric(merged[col], errors='coerce').dropna()
        if len(vals) > 0:
            print(f'  {col}: mean={vals.mean():.4f}, median={vals.median():.4f}')
"
echo ""

fi  # Phase 3

# ============================================================================
# Phase 4: Filter by RF3 criteria (Proteina-Complexa standard)
# ============================================================================
if [[ "${PHASE}" == "4" ]]; then

echo "[Phase 4] Filtering by Proteina-Complexa RF3 criteria..."

cd "${LOBSTER_DIR}"
uv run python scripts/filter_plinder_redesigns.py \
    --input_csv "${RF3_MERGED_CSV}" \
    --output_csv "${BASE_DIR}/plinder_redesigns_passing.csv" \
    --max_min_ipae ${MAX_MIN_IPAE} \
    --max_binder_scrmsd ${MAX_BINDER_SCRMSD} \
    --max_ligand_scrmsd ${MAX_LIGAND_SCRMSD}

echo ""
echo "Also saving best-per-complex version..."
uv run python scripts/filter_plinder_redesigns.py \
    --input_csv "${RF3_MERGED_CSV}" \
    --output_csv "${BASE_DIR}/plinder_redesigns_passing_best.csv" \
    --max_min_ipae ${MAX_MIN_IPAE} \
    --max_binder_scrmsd ${MAX_BINDER_SCRMSD} \
    --max_ligand_scrmsd ${MAX_LIGAND_SCRMSD} \
    --best_per_complex

echo ""

fi  # Phase 4

# ============================================================================
# Phase 5: Convert passing designs to Gen-UME .pt with alternative_sequences
# ============================================================================
if [[ "${PHASE}" == "5" ]]; then

echo "[Phase 5] Converting passing designs to Gen-UME .pt format..."
echo "          Input:  ${PASSING_CSV}"
echo "          Source: ${DATA_DIR}"
echo "          Output: ${PT_OUTPUT_DIR}"

cd "${LOBSTER_DIR}"
uv run python scripts/convert_redesigns_to_pt.py \
    --passing_csv "${PASSING_CSV}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${PT_OUTPUT_DIR}" \
    --include_gt_sequence

echo ""
echo "Done. Redesigned .pt files at: ${PT_OUTPUT_DIR}"
echo "Use with structure_backbone_aa_tokenizer_alt_seq_transform.yaml to"
echo "randomly sample from alternative_sequences during training."

fi  # Phase 5

echo "============================================================"
echo " Pipeline setup complete."
echo " Base directory: ${BASE_DIR}"
echo ""
echo " Execution order:"
echo "   PHASE=1  → LigandMPNN redesign (SLURM array)"
echo "   PHASE=1b → Merge redesign chunks"
echo "   PHASE=2  → RF3 evaluation (SLURM array, 1 GPU/task)"
echo "   PHASE=3  → Merge RF3 results"
echo "   PHASE=4  → Filter by Proteina-Complexa criteria"
echo "   PHASE=5  → Convert passing designs to Gen-UME .pt"
echo "============================================================"
