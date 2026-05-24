#!/bin/bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/plinder_logs/extract_ligands_%A.out
#SBATCH -e /cv/scratch/u/lisanzas/plinder_logs/extract_ligands_%A.err
#SBATCH --job-name=plinder-lig
#SBATCH -t 04:00:00

set -euo pipefail

source /cv/scratch/u/lisanzas/uv_env/plinder/.venv/bin/activate

python /cv/home/lisanzas/lobster/scripts/merge_plinder_metadata_and_extract_ligands.py \
    --processed-dir /cv/scratch/u/lisanzas/plinder_processed/ \
    --ligand-output-dir /cv/scratch/u/lisanzas/plinder_ligands/

echo "=== Done ==="
