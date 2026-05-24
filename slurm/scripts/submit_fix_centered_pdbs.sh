#!/bin/bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/plinder_logs/fix_pdbs_%A.out
#SBATCH -e /cv/scratch/u/lisanzas/plinder_logs/fix_pdbs_%A.err
#SBATCH --job-name=fix-pdbs
#SBATCH -t 02:00:00

set -euo pipefail

source /cv/scratch/u/lisanzas/uv_env/plinder/.venv/bin/activate

echo "=== Regenerating centered PDBs with unique atom names ==="

python /cv/home/lisanzas/lobster/scripts/merge_plinder_metadata_and_extract_ligands.py \
    --processed-dir /cv/scratch/u/lisanzas/plinder_processed/ \
    --ligand-output-dir /cv/scratch/u/lisanzas/plinder_ligands/

echo "=== Done ==="
