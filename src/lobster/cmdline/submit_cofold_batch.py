"""Prepare cofold inputs and submit a SLURM array job for batch local execution.

This script handles two workflows:

1. **From evaluation CSV**: Extract (sequence, smiles) pairs from a Gen-UME
   evaluation CSV (produced by forward_folding, inverse_folding, or conditioned
   generation evaluators).

2. **From raw inputs**: Take a directory of pre-prepared input JSONs.

It writes per-prediction input JSONs, generates a SLURM array job, and submits
it. Each array task runs one cofold prediction on one GPU.

Usage:
    # From an evaluation CSV (forward folding)
    python -m lobster.cmdline.submit_cofold_batch \
        --eval_csv results.csv \
        --sequence_col sequence \
        --smiles_col smiles \
        --id_col pdb_id \
        --output_dir /scratch/cofold_batch_001 \
        --backend protenix \
        --submit

    # From raw posebusters data
    python -m lobster.cmdline.submit_cofold_batch \
        --data_dir /path/to/posebusters/processed/ \
        --raw_data_dir /path/to/posebusters/raw/ \
        --output_dir /scratch/cofold_batch_001 \
        --backend protenix \
        --submit
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def prepare_from_csv(
    csv_path: str,
    output_dir: Path,
    id_col: str = "pdb_id",
    sequence_col: str = "sequence",
    smiles_col: str = "smiles",
) -> int:
    """Extract cofold inputs from an evaluation CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for _, row in df.iterrows():
        sample_id = str(row[id_col])
        sequence = str(row[sequence_col])
        smiles = row.get(smiles_col)
        if smiles is not None and (str(smiles) == "nan" or str(smiles) == ""):
            smiles = None
        else:
            smiles = str(smiles)

        inp = {"id": sample_id, "sequence": sequence, "smiles": smiles}
        with open(input_dir / f"{count}.json", "w") as f:
            json.dump(inp, f)
        count += 1

    print(f"Prepared {count} input files in {input_dir}")
    return count


def prepare_from_data_dir(
    data_dir: str,
    raw_data_dir: str,
    output_dir: Path,
    num_samples: int | None = None,
) -> int:
    """Extract cofold inputs from posebusters processed + raw data directories."""
    import torch

    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    from glob import glob

    protein_files = sorted(glob(os.path.join(data_dir, "*_protein.pt")))
    if num_samples is not None:
        protein_files = protein_files[:num_samples]

    count = 0
    for pf in protein_files:
        pdb_id = os.path.basename(pf).replace("_protein.pt", "")
        protein_data = torch.load(pf, weights_only=False, map_location="cpu")
        seq_tensor = protein_data.get("sequence")
        if seq_tensor is None:
            continue

        aa_map = {
            0: "A",
            1: "R",
            2: "N",
            3: "D",
            4: "C",
            5: "Q",
            6: "E",
            7: "G",
            8: "H",
            9: "I",
            10: "L",
            11: "K",
            12: "M",
            13: "F",
            14: "P",
            15: "S",
            16: "T",
            17: "W",
            18: "Y",
            19: "V",
            20: "X",
        }
        sequence = "".join([aa_map.get(int(s), "X") for s in seq_tensor.tolist()])

        smiles = None
        sdf_path = os.path.join(raw_data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
        if os.path.exists(sdf_path):
            try:
                from rdkit import Chem

                suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
                mol = next(iter(suppl), None)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
            except Exception:
                pass

        inp = {"id": pdb_id, "sequence": sequence, "smiles": smiles}
        with open(input_dir / f"{count}.json", "w") as f:
            json.dump(inp, f)
        count += 1

    print(f"Prepared {count} input files in {input_dir}")
    return count


def write_slurm_script(
    output_dir: Path,
    n_tasks: int,
    backend: str,
    partition: str = "ai4dd-b200",
    account: str = "llm",
    queue: str = "llm",
    time_limit: str = "4:00:00",
    mem: str = "64G",
    protenix_model: str = "protenix_base_default_v1.0.0",
    boltz_model: str = "boltz2",
    protenix_venv: str = "/cv/scratch/u/lisanzas/uv_envs/protenix/.venv",
    boltz_venv: str = "/cv/scratch/u/lisanzas/uv_envs/boltz/.venv",
    max_concurrent: int | None = None,
) -> Path:
    """Write the SLURM array job script."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_base = output_dir / "work"
    work_base.mkdir(parents=True, exist_ok=True)

    script = f"""#!/usr/bin/env bash
#SBATCH --partition {partition}
#SBATCH --account {account}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 8
#SBATCH --mem={mem}
#SBATCH --job-name=cofold_{backend}
#SBATCH --array=0-{n_tasks - 1}%{max_concurrent if max_concurrent else n_tasks}
#SBATCH -o {log_dir}/%A_%a.out
#SBATCH -e {log_dir}/%A_%a.err
#SBATCH -t {time_limit}
#SBATCH -q {queue}

export PROTENIX_VENV="{protenix_venv}"
export BOLTZ_VENV="{boltz_venv}"
export PROTENIX_ROOT_DIR="/cv/scratch/u/lisanzas/protenix_data"

INPUT_DIR="{output_dir / "inputs"}"
RESULTS_DIR="{results_dir}"
WORK_BASE="{work_base}"
BACKEND="{backend}"

TASK_ID=${{SLURM_ARRAY_TASK_ID}}
INPUT_JSON="${{INPUT_DIR}}/${{TASK_ID}}.json"
OUTPUT_JSON="${{RESULTS_DIR}}/${{TASK_ID}}.json"
WORK_DIR="${{WORK_BASE}}/task_${{TASK_ID}}"

if [ ! -f "${{INPUT_JSON}}" ]; then
    echo "Input file not found: ${{INPUT_JSON}}"
    exit 0
fi

# Skip if result already exists AND is complete (has structure field)
if [ -f "${{OUTPUT_JSON}}" ]; then
    # Check the result has a non-null structure (not a partial/failed result)
    HAS_STRUCT=$(python3 -c "import json; d=json.load(open('${{OUTPUT_JSON}}')); print('yes' if d.get('structure') else 'no')" 2>/dev/null || echo "no")
    if [ "${{HAS_STRUCT}}" = "yes" ]; then
        echo "Result already complete: ${{OUTPUT_JSON}}, skipping"
        exit 0
    fi
fi

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.run_cofold_local \\
    --input_json "${{INPUT_JSON}}" \\
    --output_json "${{OUTPUT_JSON}}" \\
    --backend "${{BACKEND}}" \\
    --work_dir "${{WORK_DIR}}" \\
    --protenix_model "{protenix_model}" \\
    --boltz_model "{boltz_model}"

echo "Task ${{TASK_ID}} completed"
"""

    script_path = output_dir / "run_cofold_array.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Prepare and submit batch cofold SLURM array job")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eval_csv", type=str, help="Path to evaluation CSV with sequences")
    source.add_argument("--data_dir", type=str, help="Path to processed posebusters data directory")

    parser.add_argument(
        "--raw_data_dir", type=str, default=None, help="Path to raw data dir (for SDF SMILES extraction)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for batch job")
    parser.add_argument("--backend", type=str, default="protenix", choices=["protenix", "boltz"])

    parser.add_argument("--id_col", type=str, default="pdb_id", help="Column name for sample ID in CSV")
    parser.add_argument("--sequence_col", type=str, default="sequence", help="Column name for sequence in CSV")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Column name for SMILES in CSV")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples")

    parser.add_argument("--partition", type=str, default="ai4dd-b200")
    parser.add_argument("--account", type=str, default="llm")
    parser.add_argument("--queue", type=str, default="llm")
    parser.add_argument("--time_limit", type=str, default="4:00:00", help="Time limit per task")
    parser.add_argument("--mem", type=str, default="64G", help="Memory per task")

    parser.add_argument("--protenix_model", type=str, default="protenix_base_default_v1.0.0")
    parser.add_argument("--boltz_model", type=str, default="boltz2")
    parser.add_argument("--protenix_venv", type=str, default="/cv/scratch/u/lisanzas/uv_envs/protenix/.venv")
    parser.add_argument("--boltz_venv", type=str, default="/cv/scratch/u/lisanzas/uv_envs/boltz/.venv")
    parser.add_argument("--submit", action="store_true", help="Actually submit the SLURM job (otherwise just prepare)")
    parser.add_argument(
        "--max_concurrent", type=int, default=None, help="Max concurrent SLURM array tasks (e.g. 80 for 10 nodes)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_csv:
        n_tasks = prepare_from_csv(
            args.eval_csv,
            output_dir,
            id_col=args.id_col,
            sequence_col=args.sequence_col,
            smiles_col=args.smiles_col,
        )
    else:
        if args.raw_data_dir is None:
            parser.error("--raw_data_dir is required when using --data_dir")
        n_tasks = prepare_from_data_dir(
            args.data_dir,
            args.raw_data_dir,
            output_dir,
            num_samples=args.num_samples,
        )

    if n_tasks == 0:
        print("No tasks to submit")
        sys.exit(0)

    script_path = write_slurm_script(
        output_dir,
        n_tasks,
        args.backend,
        partition=args.partition,
        account=args.account,
        queue=args.queue,
        time_limit=args.time_limit,
        mem=args.mem,
        protenix_model=args.protenix_model,
        boltz_model=args.boltz_model,
        protenix_venv=args.protenix_venv,
        boltz_venv=args.boltz_venv,
        max_concurrent=args.max_concurrent,
    )
    print(f"SLURM script written to {script_path}")
    print(f"Array size: {n_tasks} tasks (0-{n_tasks - 1})")

    if args.submit:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"sbatch failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"\nTo submit:\n  sbatch {script_path}")


if __name__ == "__main__":
    main()
