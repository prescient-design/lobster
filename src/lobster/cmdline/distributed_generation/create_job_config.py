#!/usr/bin/env python3
"""
Helper script to generate wandb_config.yaml for distributed generation.
Creates job distribution configurations for different sample counts and parallelization strategies.

Supports two modes:
1. Unconditional: Distribute samples across jobs (e.g., 100 samples → 20 jobs × 5)
2. Structure-based (inverse/forward folding): Distribute input structure files across jobs
"""

import glob
import yaml
from pathlib import Path


def create_job_config(
    total_samples: int,
    samples_per_job: int,
    base_config_path: str = "src/lobster/hydra_config/experiment/generate_unconditional.yaml",
    output_file: str = "src/lobster/cmdline/distributed_generation/wandb_config.yaml",
    lengths: list[int] | None = None,
):
    """
    Generate job distribution config.

    Args:
        total_samples: Total number of samples to generate
        samples_per_job: Number of samples per job
        base_config_path: Path to base config file
        output_file: Output file name
        lengths: Optional list of lengths for multi-length generation
    """
    num_jobs = (total_samples + samples_per_job - 1) // samples_per_job

    print(f"Creating config for {total_samples} samples")
    print(f"Samples per job: {samples_per_job}")
    print(f"Number of jobs: {num_jobs}")

    job_ids = list(range(num_jobs))

    config = {
        "program": "src/lobster/cmdline/distributed_generate.py",
        "method": "grid",
        "project": "lobster-distributed-generation",
        "entity": "prescient-design",
        "metric": {"name": "job_completed", "goal": "maximize"},
        "parameters": {
            "base_config_path": {"value": base_config_path},
            "job_id": {"values": job_ids},
            "samples_per_job": {"value": samples_per_job},
            "total_samples": {"value": total_samples},
        },
        "command": ["${env}", "python", "${program}"],
    }

    # Add lengths if specified
    if lengths:
        config["parameters"]["length"] = {"value": lengths}

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfig saved to: {output_file}")
    print("\nNext steps:")
    print("1. Review the config file")
    print(f"2. Initialize: wandb sweep {output_file}")
    print(f"3. Update submit_slurm.sh with sweep ID and --array=1-{num_jobs}")
    print("4. Submit: sbatch src/lobster/cmdline/distributed_generation/submit_slurm.sh")


def create_structure_based_job_config(
    input_structures: str,
    structures_per_job: int,
    base_config_path: str,
    output_file: str = "src/lobster/cmdline/distributed_generation/wandb_config.yaml",
    mode: str = "inverse_folding",
):
    """
    Generate job distribution config for structure-based modes (inverse_folding, forward_folding).

    Args:
        input_structures: Glob pattern or path to structure files
        structures_per_job: Number of structures each job processes
        base_config_path: Path to base config (inverse_folding or forward_folding yaml)
        output_file: Output wandb config file
        mode: Generation mode ("inverse_folding" or "forward_folding")
    """
    # Expand glob to count files
    if "*" in input_structures or "?" in input_structures:
        structure_files = sorted(glob.glob(input_structures))
    else:
        path = Path(input_structures)
        if path.is_file():
            structure_files = [str(path)]
        elif path.is_dir():
            structure_files = []
            structure_files.extend(sorted(glob.glob(str(path / "*.pdb"))))
            structure_files.extend(sorted(glob.glob(str(path / "*.cif"))))
            structure_files.extend(sorted(glob.glob(str(path / "*.pt"))))
        else:
            raise ValueError(f"Input path does not exist: {input_structures}")

    total_structures = len(structure_files)

    if total_structures == 0:
        raise ValueError(f"No structure files found matching: {input_structures}")

    # Calculate number of jobs needed
    num_jobs = (total_structures + structures_per_job - 1) // structures_per_job

    print(f"Creating config for {mode} mode")
    print(f"Input pattern: {input_structures}")
    print(f"Total structures found: {total_structures}")
    print(f"Structures per job: {structures_per_job}")
    print(f"Number of jobs: {num_jobs}")

    job_ids = list(range(num_jobs))

    config = {
        "program": "src/lobster/cmdline/distributed_generate.py",
        "method": "grid",
        "project": f"lobster-distributed-{mode.replace('_', '-')}",
        "entity": "prescient-design",
        "metric": {"name": "job_completed", "goal": "maximize"},
        "parameters": {
            "base_config_path": {"value": base_config_path},
            "job_id": {"values": job_ids},
            "structures_per_job": {"value": structures_per_job},
            "total_structures": {"value": total_structures},
            "mode": {"value": mode},
        },
        "command": ["${env}", "python", "${program}"],
    }

    # Save config
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfig saved to: {output_file}")
    print("\nStructure distribution:")
    print(f"  Jobs 0-{num_jobs - 2}: {structures_per_job} structures each")
    last_job_count = total_structures - (num_jobs - 1) * structures_per_job
    print(f"  Job {num_jobs - 1}: {last_job_count} structures")
    print("\nNext steps:")
    print("1. Review the config file")
    print(f"2. Initialize: wandb sweep {output_file}")
    print(f"3. Update submit_slurm.sh with sweep ID and --array=1-{num_jobs}")
    print("4. Submit: sbatch src/lobster/cmdline/distributed_generation/submit_slurm.sh")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate wandb distributed generation config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unconditional generation
  python create_job_config.py --mode unconditional --total_samples 100 --samples_per_job 5

  # Inverse folding
  python src/lobster/cmdline/distributed_generation/create_job_config.py --mode inverse_folding \
    --input_structures "/data2/lisanzas/multi_flow_data/test_set_filtered_pt/*.pt" \
    --structures_per_job 5 \
    --base_config src/lobster/hydra_config/experiment/generate_inverse_folding_450M.yaml

  # Forward folding
  python src/lobster/cmdline/distributed_generation/create_job_config.py --mode forward_folding \
    --input_structures "/data2/lisanzas/multi_flow_data/test_set_filtered_pt/*.pt" \
    --structures_per_job 5 \
    --base_config src/lobster/hydra_config/experiment/generate_forward_folding_450M.yaml
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["unconditional", "inverse_folding", "forward_folding"],
        required=True,
        help="Generation mode",
    )
    parser.add_argument(
        "--output", default="src/lobster/cmdline/distributed_generation/wandb_config.yaml", help="Output config file"
    )
    parser.add_argument(
        "--base_config",
        help="Path to base config file (auto-detected if not provided)",
    )

    # Unconditional mode arguments
    unconditional_group = parser.add_argument_group("unconditional mode arguments")
    unconditional_group.add_argument("--total_samples", type=int, help="Total samples to generate")
    unconditional_group.add_argument("--samples_per_job", type=int, default=50, help="Samples per job")
    unconditional_group.add_argument("--lengths", type=int, nargs="+", help="Optional: lengths to generate")

    # Structure-based mode arguments (inverse_folding, forward_folding)
    structure_group = parser.add_argument_group("inverse_folding/forward_folding mode arguments")
    structure_group.add_argument("--input_structures", help="Glob pattern or path to structure files")
    structure_group.add_argument("--structures_per_job", type=int, default=5, help="Structures per job")

    args = parser.parse_args()

    # Auto-detect base config if not provided
    if not args.base_config:
        if args.mode == "unconditional":
            args.base_config = "src/lobster/hydra_config/experiment/generate_unconditional.yaml"
        elif args.mode == "inverse_folding":
            args.base_config = "src/lobster/hydra_config/experiment/generate_inverse_folding_450M.yaml"
        elif args.mode == "forward_folding":
            args.base_config = "src/lobster/hydra_config/experiment/generate_forward_folding_450M.yaml"

    # Call appropriate function based on mode
    if args.mode == "unconditional":
        if not args.total_samples:
            parser.error("--total_samples is required for unconditional mode")
        create_job_config(
            total_samples=args.total_samples,
            samples_per_job=args.samples_per_job,
            base_config_path=args.base_config,
            output_file=args.output,
            lengths=args.lengths,
        )
    elif args.mode in ["inverse_folding", "forward_folding"]:
        if not args.input_structures:
            parser.error("--input_structures is required for inverse_folding/forward_folding mode")
        create_structure_based_job_config(
            input_structures=args.input_structures,
            structures_per_job=args.structures_per_job,
            base_config_path=args.base_config,
            output_file=args.output,
            mode=args.mode,
        )
