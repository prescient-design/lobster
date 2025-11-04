#!/usr/bin/env python3
"""
Helper script to generate wandb_config.yaml for distributed generation.
Creates job distribution configurations for different sample counts and parallelization strategies.
"""

import yaml


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate wandb distributed generation config")
    parser.add_argument("--total_samples", type=int, required=True, help="Total samples to generate")
    parser.add_argument("--samples_per_job", type=int, default=50, help="Samples per job")
    parser.add_argument("--base_config", default="src/lobster/hydra_config/experiment/generate_unconditional.yaml")
    parser.add_argument(
        "--output", default="src/lobster/cmdline/distributed_generation/wandb_config.yaml", help="Output config file"
    )
    parser.add_argument("--lengths", type=int, nargs="+", help="Optional: lengths to generate")

    args = parser.parse_args()

    create_job_config(
        total_samples=args.total_samples,
        samples_per_job=args.samples_per_job,
        base_config_path=args.base_config,
        output_file=args.output,
        lengths=args.lengths,
    )
