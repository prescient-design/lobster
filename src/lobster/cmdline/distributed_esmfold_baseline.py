#!/usr/bin/env python3
"""
WandB Distributed ESMFold Baseline Script

Uses wandb agents as a distributed job queue to parallelize ESMFold baseline evaluation.
Compatible with the existing aggregation script (aggregate_results.py) for forward_folding mode.

Usage:
    # Initialize the job queue
    wandb sweep src/lobster/cmdline/distributed_generation/wandb_config_esmfold_baseline.yaml

    # Submit SLURM array to process jobs
    # Update submit_slurm.sh with sweep ID
    sbatch src/lobster/cmdline/distributed_generation/submit_slurm_esmfold.sh
"""

import glob
from pathlib import Path
import pandas as pd
from loguru import logger
import wandb
from omegaconf import OmegaConf

# Import the main ESMFold baseline function
from lobster.cmdline.esmfold_baseline import main as run_esmfold_baseline


def main():
    """
    Main distributed ESMFold baseline function.
    Each wandb agent runs this and gets assigned a job_id.
    """
    # Initialize wandb run - this gets config from the sweep
    with wandb.init() as run:
        config = run.config

        job_id = config.job_id

        # Load base configuration
        base_config_path = config.get("base_config_path", "src/lobster/hydra_config/experiment/esmfold_baseline.yaml")

        logger.info("Starting distributed ESMFold baseline job")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Loading base config from: {base_config_path}")
        baseline_config = OmegaConf.load(base_config_path)

        # Set common parameters
        output_base = baseline_config.get("output_dir", "./examples/esmfold_baseline")
        baseline_config.output_dir = f"{output_base}/job_{job_id}"

        base_seed = baseline_config.get("seed", 12345)
        baseline_config.seed = base_seed + job_id

        # Setup job configuration for structure-based processing
        structures_per_job = config.structures_per_job
        total_structures = config.total_structures

        # Get input structure pattern from base config
        input_structures_pattern = baseline_config.generation.input_structures

        if not input_structures_pattern:
            raise ValueError("input_structures must be set in base config")

        logger.info(f"Input structures pattern: {input_structures_pattern}")

        # Expand glob pattern to get all structure files
        if isinstance(input_structures_pattern, str):
            if "*" in input_structures_pattern or "?" in input_structures_pattern:
                # Glob pattern
                all_structure_files = sorted(glob.glob(input_structures_pattern))
            else:
                # Single file or directory
                path = Path(input_structures_pattern)
                if path.is_file():
                    all_structure_files = [str(path)]
                elif path.is_dir():
                    # Find all structure files in directory (PDB, CIF, PT)
                    all_structure_files = []
                    all_structure_files.extend(sorted(glob.glob(str(path / "*.pdb"))))
                    all_structure_files.extend(sorted(glob.glob(str(path / "*.cif"))))
                    all_structure_files.extend(sorted(glob.glob(str(path / "*.pt"))))
                else:
                    raise ValueError(f"Input path does not exist: {input_structures_pattern}")
        elif isinstance(input_structures_pattern, (list, tuple)):
            # Already a list of files
            all_structure_files = sorted([str(p) for p in input_structures_pattern if Path(p).is_file()])
        else:
            raise ValueError(f"Invalid input_structures format: {type(input_structures_pattern)}")

        if not all_structure_files:
            raise ValueError(f"No structure files found matching: {input_structures_pattern}")

        logger.info(f"Found {len(all_structure_files)} total structure files")

        # Calculate this job's slice of structures
        start_idx = job_id * structures_per_job
        end_idx = min((job_id + 1) * structures_per_job, total_structures)

        # Ensure we don't exceed available files
        end_idx = min(end_idx, len(all_structure_files))

        job_structure_files = all_structure_files[start_idx:end_idx]
        num_structures = len(job_structure_files)

        if num_structures == 0:
            raise ValueError(
                f"Job {job_id} has no structures to process (start_idx={start_idx}, total={len(all_structure_files)})"
            )

        logger.info(f"Structure range: {start_idx}-{end_idx} ({num_structures} structures)")
        logger.info(f"First file: {job_structure_files[0]}")
        logger.info(f"Last file: {job_structure_files[-1]}")

        # Override config with this job's structure subset
        baseline_config.generation.input_structures = job_structure_files

        logger.info("Configuration for this job:")
        logger.info(f"  Output: {baseline_config.output_dir}")
        logger.info(f"  Structures to process: {num_structures} (indices {start_idx}-{end_idx})")
        logger.info(f"  Seed: {baseline_config.seed}")

        # Run ESMFold baseline
        logger.info("Starting ESMFold baseline evaluation...")
        run_esmfold_baseline(baseline_config)
        logger.info("ESMFold baseline evaluation complete!")

        # Collect and log metrics to wandb
        metrics = collect_job_metrics(baseline_config.output_dir)

        # Log to wandb
        wandb.log({"job_id": job_id, "mode": "esmfold_baseline", **metrics})

        logger.info(f"Job {job_id} completed successfully")
        logger.info(f"Metrics: {metrics}")


def collect_job_metrics(output_dir: str) -> dict:
    """
    Collect metrics from this job's outputs.

    Args:
        output_dir: Path to job output directory

    Returns:
        Dictionary of aggregated metrics
    """

    output_path = Path(output_dir)
    metrics = {}

    # Find metrics CSV
    csv_files = list(output_path.glob("*_metrics_*.csv"))

    if not csv_files:
        logger.warning(f"No metrics CSV found in {output_dir}")
        return metrics

    # Load most recent CSV
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_csv)

    logger.info(f"Loaded metrics from {latest_csv}")
    logger.info(f"Found {len(df)} samples")

    # Collect key metrics
    metric_columns = ["plddt", "tm_score", "rmsd"]

    for metric in metric_columns:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(values) > 0:
                metrics[f"avg_{metric}"] = float(values.mean())
                metrics[f"std_{metric}"] = float(values.std())
                metrics[f"min_{metric}"] = float(values.min())
                metrics[f"max_{metric}"] = float(values.max())

    return metrics


if __name__ == "__main__":
    main()
