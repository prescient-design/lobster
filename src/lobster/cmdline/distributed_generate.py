#!/usr/bin/env python3
"""
WandB Distributed Generation Script for genUME

Uses wandb agents as a distributed job queue to parallelize structure generation.
Supports three modes:
  1. Unconditional: Each agent generates a subset of samples (e.g., 100 samples → 20 jobs × 5)
  2. Inverse Folding: Each agent processes a subset of input structures
  3. Forward Folding: Each agent processes a subset of input structures

Usage:
    # Initialize the job queue
    wandb sweep src/lobster/cmdline/distributed_generation/wandb_config.yaml

    # Submit SLURM array to process jobs
    # Update submit_slurm.sh with sweep ID
    sbatch src/lobster/cmdline/distributed_generation/submit_slurm.sh
"""

import glob
from pathlib import Path
from loguru import logger
import wandb
from omegaconf import OmegaConf

# Import the original generation function
from lobster.cmdline.generate import generate as run_generation


def main():
    """
    Main distributed generation function.
    Each wandb agent runs this and gets assigned a job_id.
    Supports unconditional, inverse_folding, and forward_folding modes.
    """
    # Initialize wandb run - this gets config from the sweep
    with wandb.init() as run:
        config = run.config

        job_id = config.job_id

        # Load base configuration
        base_config_path = config.get(
            "base_config_path", "src/lobster/hydra_config/experiment/generate_unconditional.yaml"
        )

        logger.info("Starting distributed generation job")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Loading base config from: {base_config_path}")
        gen_config = OmegaConf.load(base_config_path)

        # Detect generation mode from config
        mode = gen_config.generation.get("mode", "unconditional")
        logger.info(f"Generation mode: {mode}")

        # Set common parameters
        output_base = gen_config.get("output_dir", "./examples/generated")
        gen_config.output_dir = f"{output_base}/job_{job_id}"

        base_seed = gen_config.get("seed", 12345)
        gen_config.seed = base_seed + job_id

        # CRITICAL: Disable Foldseek clustering during distributed generation
        gen_config.generation.calculate_foldseek_diversity = False
        logger.info("Foldseek diversity calculation disabled (will run post-aggregation)")

        # Branch based on mode
        if mode == "unconditional":
            _setup_unconditional_job(gen_config, config, job_id)
        elif mode in ["inverse_folding", "forward_folding"]:
            _setup_structure_based_job(gen_config, config, job_id, mode)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        # Run generation
        logger.info("Starting generation...")
        run_generation(gen_config)
        logger.info("Generation complete!")

        # Collect and log metrics to wandb
        metrics = collect_job_metrics(gen_config.output_dir)

        # Log to wandb
        wandb.log({"job_id": job_id, "mode": mode, **metrics})

        logger.info(f"Job {job_id} completed successfully")
        logger.info(f"Metrics: {metrics}")


def _setup_unconditional_job(gen_config, config, job_id):
    """
    Setup job configuration for unconditional generation mode.

    Args:
        gen_config: OmegaConf config to modify
        config: WandB sweep config
        job_id: Job ID for this worker
    """
    samples_per_job = config.samples_per_job
    total_samples = config.total_samples

    start_sample = job_id * samples_per_job
    end_sample = min((job_id + 1) * samples_per_job, total_samples)
    num_samples = end_sample - start_sample

    logger.info(f"Sample range: {start_sample}-{end_sample} ({num_samples} samples per length)")

    # Set number of samples for this job chunk
    # IMPORTANT: num_samples is PER LENGTH in the config
    # If config has length: [100, 200, 300], each job generates num_samples × 3 structures
    gen_config.generation.num_samples = num_samples

    # Optional: Override any parameters from wandb config
    if "length" in config:
        gen_config.generation.length = config.length
    if "nsteps" in config:
        gen_config.generation.nsteps = config.nsteps

    # Calculate actual number of structures
    lengths = gen_config.generation.length
    if isinstance(lengths, list):
        num_lengths = len(lengths)
        total_structures = num_samples * num_lengths
    else:
        num_lengths = 1
        total_structures = num_samples

    logger.info("Configuration for this job:")
    logger.info(f"  Output: {gen_config.output_dir}")
    logger.info(f"  Samples per length: {num_samples} (indices {start_sample}-{end_sample})")
    logger.info(f"  Lengths: {gen_config.generation.length} ({num_lengths} lengths)")
    logger.info(f"  Total structures this job: {total_structures}")
    logger.info(f"  Seed: {gen_config.seed}")
    logger.info(f"  Steps: {gen_config.generation.nsteps}")


def _setup_structure_based_job(gen_config, config, job_id, mode):
    """
    Setup job configuration for inverse_folding or forward_folding modes.

    Args:
        gen_config: OmegaConf config to modify
        config: WandB sweep config
        job_id: Job ID for this worker
        mode: "inverse_folding" or "forward_folding"
    """
    structures_per_job = config.structures_per_job
    total_structures = config.total_structures

    # Get input structure pattern from base config
    input_structures_pattern = gen_config.generation.input_structures

    if not input_structures_pattern:
        raise ValueError(f"input_structures must be set in base config for {mode} mode")

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
    gen_config.generation.input_structures = job_structure_files

    # Optional: Override any parameters from wandb config
    if "nsteps" in config:
        gen_config.generation.nsteps = config.nsteps

    logger.info("Configuration for this job:")
    logger.info(f"  Output: {gen_config.output_dir}")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Structures to process: {num_structures} (indices {start_idx}-{end_idx})")
    logger.info(f"  Seed: {gen_config.seed}")
    logger.info(f"  Steps: {gen_config.generation.nsteps}")


def collect_job_metrics(output_dir: str) -> dict:
    """
    Collect metrics from this job's outputs.

    Args:
        output_dir: Path to job output directory

    Returns:
        Dictionary of aggregated metrics
    """
    import pandas as pd

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
    metric_columns = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]

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
