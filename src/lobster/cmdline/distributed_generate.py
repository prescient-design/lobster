#!/usr/bin/env python3
"""
WandB Distributed Generation Script for genUME

Uses wandb agents as a distributed job queue to parallelize structure generation.
Each agent generates a subset of samples from the total workload.

Usage:
    # Initialize the job queue
    wandb sweep src/lobster/cmdline/distributed_generation/wandb_config.yaml

    # Submit SLURM array to process jobs
    # Update submit_slurm.sh with sweep ID
    sbatch src/lobster/cmdline/distributed_generation/submit_slurm.sh
"""

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
    """
    # Initialize wandb run - this gets config from the sweep
    with wandb.init() as run:
        config = run.config

        # Calculate start/end samples from job_id and samples_per_job
        job_id = config.job_id
        samples_per_job = config.samples_per_job
        total_samples = config.total_samples

        start_sample = job_id * samples_per_job
        end_sample = min((job_id + 1) * samples_per_job, total_samples)
        num_samples = end_sample - start_sample

        logger.info("Starting distributed generation job")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Sample range: {start_sample}-{end_sample} ({num_samples} samples per length)")

        # Load base configuration from your generate_unconditional.yaml
        base_config_path = config.get(
            "base_config_path", "src/lobster/hydra_config/experiment/generate_unconditional.yaml"
        )

        logger.info(f"Loading base config from: {base_config_path}")
        gen_config = OmegaConf.load(base_config_path)

        # Override specific parameters for this job
        # 1. Set output directory for this job
        output_base = gen_config.get("output_dir", "./examples/generated_unconditional")
        gen_config.output_dir = f"{output_base}/job_{job_id}"

        # 2. Set number of samples for this job chunk
        # IMPORTANT: num_samples is PER LENGTH in the config
        # If config has length: [100, 200, 300], each job generates num_samples × 3 structures
        gen_config.generation.num_samples = num_samples

        # 3. Set seed to ensure reproducibility per job
        # Each job gets a unique seed based on job_id
        base_seed = gen_config.get("seed", 12345)
        gen_config.seed = base_seed + job_id

        # 4. CRITICAL: Disable Foldseek clustering during distributed generation
        # Foldseek must be run on ALL samples together after aggregation
        gen_config.generation.calculate_foldseek_diversity = False
        logger.info("Foldseek diversity calculation disabled (will run post-aggregation)")

        # 5. Optional: Override any parameters from wandb config
        # This allows flexibility without doing full sweeps
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

        # Run generation
        logger.info("Starting generation...")
        run_generation(gen_config)
        logger.info("Generation complete!")

        # Collect and log metrics to wandb
        metrics = collect_job_metrics(gen_config.output_dir)

        # Log to wandb
        wandb.log({"job_id": job_id, "num_samples_generated": num_samples, **metrics})

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
