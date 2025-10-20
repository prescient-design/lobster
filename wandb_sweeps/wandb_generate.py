#!/usr/bin/env python3
"""
WandB-Integrated Generation Script for genUME Parameter Optimization

This script integrates with wandb sweeps to optimize generation parameters.
Based on the wandb sweeps walkthrough: https://docs.wandb.ai/guides/sweeps/walkthrough/

Usage:
    python wandb_generate.py --config_path=../hydra_config --config_name=generate
"""

from pathlib import Path

from loguru import logger
import pandas as pd

import wandb
from omegaconf import DictConfig, OmegaConf

# Import the original generation function
from lobster.cmdline.generate import generate as run_generation


def objective(config):
    """
    Objective function for wandb sweep optimization.

    Args:
        config: wandb config object containing sweep parameters

    Returns:
        float: Composite score for optimization
    """
    # Create config from wandb parameters
    gen_config = create_config_from_wandb(config)

    # Run generation
    run_generation(gen_config)

    # Collect metrics
    metrics = collect_metrics_from_output(gen_config.output_dir)

    # Calculate composite score
    composite_score = calculate_composite_score(metrics)

    # Log metrics to wandb
    wandb.log({**metrics, "composite_score": composite_score})

    logger.info(f"Run completed with composite score: {composite_score:.4f}")
    return composite_score


def create_config_from_wandb(config) -> DictConfig:
    """Create genUME config from wandb sweep parameters."""

    # Base config structure
    config_dict = {
        "output_dir": f"./wandb_outputs/{wandb.run.id}",
        "seed": 12345,
        "model": {
            "_target_": "lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule",
            "ckpt_path": "/data2/ume/gen_ume/runs//2025-10-08T23-54-39/last.ckpt",
        },
        "generation": {
            "mode": "unconditional",
            "length": config.get("protein_length", 200),
            "num_samples": config.get("num_generation_samples", 10),
            "nsteps": config.get("generation_steps", 200),
            "temperature_seq": config.get("temperature_seq", 0.5),
            "temperature_struc": config.get("temperature_struc", 0.5),
            "stochasticity_seq": config.get("stochasticity_seq", 20),
            "stochasticity_struc": config.get("stochasticity_struc", 20),
            "use_esmfold": True,
            "max_length": 512,
            "save_csv_metrics": True,
            "create_plots": False,
            "batch_size": 1,
            "n_trials": 1,
            "input_structures": None,
        },
    }

    return OmegaConf.create(config_dict)


def collect_metrics_from_output(output_dir: str) -> dict[str, float]:
    """Collect metrics from generation output CSV files."""
    output_path = Path(output_dir)
    metrics = {}

    # Look for metrics CSV files
    csv_files = list(output_path.glob("*_metrics_*.csv"))

    if not csv_files:
        logger.warning(f"No metrics CSV files found in {output_dir}")
        return metrics

    # Use the most recent CSV file
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)

    df = pd.read_csv(latest_csv)

    # Calculate metrics
    metric_columns = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]

    for metric in metric_columns:
        if metric in df.columns:
            # Convert to numeric and remove NaN values
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(values) > 0:
                metrics[f"avg_{metric}"] = float(values.mean())
                metrics[f"std_{metric}"] = float(values.std())
                metrics[f"min_{metric}"] = float(values.min())
                metrics[f"max_{metric}"] = float(values.max())
                metrics[f"count_{metric}"] = len(values)

    # Calculate additional metrics
    if "sequence_length" in df.columns:
        lengths = pd.to_numeric(df["sequence_length"], errors="coerce").dropna()
        if len(lengths) > 0:
            metrics["avg_sequence_length"] = float(lengths.mean())
            metrics["std_sequence_length"] = float(lengths.std())

    logger.info(f"Collected metrics: {metrics}")

    return metrics


def calculate_composite_score(metrics: dict[str, float]) -> float:
    """
    Calculate composite score for optimization.

    Higher is better: plddt, tm_score
    Lower is better: predicted_aligned_error, rmsd
    """
    score = 0.0

    # Higher is better: plddt, tm_score
    if "avg_plddt" in metrics:
        score += metrics["avg_plddt"] * 0.3

    if "avg_tm_score" in metrics:
        score += metrics["avg_tm_score"] * 0.3

    # Lower is better: predicted_aligned_error, rmsd
    if "avg_predicted_aligned_error" in metrics:
        score -= metrics["avg_predicted_aligned_error"] / 100 * 0.2

    if "avg_rmsd" in metrics:
        score -= metrics["avg_rmsd"] / 10 * 0.2

    return score


def main():
    """
    Main function for wandb-integrated generation.
    Based on the wandb sweeps walkthrough pattern.
    """
    # Initialize wandb run
    with wandb.init(project="genume-parameter-optimization") as run:
        # Run objective function
        score = objective(run.config)

        # Log final score
        wandb.log({"final_score": score})

        logger.info(f"WandB run completed successfully with score: {score:.4f}")


if __name__ == "__main__":
    main()
