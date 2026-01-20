#!/usr/bin/env python3
"""
WandB-Integrated Generation Script for genUME Parameter Optimization

This script integrates with wandb sweeps to optimize generation parameters.
Based on the wandb sweeps walkthrough: https://docs.wandb.ai/guides/sweeps/walkthrough/

Usage:
    wandb sweep <path to yaml>
    Update the id in the wandb_slurm.sh script to the wandb run id:
    srun -u --cpus-per-task $SLURM_CPUS_PER_TASK --cpu-bind=cores,verbose wandb agent prescient-design/lobster-wandb_sweeps/<wandb run id>
    sbatch wandb_slurm.sh
"""

import logging
from pathlib import Path
import logging

import pandas as pd

import wandb

from omegaconf import DictConfig, OmegaConf

# Import the original generation function
from lobster.cmdline.generate import generate as run_generation

logger = logging.getLogger(__name__)


def objective(config):
    """
    Objective function for wandb sweep optimization.

    Args:
        config: wandb config object containing sweep parameters

    Returns:
        float: Composite score for optimization
    """
    # Get generation lengths (list of lengths to test)
    lengths = config.get("generation_lengths", [200])
    if not isinstance(lengths, list):
        lengths = [lengths]

    logger.info(f"Running generation for {len(lengths)} lengths: {lengths}")

    # Run generation for each length
    output_dirs = []
    for length in lengths:
        logger.info(f"Starting generation for length {length}")

        # Create config from wandb parameters for this length
        gen_config = create_config_from_wandb(config, length=length)

        # Run generation
        run_generation(gen_config)

        output_dirs.append(gen_config.output_dir)
        logger.info(f"Completed generation for length {length}, output: {gen_config.output_dir}")

    # Collect metrics from all length runs
    metrics = collect_metrics_from_all_lengths(output_dirs, lengths)

    # Extract score weights from config
    score_weights = {
        "diversity": config.get("score_weight_diversity", 1.0),
        "plddt": config.get("score_weight_plddt", 0.05),
        "tm_score": config.get("score_weight_tm_score", 0.05),
        "pae": config.get("score_weight_pae", 0.05),
        "rmsd": config.get("score_weight_rmsd", 0.05),
    }

    # Calculate composite score (now aggregated across lengths)
    composite_score = calculate_composite_score(metrics, score_weights)

    # Log metrics to wandb (includes per-length and aggregated)
    log_data = {**metrics, "composite_score": composite_score, **{f"weight_{k}": v for k, v in score_weights.items()}}

    # Add length-specific diversity metrics for visualization
    for length in lengths:
        cluster_key = f"length_{length}_diversity_num_clusters_length_{length}"
        if cluster_key in metrics:
            log_data[f"diversity_by_length/{length}"] = metrics[cluster_key]

    wandb.log(log_data)

    logger.info(f"Run completed with composite score: {composite_score:.4f}")
    return composite_score


def create_config_from_wandb(config, length: int | None = None) -> DictConfig:
    """
    Create genUME config from wandb sweep parameters.

    Args:
        config: wandb config object
        length: Optional length override for multi-length generation
    """

    # Get generation mode from config
    mode = config.get("mode", "unconditional")

    # Determine length to use
    if length is None:
        # Single length mode (backward compatible)
        length = config.get("length", 200)

    # Base config structure
    config_dict = {
        "output_dir": f"./wandb_outputs/{wandb.run.id}/length_{length}",
        "seed": 12345,
        "model": {
            "_target_": "lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule",
            "ckpt_path": "/data2/ume/gen_ume/runs//2025-10-08T23-54-39/last.ckpt",
        },
        "generation": {
            "mode": mode,
            "length": length,
            "num_samples": config.get("num_samples", 10),
            "nsteps": config.get("nsteps", 200),
            "temperature_seq": config.get("temperature_seq", 0.5),
            "temperature_struc": config.get("temperature_struc", 0.5),
            "stochasticity_seq": config.get("stochasticity_seq", 20),
            "stochasticity_struc": config.get("stochasticity_struc", 20),
            "use_esmfold": True,
            "max_length": 512,
            "save_csv_metrics": True,
            "create_plots": False,
            "batch_size": config.get("batch_size", 1),
            "n_trials": config.get("n_trials", 1),
            "input_structures": config.get("input_structures", None),
            # Foldseek Diversity Analysis
            "calculate_foldseek_diversity": config.get("calculate_foldseek_diversity", True),
            "foldseek_bin_path": config.get(
                "foldseek_bin_path", "/homefs/home/lisanzas/scratch/Develop/lobster/src/lobster/metrics/foldseek/bin"
            ),
            "foldseek_tmscore_threshold": config.get("foldseek_tmscore_threshold", 0.5),
            "rmsd_threshold_for_diversity": config.get("rmsd_threshold_for_diversity", 2.0),
        },
    }

    # Mode-specific adjustments
    if mode == "inverse_folding":
        # Inverse folding specific settings
        if config_dict["generation"]["input_structures"] is None:
            raise ValueError("input_structures must be provided for inverse folding mode")
        logger.info(f"Using input structures: {config_dict['generation']['input_structures']}")

    elif mode == "forward_folding":
        # Forward folding specific settings
        if config_dict["generation"]["input_structures"] is None:
            raise ValueError("input_structures must be provided for forward folding mode")
        logger.info(f"Using input structures: {config_dict['generation']['input_structures']}")

    elif mode == "inpainting":
        # Inpainting specific settings
        if config_dict["generation"]["input_structures"] is None:
            raise ValueError("input_structures must be provided for inpainting mode")

        # Add inpainting-specific parameters
        config_dict["generation"]["mask_indices_sequence"] = config.get("mask_indices_sequence", "")
        config_dict["generation"]["mask_indices_structure"] = config.get("mask_indices_structure", "")

        logger.info(f"Using input structures: {config_dict['generation']['input_structures']}")
        logger.info(f"Sequence mask indices: {config_dict['generation']['mask_indices_sequence']}")
        logger.info(f"Structure mask indices: {config_dict['generation']['mask_indices_structure']}")

    elif mode == "unconditional":
        # Unconditional generation - no input structures needed
        config_dict["generation"]["input_structures"] = None

        # Add self-reflection parameters if enabled
        if config.get("enable_self_reflection", False):
            logger.info("Enabling self-reflection refinement for unconditional generation")
            config_dict["generation"]["enable_self_reflection"] = True
            config_dict["generation"]["self_reflection"] = {
                "use_esmfold_validation": False,  # Optional ESMFold validation within self-reflection
                "forward_folding": {
                    "nsteps": config.get("self_reflection_forward_nsteps", 100),
                    "temperature_seq": config.get("self_reflection_forward_temp_seq", 0.2967457760634187),
                    "temperature_struc": config.get("self_reflection_forward_temp_struc", 0.1102551183666233),
                    "stochasticity_seq": config.get("self_reflection_forward_stoch_seq", 10),
                    "stochasticity_struc": config.get("self_reflection_forward_stoch_struc", 30),
                },
                "inverse_folding": {
                    "nsteps": config.get("self_reflection_inverse_nsteps", 200),
                    "temperature_seq": config.get("self_reflection_inverse_temp_seq", 0.16423763902324678),
                    "temperature_struc": config.get("self_reflection_inverse_temp_struc", 1.0),
                    "stochasticity_seq": config.get("self_reflection_inverse_stoch_seq", 20),
                    "stochasticity_struc": config.get("self_reflection_inverse_stoch_struc", 10),
                },
                "quality_control": {
                    "enable_tm_threshold": True,
                    "min_tm_score_forward": config.get("self_reflection_min_tm_score", 0.8),
                    "enable_min_percent_identity_threshold": True,
                    "min_percent_identity": config.get("self_reflection_min_percent_identity", 20.0),
                    "enable_max_percent_identity_threshold": True,
                    "max_percent_identity": config.get("self_reflection_max_percent_identity", 90.0),
                    "enable_sequence_token_check": True,
                    "max_retries": config.get("self_reflection_max_retries", 30),
                },
            }
            logger.info(f"Self-reflection config: {config_dict['generation']['self_reflection']}")

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

    # Define metrics based on generation mode
    mode = df["mode"].iloc[0] if "mode" in df.columns and len(df) > 0 else "unconditional"

    if mode == "unconditional":
        metric_columns = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]
    elif mode == "inverse_folding":
        metric_columns = ["percent_identity", "plddt", "predicted_aligned_error", "tm_score", "rmsd"]
    elif mode == "forward_folding":
        metric_columns = ["tm_score", "rmsd"]
    elif mode == "inpainting":
        metric_columns = [
            "percent_identity_masked",
            "percent_identity_unmasked",
            "rmsd_inpainted",
            "plddt",
            "predicted_aligned_error",
            "tm_score",
            "rmsd",
        ]
    else:
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

    logger.info(f"Collected metrics for {mode} mode: {metrics}")

    # Collect Foldseek diversity metrics (unconditional mode only)
    if mode == "unconditional":
        try:
            from biotite.sequence.io import fasta

            # Look for diversity results files
            diversity_files = list(output_path.glob("foldseek_results/length_*/res_rep_seq.fasta"))

            if diversity_files:
                logger.info(f"Found {len(diversity_files)} Foldseek diversity result files")

                for div_file in diversity_files:
                    # Count clusters from Foldseek output
                    fasta_content = fasta.FastaFile.read(str(div_file))
                    num_clusters = len(fasta_content)

                    # Extract length from path: foldseek_results/length_500/res_rep_seq.fasta
                    length = int(div_file.parent.name.split("_")[1])

                    # Get total structures passing RMSD threshold from CSV
                    if "rmsd" in df.columns:
                        rmsd_values = pd.to_numeric(df["rmsd"], errors="coerce").dropna()
                        total_passing = len(rmsd_values[rmsd_values < 2.0])
                    else:
                        total_passing = 0

                    # Store diversity metrics
                    metrics[f"diversity_num_clusters_length_{length}"] = num_clusters
                    metrics[f"diversity_total_structures_length_{length}"] = total_passing

                    if total_passing > 0:
                        diversity_pct = (num_clusters / total_passing) * 100
                        metrics[f"diversity_percentage_length_{length}"] = diversity_pct
                        diversity_str = f"{diversity_pct:.1f}%"
                    else:
                        diversity_str = "N/A"

                    logger.info(
                        f"Diversity metrics for length {length}: "
                        f"{num_clusters}/{total_passing} clusters ({diversity_str})"
                    )
            else:
                logger.debug("No Foldseek diversity results found")

        except Exception as e:
            logger.warning(f"Failed to collect diversity metrics: {e}")

    return metrics


def collect_metrics_from_all_lengths(output_dirs: list[str], lengths: list[int]) -> dict[str, float]:
    """
    Collect metrics from multiple length-specific output directories.

    Args:
        output_dirs: List of output directory paths, one per length
        lengths: List of generation lengths corresponding to output_dirs

    Returns:
        Dictionary with both per-length metrics and aggregated metrics
    """
    all_metrics = {}
    per_length_metrics = {}

    logger.info(f"Collecting metrics from {len(output_dirs)} length runs")

    # Collect metrics for each length
    for output_dir, length in zip(output_dirs, lengths):
        logger.info(f"Collecting metrics for length {length} from {output_dir}")
        length_metrics = collect_metrics_from_output(output_dir)
        per_length_metrics[length] = length_metrics

        # Store per-length metrics with prefix
        for metric_name, value in length_metrics.items():
            all_metrics[f"length_{length}_{metric_name}"] = value

    # Calculate aggregated metrics across all lengths
    logger.info("Aggregating metrics across all lengths")
    aggregate_metrics = aggregate_across_lengths(per_length_metrics)
    all_metrics.update(aggregate_metrics)

    logger.info(f"Total metrics collected: {len(all_metrics)} (per-length + aggregated)")
    return all_metrics


def aggregate_across_lengths(per_length_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    """
    Aggregate metrics across all lengths using simple averaging.

    Args:
        per_length_metrics: Dict mapping length → metrics dict

    Returns:
        Dictionary of aggregated metrics with 'agg_' prefix
    """
    aggregated = {}

    if not per_length_metrics:
        logger.warning("No per-length metrics to aggregate")
        return aggregated

    # Collect all unique metric keys
    all_metric_keys = set()
    for metrics in per_length_metrics.values():
        all_metric_keys.update(metrics.keys())

    logger.debug(f"Aggregating {len(all_metric_keys)} unique metric types across {len(per_length_metrics)} lengths")

    # For each metric, calculate mean across lengths
    for metric_key in all_metric_keys:
        values = []
        for length, metrics in per_length_metrics.items():
            if metric_key in metrics:
                values.append(metrics[metric_key])

        if values:
            aggregated[f"agg_{metric_key}"] = sum(values) / len(values)

    # Special handling for diversity: sum total clusters across ALL lengths
    cluster_keys = [k for k in all_metric_keys if k.startswith("diversity_num_clusters_")]
    if cluster_keys:
        total_clusters = 0
        for length, metrics in per_length_metrics.items():
            for k in cluster_keys:
                if k in metrics:
                    total_clusters += metrics[k]

        aggregated["agg_total_clusters_all_lengths"] = total_clusters
        logger.info(f"Total clusters across all lengths: {total_clusters}")

    # Also calculate total structures across all lengths
    total_structures = 0
    for length, metrics in per_length_metrics.items():
        for k in all_metric_keys:
            if k.startswith("diversity_total_structures_"):
                total_structures += metrics.get(k, 0)

    if total_structures > 0:
        aggregated["agg_total_structures_all_lengths"] = total_structures
        if total_clusters > 0:
            aggregated["agg_diversity_percentage_all_lengths"] = (total_clusters / total_structures) * 100
            logger.info(
                f"Overall diversity: {total_clusters}/{total_structures} = {aggregated['agg_diversity_percentage_all_lengths']:.1f}%"
            )

    return aggregated


def calculate_composite_score(metrics: dict[str, float], score_weights: dict[str, float] | None = None) -> float:
    """
    Calculate composite score for optimization.

    Different scoring strategies for different modes:
    - Unconditional: Focus on diversity (foldseek clusters), with configurable weights for other metrics
    - Inverse folding: Focus on percent_identity, plddt, tm_score
    - Forward folding: Focus on tm_score, minimize RMSD
    - Inpainting: Focus on percent_identity_masked, rmsd_inpainted (post-alignment), tm_score, plddt

    Args:
        metrics: Dictionary of collected metrics
        score_weights: Optional dictionary of weights for unconditional mode. Keys:
            - diversity: weight per cluster (default 1.0)
            - plddt: weight for pLDDT (default 0.05)
            - tm_score: weight for TM-score (default 0.05)
            - pae: weight for PAE penalty (default 0.05)
            - rmsd: weight for RMSD penalty (default 0.05)
    """
    score = 0.0

    # Default weights for unconditional mode
    if score_weights is None:
        score_weights = {
            "diversity": 1.0,
            "plddt": 0.05,
            "tm_score": 0.05,
            "pae": 0.05,
            "rmsd": 0.05,
        }

    # Determine mode from available metrics
    mode = "unconditional"  # default
    if "avg_percent_identity_masked" in metrics or "avg_rmsd_inpainted" in metrics:
        mode = "inpainting"
    elif "avg_percent_identity" in metrics:
        mode = "inverse_folding"
    elif "avg_tm_score" in metrics and "avg_plddt" not in metrics:
        mode = "forward_folding"

    if mode == "unconditional":
        # MAIN METRIC: Number of foldseek clusters (diversity) across ALL lengths
        # Try to use aggregated metrics first (multi-length mode)
        if "agg_total_clusters_all_lengths" in metrics:
            # Multi-length mode: use aggregated total clusters
            total_clusters = metrics["agg_total_clusters_all_lengths"]
            diversity_score = total_clusters * score_weights["diversity"]
            score += diversity_score
            logger.info(
                f"Diversity contribution to score (MAIN METRIC, ALL LENGTHS): {total_clusters} clusters * "
                f"{score_weights['diversity']:.2f} weight = {diversity_score:.2f} points"
            )
        else:
            # Single-length mode (backward compatible): sum clusters from individual length keys
            cluster_keys = [k for k in metrics.keys() if k.startswith("diversity_num_clusters_")]
            if cluster_keys:
                total_clusters = sum(metrics[k] for k in cluster_keys)
                diversity_score = total_clusters * score_weights["diversity"]
                score += diversity_score
                logger.info(
                    f"Diversity contribution to score (MAIN METRIC): {total_clusters} clusters * "
                    f"{score_weights['diversity']:.2f} weight = {diversity_score:.2f} points"
                )

        # Secondary metrics (configurable weights)
        # Use aggregated metrics if available (multi-length), otherwise use direct metrics (single-length)

        # Higher is better: plddt, tm_score
        plddt_key = "agg_avg_plddt" if "agg_avg_plddt" in metrics else "avg_plddt"
        if plddt_key in metrics:
            plddt_score = metrics[plddt_key] * score_weights["plddt"]
            score += plddt_score
            logger.debug(
                f"pLDDT contribution: {metrics[plddt_key]:.2f} * {score_weights['plddt']:.2f} = {plddt_score:.2f}"
            )

        tm_key = "agg_avg_tm_score" if "agg_avg_tm_score" in metrics else "avg_tm_score"
        if tm_key in metrics:
            tm_score = metrics[tm_key] * score_weights["tm_score"]
            score += tm_score
            logger.debug(
                f"TM-score contribution: {metrics[tm_key]:.2f} * {score_weights['tm_score']:.2f} = {tm_score:.2f}"
            )

        # Lower is better: predicted_aligned_error, rmsd
        pae_key = (
            "agg_avg_predicted_aligned_error"
            if "agg_avg_predicted_aligned_error" in metrics
            else "avg_predicted_aligned_error"
        )
        if pae_key in metrics:
            pae_penalty = metrics[pae_key] / 100 * score_weights["pae"]
            score -= pae_penalty
            logger.debug(f"PAE penalty: {metrics[pae_key]:.2f}/100 * {score_weights['pae']:.2f} = -{pae_penalty:.2f}")

        rmsd_key = "agg_avg_rmsd" if "agg_avg_rmsd" in metrics else "avg_rmsd"
        if rmsd_key in metrics:
            rmsd_penalty = metrics[rmsd_key] / 10 * score_weights["rmsd"]
            score -= rmsd_penalty
            logger.debug(
                f"RMSD penalty: {metrics[rmsd_key]:.2f}/10 * {score_weights['rmsd']:.2f} = -{rmsd_penalty:.2f}"
            )

    elif mode == "inverse_folding":
        # Higher is better: percent_identity, plddt, tm_score
        if "avg_percent_identity" in metrics:
            score += metrics["avg_percent_identity"] * 0.4  # Most important for inverse folding

        if "avg_plddt" in metrics:
            score += metrics["avg_plddt"] * 0.2

        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.4

        # Lower is better: predicted_aligned_error, rmsd
        if "avg_predicted_aligned_error" in metrics:
            score -= metrics["avg_predicted_aligned_error"] / 100 * 0.1

        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.1

    elif mode == "forward_folding":
        # Higher is better: tm_score
        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.7  # Most important for forward folding

        # Lower is better: rmsd
        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.3

    elif mode == "inpainting":
        # Higher is better: percent_identity_masked, percent_identity_unmasked, plddt, tm_score
        if "avg_percent_identity_masked" in metrics:
            score += metrics["avg_percent_identity_masked"] * 0.25  # How well we recovered masked sequence

        if "avg_percent_identity_unmasked" in metrics:
            score += metrics["avg_percent_identity_unmasked"] * 0.15  # Preserved unmasked regions

        if "avg_plddt" in metrics:
            score += metrics["avg_plddt"] * 0.0015  # Structure quality (scale to 0-100 range)

        if "avg_tm_score" in metrics:
            score += metrics["avg_tm_score"] * 0.20  # Overall structural similarity

        # Lower is better: rmsd_inpainted, predicted_aligned_error, rmsd
        if "avg_rmsd_inpainted" in metrics:
            score -= metrics["avg_rmsd_inpainted"] / 5 * 0.20  # KEY: minimize structural deviation in inpainted region

        if "avg_predicted_aligned_error" in metrics:
            score -= metrics["avg_predicted_aligned_error"] / 100 * 0.05  # Minimize PAE

        if "avg_rmsd" in metrics:
            score -= metrics["avg_rmsd"] / 10 * 0.05  # Minimize overall RMSD

    logger.info(f"Calculated composite score for {mode} mode: {score:.4f}")
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
