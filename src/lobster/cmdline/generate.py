import logging
from pathlib import Path
import glob
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from lobster.model.latent_generator.io import writepdb, load_pdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.callbacks._folding_structure_utils import get_folded_structure_metrics
from lobster.transforms._structure_transforms import StructureBackboneTransform, AminoAcidTokenizerTransform
from tmtools import tm_align

# Set up logging
logging.basicConfig(level=logging.INFO)
# warning emsfolding does not take into account multiple chains
# todo: add support for multiple chains


class MetricsPlotter:
    """Helper class to create plots from metrics data."""

    def __init__(self, output_dir: Path, mode: str):
        """Initialize plotter for a specific generation mode.

        Args:
            output_dir: Directory to save plot files
            mode: Generation mode (unconditional, inverse_folding, forward_folding)
        """
        self.output_dir = output_dir
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_box_plots_from_csv(self, csv_path: Path):
        """Create box and whisker plots from CSV metrics data.

        Args:
            csv_path: Path to the CSV file containing metrics
        """

        # Read CSV data
        df = pd.read_csv(csv_path)

        # Define metrics to plot based on mode
        if self.mode == "unconditional":
            metrics = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]
            length_col = "sequence_length"
        elif self.mode == "inverse_folding":
            metrics = ["percent_identity", "plddt", "predicted_aligned_error", "tm_score", "rmsd"]
            length_col = "sequence_length"
        elif self.mode == "forward_folding":
            metrics = ["tm_score", "rmsd"]
            length_col = "sequence_length"
        else:
            logger.warning(f"Unknown mode for plotting: {self.mode}")
            return

        # Filter out empty values and convert to numeric
        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")

        # Remove rows with NaN values
        df = df.dropna(subset=metrics)

        if df.empty:
            logger.warning("No valid data found for plotting")
            return

        # Create plots for each metric
        for metric in metrics:
            if metric not in df.columns:
                continue

            self._create_single_box_plot(df, metric, length_col)

        # Create combined plot
        self._create_combined_box_plot(df, metrics, length_col)

    def _create_single_box_plot(self, df: pd.DataFrame, metric: str, length_col: str):
        """Create a single box plot for one metric."""
        plt.figure(figsize=(10, 6))

        # Group data by length
        lengths = sorted(df[length_col].unique())
        data_by_length = [df[df[length_col] == length][metric].dropna() for length in lengths]

        # Create box plot
        box_plot = plt.boxplot(data_by_length, labels=lengths, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(lengths)))
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title(
            f"{metric.replace('_', ' ').title()} by Sequence Length\n({self.mode.replace('_', ' ').title()} Generation)"
        )
        plt.xlabel("Sequence Length")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)

        # Rotate x-axis labels if there are many lengths
        if len(lengths) > 5:
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{self.mode}_{metric}_boxplot_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved box plot: {plot_path}")

    def _create_combined_box_plot(self, df: pd.DataFrame, metrics: list, length_col: str):
        """Create a combined subplot with all metrics."""
        lengths = sorted(df[length_col].unique())

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            ax = axes[i]

            # Group data by length
            data_by_length = [df[df[length_col] == length][metric].dropna() for length in lengths]

            # Create box plot
            box_plot = ax.boxplot(data_by_length, labels=lengths, patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(lengths)))
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels if there are many lengths
            if len(lengths) > 5:
                ax.tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"Metrics by Sequence Length - {self.mode.replace('_', ' ').title()} Generation", fontsize=16)
        plt.tight_layout()

        # Save combined plot
        plot_path = self.output_dir / f"{self.mode}_combined_boxplots_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved combined box plots: {plot_path}")


class MetricsCSVWriter:
    """Helper class to write metrics to CSV files."""

    def __init__(self, output_dir: Path, mode: str):
        """Initialize CSV writer for a specific generation mode.

        Args:
            output_dir: Directory to save CSV files
            mode: Generation mode (unconditional, inverse_folding, forward_folding)
        """
        self.output_dir = output_dir
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create CSV file path
        self.csv_path = output_dir / f"{mode}_metrics_{self.timestamp}.csv"

        # Initialize CSV file with headers
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with appropriate headers based on mode."""
        headers = ["run_id", "timestamp", "mode"]

        if self.mode == "unconditional":
            headers.extend(["plddt", "predicted_aligned_error", "tm_score", "rmsd", "sequence_length", "num_samples"])
        elif self.mode == "inverse_folding":
            headers.extend(
                [
                    "percent_identity",
                    "plddt",
                    "predicted_aligned_error",
                    "tm_score",
                    "rmsd",
                    "sequence_length",
                    "input_file",
                ]
            )
        elif self.mode == "forward_folding":
            headers.extend(["tm_score", "rmsd", "sequence_length", "input_file"])

        # Write headers to CSV
        with open(self.csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        logger.info(f"Initialized CSV metrics file: {self.csv_path}")

    def write_batch_metrics(self, metrics: dict, run_id: str, **kwargs):
        """Write batch metrics to CSV.

        Args:
            metrics: Dictionary containing metric values
            run_id: Unique identifier for this run
            **kwargs: Additional data to include (input_file, sequence_length, etc.)
        """

        def _to_scalar(value):
            """Convert tensor to scalar value."""
            if value is None or value == "":
                return ""
            if hasattr(value, "item"):
                return value.item()
            elif hasattr(value, "cpu"):
                return value.cpu().item()
            else:
                return float(value)

        row = [run_id, datetime.now().isoformat(), self.mode]

        if self.mode == "unconditional":
            row.extend(
                [
                    _to_scalar(metrics.get("_plddt", "")),
                    _to_scalar(metrics.get("_predicted_aligned_error", "")),
                    _to_scalar(metrics.get("_tm_score", "")),
                    _to_scalar(metrics.get("_rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("num_samples", ""),
                ]
            )
        elif self.mode == "inverse_folding":
            row.extend(
                [
                    _to_scalar(kwargs.get("percent_identity", "")),
                    _to_scalar(metrics.get("_plddt", "")),
                    _to_scalar(metrics.get("_predicted_aligned_error", "")),
                    _to_scalar(metrics.get("_tm_score", "")),
                    _to_scalar(metrics.get("_rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("input_file", ""),
                ]
            )
        elif self.mode == "forward_folding":
            row.extend(
                [
                    _to_scalar(metrics.get("tm_score", "")),
                    _to_scalar(metrics.get("rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("input_file", ""),
                ]
            )

        # Write row to CSV
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def write_aggregate_stats(self, aggregate_stats: dict, length: int = None):
        """Write aggregate statistics to a separate summary CSV.

        Args:
            aggregate_stats: Dictionary containing aggregate statistics
            length: Optional length parameter for per-length aggregation
        """
        if length is not None:
            summary_csv_path = self.output_dir / f"{self.mode}_summary_length_{length}_{self.timestamp}.csv"
        else:
            summary_csv_path = self.output_dir / f"{self.mode}_summary_{self.timestamp}.csv"

        headers = ["metric", "value", "count", "mode", "timestamp"]
        if length is not None:
            headers.insert(-1, "length")

        with open(summary_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for metric_name, (value, count) in aggregate_stats.items():
                row = [metric_name, value, count, self.mode]
                if length is not None:
                    row.append(length)
                row.append(datetime.now().isoformat())
                writer.writerow(row)

        logger.info(f"Saved aggregate statistics to: {summary_csv_path}")


def _calculate_aggregate_stats(metric_lists: dict) -> dict:
    """Calculate aggregate statistics from lists of metrics.

    Args:
        metric_lists: Dictionary mapping metric names to lists of values

    Returns:
        Dictionary mapping metric names to (average, count) tuples
    """
    aggregate_stats = {}

    for metric_name, values in metric_lists.items():
        if values:
            # Convert tensors to scalars and filter out invalid values
            valid_values = []
            for v in values:
                # Convert tensor to scalar if needed
                if hasattr(v, "item"):
                    scalar_v = v.item()
                elif hasattr(v, "cpu"):
                    scalar_v = v.cpu().item()
                else:
                    scalar_v = float(v)

                # Filter out invalid values (inf, nan)
                if scalar_v != float("inf") and not (isinstance(scalar_v, float) and scalar_v != scalar_v):
                    valid_values.append(scalar_v)

            if valid_values:
                avg_value = sum(valid_values) / len(valid_values)
                aggregate_stats[metric_name] = (avg_value, len(valid_values))
            else:
                aggregate_stats[metric_name] = (0.0, 0)
        else:
            aggregate_stats[metric_name] = (0.0, 0)

    return aggregate_stats


def calculate_percent_identity(ground_truth_seq, generated_seq, mask=None):
    """
    Calculate percent identity between ground truth and generated sequences.

    Args:
        ground_truth_seq: Ground truth sequence tensor of shape (B, L)
        generated_seq: Generated sequence tensor of shape (B, L)
        mask: Optional mask tensor of shape (B, L) to ignore padded positions

    Returns:
        Tensor of percent identities for each sequence in the batch
    """
    # Ensure both sequences have the same shape
    assert ground_truth_seq.shape == generated_seq.shape, "Sequences must have the same shape"

    # Calculate matches
    matches = (ground_truth_seq == generated_seq).float()

    if mask is not None:
        # Only consider positions where mask is 1
        matches = matches * mask.float()
        valid_positions = mask.sum(dim=1).float()
        # Avoid division by zero
        valid_positions = torch.clamp(valid_positions, min=1.0)
        percent_identity = (matches.sum(dim=1) / valid_positions) * 100.0
    else:
        # Consider all positions
        sequence_length = ground_truth_seq.shape[1]
        percent_identity = (matches.sum(dim=1) / sequence_length) * 100.0

    return percent_identity


@hydra.main(version_base=None, config_path="../hydra_config", config_name="generate")
def generate(cfg: DictConfig) -> None:
    """Generate protein structures using genUME model.

    This command-line interface supports:
    - Unconditional generation: Generate novel protein structures from scratch
    - Inverse folding: Generate sequences for given protein structures
    - Optional ESMFold validation of generated structures
    """
    logger.info("Starting genUME structure generation")
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    # Load model
    logger.info("Loading genUME model...")
    if hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        logger.info(f"Loading model from checkpoint: {cfg.model.ckpt_path}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        model = model_cls.load_from_checkpoint(cfg.model.ckpt_path)
    else:
        logger.info("Instantiating fresh model (no checkpoint provided)")
        model = hydra.utils.instantiate(cfg.model)

    model.to(device)
    model.eval()
    logger.info("✓ Model loaded successfully")

    # Initialize ESMFold if requested
    plm_fold = None
    if cfg.generation.get("use_esmfold", False):
        logger.info("Loading ESMFold for structure validation...")
        from lobster.model import LobsterPLMFold

        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=cfg.generation.get("max_length", 512))
        plm_fold.to(device)
        logger.info("✓ ESMFold loaded successfully")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize CSV logging and plotting if enabled
    csv_writer = None
    plotter = None
    if cfg.generation.get("save_csv_metrics", True):
        generation_mode = cfg.generation.mode
        csv_writer = MetricsCSVWriter(output_dir, generation_mode)
        logger.info(f"CSV metrics logging enabled for {generation_mode} mode")

        # Initialize plotter if plotting is enabled
        if cfg.generation.get("create_plots", True):
            plotter = MetricsPlotter(output_dir, generation_mode)
            logger.info(f"Plotting enabled for {generation_mode} mode")

    # Generate structures
    generation_mode = cfg.generation.mode
    logger.info(f"Generation mode: {generation_mode}")

    if generation_mode == "unconditional":
        _generate_unconditional(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inverse_folding":
        _generate_inverse_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "forward_folding":
        _generate_forward_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    else:
        raise ValueError(f"Unknown generation mode: {generation_mode}")

    logger.info("Generation completed successfully!")


def _generate_unconditional(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate structures unconditionally."""
    logger.info("Starting unconditional generation...")

    gen_cfg = cfg.generation
    length = gen_cfg.length
    num_samples = gen_cfg.num_samples
    nsteps = gen_cfg.get("nsteps", 200)
    batch_size = gen_cfg.get("batch_size", 1)

    # Handle both single length and list of lengths
    # Check for ListConfig, list, or tuple
    if hasattr(length, "__iter__") and not isinstance(length, (str, int, float)):
        # Convert ListConfig/list/tuple to regular list if needed
        lengths = list(length)
        logger.info(f"Generating {num_samples} structures for each length in {lengths}")
    else:
        lengths = [int(length)]
        logger.info(f"Generating {num_samples} structures of length {length}")

    # Process each length
    for current_length in lengths:
        # Ensure current_length is an integer
        current_length = int(current_length)

        logger.info("=" * 60)
        logger.info(f"PROCESSING LENGTH: {current_length}")
        logger.info("=" * 60)

        n_iterations = num_samples // batch_size
        logger.info(
            f"Generating {num_samples} structures of length {current_length} with {nsteps} steps, will run with batch size {batch_size} for {n_iterations} iterations"
        )

        # Initialize metrics collection for this length
        all_metrics = []

        for n_iter in range(n_iterations):
            logger.info(f"Iteration {n_iter + 1}/{n_iterations}")

            with torch.no_grad():
                # Generate samples
                generate_sample = model.generate_sample(
                    length=current_length,
                    num_samples=batch_size,
                    nsteps=nsteps,
                    temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                    temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                    stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                    stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                )

                # Create mask for decoding
                mask = torch.ones((batch_size, current_length), device=device)

                # Decode structures
                decoded_x = model.decode_structure(generate_sample, mask)

                # Extract coordinates
                x_recon_xyz = None
                for decoder_name in decoded_x:
                    if "vit_decoder" == decoder_name:
                        x_recon_xyz = decoded_x[decoder_name]
                        break

                if x_recon_xyz is None:
                    raise RuntimeError("No structure decoder found in model output")

                # Extract sequences
                if generate_sample["sequence_logits"].shape[-1] == 33:
                    seq = convert_lobster_aa_tokenization_to_standard_aa(
                        generate_sample["sequence_logits"], device=device
                    )
                else:
                    seq = generate_sample["sequence_logits"].argmax(dim=-1)
                    seq[seq > 21] = 20

                # Save generated structures
                logger.info("Saving generated structures...")
                for i in range(batch_size):
                    filename = (
                        output_dir / f"generated_structure_length_{current_length}_{n_iter * batch_size + i:03d}.pdb"
                    )
                    writepdb(str(filename), x_recon_xyz[i], seq[i])
                    logger.info(f"Saved: {filename}")

                # Optional ESMFold validation
                if plm_fold is not None:
                    logger.info("Validating structures with ESMFold...")
                    batch_metrics = _validate_with_esmfold(
                        seq,
                        x_recon_xyz,
                        plm_fold,
                        device,
                        output_dir,
                        f"generated_structure_length_{current_length}_{n_iter * batch_size + i:03d}",
                        max_length=current_length,
                    )

                    # Log metrics for unconditional generation
                    if batch_metrics:
                        logger.info("ESMFold validation metrics for unconditional generation:")
                        for key, value in batch_metrics.items():
                            logger.info(f"  {key}: {value:.4f}")

                        # Store metrics for CSV logging
                        if csv_writer is not None:
                            run_id = f"unconditional_length_{current_length}_iter_{n_iter:03d}"
                            csv_writer.write_batch_metrics(
                                batch_metrics, run_id, sequence_length=current_length, num_samples=batch_size
                            )

                        # Always collect metrics for aggregate statistics
                        all_metrics.append(batch_metrics)

        # Calculate and log aggregate statistics for this length
        if all_metrics:
            logger.info(f"Calculating aggregate statistics for length {current_length}...")

            # Collect all metric values
            metric_lists = {"_plddt": [], "_predicted_aligned_error": [], "_tm_score": [], "_rmsd": []}

            for metrics in all_metrics:
                for key in metric_lists:
                    if key in metrics:
                        metric_lists[key].append(metrics[key])

            # Calculate aggregate statistics
            aggregate_stats = _calculate_aggregate_stats(metric_lists)

            # Log aggregate statistics
            logger.info("=" * 80)
            logger.info(f"UNCONDITIONAL GENERATION AGGREGATE STATISTICS - LENGTH {current_length}")
            logger.info("=" * 80)

            for metric_name, (avg_value, count) in aggregate_stats.items():
                logger.info(f"Average {metric_name}: {avg_value:.4f} (n={count})")

            logger.info("=" * 80)

            # Write aggregate statistics to CSV if writer is available
            if csv_writer is not None:
                csv_writer.write_aggregate_stats(aggregate_stats, length=current_length)

    # Create plots from CSV data if plotter is available
    if plotter is not None and csv_writer is not None:
        logger.info("Creating box and whisker plots from CSV data...")
        try:
            plotter.create_box_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Box plots created successfully")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")


def _generate_inverse_folding(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate sequences for given structures (inverse folding)."""
    logger.info("Starting inverse folding generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for inverse folding mode")

    # Handle different input formats
    structure_paths = []
    if isinstance(input_structures, str):
        # Single path or glob pattern
        if "*" in input_structures or "?" in input_structures:
            # Glob pattern
            structure_paths = glob.glob(input_structures)
        else:
            # Single file or directory
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                # Find all structure files in directory (PDB, CIF, PT)
                structure_paths = list(glob.glob(str(path / "*.pdb")))
                structure_paths.extend(glob.glob(str(path / "*.cif")))
                structure_paths.extend(glob.glob(str(path / "*.pt")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple)):
        # List of paths
        for path_str in input_structures:
            path = Path(path_str)
            if path.is_file():
                structure_paths.append(str(path))
            else:
                logger.warning(f"Skipping non-existent file: {path_str}")
    else:
        raise ValueError(f"Invalid input_structures format: {type(input_structures)}")

    if not structure_paths:
        raise ValueError("No valid structure files found in input_structures")

    logger.info(f"Found {len(structure_paths)} structure files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 100)
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection

    logger.info(f"Processing structures with {nsteps} generation steps, batch size {batch_size}, n_trials {n_trials}")

    # Initialize StructureBackboneTransform
    structure_transform = StructureBackboneTransform(max_length=cfg.generation.get("max_length", 512))

    # Initialize aggregate statistics collection
    all_percent_identities = []
    all_plddt_scores = []
    all_predicted_aligned_errors = []
    all_tm_scores = []
    all_rmsd_scores = []

    with torch.no_grad():
        # Process structure files in batches
        for batch_start in range(0, len(structure_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(structure_paths))
            batch_paths = structure_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size

            logger.info(f"Processing batch {batch_idx + 1}/{(len(structure_paths) + batch_size - 1) // batch_size}")

            # Load structures from files
            batch_data = []
            valid_indices = []

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    try:
                        structure_data = torch.load(structure_path, map_location="cpu")
                        if structure_data is not None:
                            # Apply StructureBackboneTransform
                            structure_data = structure_transform(structure_data)
                            batch_data.append(structure_data)
                            valid_indices.append(i)
                        else:
                            logger.warning(f"Failed to load structure from {structure_path} - data is None")
                    except Exception as e:
                        logger.warning(f"Failed to load .pt file {structure_path}: {e}")
                else:
                    # Load PDB/CIF file using existing method
                    structure_data = load_pdb(structure_path, add_batch_dim=False)
                    if structure_data is not None:
                        # Apply StructureBackboneTransform
                        structure_data = structure_transform(structure_data)
                        batch_data.append(structure_data)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"Failed to load structure from {structure_path}")

            if not batch_data:
                logger.warning(f"No valid structures in batch {batch_idx + 1}, skipping")
                continue

            # Filter structures by minimum length (30 residues) and make sure sequence tensor does not contain more than 10% 20s
            filtered_batch_data = []
            filtered_valid_indices = []
            for i, data in enumerate(batch_data):
                if data["coords_res"].shape[0] >= 30:
                    percent_20s = (data["sequence"] == 20).sum() / data["sequence"].shape[0]
                    if percent_20s > 0.1:
                        logger.info(
                            f"Skipping structure {batch_paths[valid_indices[i]]} - sequence tensor contains more than 10% 20s"
                        )
                        continue
                    filtered_batch_data.append(data)
                    filtered_valid_indices.append(valid_indices[i])
                else:
                    logger.info(
                        f"Skipping structure {batch_paths[valid_indices[i]]} - too short ({data['coords_res'].shape[0]} residues, minimum 30)"
                    )

            if not filtered_batch_data:
                logger.warning(f"No structures with sufficient length in batch {batch_idx + 1}, skipping")
                continue

            # Prepare batch tensors
            max_length = max(data["coords_res"].shape[0] for data in filtered_batch_data)
            B = len(filtered_batch_data)

            # Initialize tensors
            coords_res = torch.zeros((B, max_length, 3, 3), device=device)
            mask = torch.zeros((B, max_length), device=device)
            indices = torch.zeros((B, max_length), device=device, dtype=torch.long)

            # Fill batch tensors
            for i, data in enumerate(filtered_batch_data):
                L = data["coords_res"].shape[0]
                coords_res[i, :L] = data["coords_res"].to(device)
                mask[i, :L] = data["mask"].to(device)
                indices[i, :L] = data["indices"].to(device)

            # Handle NaN coordinates
            nan_indices = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
            mask[nan_indices] = 0
            coords_res[nan_indices] = 0

            logger.info(f"Batch {batch_idx + 1}: {B} structures, max length {max_length}")

            # Run multiple trials and select best based on TM-score
            best_trial_results = []

            for trial in range(n_trials):
                logger.info(f"Trial {trial + 1}/{n_trials} for batch {batch_idx + 1}")

                # Generate sequences
                generate_sample = model.generate_sample(
                    length=max_length,
                    num_samples=B,
                    inverse_folding=True,
                    nsteps=nsteps,
                    input_structure_coords=coords_res,
                    input_mask=mask,
                    input_indices=indices,
                    temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                    stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                )

                # Decode structures
                decoded_x = model.decode_structure(generate_sample, mask)

                # Extract coordinates
                x_recon_xyz = None
                for decoder_name in decoded_x:
                    if "vit_decoder" == decoder_name:
                        x_recon_xyz = decoded_x[decoder_name]
                        break

                # Extract sequences
                if generate_sample["sequence_logits"].shape[-1] == 33:
                    seq = convert_lobster_aa_tokenization_to_standard_aa(
                        generate_sample["sequence_logits"], device=device
                    )
                else:
                    seq = generate_sample["sequence_logits"].argmax(dim=-1)
                    seq[seq > 21] = 20

                # Calculate TM-scores for this trial
                trial_tm_scores = []
                outputs = None
                pred_coords = None
                trial_folded_structure_metrics = None

                for i in range(B):
                    # Get original coordinates
                    orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure

                    # Get generated sequence
                    seq_i = seq[i, mask[i] == 1]
                    sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                    # For inverse folding, we need to fold the generated sequence with ESMFold
                    # and compare with the original structure
                    if plm_fold is not None:
                        # Tokenize the generated sequence
                        tokenized_input = plm_fold.tokenizer.encode_plus(
                            sequence_str,
                            padding=True,
                            truncation=True,
                            max_length=cfg.generation.get("max_length", 512),
                            add_special_tokens=False,
                            return_tensors="pt",
                        )["input_ids"].to(device)

                        # Fold with ESMFold
                        with torch.no_grad():
                            outputs = plm_fold.model(tokenized_input)

                        # Get folded structure coordinates
                        folded_structure_metrics, pred_coords = get_folded_structure_metrics(
                            outputs, orig_coords[None], [sequence_str], mask=mask[i : i + 1]
                        )

                        trial_tm_scores.append(folded_structure_metrics["_tm_score"])
                        trial_folded_structure_metrics = folded_structure_metrics  # Store for reuse
                        print("TM-score: ", folded_structure_metrics["_tm_score"])

                    else:
                        # If ESMFold is not available, use generated structure as fallback
                        gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure
                        tm_out = tm_align(
                            gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                            orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                            sequence_str,
                            sequence_str,
                        )
                        trial_tm_scores.append(tm_out.tm_norm_chain1)

                # Store trial results
                best_trial_results.append(
                    {
                        "trial": trial,
                        "tm_scores": trial_tm_scores,
                        "avg_tm_score": sum(trial_tm_scores) / len(trial_tm_scores),
                        "generate_sample": generate_sample,
                        "x_recon_xyz": x_recon_xyz,
                        "seq": seq,
                        "esmfold_outputs": outputs,
                        "esmfold_pred_coords": pred_coords,
                        "folded_structure_metrics": trial_folded_structure_metrics,
                    }
                )

            # Select best trial based on average TM-score
            best_trial = max(best_trial_results, key=lambda x: x["avg_tm_score"])
            logger.info(
                f"Selected trial {best_trial['trial'] + 1} with average TM-score: {best_trial['avg_tm_score']:.3f}"
            )

            # Use best trial results
            generate_sample = best_trial["generate_sample"]
            x_recon_xyz = best_trial["x_recon_xyz"]
            seq = best_trial["seq"]

            # Calculate percent identity for inverse folding (compare generated sequence with original)
            # For inverse folding, we need to get the original sequence from the input structure
            original_sequences = []
            for i, valid_idx in enumerate(filtered_valid_indices):
                structure_path = batch_paths[valid_idx]
                if structure_path.endswith(".pt"):
                    # For .pt files, the sequence should be in the loaded data
                    structure_data = torch.load(structure_path, map_location="cpu")
                    if "sequence" in structure_data:
                        orig_seq = structure_data["sequence"]
                        if orig_seq.dim() > 1:
                            orig_seq = orig_seq.squeeze()
                        original_sequences.append(orig_seq)
                    else:
                        raise ValueError(f"No sequence found for structure: {structure_path}")
                else:
                    # For PDB/CIF files, we need to extract sequence from the loaded structure
                    # This is already done in the structure_transform, so we can get it from batch_data
                    if i < len(batch_data) and "sequence" in batch_data[i]:
                        orig_seq = batch_data[i]["sequence"]
                        if orig_seq.dim() > 1:
                            orig_seq = orig_seq.squeeze()
                        original_sequences.append(orig_seq)
                    else:
                        raise ValueError(f"No sequence found for structure: {structure_path}")

            # Calculate percent identity for this batch
            if original_sequences:
                batch_percent_identities = []

                for i, (orig_seq, gen_seq) in enumerate(zip(original_sequences, seq)):
                    # Get the actual length of the original sequence (excluding padding)
                    orig_len = len(orig_seq)
                    gen_len = len(gen_seq)

                    # Use the minimum length to avoid dimension mismatches
                    min_len = min(orig_len, gen_len)

                    if min_len > 0:
                        # Truncate both sequences to the same length and ensure they're on the same device
                        orig_seq_truncated = orig_seq[:min_len].to(device)
                        gen_seq_truncated = gen_seq[:min_len].to(device)

                        # Calculate percent identity for this single sequence
                        percent_identity = calculate_percent_identity(
                            orig_seq_truncated.unsqueeze(0), gen_seq_truncated.unsqueeze(0)
                        )
                        batch_percent_identities.append(percent_identity.item())
                    else:
                        # If sequences are empty, set percent identity to 0
                        batch_percent_identities.append(0.0)

                all_percent_identities.extend(batch_percent_identities)

            # Save results
            logger.info(f"Saving inverse folding results for batch {batch_idx + 1}...")
            for i, valid_idx in enumerate(filtered_valid_indices):
                original_path = batch_paths[valid_idx]
                original_name = Path(original_path).stem
                x_recon_xyz_i_masked = x_recon_xyz[i, mask[i] == 1]
                seq_i_masked = seq[i, mask[i] == 1]

                # Save generated structure
                filename = output_dir / f"inverse_folding_{original_name}_generated.pdb"
                writepdb(str(filename), x_recon_xyz_i_masked, seq_i_masked)
                logger.info(f"Saved: {filename}")

            # Optional ESMFold validation - reuse results from trial selection
            if plm_fold is not None:
                logger.info(f"Validating batch {batch_idx + 1} with ESMFold (reusing trial results)...")

                # Reuse ESMFold results from the best trial
                if best_trial["folded_structure_metrics"] is not None and best_trial["esmfold_pred_coords"] is not None:
                    # Use stored metrics without recalculation
                    folded_structure_metrics = best_trial["folded_structure_metrics"]
                    pred_coords = best_trial["esmfold_pred_coords"]

                    # Log metrics
                    logger.info("ESMFold validation metrics:")
                    for key, value in folded_structure_metrics.items():
                        logger.info(f"  {key}: {value:.4f}")

                    # Save folded structures
                    for i in range(seq.shape[0]):
                        original_name = Path(batch_paths[filtered_valid_indices[i]]).stem
                        filename = output_dir / f"inverse_folding_{original_name}_esmfold.pdb"
                        pred_coords_i_masked = pred_coords[i, mask[i] == 1]
                        seq_i_masked = seq[i, mask[i] == 1]
                        writepdb(str(filename), pred_coords_i_masked, seq_i_masked)
                        logger.info(f"Saved ESMFold structure: {filename}")

                    batch_metrics = folded_structure_metrics
                else:
                    # Fallback to original validation if no stored results
                    logger.warning("No stored ESMFold results, running validation...")
                    batch_metrics = _validate_with_esmfold(
                        seq,
                        x_recon_xyz,
                        plm_fold,
                        device,
                        output_dir,
                        f"inverse_folding_batch{batch_idx:03d}",
                        original_paths=[batch_paths[i] for i in filtered_valid_indices],
                        mask=mask,
                        max_length=max_length,
                    )

                # Collect metrics for aggregate statistics
                if batch_metrics:
                    all_plddt_scores.append(batch_metrics["_plddt"])
                    all_predicted_aligned_errors.append(batch_metrics["_predicted_aligned_error"])
                    all_tm_scores.append(batch_metrics["_tm_score"])
                    all_rmsd_scores.append(batch_metrics["_rmsd"])

                    # Write batch metrics to CSV
                    if csv_writer is not None:
                        run_id = f"inverse_folding_batch_{batch_idx:03d}"
                        csv_writer.write_batch_metrics(
                            batch_metrics,
                            run_id,
                            percent_identity=0.0,
                            sequence_length=max_length,
                            input_file=f"batch_{batch_idx:03d}",
                        )

    # Calculate and report aggregate statistics
    logger.info("=" * 80)
    logger.info("INVERSE FOLDING AGGREGATE STATISTICS")
    logger.info("=" * 80)

    if all_percent_identities:
        avg_percent_identity = sum(all_percent_identities) / len(all_percent_identities)
        logger.info(f"Average Percent Identity: {avg_percent_identity:.2f}% (n={len(all_percent_identities)})")
    else:
        logger.warning("No percent identity data collected")

    if all_plddt_scores:
        avg_plddt = sum(all_plddt_scores) / len(all_plddt_scores)
        logger.info(f"Average pLDDT: {avg_plddt:.2f} (n={len(all_plddt_scores)})")
    else:
        logger.warning("No pLDDT data collected")

    if all_predicted_aligned_errors:
        avg_pae = sum(all_predicted_aligned_errors) / len(all_predicted_aligned_errors)
        logger.info(f"Average Predicted Aligned Error: {avg_pae:.2f} (n={len(all_predicted_aligned_errors)})")
    else:
        logger.warning("No Predicted Aligned Error data collected")

    if all_tm_scores:
        avg_tm_score = sum(all_tm_scores) / len(all_tm_scores)
        logger.info(f"Average TM-Score: {avg_tm_score:.3f} (n={len(all_tm_scores)})")
    else:
        logger.warning("No TM-Score data collected")

    if all_rmsd_scores:
        avg_rmsd = sum(all_rmsd_scores) / len(all_rmsd_scores)
        logger.info(f"Average RMSD: {avg_rmsd:.2f} Å (n={len(all_rmsd_scores)})")
    else:
        logger.warning("No RMSD data collected")

    logger.info("=" * 80)

    # Write aggregate statistics to CSV
    if csv_writer is not None:
        logger.info("Writing inverse folding aggregate statistics to CSV...")

        # Collect all metric values
        metric_lists = {
            "percent_identity": all_percent_identities,
            "plddt": all_plddt_scores,
            "predicted_aligned_error": all_predicted_aligned_errors,
            "tm_score": all_tm_scores,
            "rmsd": all_rmsd_scores,
        }

        # Calculate aggregate statistics
        aggregate_stats = _calculate_aggregate_stats(metric_lists)

        # Write aggregate statistics to CSV
        csv_writer.write_aggregate_stats(aggregate_stats)

    # Create plots from CSV data if plotter is available
    if plotter is not None and csv_writer is not None:
        logger.info("Creating box and whisker plots from CSV data...")
        try:
            plotter.create_box_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Box plots created successfully")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")


def _generate_forward_folding(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate structures from given input structures (forward folding)."""
    logger.info("Starting forward folding generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for forward folding mode")

    # Handle different input formats (same as inverse folding)
    structure_paths = []
    if isinstance(input_structures, str):
        # Single path or glob pattern
        if "*" in input_structures or "?" in input_structures:
            # Glob pattern
            structure_paths = glob.glob(input_structures)
        else:
            # Single file or directory
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                # Find all structure files in directory (PDB, CIF, PT)
                structure_paths = list(glob.glob(str(path / "*.pdb")))
                structure_paths.extend(glob.glob(str(path / "*.cif")))
                structure_paths.extend(glob.glob(str(path / "*.pt")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple)):
        # List of paths
        for path_str in input_structures:
            path = Path(path_str)
            if path.is_file():
                structure_paths.append(str(path))
            else:
                logger.warning(f"Skipping non-existent file: {path_str}")
    else:
        raise ValueError(f"Invalid input_structures format: {type(input_structures)}")

    if not structure_paths:
        raise ValueError("No valid structure files found in input_structures")

    logger.info(f"Found {len(structure_paths)} structure files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 200)  # More steps for forward folding
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection

    logger.info(f"Processing structures with {nsteps} generation steps, batch size {batch_size}, n_trials {n_trials}")

    # Initialize transforms
    structure_transform = StructureBackboneTransform(max_length=cfg.generation.get("max_length", 512))
    tokenizer_transform = AminoAcidTokenizerTransform(max_length=cfg.generation.get("max_length", 512))

    # Initialize aggregate statistics collection
    all_tm_scores = []
    all_rmsd_scores = []

    with torch.no_grad():
        # Process structure files in batches
        for batch_start in range(0, len(structure_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(structure_paths))
            batch_paths = structure_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size

            logger.info(f"Processing batch {batch_idx + 1}/{(len(structure_paths) + batch_size - 1) // batch_size}")

            # Load structures from files
            batch_data = []
            valid_indices = []

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    structure_data = torch.load(structure_path, map_location="cpu")
                    if structure_data is not None:
                        # Apply StructureBackboneTransform
                        structure_data = structure_transform(structure_data)
                        batch_data.append(structure_data)
                        valid_indices.append(i)
                    else:
                        raise ValueError(f"Failed to load structure from {structure_path} - data is None")

                else:
                    # Load PDB/CIF file using existing method
                    structure_data = load_pdb(structure_path, add_batch_dim=False)
                    if structure_data is not None:
                        # Apply StructureBackboneTransform
                        structure_data = structure_transform(structure_data)
                        batch_data.append(structure_data)
                        valid_indices.append(i)
                    else:
                        raise ValueError(f"Failed to load structure from {structure_path}")

            if not batch_data:
                raise ValueError(f"No valid structures in batch {batch_idx + 1}, skipping")

            # Filter structures by minimum length (30 residues) and make sure sequence tensor does not contain more than 10% 20s
            filtered_batch_data = []
            filtered_valid_indices = []
            for i, data in enumerate(batch_data):
                if data["coords_res"].shape[0] >= 30:
                    percent_20s = (data["sequence"] == 20).sum() / data["sequence"].shape[0]
                    if percent_20s > 0.1:
                        logger.info(
                            f"Skipping structure {batch_paths[valid_indices[i]]} - sequence tensor contains more than 10% 20s"
                        )
                        continue
                    filtered_batch_data.append(data)
                    filtered_valid_indices.append(valid_indices[i])
                else:
                    logger.info(
                        f"Skipping structure {batch_paths[valid_indices[i]]} - too short ({data['coords_res'].shape[0]} residues, minimum 30)"
                    )

            if not filtered_batch_data:
                logger.warning(f"No structures with sufficient length in batch {batch_idx + 1}, skipping")
                continue

            # Prepare batch tensors
            max_length = max(data["coords_res"].shape[0] for data in filtered_batch_data)
            B = len(filtered_batch_data)

            # Initialize tensors
            coords_res = torch.zeros((B, max_length, 3, 3), device=device)
            mask = torch.zeros((B, max_length), device=device)
            indices = torch.zeros((B, max_length), device=device, dtype=torch.long)

            # Fill batch tensors
            for i, data in enumerate(filtered_batch_data):
                L = data["coords_res"].shape[0]
                coords_res[i, :L] = data["coords_res"].to(device)
                mask[i, :L] = data["mask"].to(device)
                indices[i, :L] = data["indices"].to(device)

            # Handle NaN coordinates
            nan_indices = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
            mask[nan_indices] = 0
            coords_res[nan_indices] = 0

            logger.info(f"Batch {batch_idx + 1}: {B} structures, max length {max_length}")

            # Extract and tokenize sequences from input structures for forward folding
            input_sequences = []
            for i, data in enumerate(filtered_batch_data):
                if "sequence" in data:
                    seq_tensor = data["sequence"]
                    if seq_tensor.dim() > 1:
                        seq_tensor = seq_tensor.squeeze()

                    # Apply tokenizer transform to the sequence
                    tokenized_data = tokenizer_transform({"sequence": seq_tensor})
                    tokenized_seq = tokenized_data["sequence"]
                    input_sequences.append(tokenized_seq)
                else:
                    raise ValueError(f"No sequence found for structure: {structure_path}")

            # Pad sequences to same length
            padded_sequences = torch.zeros((B, max_length), device=device, dtype=torch.long)
            for i, seq in enumerate(input_sequences):
                seq_len = min(len(seq), max_length)
                padded_sequences[i, :seq_len] = seq[:seq_len]

            # Run multiple trials and select best based on TM-score
            best_trial_results = []

            for trial in range(n_trials):
                logger.info(f"Trial {trial + 1}/{n_trials} for batch {batch_idx + 1}")

                # Generate new structures (forward folding)
                generate_sample = model.generate_sample(
                    length=max_length,
                    num_samples=B,
                    nsteps=nsteps,
                    temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                    temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                    stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                    stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                    forward_folding=True,
                    input_sequence_tokens=padded_sequences,
                    input_mask=mask,
                    input_indices=indices,
                )

                # Decode structures
                decoded_x = model.decode_structure(generate_sample, mask)

                # Extract coordinates
                x_recon_xyz = None
                for decoder_name in decoded_x:
                    if "vit_decoder" == decoder_name:
                        x_recon_xyz = decoded_x[decoder_name]
                        break

                if x_recon_xyz is None:
                    raise RuntimeError("No structure decoder found in model output")

                # Extract sequences
                if generate_sample["sequence_logits"].shape[-1] == 33:
                    seq = convert_lobster_aa_tokenization_to_standard_aa(
                        generate_sample["sequence_logits"], device=device
                    )
                else:
                    seq = generate_sample["sequence_logits"].argmax(dim=-1)
                    seq[seq > 21] = 20

                # Calculate TM-scores for this trial
                trial_tm_scores = []
                for i in range(B):
                    # Get original and generated coordinates
                    orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure
                    gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure

                    # Get sequence for TM-align
                    seq_i = seq[i, mask[i] == 1]
                    sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                    # Calculate TM-Score using TM-align

                    tm_out = tm_align(
                        gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                        orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                        sequence_str,
                        sequence_str,
                    )
                    trial_tm_scores.append(tm_out.tm_norm_chain1)
                    logger.info(f"TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {tm_out.rmsd:.2f} Å")

                # Store trial results
                best_trial_results.append(
                    {
                        "trial": trial,
                        "tm_scores": trial_tm_scores,
                        "avg_tm_score": sum(trial_tm_scores) / len(trial_tm_scores),
                        "generate_sample": generate_sample,
                        "x_recon_xyz": x_recon_xyz,
                        "seq": seq,
                    }
                )

            # Select best trial based on average TM-score
            best_trial = max(best_trial_results, key=lambda x: x["avg_tm_score"])
            logger.info(
                f"Selected trial {best_trial['trial'] + 1} with average TM-score: {best_trial['avg_tm_score']:.3f}"
            )

            # Use best trial results
            generate_sample = best_trial["generate_sample"]
            x_recon_xyz = best_trial["x_recon_xyz"]
            seq = best_trial["seq"]

            # Save generated and original structures
            logger.info(f"Saving forward folding results for batch {batch_idx + 1}...")
            for i, valid_idx in enumerate(filtered_valid_indices):
                original_path = batch_paths[valid_idx]
                original_name = Path(original_path).stem
                x_recon_xyz_i_masked = x_recon_xyz[i, mask[i] == 1]
                seq_i_masked = seq[i, mask[i] == 1]

                # Get original structure coordinates and sequence
                orig_coords_i_masked = coords_res[i, mask[i] == 1, :, :]

                # Save generated structure
                generated_filename = output_dir / f"forward_folding_{original_name}_generated.pdb"
                writepdb(str(generated_filename), x_recon_xyz_i_masked, seq_i_masked)
                logger.info(f"Saved generated: {generated_filename}")

                # Save original structure
                original_filename = output_dir / f"forward_folding_{original_name}_original.pdb"
                writepdb(str(original_filename), orig_coords_i_masked, seq_i_masked)
                logger.info(f"Saved original: {original_filename}")

            # Calculate TM-Score and RMSD between generated and original structures
            logger.info(f"Calculating structural metrics for batch {batch_idx + 1}...")
            batch_tm_scores = []
            batch_rmsd_scores = []

            for i, valid_idx in enumerate(filtered_valid_indices):
                # Get original and generated coordinates
                orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure
                gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure

                # Get sequence for TM-align
                seq_i = seq[i, mask[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # Calculate TM-Score and RMSD using TM-align

                tm_out = tm_align(
                    gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                    orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                    sequence_str,
                    sequence_str,
                )
                logger.info(f"Sequence: {sequence_str}")
                logger.info(f"TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {tm_out.rmsd:.2f} Å")
                batch_tm_scores.append(tm_out.tm_norm_chain1)
                batch_rmsd_scores.append(tm_out.rmsd)

            # Collect metrics for aggregate statistics
            all_tm_scores.extend(batch_tm_scores)
            all_rmsd_scores.extend(batch_rmsd_scores)

            # Write batch metrics to CSV
            if csv_writer is not None:
                run_id = f"forward_folding_batch_{batch_idx:03d}"
                batch_metrics = {
                    "tm_score": sum(batch_tm_scores) / len(batch_tm_scores) if batch_tm_scores else 0.0,
                    "rmsd": sum(batch_rmsd_scores) / len(batch_rmsd_scores) if batch_rmsd_scores else 0.0,
                }
                csv_writer.write_batch_metrics(
                    batch_metrics, run_id, sequence_length=max_length, input_file=f"batch_{batch_idx:03d}"
                )

    # Calculate and report aggregate statistics
    logger.info("=" * 80)
    logger.info("FORWARD FOLDING AGGREGATE STATISTICS")
    logger.info("=" * 80)

    if all_tm_scores:
        avg_tm_score = sum(all_tm_scores) / len(all_tm_scores)
        logger.info(f"Average TM-Score: {avg_tm_score:.3f} (n={len(all_tm_scores)})")
    else:
        logger.warning("No TM-Score data collected")

    if all_rmsd_scores:
        # Filter out infinite RMSD values
        valid_rmsd = [r for r in all_rmsd_scores if r != float("inf")]
        if valid_rmsd:
            avg_rmsd = sum(valid_rmsd) / len(valid_rmsd)
            logger.info(f"Average RMSD: {avg_rmsd:.2f} Å (n={len(valid_rmsd)})")
        else:
            logger.warning("No valid RMSD data collected")
    else:
        logger.warning("No RMSD data collected")

    logger.info("=" * 80)

    # Write aggregate statistics to CSV
    if csv_writer is not None:
        logger.info("Writing forward folding aggregate statistics to CSV...")

        # Collect all metric values
        metric_lists = {"tm_score": all_tm_scores, "rmsd": all_rmsd_scores}

        # Calculate aggregate statistics
        aggregate_stats = _calculate_aggregate_stats(metric_lists)

        # Write aggregate statistics to CSV
        csv_writer.write_aggregate_stats(aggregate_stats)

    # Create plots from CSV data if plotter is available
    if plotter is not None and csv_writer is not None:
        logger.info("Creating box and whisker plots from CSV data...")
        try:
            plotter.create_box_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Box plots created successfully")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")


def _generate_binders(model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None) -> None:
    """Generate binders."""
    raise NotImplementedError("Binder generation is not implemented")


def _validate_with_esmfold(
    seq: torch.Tensor,
    x_recon_xyz: torch.Tensor,
    plm_fold,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    original_paths: list[str] | None = None,
    mask: torch.Tensor | None = None,
    max_length: int | None = 512,
) -> dict[str, float] | None:
    """Validate generated structures using ESMFold."""
    # Convert sequences to strings
    sequence_str = []
    for i in range(seq.shape[0]):
        if mask is not None:
            # do not include the padded positions in the sequence
            seq_i = seq[i, mask[i] == 1]
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq_i]))
        else:
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq[i]]))

    # Tokenize sequences
    tokenized_input = plm_fold.tokenizer.batch_encode_plus(
        sequence_str,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    # Fold with ESMFold
    with torch.no_grad():
        outputs = plm_fold.model(tokenized_input)

    # Get folding metrics
    folded_structure_metrics, pred_coords = get_folded_structure_metrics(outputs, x_recon_xyz, sequence_str, mask=mask)

    # Log metrics
    logger.info("ESMFold validation metrics:")
    for key, value in folded_structure_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save folded structures
    for i in range(len(sequence_str)):
        if original_paths and i < len(original_paths):
            # Use original filename for inverse folding
            original_name = Path(original_paths[i]).stem
            filename = output_dir / f"inverse_folding_{original_name}_esmfold.pdb"
        else:
            # Use generic naming for unconditional generation
            filename = output_dir / f"{prefix}_esmfold_{i:03d}.pdb"
        if mask is not None:
            pred_coords_i_masked = pred_coords[i, mask[i] == 1]
            seq_i_masked = seq[i, mask[i] == 1]
        else:
            pred_coords_i_masked = pred_coords[i]
            seq_i_masked = seq[i]
        writepdb(str(filename), pred_coords_i_masked, seq_i_masked)
        logger.info(f"Saved ESMFold structure: {filename}")

    # Return the metrics for aggregate statistics
    return folded_structure_metrics


if __name__ == "__main__":
    generate()
