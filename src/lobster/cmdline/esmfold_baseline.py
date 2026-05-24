#!/usr/bin/env python3
"""
ESMFold Baseline for Forward Folding Comparison

This script runs ESMFold as a baseline for forward folding tasks.
It takes input structures, extracts sequences, predicts structures using ESMFold,
and compares predictions to ground truth structures.

Outputs the same CSV format as forward_folding mode for easy comparison.

Usage:
    uv run python -m lobster.cmdline.esmfold_baseline \\
        --config-path "../hydra_config/experiment" \\
        --config-name esmfold_baseline
"""

import glob
from pathlib import Path
from datetime import datetime
import torch
from loguru import logger
import hydra
from omegaconf import DictConfig, ListConfig
import csv

from lobster.model._lobster_fold import LobsterPLMFold
from lobster.transforms._structure_transforms import StructureBackboneTransform
from lobster.model.latent_generator.io import writepdb, load_pdb
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
from lobster.metrics import align_and_compute_rmsd
from tmtools import tm_align


@hydra.main(version_base=None, config_path="../hydra_config", config_name="experiment/esmfold_baseline")
def main(cfg: DictConfig) -> None:
    """
    Run ESMFold baseline for forward folding comparison.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("ESMFold Baseline for Forward Folding")
    logger.info("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed
    seed = cfg.get("seed", 12345)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load ESMFold model
    logger.info("Loading ESMFold model...")
    max_length = cfg.generation.get("max_length", 512)
    plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=max_length)
    plm_fold.to(device)
    plm_fold.eval()
    logger.info("✓ ESMFold loaded successfully")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided")

    # Handle different input formats
    structure_paths = []
    if isinstance(input_structures, str):
        # Single path or glob pattern
        if "*" in input_structures or "?" in input_structures:
            # Glob pattern
            structure_paths = sorted(glob.glob(input_structures))
        else:
            # Single file or directory
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                # Find all structure files in directory (PDB, CIF, PT)
                structure_paths = sorted(list(glob.glob(str(path / "*.pdb"))))
                structure_paths.extend(sorted(glob.glob(str(path / "*.cif"))))
                structure_paths.extend(sorted(glob.glob(str(path / "*.pt"))))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple, ListConfig)):
        # List of paths (includes OmegaConf ListConfig)
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

    # Initialize structure transform
    structure_transform = StructureBackboneTransform()

    # Initialize CSV writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"esmfold_baseline_metrics_{timestamp}.csv"
    sequences_csv_path = output_dir / f"sequences_esmfold_baseline_{timestamp}.csv"

    # Write CSV headers
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_id", "timestamp", "mode", "plddt", "tm_score", "rmsd", "sequence_length", "input_file"])

    with open(sequences_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_id", "sample_idx", "sequence", "original_sequence", "length", "input_structure"])

    logger.info(f"Initialized CSV metrics file: {csv_path}")
    logger.info(f"Initialized sequences CSV file: {sequences_csv_path}")

    # Process structures
    batch_size = cfg.generation.get("batch_size", 1)
    all_tm_scores = []
    all_rmsd_scores = []
    all_plddt_scores = []

    with torch.no_grad():
        # Process structure files in batches
        for batch_start in range(0, len(structure_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(structure_paths))
            batch_paths = structure_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size

            logger.info(f"\nProcessing batch {batch_idx + 1}/{(len(structure_paths) + batch_size - 1) // batch_size}")

            # Load structures from files
            batch_data = []
            valid_indices = []

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    try:
                        structure_data = torch.load(structure_path, map_location="cpu", weights_only=False)
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
                    # Load PDB/CIF file
                    try:
                        structure_data = load_pdb(structure_path, add_batch_dim=False)
                        if structure_data is not None:
                            # Apply StructureBackboneTransform
                            structure_data = structure_transform(structure_data)
                            batch_data.append(structure_data)
                            valid_indices.append(i)
                        else:
                            logger.warning(f"Failed to load structure from {structure_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load structure {structure_path}: {e}")

            if not batch_data:
                logger.warning(f"No valid structures in batch {batch_idx + 1}, skipping")
                continue

            # Filter structures by minimum length (30 residues) and check sequence quality
            filtered_batch_data = []
            filtered_valid_indices = []
            for i, data in enumerate(batch_data):
                if data["coords_res"].shape[0] >= 30:
                    percent_20s = (data["sequence"] == 20).sum() / data["sequence"].shape[0]
                    if percent_20s > 0.1:
                        logger.info(
                            f"Skipping structure {batch_paths[valid_indices[i]]} - sequence contains more than 10% unknown residues"
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

            # Process each structure in the batch
            for i, (data, valid_idx) in enumerate(zip(filtered_batch_data, filtered_valid_indices)):
                original_path = batch_paths[valid_idx]
                original_name = Path(original_path).stem

                # Extract sequence from structure
                seq_tensor = data["sequence"]
                if seq_tensor.dim() > 1:
                    seq_tensor = seq_tensor.squeeze()

                # Convert sequence to string
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_tensor])
                seq_length = len(sequence_str)

                logger.info(f"\nStructure {batch_idx * batch_size + i + 1}: {original_name}")
                logger.info(f"  Sequence length: {seq_length}")
                logger.info(f"  Sequence: {sequence_str[:50]}{'...' if len(sequence_str) > 50 else ''}")

                # Get ground truth coordinates and move to device
                ground_truth_coords = data["coords_res"].to(device)  # Shape: [L, 3, 3]

                # Tokenize sequence for ESMFold
                try:
                    tokenized_input = plm_fold.tokenizer(
                        [sequence_str],
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["input_ids"].to(device)

                    # Predict structure with ESMFold
                    logger.info("  Running ESMFold prediction...")
                    outputs = plm_fold.model(tokenized_input)

                    # Extract predicted coordinates
                    pred_coords = outputs["positions"][-1]  # Shape: [B, L, 14, 3], last recycle
                    pred_coords_ca = pred_coords[0, :, 1, :]  # CA atoms, Shape: [L, 3]
                    pred_coords_backbone = pred_coords[0, :, [0, 1, 2], :]  # N, CA, C atoms, Shape: [L, 3, 3]

                    # Extract pLDDT scores
                    plddt = outputs["plddt"][0]  # Shape: [L]
                    mean_plddt = plddt.mean().item()
                    logger.info(f"  Mean pLDDT: {mean_plddt:.2f}")

                    # Get ground truth CA coordinates for TM-align
                    ground_truth_ca = ground_truth_coords[:, 1, :]  # CA atoms, Shape: [L, 3]

                    # Calculate TM-score using TM-align (CA atoms only)
                    tm_out = tm_align(
                        pred_coords_ca.cpu().numpy(),
                        ground_truth_ca.cpu().numpy(),
                        sequence_str,
                        sequence_str,
                    )

                    tm_score = tm_out.tm_norm_chain1

                    # Calculate RMSD using Kabsch alignment (all backbone atoms: N, CA, C)
                    # This matches the approach in the base generation script
                    # Ensure both tensors are on the same device
                    rmsd = align_and_compute_rmsd(
                        coords1=pred_coords_backbone.to(device),  # ESMFold prediction [L, 3, 3]
                        coords2=ground_truth_coords.to(device),  # Ground truth [L, 3, 3]
                        mask=None,  # Use all positions
                        return_aligned=False,
                        device=device,
                    )

                    logger.info(f"  TM-score: {tm_score:.3f}")
                    logger.info(f"  RMSD (Kabsch): {rmsd:.2f} Å")

                    # Collect metrics
                    all_tm_scores.append(tm_score)
                    all_rmsd_scores.append(rmsd)
                    all_plddt_scores.append(mean_plddt)

                    # Save ESMFold predicted structure
                    esmfold_filename = output_dir / f"esmfold_baseline_{original_name}_predicted.pdb"
                    writepdb(str(esmfold_filename), pred_coords_backbone.cpu(), seq_tensor)
                    logger.info(f"  Saved ESMFold prediction: {esmfold_filename}")

                    # Save ground truth structure
                    ground_truth_filename = output_dir / f"esmfold_baseline_{original_name}_ground_truth.pdb"
                    writepdb(str(ground_truth_filename), ground_truth_coords, seq_tensor)
                    logger.info(f"  Saved ground truth: {ground_truth_filename}")

                    # Write metrics to CSV
                    run_id = f"esmfold_baseline_batch_{batch_idx:03d}_{i}"
                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                run_id,
                                current_timestamp,
                                "esmfold_baseline",
                                round(mean_plddt, 4),
                                round(tm_score, 4),
                                round(rmsd, 4),
                                seq_length,
                                original_name,
                            ]
                        )

                    # Write sequences to CSV
                    with open(sequences_csv_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                run_id,
                                i,
                                sequence_str,
                                sequence_str,  # For ESMFold baseline, input and output sequences are the same
                                seq_length,
                                original_name,
                            ]
                        )

                except Exception as e:
                    logger.error(f"  Error processing structure {original_name}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

    # Calculate and report aggregate statistics
    logger.info("\n" + "=" * 80)
    logger.info("ESMFOLD BASELINE AGGREGATE STATISTICS")
    logger.info("=" * 80)

    if all_tm_scores:
        avg_tm_score = sum(all_tm_scores) / len(all_tm_scores)
        min_tm_score = min(all_tm_scores)
        max_tm_score = max(all_tm_scores)
        logger.info(f"TM-Score Statistics (n={len(all_tm_scores)}):")
        logger.info(f"  Average: {avg_tm_score:.3f}")
        logger.info(f"  Min: {min_tm_score:.3f}")
        logger.info(f"  Max: {max_tm_score:.3f}")
    else:
        logger.warning("No TM-Score data collected")

    if all_rmsd_scores:
        # Filter out infinite RMSD values
        valid_rmsd = [r for r in all_rmsd_scores if r != float("inf")]
        if valid_rmsd:
            avg_rmsd = sum(valid_rmsd) / len(valid_rmsd)
            min_rmsd = min(valid_rmsd)
            max_rmsd = max(valid_rmsd)
            logger.info(f"\nRMSD Statistics (n={len(valid_rmsd)}):")
            logger.info(f"  Average: {avg_rmsd:.2f} Å")
            logger.info(f"  Min: {min_rmsd:.2f} Å")
            logger.info(f"  Max: {max_rmsd:.2f} Å")

            # Calculate RMSD pass rate (< 2.0Å threshold)
            rmsd_threshold = 2.0
            pass_count = sum(1 for rmsd in valid_rmsd if rmsd < rmsd_threshold)
            total_count = len(valid_rmsd)
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0.0
            logger.info(f"  RMSD Pass Rate (< {rmsd_threshold:.1f}Å): {pass_count}/{total_count} ({pass_rate:.1f}%)")
        else:
            logger.warning("No valid RMSD data collected")
    else:
        logger.warning("No RMSD data collected")

    if all_plddt_scores:
        avg_plddt = sum(all_plddt_scores) / len(all_plddt_scores)
        min_plddt = min(all_plddt_scores)
        max_plddt = max(all_plddt_scores)
        logger.info(f"\npLDDT Statistics (n={len(all_plddt_scores)}):")
        logger.info(f"  Average: {avg_plddt:.2f}")
        logger.info(f"  Min: {min_plddt:.2f}")
        logger.info(f"  Max: {max_plddt:.2f}")
    else:
        logger.warning("No pLDDT data collected")

    logger.info("=" * 80)
    logger.info("\n✓ ESMFold baseline completed successfully!")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info(f"  Metrics CSV: {csv_path}")
    logger.info(f"  Sequences CSV: {sequences_csv_path}")
    logger.info(f"  Total structures processed: {len(all_tm_scores)}")


if __name__ == "__main__":
    main()
