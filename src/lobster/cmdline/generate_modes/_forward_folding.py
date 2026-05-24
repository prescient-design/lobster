"""LeFlur forward-folding mode.

Predicts a structure given an input sequence. The sequence tokens are held
fixed while the structure stream is sampled, and decoded back to atom
coordinates via the ViT decoder. Comparable in scope to single-sequence
folding baselines like ESMFold but using the LeFlur shared latent.

Called from :func:`lobster.cmdline.generate.generate` when
``cfg.generation.mode == "forward_folding"``.
"""

from __future__ import annotations

from pathlib import Path
import glob

from loguru import logger
from omegaconf import (
    DictConfig,
    ListConfig,
)
from tmtools import tm_align
import torch

from lobster.metrics import (
    calculate_aggregate_stats,
    align_and_compute_rmsd,
)
from lobster.model.latent_generator.io import (
    writepdb,
    load_pdb,
)
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.transforms._structure_transforms import (
    StructureBackboneTransform,
    AminoAcidTokenizerTransform,
)


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

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 200)  # More steps for forward folding
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection

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
                    asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
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

                # Extract structure tokens (argmax)
                structure_tokens = generate_sample["structure_logits"].argmax(dim=-1)  # Shape: [batch_size, length]

                # Calculate TM-scores and RMSDs for this trial
                trial_tm_scores = []
                trial_rmsd_scores = []
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

                    # Calculate RMSD using Kabsch alignment (all backbone atoms)
                    rmsd = align_and_compute_rmsd(
                        coords1=gen_coords,
                        coords2=orig_coords,
                        mask=None,  # Use all positions
                        return_aligned=False,
                        device=device,
                    )
                    trial_rmsd_scores.append(rmsd)
                    logger.info(f"TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {rmsd:.2f} Å")

                # Store trial results
                best_trial_results.append(
                    {
                        "trial": trial,
                        "tm_scores": trial_tm_scores,
                        "rmsd_scores": trial_rmsd_scores,
                        "avg_tm_score": sum(trial_tm_scores) / len(trial_tm_scores),
                        "avg_rmsd": sum(trial_rmsd_scores) / len(trial_rmsd_scores),
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

            # Extract structure tokens from best trial (argmax)
            structure_tokens = generate_sample["structure_logits"].argmax(dim=-1)  # Shape: [batch_size, length]

            # Write sequences to CSV
            if csv_writer is not None:
                # Convert generated sequences to strings
                generated_sequence_strs = []
                structure_token_strs = []
                for i in range(B):
                    seq_i = seq[i, mask[i] == 1]
                    sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                    generated_sequence_strs.append(sequence_str)

                    # Convert structure tokens to comma-separated string
                    tokens_i = structure_tokens[i, mask[i] == 1]
                    tokens_str = ",".join([str(t.item()) for t in tokens_i])
                    structure_token_strs.append(tokens_str)

                # Convert original sequences to strings (from input structures)
                original_sequence_strs = []
                for i, data in enumerate(filtered_batch_data):
                    orig_seq = data["sequence"]
                    if orig_seq.dim() > 1:
                        orig_seq = orig_seq.squeeze()
                    orig_seq_str = "".join([restype_order_with_x_inv[j.item()] for j in orig_seq])
                    original_sequence_strs.append(orig_seq_str)

                # Write to sequences CSV
                csv_writer.write_sequences(
                    sequences=generated_sequence_strs,
                    original_sequences=original_sequence_strs,
                    run_id=f"forward_folding_batch_{batch_idx:03d}",
                    input_structure=[Path(batch_paths[i]).stem for i in filtered_valid_indices],
                    trial_number=best_trial["trial"] + 1,
                    latent_generator_tokens=structure_token_strs,
                )

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

                # Calculate TM-Score using TM-align
                tm_out = tm_align(
                    gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                    orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                    sequence_str,
                    sequence_str,
                )
                batch_tm_scores.append(tm_out.tm_norm_chain1)

                # Calculate RMSD using Kabsch alignment (all backbone atoms)
                rmsd = align_and_compute_rmsd(
                    coords1=gen_coords,
                    coords2=orig_coords,
                    mask=None,  # Use all positions
                    return_aligned=False,
                    device=device,
                )
                batch_rmsd_scores.append(rmsd)

                logger.info(f"Sequence: {sequence_str}")
                logger.info(f"TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {rmsd:.2f} Å")

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

    # Calculate RMSD pass rate (< 2.0Å threshold)
    rmsd_threshold = 2.0
    rmsd_pass_rates = {}

    if all_rmsd_scores:
        # Filter out infinite RMSD values
        valid_rmsd = [r for r in all_rmsd_scores if r != float("inf")]
        if valid_rmsd:
            avg_rmsd = sum(valid_rmsd) / len(valid_rmsd)
            logger.info(f"Average RMSD: {avg_rmsd:.2f} Å (n={len(valid_rmsd)})")

            pass_count = sum(1 for rmsd in valid_rmsd if rmsd < rmsd_threshold)
            total_count = len(valid_rmsd)
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0.0
            rmsd_pass_rates["rmsd"] = (pass_count, total_count, pass_rate)
            logger.info(f"RMSD Pass Rate (< {rmsd_threshold:.1f}Å): {pass_count}/{total_count} ({pass_rate:.1f}%)")
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
        aggregate_stats = calculate_aggregate_stats(metric_lists)

        # Write aggregate statistics to CSV
        csv_writer.write_aggregate_stats(aggregate_stats)

        # Write pass rate statistics to CSV if available
        if rmsd_pass_rates:
            csv_writer.write_pass_rates(rmsd_pass_rates, threshold=rmsd_threshold)

    # Create plots from CSV data if plotter is available
    if plotter is not None and csv_writer is not None:
        logger.info("Creating box and whisker plots from CSV data...")
        try:
            plotter.create_box_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Box plots created successfully")
        except Exception as e:
            logger.error(f"Error creating box plots: {e}")

        # Create correlation plots (only for unconditional mode)
        try:
            plotter.create_correlation_plots_from_csv(csv_writer.csv_path)
        except Exception as e:
            logger.debug(f"Correlation plots not applicable: {e}")
