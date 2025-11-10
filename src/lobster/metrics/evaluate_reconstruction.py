#!/usr/bin/env python3
"""
Script to evaluate reconstruction quality of LatentGenerator models.
Takes structures from a directory, encodes and reconstructs them with multiple models,
and reports average aligned RMSD between original and reconstructed structures.
"""

import os
import glob
import torch
import numpy as np
import argparse
from loguru import logger
from tqdm import tqdm
import json

# Import from latent_generator
from lobster.model.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.io import load_pdb, load_ligand, writepdb_ligand_complex, writepdb
from lobster.model.latent_generator.utils._utils import kabsch_torch_batched
from lobster.transforms._structure_transforms import StructureLigandTransform

ligand_transform = StructureLigandTransform(max_length=512)


def check_file_not_empty(file_path: str) -> bool:
    """Check if a file exists and is not empty."""
    if not os.path.exists(file_path):
        return False

    # Check file size
    if os.path.getsize(file_path) == 0:
        return False

    # For PDB files, check if they contain actual structure data
    if file_path.endswith(".pdb"):
        with open(file_path) as f:
            lines = f.readlines()
            # Check if file contains ATOM or HETATM records
            has_atoms = any(line.startswith(("ATOM", "HETATM")) for line in lines)
            return has_atoms

    # For SDF files, check if they contain structure data
    elif file_path.endswith(".sdf"):
        with open(file_path) as f:
            content = f.read()
            # Check if file contains atom count and coordinates
            return "$$$$" in content and any(line.strip().isdigit() for line in content.split("\n")[:10])

    # For PT files, check if they can be loaded
    elif file_path.endswith(".pt"):
        # Try to load the file to check if it's valid
        data = torch.load(file_path, map_location="cpu")
        return data is not None

    return True


def load_structure_data(file_path: str) -> dict:
    """Load structure data from file (PDB, SDF, or processed PT files)."""
    if file_path.endswith(".pt"):
        # Load already processed data
        structure_data = torch.load(file_path, map_location="cpu")
        if isinstance(structure_data, dict):
            return structure_data
        else:
            # If it's a tensor, wrap it in a dict
            return {"protein_coords": structure_data}

    elif file_path.endswith(".pdb"):
        return load_pdb(file_path)
    elif file_path.endswith(".sdf"):
        return load_ligand(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def compute_aligned_rmsd(
    original_coords: torch.Tensor,
    reconstructed_coords: torch.Tensor,
    mask: torch.Tensor = None,
    structure_type: str = "protein",
) -> float:
    """Compute aligned RMSD between original and reconstructed coordinates.

    Args:
        original_coords: Original coordinates tensor
        reconstructed_coords: Reconstructed coordinates tensor
        mask: Mask indicating valid positions
        structure_type: "protein" or "ligand" - determines alignment strategy
    """
    if mask is None:
        mask = torch.ones(original_coords.shape[:2], device=original_coords.device)

    B, L = original_coords.shape[:2]
    if structure_type == "protein":
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 3).to(reconstructed_coords.device)
    else:
        mask_expanded = mask.to(reconstructed_coords.device)
    original_coords = original_coords.to(reconstructed_coords.device)

    # Align structures using Kabsch algorithm
    aligned_coords = kabsch_torch_batched(
        reconstructed_coords.reshape(B, -1, 3), original_coords.reshape(B, -1, 3), mask_expanded.reshape(B, -1)
    )

    if structure_type == "protein":
        aligned_coords = aligned_coords.reshape(B, L, 3, 3)
        # For proteins, use CA atoms (index 1) for RMSD calculation
        atoms_for_rmsd_original = original_coords[:, :, 1, :]  # CA atoms
        atoms_for_rmsd_aligned = aligned_coords[:, :, 1, :]  # CA atoms
        mask_expanded = mask[:, :].to(original_coords.device)
    elif structure_type == "ligand":
        # For ligands, use all atoms (average across atom dimension)
        atoms_for_rmsd_original = original_coords
        atoms_for_rmsd_aligned = aligned_coords
    else:  # not tested yet
        # Default to all atoms
        atoms_for_rmsd_original = original_coords.reshape(B, L * original_coords.shape[2], 3)
        atoms_for_rmsd_aligned = aligned_coords.reshape(B, L * aligned_coords.shape[2], 3)
        # Adjust mask for flattened coordinates
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, original_coords.shape[2]).reshape(B, -1)

    # Compute RMSD
    squared_diff = torch.sum((atoms_for_rmsd_original - atoms_for_rmsd_aligned) ** 2, dim=-1)
    masked_squared_diff = squared_diff * mask_expanded
    total_atoms = torch.sum(mask_expanded)

    if total_atoms > 0:
        rmsd = torch.sqrt(torch.sum(masked_squared_diff) / total_atoms)
        return rmsd.item(), original_coords, aligned_coords
    else:
        return float("inf"), None, None


def evaluate_model_on_structure(
    model_name: str,
    structure_data: dict,
    structure_path: str,
    save_structures: bool = False,
    output_dir: str = None,
    use_canonical_pose: bool = False,
    num_steps: int = None,
) -> dict:
    """Evaluate a single model on a single structure."""
    results = {
        "model_name": model_name,
        "structure_path": structure_path,
        "num_steps": num_steps,
        "rmsd": float("inf"),
        "success": False,
        "error": None,
    }

    # Check protein length and skip if > 512
    protein_length = None
    if "protein_coords" in structure_data:
        protein_length = structure_data["protein_coords"].shape[1]
    elif "coords_res" in structure_data:
        protein_length = structure_data["coords_res"].shape[1]

    if protein_length is not None and protein_length > 512:
        results["error"] = f"Protein length ({protein_length}) exceeds maximum allowed length (512)"
        logger.warning(f"Skipping {structure_path} - protein length {protein_length} > 512")
        return results

    # Handle different model types
    if "Ligand" in model_name:
        structure_data = ligand_transform(structure_data)
        if "conformers" in structure_data:
            structure_data = structure_data["conformers"][np.random.randint(0, len(structure_data["conformers"]))]
        structure_data["ligand_coords"] = structure_data["atom_coords"][None]
        structure_data["ligand_mask"] = structure_data["mask"][None]
        structure_data["ligand_residue_index"] = structure_data["atom_indices"][None]
        structure_data["ligand_atom_names"] = structure_data["atom_names"]
        structure_data["ligand_indices"] = structure_data["atom_indices"][None]

    # Encode and decode
    if use_canonical_pose:
        frame_type = "mol_frame"
    else:
        frame_type = None

    latents, embeddings = encode(structure_data, return_embeddings=True, frame_type=frame_type)

    # Pass num_steps to decode if provided
    if num_steps is not None:
        decoded_outputs, sequence_outputs = decode(latents, x_emb=embeddings, num_steps=num_steps)
    else:
        decoded_outputs, sequence_outputs = decode(latents, x_emb=embeddings)

    # Extract coordinates from decoded output
    if isinstance(decoded_outputs, dict):
        if "protein_coords" in decoded_outputs and decoded_outputs["protein_coords"] is not None:
            reconstructed_coords = decoded_outputs["protein_coords"]
            if "protein_coords_refinement" in decoded_outputs:
                refined_reconstructed_coords = decoded_outputs["protein_coords_refinement"]
            else:
                refined_reconstructed_coords = None
        else:
            # Ligand-only case
            reconstructed_coords = decoded_outputs["ligand_coords"]
            refined_reconstructed_coords = None
    else:
        reconstructed_coords = decoded_outputs
        refined_reconstructed_coords = None

    # Get original coordinates and determine structure type
    structure_type = "protein"  # default
    if "protein_coords" in structure_data and structure_data["protein_coords"] is not None:
        original_coords = structure_data["protein_coords"]
        structure_type = "protein"
    elif "ligand_coords" in structure_data:
        original_coords = structure_data["ligand_coords"]
        structure_type = "ligand"
    elif "atom_coords" in structure_data:
        original_coords = structure_data["atom_coords"]
        structure_type = "ligand"  # Assume ligand if atom_coords present
    elif "coords_res" in structure_data:
        original_coords = structure_data["coords_res"]
        structure_type = "protein"  # Assume protein for coords_res
    else:
        raise ValueError("No coordinates found in structure data")

    # Get mask
    if structure_type == "protein":
        mask = structure_data.get("mask")
        if mask is None:
            mask = structure_data.get("protein_mask")
    elif structure_type == "ligand":
        mask = structure_data.get("ligand_mask")
    else:
        raise ValueError("Invalid structure type")

    # Compute RMSD with structure type
    rmsd, gt_coords, aligned_pred_coords = compute_aligned_rmsd(
        original_coords, reconstructed_coords, mask, structure_type
    )
    if refined_reconstructed_coords is not None:
        rmsd_refined, gt_coords_refined, aligned_pred_coords_refined = compute_aligned_rmsd(
            original_coords, refined_reconstructed_coords, mask, structure_type
        )

    if rmsd != float("inf"):
        results["rmsd"] = rmsd
        results["success"] = True
        if refined_reconstructed_coords is not None:
            results["rmsd_refined"] = rmsd_refined
            results["success_refined"] = True

        # Save aligned structures if requested
        if save_structures and output_dir is not None and gt_coords is not None and aligned_pred_coords is not None:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Generate base filename from structure path
                structure_name = os.path.splitext(os.path.basename(structure_path))[0]
                model_safe_name = model_name.replace(" ", "_").replace("/", "_")

                # Save ground truth structure
                gt_filename = os.path.join(output_dir, f"{structure_name}_{model_safe_name}_gt.pdb")
                if structure_type == "protein":
                    # Get sequence or create default sequence
                    seq = structure_data.get("seq")
                    if seq is None:
                        # Create default sequence (UNK) for each residue
                        seq = torch.full((gt_coords.shape[1],), 20, dtype=torch.long)  # UNK = 20
                    writepdb(gt_filename, gt_coords, seq)
                else:
                    # For ligands, write as ligand complex
                    writepdb_ligand_complex(
                        gt_filename,
                        ligand_atoms=gt_coords.mean(dim=2),  # Average across atom dimension
                        ligand_atom_names=structure_data.get("ligand_atom_names", None),
                        ligand_resname="GT",
                    )

                # Save aligned prediction structure
                pred_filename = os.path.join(
                    output_dir,
                    f"{structure_name}_{model_safe_name}_{num_steps if num_steps is not None else 'default'}_pred.pdb",
                )
                if structure_type == "protein":
                    # Get sequence or create default sequence
                    seq = structure_data.get("seq")
                    if seq is None:
                        # Create default sequence (UNK) for each residue
                        seq = torch.full((aligned_pred_coords.shape[1],), 20, dtype=torch.long)  # UNK = 20
                    writepdb(pred_filename, aligned_pred_coords, seq)
                else:
                    # For ligands, write as ligand complex
                    writepdb_ligand_complex(
                        pred_filename,
                        ligand_atoms=aligned_pred_coords.mean(dim=2),  # Average across atom dimension
                        ligand_atom_names=structure_data.get("ligand_atom_names", None),
                        ligand_resname="PRED",
                    )

                results["saved_structures"] = {"ground_truth": gt_filename, "prediction": pred_filename}

            except Exception as e:
                logger.warning(f"Failed to save structures for {structure_path}: {e}")
                results["saved_structures"] = None

    else:
        results["success"] = False
        results["error"] = "RMSD computation failed"

    return results


def evaluate_models_on_directory(
    models: list[str],
    data_dir: str,
    output_file: str = "reconstruction_evaluation.json",
    save_structures: bool = False,
    structures_output_dir: str = None,
    use_canonical_pose: bool = False,
    num_steps_list: list[int] = None,
) -> dict:
    """Evaluate multiple models on all structures in a directory."""

    if num_steps_list is None:
        num_steps_list = [None]  # Default to no num_steps specified

    # Find all structure files (including processed PT files)
    pdb_files = glob.glob(os.path.join(data_dir, "*.pdb"))
    sdf_files = glob.glob(os.path.join(data_dir, "*.sdf"))
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    structure_files = pdb_files + sdf_files + pt_files

    # Filter out empty files
    valid_files = []
    for file_path in tqdm(structure_files, desc="Checking files"):
        if check_file_not_empty(file_path):
            valid_files.append(file_path)
        else:
            logger.warning(f"Skipping empty or invalid file: {file_path}")

    logger.info(f"Found {len(pdb_files)} PDB files, {len(sdf_files)} SDF files, and {len(pt_files)} PT files")
    logger.info(f"Total: {len(valid_files)} valid structure files")

    if len(valid_files) == 0:
        logger.error("No valid structure files found!")
        return {}

    # Results storage - now includes step information
    all_results = {"models": {}, "structures": {}, "summary": {}}

    # Initialize model results for each step count
    for model_name in models:
        all_results["models"][model_name] = {}
        for num_steps in num_steps_list:
            step_key = f"steps_{num_steps}" if num_steps is not None else "default"
            all_results["models"][model_name][step_key] = {
                "rmsd_values": [],
                "successful_reconstructions": 0,
                "failed_reconstructions": 0,
                "average_rmsd": float("inf"),
                "std_rmsd": float("inf"),
                "rmsd_values_refined": [],
                "successful_reconstructions_refined": 0,
                "failed_reconstructions_refined": 0,
                "average_rmsd_refined": float("inf"),
                "std_rmsd_refined": float("inf"),
            }

    # Initialize structures dictionary
    for structure_path in valid_files:
        structure_name = os.path.basename(structure_path)
        all_results["structures"][structure_name] = {}

    # Process one model at a time
    for model_name in tqdm(models, desc="Processing models"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading and evaluating model: {model_name}")
        logger.info(f"{'=' * 60}")

        # Load the current model
        logger.info(f"Loading model: {model_name}")
        load_model(
            methods[model_name].model_config.checkpoint,
            methods[model_name].model_config.config_path,
            methods[model_name].model_config.config_name,
            overrides=methods[model_name].model_config.overrides,
        )
        logger.info(f"Model {model_name} loaded successfully")

        # Evaluate this model on all structures with all step counts
        for structure_path in tqdm(valid_files, desc=f"Evaluating {model_name}"):
            structure_name = os.path.basename(structure_path)

            if structure_name not in all_results["structures"]:
                all_results["structures"][structure_name] = {}
            if model_name not in all_results["structures"][structure_name]:
                all_results["structures"][structure_name][model_name] = {}

            # Load structure data once
            structure_data = load_structure_data(structure_path)

            # Test each num_steps value
            for num_steps in num_steps_list:
                step_key = f"steps_{num_steps}" if num_steps is not None else "default"

                # Evaluate model on this structure with this num_steps
                result = evaluate_model_on_structure(
                    model_name,
                    structure_data,
                    structure_path,
                    save_structures,
                    structures_output_dir,
                    use_canonical_pose,
                    num_steps,
                )

                all_results["structures"][structure_name][model_name][step_key] = result

                # Update model statistics
                if result["success"]:
                    all_results["models"][model_name][step_key]["rmsd_values"].append(result["rmsd"])
                    all_results["models"][model_name][step_key]["successful_reconstructions"] += 1
                    if "success_refined" in result and result["success_refined"]:
                        all_results["models"][model_name][step_key]["rmsd_values_refined"].append(
                            result["rmsd_refined"]
                        )
                        all_results["models"][model_name][step_key]["successful_reconstructions_refined"] += 1
                else:
                    all_results["models"][model_name][step_key]["failed_reconstructions"] += 1
                    all_results["models"][model_name][step_key]["failed_reconstructions_refined"] += 1

        # Clear model from memory after evaluation
        logger.info(f"Clearing model {model_name} from memory")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute summary statistics
    for model_name in models:
        for num_steps in num_steps_list:
            step_key = f"steps_{num_steps}" if num_steps is not None else "default"
            model_results = all_results["models"][model_name][step_key]
            if model_results["rmsd_values"]:
                model_results["average_rmsd"] = np.mean(model_results["rmsd_values"])
                model_results["std_rmsd"] = np.std(model_results["rmsd_values"])
                model_results["min_rmsd"] = np.min(model_results["rmsd_values"])
                model_results["max_rmsd"] = np.max(model_results["rmsd_values"])
                if model_results["rmsd_values_refined"]:
                    model_results["average_rmsd_refined"] = np.mean(model_results["rmsd_values_refined"])
                    model_results["std_rmsd_refined"] = np.std(model_results["rmsd_values_refined"])
                    model_results["min_rmsd_refined"] = np.min(model_results["rmsd_values_refined"])
                    model_results["max_rmsd_refined"] = np.max(model_results["rmsd_values_refined"])
            else:
                model_results["average_rmsd"] = float("inf")
                model_results["std_rmsd"] = float("inf")
                model_results["min_rmsd"] = float("inf")
                model_results["max_rmsd"] = float("inf")
                # filler values
                model_results["average_rmsd_refined"] = float("inf")
                model_results["std_rmsd_refined"] = float("inf")
                model_results["min_rmsd_refined"] = float("inf")
                model_results["max_rmsd_refined"] = float("inf")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("RECONSTRUCTION EVALUATION SUMMARY")
    logger.info("=" * 80)

    for model_name in models:
        for num_steps in num_steps_list:
            step_key = f"steps_{num_steps}" if num_steps is not None else "default"
            model_results = all_results["models"][model_name][step_key]
            logger.info(f"\nModel: {model_name} (Steps: {num_steps if num_steps is not None else 'Default'})")
            logger.info(f"  Successful reconstructions: {model_results['successful_reconstructions']}")
            logger.info(f"  Failed reconstructions: {model_results['failed_reconstructions']}")
            if model_results["average_rmsd"] != float("inf"):
                logger.info(f"  Average RMSD: {model_results['average_rmsd']:.3f} ± {model_results['std_rmsd']:.3f} Å")
                logger.info(f"  Min RMSD: {model_results['min_rmsd']:.3f} Å")
                logger.info(f"  Max RMSD: {model_results['max_rmsd']:.3f} Å")
            else:
                logger.info("  Average RMSD: N/A (no successful reconstructions)")
            if model_results["rmsd_values_refined"]:
                logger.info(
                    f"  Successful refined reconstructions: {model_results['successful_reconstructions_refined']}"
                )
                logger.info(f"  Failed refined reconstructions: {model_results['failed_reconstructions_refined']}")
                if model_results["average_rmsd_refined"] != float("inf"):
                    logger.info(
                        f"  Average refined RMSD: {model_results['average_rmsd_refined']:.3f} ± {model_results['std_rmsd_refined']:.3f} Å"
                    )
                    logger.info(f"  Min refined RMSD: {model_results['min_rmsd_refined']:.3f} Å")
                    logger.info(f"  Max refined RMSD: {model_results['max_rmsd_refined']:.3f} Å")
                else:
                    logger.info("  Average refined RMSD: N/A (no successful refined reconstructions)")
    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nDetailed results saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction quality of LatentGenerator models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Models:
{chr(10).join([f"  - {name}" for name in methods.keys()])}

Example usage Protein-only model:
  python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG full attention 2" \
    --data_dir /data2/lisanzas/latent_generator_files/casp_recon/CASP15_merged/ \
    --output_file reconstruction_results_protein_only.json \
    --save_structures \
    --structures_output_dir aligned_structures_protein_only

Example usage Ligand-only model:
  python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG Ligand 20A" \
    --data_dir /data/bucket/lisanza/structures/GEOM/processed/test/ \
    --output_file reconstruction_results_ligand_only.json
        """,
    )

    parser.add_argument("--models", nargs="+", required=True, help="List of model names to evaluate")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data2/lisanzas/latent_generator_files/casp_recon/CASP15_merged/",
        help="Directory containing structure files to evaluate",
    )

    parser.add_argument(
        "--output_file", type=str, default="reconstruction_evaluation.json", help="Output file to save results"
    )

    parser.add_argument(
        "--save_structures",
        action="store_true",
        help="Save aligned ground truth and prediction structures to PDB files",
    )

    parser.add_argument(
        "--structures_output_dir",
        type=str,
        default="aligned_structures",
        help="Directory to save aligned structures (used with --save_structures)",
    )

    parser.add_argument(
        "--use_canonical_pose", action="store_true", help="Make model invariant with mol_frame to get canonical pose"
    )

    parser.add_argument(
        "--num_steps", nargs="+", type=int, help="List of num_steps values to test for each model (e.g., 1 5 10 20)"
    )

    args = parser.parse_args()

    # Validate models
    invalid_models = [model for model in args.models if model not in methods]
    if invalid_models:
        logger.error(f"Invalid model names: {invalid_models}")
        logger.error(f"Available models: {list(methods.keys())}")
        return

    # Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return

    logger.info(f"Evaluating models: {args.models}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output file: {args.output_file}")
    if args.save_structures:
        logger.info(f"Saving aligned structures to: {args.structures_output_dir}")

    # Run evaluation
    results = evaluate_models_on_directory(
        args.models,
        args.data_dir,
        args.output_file,
        args.save_structures,
        args.structures_output_dir,
        args.use_canonical_pose,
        args.num_steps,
    )
    if results:
        logger.info("Evaluation completed successfully!")
    else:
        logger.error("Evaluation failed!")


if __name__ == "__main__":
    main()
