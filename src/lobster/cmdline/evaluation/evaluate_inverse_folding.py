"""Evaluate inverse folding on protein-only structures (e.g., CAMEO).

This script evaluates sequence recovery for inverse folding without ligand context.
Useful for comparing protein-only inverse folding performance.

Usage:
    uv run python -m lobster.cmdline.evaluation.evaluate_inverse_folding \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir "/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt" \
        --num_samples 127 \
        --nsteps 100 \
        --device cuda
"""

import argparse
import glob
import os

import torch
from loguru import logger
from tqdm import tqdm

from lobster.model.leflur import LeFlurProteinLigandLightningModule
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
)
from lobster.transforms._structure_transforms import StructureBackboneTransform


def load_structures(data_dir: str, num_samples: int, max_length: int = 512) -> list[dict]:
    """Load structures from .pt files."""
    if "*" in data_dir:
        pt_files = sorted(glob.glob(data_dir))
    else:
        pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    if not pt_files:
        raise ValueError(f"No .pt files found at {data_dir}")

    logger.info(f"Found {len(pt_files)} .pt files")

    transform = StructureBackboneTransform(max_length=max_length)
    structures = []

    for pt_path in tqdm(pt_files[: num_samples * 3], desc="Loading structures"):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            data = transform(data)

            # Filter by length and unknown residues
            if data["coords_res"].shape[0] >= 30:
                percent_unknown = (data["sequence"] == 20).sum().float() / data["sequence"].shape[0]
                if percent_unknown <= 0.1:
                    structures.append(data)

            if len(structures) >= num_samples:
                break
        except Exception as e:
            logger.warning(f"Failed to load {pt_path}: {e}")

    logger.info(f"Loaded {len(structures)} valid structures")
    return structures


def create_dummy_ligand(protein_coords: torch.Tensor, mask: torch.Tensor, num_atoms: int = 10, device: str = "cuda"):
    """Create a dummy ligand near the protein centroid.

    Parameters
    ----------
    protein_coords : Tensor
        Protein coordinates [B, L, 3, 3] (N, CA, C per residue)
    mask : Tensor
        Valid residue mask [B, L]
    num_atoms : int
        Number of dummy ligand atoms
    device : str
        Device to create tensors on

    Returns
    -------
    dict with ligand_coords, ligand_atom_types, ligand_mask
    """
    B, L = mask.shape

    # Get protein centroid (using CA atoms)
    ca_coords = protein_coords[:, :, 1, :]  # [B, L, 3]

    # Compute centroid per sample
    centroids = []
    for i in range(B):
        valid = mask[i].bool()
        if valid.sum() > 0:
            centroid = ca_coords[i, valid].mean(dim=0)
        else:
            centroid = torch.zeros(3, device=device)
        centroids.append(centroid)
    centroids = torch.stack(centroids)  # [B, 3]

    # Create dummy ligand atoms around centroid (random positions within 5Å)
    ligand_coords = centroids.unsqueeze(1) + torch.randn(B, num_atoms, 3, device=device) * 2.0  # [B, num_atoms, 3]

    # Use carbon atoms (index 3 in ELEMENT_VOCAB_EXTENDED)
    ligand_atom_types = torch.full((B, num_atoms), 3, dtype=torch.long, device=device)  # All carbons

    # All atoms valid
    ligand_mask = torch.ones(B, num_atoms, device=device)

    # Simple linear bond matrix (chain of atoms)
    bond_matrix = torch.zeros(B, num_atoms, num_atoms, dtype=torch.long, device=device)
    for i in range(num_atoms - 1):
        bond_matrix[:, i, i + 1] = 1  # Single bond
        bond_matrix[:, i + 1, i] = 1  # Symmetric

    return {
        "ligand_coords": ligand_coords,
        "ligand_atom_types": ligand_atom_types,
        "ligand_mask": ligand_mask,
        "bond_matrix": bond_matrix,
    }


def evaluate_inverse_folding(
    model,
    structures: list[dict],
    device: str,
    nsteps: int = 100,
    output_dir: str | None = None,
    use_dummy_ligand: bool = False,
    dummy_ligand_atoms: int = 10,
    batch_size: int = 10,
) -> dict:
    """Run inverse folding evaluation.

    Parameters
    ----------
    use_dummy_ligand : bool
        If True, add a dummy ligand (random atoms near centroid) to test
        if the model needs ligand context to function properly.
    dummy_ligand_atoms : int
        Number of atoms in the dummy ligand.
    batch_size : int
        Batch size for evaluation.
    """
    model.eval()
    model.to(device)

    all_recoveries = []
    per_sample_results = []

    for batch_start in tqdm(range(0, len(structures), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(structures))
        batch_structures = structures[batch_start:batch_end]

        # Prepare batch
        max_len = max(s["coords_res"].shape[0] for s in batch_structures)
        B = len(batch_structures)

        sequence = torch.zeros((B, max_len), dtype=torch.long, device=device)
        coords_res = torch.zeros((B, max_len, 3, 3), device=device)
        mask = torch.zeros((B, max_len), device=device)
        indices = torch.zeros((B, max_len), dtype=torch.long, device=device)

        for i, s in enumerate(batch_structures):
            L = s["coords_res"].shape[0]
            sequence[i, :L] = s["sequence"].to(device)
            coords_res[i, :L] = s["coords_res"].to(device)
            mask[i, :L] = s["mask"].to(device)
            indices[i, :L] = s["indices"].to(device)

        # Handle NaNs
        nan_mask = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
        mask[nan_mask] = 0
        coords_res[nan_mask] = 0

        # Generate sequences
        with torch.no_grad():
            if use_dummy_ligand:
                # Create dummy ligand for this batch
                dummy = create_dummy_ligand(coords_res, mask, dummy_ligand_atoms, device)

                # Encode dummy ligand structure to tokens
                encode_result = model.encode_ligand_structure(
                    dummy["ligand_coords"],
                    dummy["ligand_mask"],
                    torch.arange(dummy_ligand_atoms, device=device).unsqueeze(0).expand(B, -1),
                    return_continuous=True,
                )
                ligand_structure_tokens, _, ligand_structure_embeddings = encode_result

                result = model.generate_sample(
                    length=max_len,
                    num_samples=B,
                    inverse_folding=True,
                    nsteps=nsteps,
                    input_structure_coords=coords_res,
                    input_mask=mask,
                    input_indices=indices,
                    # Dummy ligand as context
                    generate_ligand=True,
                    num_atoms=dummy_ligand_atoms,
                    input_ligand_atom_tokens=dummy["ligand_atom_types"],
                    input_ligand_structure_tokens=ligand_structure_tokens,
                    input_ligand_structure_embeddings=ligand_structure_embeddings,
                    input_bond_matrix=dummy["bond_matrix"],
                    ligand_is_context=True,
                )
            else:
                # Pure protein-only inverse folding
                result = model.generate_sample(
                    length=max_len,
                    num_samples=B,
                    inverse_folding=True,
                    nsteps=nsteps,
                    input_structure_coords=coords_res,
                    input_mask=mask,
                    input_indices=indices,
                    generate_ligand=False,
                )

        # Get predicted sequences
        seq_logits = result["sequence_logits"]
        if seq_logits.shape[-1] == 33:
            pred_seq = convert_lobster_aa_tokenization_to_standard_aa(seq_logits, device=device)
        else:
            pred_seq = seq_logits.argmax(dim=-1)
            pred_seq[pred_seq > 20] = 20

        # Compute per-sample recovery
        for i in range(B):
            valid_mask = mask[i].bool()
            gt = sequence[i][valid_mask]
            pred = pred_seq[i][valid_mask]
            recovery = (gt == pred).float().mean().item() * 100
            all_recoveries.append(recovery)
            per_sample_results.append(
                {
                    "sample_idx": batch_start + i,
                    "length": int(valid_mask.sum().item()),
                    "recovery": recovery,
                }
            )

        # Save first few structures
        if batch_start == 0 and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            decoded_x = model.decode_structure(result, mask)
            vit_output = decoded_x.get("vit_decoder")
            if isinstance(vit_output, dict):
                x_recon_xyz = vit_output.get("protein_coords")
            else:
                x_recon_xyz = vit_output

            for i in range(min(5, B)):
                pdb_path = os.path.join(output_dir, f"sample_{i}_recovery_{all_recoveries[i]:.1f}.pdb")
                writepdb(pdb_path, x_recon_xyz[i], pred_seq[i])
                logger.info(f"Saved {pdb_path}")

    # Summary statistics
    import numpy as np

    recoveries = np.array(all_recoveries)

    summary = {
        "mean_recovery": float(recoveries.mean()),
        "std_recovery": float(recoveries.std()),
        "median_recovery": float(np.median(recoveries)),
        "min_recovery": float(recoveries.min()),
        "max_recovery": float(recoveries.max()),
        "num_samples": len(recoveries),
    }

    return {"summary": summary, "per_sample": per_sample_results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate inverse folding on protein-only structures")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to .pt files (supports glob patterns)")
    parser.add_argument("--num_samples", type=int, default=127, help="Number of samples to evaluate")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of generation steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save structures")
    parser.add_argument("--dummy_ligand", action="store_true", help="Add dummy ligand to test model bias")
    parser.add_argument("--dummy_ligand_atoms", type=int, default=10, help="Number of atoms in dummy ligand")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for evaluation")
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = LeFlurProteinLigandLightningModule.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
        strict=False,
    )
    logger.info("Model loaded successfully")

    # Load structures
    structures = load_structures(args.data_dir, args.num_samples, args.max_length)

    # Run evaluation
    results = evaluate_inverse_folding(
        model,
        structures,
        args.device,
        args.nsteps,
        args.output_dir,
        use_dummy_ligand=args.dummy_ligand,
        dummy_ligand_atoms=args.dummy_ligand_atoms,
        batch_size=args.batch_size,
    )

    # Print results
    print("\n" + "=" * 70)
    if args.dummy_ligand:
        print(f"Inverse Folding with DUMMY LIGAND ({args.dummy_ligand_atoms} atoms)")
    else:
        print("Protein-Only Inverse Folding Evaluation Results")
    print("=" * 70)
    print(f"\nSamples evaluated: {results['summary']['num_samples']}")
    print("\n--- Sequence Recovery ---")
    print(f"  Mean:   {results['summary']['mean_recovery']:.2f}%")
    print(f"  Std:    {results['summary']['std_recovery']:.2f}%")
    print(f"  Median: {results['summary']['median_recovery']:.2f}%")
    print(f"  Min:    {results['summary']['min_recovery']:.2f}%")
    print(f"  Max:    {results['summary']['max_recovery']:.2f}%")
    print("=" * 70)

    # Show distribution
    import numpy as np

    recoveries = np.array([r["recovery"] for r in results["per_sample"]])
    print("\n--- Recovery Distribution ---")
    for threshold in [10, 20, 30, 40, 50, 60]:
        pct = (recoveries >= threshold).mean() * 100
        print(f"  >= {threshold}%: {pct:.1f}% of samples")


if __name__ == "__main__":
    main()
