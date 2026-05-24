"""Merge local cofold batch results back into an evaluation CSV.

After a SLURM array job completes, this script reads the per-task output JSONs
from the results directory and either:
1. Merges cofold metrics into an existing evaluation CSV (--eval_csv)
2. Creates a standalone cofold results CSV (--output)

When --data_dir is provided, structural comparison metrics (TM-score, RMSD,
pocket RMSD, ligand RMSD, ligand-in-pocket) are computed against ground truth.

Usage:
    python -m lobster.cmdline.merge_cofold_results \
        --results_dir /scratch/cofold_batch_001/results \
        --eval_csv /scratch/eval_results/forward_folding_results.csv \
        --data_dir /path/to/posebusters_benchmark_no_overlap \
        --id_col pdb_id \
        --output merged_results.csv \
        --parse_structures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def load_cofold_results(results_dir: str) -> pd.DataFrame:
    """Load all output JSONs from a results directory into a DataFrame."""
    results_path = Path(results_dir)
    rows = []

    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        row = {"cofold_id": data["id"], "cofold_error": data.get("error")}

        confidence = data.get("confidence", {})
        for key, value in confidence.items():
            if isinstance(value, (int, float)):
                row[f"cofold_{key}"] = value

        row["cofold_has_structure"] = data.get("structure") is not None

        if data.get("structure"):
            row["cofold_structure_length"] = len(data["structure"])

        rows.append(row)

    if not rows:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    n_success = df["cofold_error"].isna().sum()
    n_failed = df["cofold_error"].notna().sum()
    print(f"Loaded {len(df)} results: {n_success} succeeded, {n_failed} failed")
    return df


def extract_structure_metrics(
    results_dir: str,
    data_dir: str | None = None,
    pocket_threshold: float = 5.0,
) -> pd.DataFrame:
    """Parse cofold structures and compute structural metrics.

    When ``data_dir`` is provided, computes TM-score, RMSD, pocket RMSD, and
    ligand placement metrics against ground-truth coordinates. Otherwise only
    counts residues and ligand atoms in the predicted structure.
    """
    import torch
    from lobster.metrics.pylon_client import (
        parse_structure_to_coords,
        parse_mmcif_ligand_coords,
    )
    from lobster.model.latent_generator.utils.residue_constants import (
        restype_order_with_x_inv,
    )

    results_path = Path(results_dir)
    rows = []

    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if data.get("error") or not data.get("structure"):
            continue

        sample_id = data["id"]
        row = {"cofold_id": sample_id}
        structure_text = data["structure"]

        try:
            pred_backbone = parse_structure_to_coords(structure_text)
            row["cofold_n_residues"] = len(pred_backbone)
        except Exception as e:
            row["cofold_parse_error"] = str(e)
            rows.append(row)
            continue

        try:
            pred_ligand = parse_mmcif_ligand_coords(structure_text)
            row["cofold_n_ligand_atoms"] = len(pred_ligand)
        except Exception:
            pred_ligand = None
            row["cofold_n_ligand_atoms"] = 0

        if data_dir is None:
            rows.append(row)
            continue

        # Resolve GT target ID. Best-of-N drivers append a `_d<idx>` suffix
        # to make per-candidate UIDs (e.g. `5S8I_2LY_d3`); strip it to find
        # the shared GT files (`5S8I_2LY_protein.pt`).
        import re as _re
        gt_id = sample_id
        gt_prot_path = os.path.join(data_dir, f"{gt_id}_protein.pt")
        if not os.path.exists(gt_prot_path):
            stripped = _re.sub(r"_d\d+$", "", sample_id)
            if stripped != sample_id:
                gt_id = stripped
                gt_prot_path = os.path.join(data_dir, f"{gt_id}_protein.pt")
        gt_lig_path = os.path.join(data_dir, f"{gt_id}_ligand.pt")
        if not os.path.exists(gt_prot_path):
            rows.append(row)
            continue

        gt_prot = torch.load(gt_prot_path, weights_only=False, map_location="cpu")
        gt_coords = gt_prot.get("coords_res", gt_prot.get("coords"))
        gt_seq = gt_prot.get("sequence")
        gt_mask = gt_prot.get("mask", torch.ones(gt_coords.shape[0]))
        if gt_coords is None or gt_seq is None:
            rows.append(row)
            continue

        min_len = min(len(pred_backbone), len(gt_coords))
        if min_len < 10:
            rows.append(row)
            continue

        pred_bb = pred_backbone[:min_len]
        gt_bb = gt_coords[:min_len].cpu()
        seq_t = gt_seq[:min_len]
        mask_t = gt_mask[:min_len].bool()

        seq_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in seq_t.cpu().tolist())

        # TM-score (tm_align does its own alignment)
        try:
            from tmtools import tm_align

            pred_ca = pred_bb[:, 1, :].detach().numpy()
            gt_ca = gt_bb[:, 1, :].detach().numpy()
            tm_out = tm_align(pred_ca, gt_ca, seq_str, seq_str)
            row["cofold_tm_score"] = float(tm_out.tm_norm_chain1)
        except Exception:
            row["cofold_tm_score"] = float("nan")

        # Kabsch-aligned RMSD (overall, pocket, ligand)
        from lobster.metrics import align_and_compute_rmsd

        valid = mask_t
        pred_aligned = None
        if valid.sum() > 0:
            try:
                aligned, rmsd_val = align_and_compute_rmsd(
                    pred_bb.float(),
                    gt_bb.float(),
                    mask=valid,
                    return_aligned=True,
                )
                row["cofold_rmsd_overall"] = float(rmsd_val)
                pred_aligned = aligned
            except Exception:
                row["cofold_rmsd_overall"] = float("nan")
        else:
            row["cofold_rmsd_overall"] = float("nan")

        # Pocket RMSD (computed on the already-aligned structure)
        pocket_mask = torch.zeros(min_len, dtype=torch.bool)
        gt_lig_coords = None
        if os.path.exists(gt_lig_path):
            gt_lig = torch.load(gt_lig_path, weights_only=False, map_location="cpu")
            gt_lig_coords = gt_lig.get("atom_coords", gt_lig.get("coords"))
            if gt_lig_coords is not None:
                ca_gt = gt_bb[:, 1, :].float()
                dists = torch.cdist(ca_gt.unsqueeze(0), gt_lig_coords.unsqueeze(0).float()).squeeze(0)
                pocket_mask = (dists.min(dim=1).values < pocket_threshold) & mask_t

        if pocket_mask.sum() > 0 and pred_aligned is not None:
            diff_pkt = pred_aligned[pocket_mask].float() - gt_bb[pocket_mask].float()
            row["cofold_rmsd_pocket"] = float(torch.sqrt((diff_pkt**2).sum(dim=-1).mean()).item())
        else:
            row["cofold_rmsd_pocket"] = float("nan")

        row["cofold_n_pocket_residues"] = int(pocket_mask.sum())

        # Ligand placement metrics
        if pred_ligand is not None and gt_lig_coords is not None and len(pred_ligand) > 0:
            ml = min(len(pred_ligand), len(gt_lig_coords))
            if ml > 0:
                pl = pred_ligand[:ml].float()
                gl = gt_lig_coords[:ml].float()
                row["cofold_ligand_centroid_dist"] = float(torch.norm(pl.mean(0) - gl.mean(0)).item())
                diff_lig = pl - gl
                row["cofold_ligand_rmsd"] = float(torch.sqrt((diff_lig**2).sum(dim=-1).mean()).item())

                # Contact-based ligand placement (CA within 6A of ligand atoms)
                from lobster.metrics._generation_utils import compute_protein_ligand_contacts

                pred_ca = pred_bb[:, 1, :].float()
                contact_out = compute_protein_ligand_contacts(pred_ca, pl, contact_threshold=6.0)
                row["cofold_n_protein_ligand_contacts"] = contact_out["n_contacts"]
                row["cofold_ligand_contacts_protein"] = contact_out["n_contacts"] > 0

                # Ligand in correct pocket: contacts GT pocket residues
                if pocket_mask.any():
                    clen = min(len(contact_out["contact_mask"]), len(pocket_mask))
                    pocket_contact = contact_out["contact_mask"][:clen] & pocket_mask[:clen]
                    row["cofold_n_pocket_contacts"] = int(pocket_contact.sum().item())
                    row["cofold_ligand_in_pocket"] = pocket_contact.any().item()
                else:
                    row["cofold_n_pocket_contacts"] = 0
                    row["cofold_ligand_in_pocket"] = False

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Merge cofold batch results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with output JSONs")
    parser.add_argument("--eval_csv", type=str, default=None, help="Existing evaluation CSV to merge into")
    parser.add_argument("--id_col", type=str, default="pdb_id", help="Column name for sample ID in eval CSV")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--parse_structures", action="store_true", help="Also parse structures for metrics")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to processed data dir (*_protein.pt, *_ligand.pt) for structural comparison vs GT",
    )
    parser.add_argument(
        "--pocket_threshold",
        type=float,
        default=5.0,
        help="Distance threshold (A) for binding pocket definition (default: 5.0)",
    )
    args = parser.parse_args()

    cofold_df = load_cofold_results(args.results_dir)
    if cofold_df.empty:
        sys.exit(1)

    if args.parse_structures:
        struct_df = extract_structure_metrics(
            args.results_dir,
            data_dir=args.data_dir,
            pocket_threshold=args.pocket_threshold,
        )
        if not struct_df.empty:
            cofold_df = cofold_df.merge(struct_df, on="cofold_id", how="left")

    if args.eval_csv:
        eval_df = pd.read_csv(args.eval_csv)
        print(f"Loaded evaluation CSV: {len(eval_df)} rows, columns: {list(eval_df.columns)}")
        merged = eval_df.merge(cofold_df, left_on=args.id_col, right_on="cofold_id", how="left")
        merged.drop(columns=["cofold_id"], inplace=True, errors="ignore")
        print(f"Merged result: {len(merged)} rows")
    else:
        merged = cofold_df

    merged.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    n_with_confidence = merged.filter(like="cofold_").notna().any(axis=1).sum()
    print(f"\nSummary: {n_with_confidence}/{len(merged)} rows have cofold results")

    cofold_cols = [
        c
        for c in merged.columns
        if c.startswith("cofold_")
        and c
        not in (
            "cofold_id",
            "cofold_error",
            "cofold_has_structure",
            "cofold_structure_length",
            "cofold_parse_error",
            "cofold_ligand_in_pocket",
        )
    ]
    if cofold_cols:
        print("\nCofold metric averages:")
        for col in cofold_cols:
            if merged[col].dtype in ("float64", "float32", "int64"):
                print(f"  {col}: {merged[col].mean():.4f}")

    if "cofold_ligand_contacts_protein" in merged.columns:
        lcp = merged["cofold_ligand_contacts_protein"].dropna()
        if len(lcp) > 0:
            print(f"  cofold_ligand_contacts_protein: {lcp.mean():.1%} ({int(lcp.sum())}/{len(lcp)})")
    if "cofold_ligand_in_pocket" in merged.columns and "cofold_tm_score" in merged.columns:
        lip = merged["cofold_ligand_in_pocket"].dropna()
        tm = merged["cofold_tm_score"]
        good_fold = tm > 0.5
        lip_good = (merged["cofold_ligand_in_pocket"] & good_fold).dropna()
        if len(lip) > 0:
            print(f"  cofold_ligand_in_pocket: {lip.mean():.1%} ({int(lip.sum())}/{len(lip)})")
            print(
                f"  cofold_good_fold_and_in_pocket (TM>0.5): {lip_good.mean():.1%} ({int(lip_good.sum())}/{len(lip_good)})"
            )


if __name__ == "__main__":
    main()
