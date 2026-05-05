#!/usr/bin/env python3
"""Convert AFDB SwissProt cluster representatives from .pt to .pdb.

Uses pdb_swissprot_clusters.pt to determine cluster representatives.
The cluster dict maps structure_id -> cluster_rep_id, where:
  - 77,519 reps are AF-style IDs (directly loadable from train_processed)
  - 707 reps are non-AF but have AF members (pick first AF member as substitute)
  - 6,347 reps are PDB-only clusters (skipped -- no .pt files available)

Usage:
    cd /cv/home/lisanzas/lobster
    uv run python scripts/convert_afdb_cluster_reps_to_pdb.py \
        --clusters-pt /cv/data/ai4dd/data2/lisanzas/AFDB/pdb_swissprot_clusters.pt \
        --pt-dir /cv/data/ai4dd/data2/lisanzas/AFDB/train_processed \
        --output-dir /cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from lobster.model.latent_generator.io import writepdb


def main():
    parser = argparse.ArgumentParser(
        description="Convert AFDB SwissProt cluster representatives from .pt to .pdb"
    )
    parser.add_argument(
        "--clusters-pt",
        type=str,
        default="/cv/data/ai4dd/data2/lisanzas/AFDB/pdb_swissprot_clusters.pt",
    )
    parser.add_argument(
        "--pt-dir",
        type=str,
        default="/cv/data/ai4dd/data2/lisanzas/AFDB/train_processed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb",
    )
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--array-size", type=int, default=None)
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Loading cluster dict from {args.clusters_pt}")
    cluster_dict = torch.load(args.clusters_pt, map_location="cpu", weights_only=True)

    cluster_to_members = defaultdict(list)
    for struct_id, cluster_id in cluster_dict.items():
        cluster_to_members[cluster_id].append(struct_id)

    all_keys = set(cluster_dict.keys())

    # For each cluster, pick one AF-style structure to convert
    reps_to_convert: list[tuple[str, str]] = []  # (cluster_label, af_structure_id)
    skipped_no_af = 0

    for cluster_label, members in sorted(cluster_to_members.items()):
        if cluster_label in all_keys:
            reps_to_convert.append((cluster_label, cluster_label))
        else:
            af_members = [m for m in members if m.startswith("AF-")]
            if af_members:
                reps_to_convert.append((cluster_label, af_members[0]))
            else:
                skipped_no_af += 1

    logger.info(
        f"Will convert {len(reps_to_convert):,} clusters "
        f"(skipped {skipped_no_af:,} PDB-only clusters with no AF member)"
    )

    if args.array_index is not None and args.array_size is not None:
        chunk_size = (len(reps_to_convert) + args.array_size - 1) // args.array_size
        start = args.array_index * chunk_size
        end = min(start + chunk_size, len(reps_to_convert))
        reps_to_convert = reps_to_convert[start:end]
        logger.info(
            f"Array task {args.array_index}/{args.array_size}: "
            f"processing {start}-{end} ({len(reps_to_convert)} reps)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    failed = 0

    for cluster_label, af_id in tqdm(reps_to_convert, desc="Converting to PDB"):
        pt_path = pt_dir / f"{af_id}.pt"
        safe_label = str(cluster_label).replace("/", "_")
        pdb_path = output_dir / f"{safe_label}.pdb"

        if pdb_path.exists():
            written += 1
            continue

        if not pt_path.exists():
            missing += 1
            continue

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            coords = data["coords_res"]
            seq = data["sequence"]

            if coords.dim() == 3:
                coords = coords.squeeze(0)
            if seq.dim() > 1:
                seq = seq.squeeze()

            if coords.shape[0] != seq.shape[0]:
                failed += 1
                continue

            writepdb(str(pdb_path), coords, seq, add_cb_o=True)
            written += 1
        except Exception:
            failed += 1

    logger.info(f"Wrote {written:,} PDB files to {output_dir}")
    if missing:
        logger.warning(f"Missing {missing:,} .pt files")
    if failed:
        logger.warning(f"Failed {failed:,} conversions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
