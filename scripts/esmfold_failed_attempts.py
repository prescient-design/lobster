#!/usr/bin/env python3
"""Run ESMFold on every QC-rejected attempt saved by the SR concordance run.

For each row in ``failed_self_reflection.csv`` we
  1. Load the saved initial backbone PDB (ground-truth backbone the model wanted to reproduce)
  2. Load the saved initial sequence (the one whose forward-fold lobster rejected)
  3. Run ESMFold on that sequence
  4. Compare ESMFold prediction to the saved backbone via TM-align (CA) and Kabsch-RMSD (N/CA/C)

The output CSV ``esmfold_failed_attempts_<timestamp>.csv`` mirrors the input
schema plus three new columns:
  - ``esmfold_plddt``
  - ``esmfold_tm``    (TM-score: ESMFold prediction vs saved initial backbone)
  - ``esmfold_rmsd``  (Kabsch-aligned all-backbone RMSD vs saved initial backbone)

Usage:
    uv run python scripts/esmfold_failed_attempts.py \\
        --concordance-dir /cv/scratch/u/lisanzas/evaluations/<...>_concordance
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from tmtools import tm_align

from lobster.metrics import align_and_compute_rmsd
from lobster.model._lobster_fold import LobsterPLMFold
from lobster.model.latent_generator.io import load_pdb
from lobster.transforms._structure_transforms import StructureBackboneTransform


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concordance-dir", type=Path, required=True,
                    help="Concordance run output directory containing failed_self_reflection.csv")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Output CSV path (default: esmfold_failed_attempts_<ts>.csv inside concordance dir)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: only process the first N rows (for quick smoke test)")
    args = ap.parse_args()

    failed_csv = args.concordance_dir / "failed_self_reflection.csv"
    if not failed_csv.exists():
        logger.error(f"Missing {failed_csv}")
        sys.exit(1)

    df = pd.read_csv(failed_csv)
    if args.limit is not None:
        df = df.head(args.limit)
    logger.info(f"Loaded {len(df)} failed-attempt rows from {failed_csv}")

    if args.out_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_csv = args.concordance_dir / f"esmfold_failed_attempts_{ts}.csv"
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading ESMFold (esmfold_v1)...")
    plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=args.max_length)
    plm_fold.to(device)
    plm_fold.eval()
    structure_transform = StructureBackboneTransform()
    logger.info("ESMFold ready")

    out_cols = list(df.columns) + ["esmfold_plddt", "esmfold_tm", "esmfold_rmsd"]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(out_cols)

    n_done = 0
    n_skipped = 0
    with torch.no_grad():
        for idx, row in df.iterrows():
            pdb_path = Path(row["pdb_path"])
            sequence = str(row["sequence"]).strip()
            length = int(row["sequence_length"])

            if not pdb_path.exists():
                logger.warning(f"[{idx}] missing PDB {pdb_path}, skipping")
                n_skipped += 1
                continue
            if len(sequence) != length or len(sequence) == 0:
                logger.warning(f"[{idx}] sequence len mismatch ({len(sequence)} vs {length}), skipping")
                n_skipped += 1
                continue

            try:
                gt = load_pdb(str(pdb_path), add_batch_dim=False)
                gt = structure_transform(gt)
                gt_coords_backbone = gt["coords_res"].to(device)  # [L, 3, 3] -- N, CA, C
                gt_coords_ca = gt_coords_backbone[:, 1, :]

                tokens = plm_fold.tokenizer(
                    [sequence],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)

                outputs = plm_fold.model(tokens)
                pred_coords = outputs["positions"][-1]            # [B, L, 14, 3]
                pred_ca = pred_coords[0, :, 1, :]
                pred_backbone = pred_coords[0, :, [0, 1, 2], :]   # [L, 3, 3]
                plddt = outputs["plddt"][0].mean().item()

                tm_out = tm_align(
                    pred_ca.cpu().numpy(),
                    gt_coords_ca.cpu().numpy(),
                    sequence,
                    sequence,
                )
                tm_score = tm_out.tm_norm_chain1

                rmsd = align_and_compute_rmsd(
                    coords1=pred_backbone.to(device),
                    coords2=gt_coords_backbone.to(device),
                    mask=None,
                    return_aligned=False,
                    device=device,
                )

                vals = list(row.values) + [round(plddt, 4), round(float(tm_score), 4), round(float(rmsd), 4)]
                with open(args.out_csv, "a", newline="") as fh:
                    csv.writer(fh).writerow(vals)
                n_done += 1
                if n_done % 25 == 0:
                    logger.info(f"  processed {n_done} | last L={length} TM={tm_score:.3f} RMSD={rmsd:.2f} pLDDT={plddt:.2f}")

            except Exception as exc:
                logger.warning(f"[{idx}] L={length} failed: {exc}")
                n_skipped += 1
                continue

    logger.info(f"Done. processed={n_done}  skipped={n_skipped}  out={args.out_csv}")


if __name__ == "__main__":
    main()
