"""PLL-score the SR-rejected attempts from a concordance run.

For every row in `failed_self_reflection.csv` we have:
  - the saved sequence (string)
  - the saved initial backbone PDB (the structure the SR forward-fold check rejected)
  - the lobster forward-fold TM score (the SR gate quantity)

We:
  1. Load the PDB and encode it into structure tokens via the model's own
     `encode_structure` (= the same latent-generator quantization that would
     have been used at training/generation time).
  2. Tokenize the sequence string.
  3. Run `score_one_sample` from scripts/score_gen_ume_pll.py to get the same
     PLL variants as the SR-accepted set.
  4. Merge with the post-hoc ESMFold-of-failed-attempts CSV so each row has
     PLL + lobster_forward_tm + ESMFold (TM, RMSD, pLDDT).

Usage:
    uv run python scripts/score_gen_ume_pll_failed_attempts.py \\
        --concordance-dir /cv/scratch/u/lisanzas/evaluations/<...>_concordance \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/<...>.ckpt \\
        --K 32
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from score_gen_ume_pll import _build_inputs, score_one_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_gen_ume_pll_failed_attempts")


_PLL_COLS = [
    "seq_score_unif",
    "seq_score_arllh",
    "struc_score_unif",
    "struc_score_arllh",
    "joint_score_unif",
    "joint_score_arllh",
    "seq_score_t0.25",
    "seq_score_t0.5",
    "seq_score_t0.75",
    "struc_score_t0.25",
    "struc_score_t0.5",
    "struc_score_t0.75",
]


def _encode_pdb_to_struc_tokens(model, structure_transform, pdb_path: Path, device: torch.device):
    """Load a PDB and turn its backbone coords into the same structure tokens
    the model would use at training time. Returns (struc_tokens [L], seq_string).
    """
    from lobster.model.latent_generator.io import load_pdb
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv

    data = load_pdb(str(pdb_path), add_batch_dim=False)
    if data is None:
        raise RuntimeError(f"Failed to load PDB: {pdb_path}")
    data = structure_transform(data)

    coords = data["coords_res"].to(device).unsqueeze(0)
    mask = data["mask"].to(device).unsqueeze(0)
    if "indices" in data:
        indices = data["indices"].to(device).unsqueeze(0)
    else:
        L = coords.shape[1]
        indices = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

    nan_pos = torch.isnan(coords).any(dim=-1).any(dim=-1)
    mask = mask.float()
    mask[nan_pos] = 0.0
    coords = torch.nan_to_num(coords, nan=0.0)

    with torch.no_grad():
        _x_quant, x_quant_emb, mask_out = model.encode_structure(coords, mask, indices)

    keep = mask_out[0].bool() if mask_out.dim() == 2 else mask_out.squeeze().bool()
    struc_tokens_full = x_quant_emb[0].argmax(dim=-1)
    struc_tokens = struc_tokens_full[keep]

    pdb_seq_str = None
    if "sequence" in data:
        seq_tensor = data["sequence"]
        if seq_tensor.dim() > 1:
            seq_tensor = seq_tensor.squeeze()
        pdb_seq_str = "".join(restype_order_with_x_inv.get(int(j), "X") for j in seq_tensor)

    return struc_tokens.to(device).long(), pdb_seq_str


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concordance-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--K", type=int, default=32, help="Random-t MC draws per modality")
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows (smoke test)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    failed_csv = args.concordance_dir / "failed_self_reflection.csv"
    if not failed_csv.exists():
        raise FileNotFoundError(f"Missing {failed_csv}")

    df = pd.read_csv(failed_csv)
    if args.limit is not None:
        df = df.head(args.limit)
    logger.info("Loaded %d failed-attempt rows from %s", len(df), failed_csv)

    esm_csvs = sorted(args.concordance_dir.glob("esmfold_failed_attempts_*.csv"))
    esm_df = None
    if esm_csvs:
        esm_df = pd.read_csv(esm_csvs[-1])
        logger.info("Joining ESMFold-of-failed-attempts CSV: %s (%d rows)", esm_csvs[-1].name, len(esm_df))

    output_path = args.output or (
        args.concordance_dir
        / f"pll_scores_failed_attempts_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
    )
    logger.info("Writing PLL scores to %s", output_path)

    device = torch.device(args.device)

    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule
    from lobster.transforms._structure_transforms import (
        AminoAcidTokenizerTransform,
        StructureBackboneTransform,
    )

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = UMESequenceStructureEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)
    model.interpolant_seq.device = device
    model.interpolant_struc.device = device
    logger.info("Model loaded in %.1fs (device=%s)", time.time() - t0, device)

    seq_mask_id = int(getattr(model, "mask_token_id"))
    struc_mask_id = int(getattr(model, "mask_index_struc_tokens"))
    aa_transform = AminoAcidTokenizerTransform(max_length=args.max_length)
    structure_transform = StructureBackboneTransform()

    out_cols = list(df.columns)
    out_cols += [c for c in _PLL_COLS]
    out_cols += ["scoring_seed", "pdb_seq_matches_csv_seq"]
    if esm_df is not None:
        for c in ("esmfold_plddt", "esmfold_tm", "esmfold_rmsd"):
            if c in esm_df.columns and c not in out_cols:
                out_cols.append(c)

    if esm_df is not None:
        join_cols = [c for c in ("length", "iteration", "retry_count") if c in df.columns and c in esm_df.columns]
        merged = df.merge(
            esm_df[join_cols + [c for c in ("esmfold_plddt", "esmfold_tm", "esmfold_rmsd") if c in esm_df.columns]],
            on=join_cols,
            how="left",
        )
    else:
        merged = df

    fh = output_path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore")
    writer.writeheader()

    n_done = 0
    n_skipped = 0
    t_start = time.time()

    try:
        for row_idx, row in merged.iterrows():
            seq_str = str(row.get("sequence", "")).strip()
            pdb_path = Path(str(row.get("pdb_path", "")).strip())
            length = int(row.get("sequence_length", row.get("length", 0)) or 0)

            if not seq_str or not pdb_path.exists() or length <= 0:
                n_skipped += 1
                logger.warning("Skipping row %s: missing seq or pdb (pdb_exists=%s, len=%d)",
                               row_idx, pdb_path.exists(), length)
                continue
            if length > args.max_length:
                n_skipped += 1
                continue

            try:
                struc_tokens, pdb_seq = _encode_pdb_to_struc_tokens(
                    model, structure_transform, pdb_path, device
                )
            except Exception as e:
                n_skipped += 1
                logger.warning("Skipping row %s: encode_structure failed: %s", row_idx, e)
                continue

            if struc_tokens.shape[0] != len(seq_str):
                n_skipped += 1
                logger.warning(
                    "Skipping row %s: token/seq length mismatch (struc=%d, seq=%d)",
                    row_idx, int(struc_tokens.shape[0]), len(seq_str),
                )
                continue

            try:
                seq_clean, struc_clean, mask_t, residue_index = _build_inputs(
                    seq_str, struc_tokens.tolist(), aa_transform, device
                )
            except Exception as e:
                n_skipped += 1
                logger.warning("Skipping row %s: _build_inputs failed: %s", row_idx, e)
                continue

            sample_seed = (args.seed * 1_000_003 + int(row_idx)) & 0x7FFFFFFF
            try:
                scores = score_one_sample(
                    model,
                    seq_clean=seq_clean,
                    struc_clean=struc_clean,
                    mask=mask_t,
                    residue_index=residue_index,
                    K=args.K,
                    seed=sample_seed,
                    seq_mask_id=seq_mask_id,
                    struc_mask_id=struc_mask_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skipped += 1
                logger.warning("OOM on row %s (L=%d); skipping", row_idx, length)
                continue

            out_row = dict(row)
            out_row["scoring_seed"] = sample_seed
            out_row["pdb_seq_matches_csv_seq"] = (pdb_seq == seq_str) if pdb_seq is not None else ""
            for k in _PLL_COLS:
                if k in scores:
                    v = scores[k]
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        out_row[k] = ""
                    else:
                        out_row[k] = v
            writer.writerow(out_row)
            fh.flush()
            n_done += 1

            if (n_done % args.log_every) == 0:
                dt = time.time() - t_start
                logger.info(
                    "[%4d/%d] L=%d  forward_tm=%.3f  struc_unif=%.3f  joint_unif=%.3f  (%.2fs/sample)",
                    n_done,
                    len(merged),
                    length,
                    float(row.get("tm_score_unconditional_to_forward", float("nan"))),
                    scores["struc_score_unif"],
                    scores["joint_score_unif"],
                    dt / max(1, n_done),
                )
    finally:
        fh.close()

    logger.info("Done. Scored %d rows; skipped %d. Output: %s", n_done, n_skipped, output_path)


if __name__ == "__main__":
    main()
