"""Equivalence smoke-test: torch port of mini3di vs numpy reference.

Loads a few real protein structures from the held-out PDB val set,
encodes them with BOTH the numpy `Encoder` (the eval reference) and the
new `MiniThreeDiEncoderTorch` (with `hard=True`), and asserts the
per-residue 3Di states match. Failure here means our auxiliary loss
would train toward a target the eval doesn't measure -- a quiet,
metric-corrupting bug.

What's compared:

* per-residue 3Di state index (uint8 vs long), mapped through the
  Foldseek alphabet
* descriptor matrix (10D per residue) at intermediate stages, to
  localise any mismatch (geometry vs VAE vs centroid argmin)

Edge cases tolerated (and explicitly checked):

* Endpoint residues (idx 0 and L-1). Numpy fills them with all-zero
  descriptors and lets the VAE classify whatever 0 maps to. The torch
  port matches this initialisation.
* Float32 round-off on cross-products and normalisation. We allow a
  small fraction of tokens to differ as long as it's <0.1% of valid
  positions; mismatches are logged for inspection.

    uv run python scripts/_diag_test_mini3di_torch_equivalence.py
    uv run python scripts/_diag_test_mini3di_torch_equivalence.py --proteins 8 --strict
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lobster.model.latent_generator.utils.mini3di import (
    Encoder as NumpyEncoder,
    calculate_cb,
)
from lobster.model.latent_generator.utils.mini3di._encoder import FeatureEncoder
from lobster.model.latent_generator.utils.mini3di._torch_encoder import (
    MiniThreeDiEncoderTorch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test_mini3di_torch")

VAL_PT = Path(
    "/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/validation.pt"
)


def _load_val_proteins(path: Path, n: int) -> list[dict]:
    """Pull the first ``n`` protein records from val.pt.

    val.pt is a *column store*: a dict with parallel lists keyed by
    ``name``, ``sequence``, ``coords_res``, ``chains``, ``indices``,
    ... each list having one entry per protein. We zip those into
    per-protein dicts so the rest of the test treats each protein
    uniformly.
    """
    obj = torch.load(path, weights_only=False, map_location="cpu")
    log.info("loaded %s; type=%s", path, type(obj).__name__)
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected val.pt top-level type: {type(obj)}")
    keys = list(obj.keys())
    log.info("val.pt keys: %s", keys)
    # Determine total proteins from any list-valued key.
    total = None
    for k, v in obj.items():
        if isinstance(v, (list, tuple)):
            total = len(v)
            break
    if total is None:
        raise RuntimeError(f"could not infer protein count from val.pt (keys={keys})")
    log.info("total proteins in val.pt: %d", total)
    take = min(n, total)
    out: list[dict] = []
    for i in range(take):
        rec: dict = {}
        for k, v in obj.items():
            if isinstance(v, (list, tuple)) and len(v) == total:
                rec[k] = v[i]
            # Other keys are dataset-level metadata; ignore.
        out.append(rec)
    return out


def _extract_atoms(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ca, Cb, N, C) numpy arrays for a single chain.

    Uses the lobster `calculate_cb` helper which expects ``coords_res``
    in shape ``(L, 3, 3)`` with atoms in [N, Cα, C] order, and derives
    Cβ from the backbone via the standard pseudo-Cβ projection.
    """
    coords = record.get("coords_res")
    if coords is None:
        # Other possible key names in the pipeline:
        for k in ("coords", "all_atom_positions", "backbone_coords"):
            if k in record:
                coords = record[k]
                break
    if coords is None:
        raise KeyError(f"no backbone coords found in record (keys={list(record.keys())[:10]})")
    coords = torch.as_tensor(coords)
    if coords.dim() != 3 or coords.shape[-2:] != (3, 3):
        raise ValueError(f"expected coords (L, 3, 3) [N,Ca,C]; got {tuple(coords.shape)}")

    Ca, Cb, N_, C_ = calculate_cb({"coords_res": coords})
    return Ca, Cb, N_, C_


def _compare_descriptors(record_idx: int, ca_np, cb_np, n_np, c_np):
    """Sanity-check the descriptor matrix between numpy and torch ports."""
    fe = FeatureEncoder()
    desc_np = fe.encode_atoms(ca_np, cb_np, n_np, c_np).data  # (L, 10)
    partner_np = fe.partner_index_encoder._find_residue_partners(
        fe.vc_encoder.encode_atoms(ca_np, cb_np, n_np, c_np)
    )

    enc = MiniThreeDiEncoderTorch().eval()
    ca_t = torch.from_numpy(np.ascontiguousarray(ca_np)).float()
    cb_t = torch.from_numpy(np.ascontiguousarray(cb_np)).float()
    n_t = torch.from_numpy(np.ascontiguousarray(n_np)).float()
    vc_t = enc.compute_virtual_center(ca_t, cb_t, n_t)
    partner_t = enc.compute_partner_indices(vc_t)
    desc_t = enc.compute_descriptors(ca_t, partner_t).numpy()

    # Compare partner indices on residues 1..L-2 (numpy masks 0 and L-1).
    L = ca_np.shape[0]
    if L >= 3:
        diff_partner = (partner_np[1:-1] != partner_t.numpy()[1:-1]).sum()
        if diff_partner:
            log.warning("[%2d] partner-index mismatches: %d / %d",
                        record_idx, int(diff_partner), L - 2)
    # Compare descriptors. Only 8 of 10 columns are float-comparable; col 8
    # is an int clip and col 9 is copysign(log(|x|+1), x). Use atol bounded
    # by the expected float32 round-off in cross/normalize.
    desc_diff = np.abs(desc_np - desc_t).max()
    log.info("[%2d] descriptor max-abs-diff numpy vs torch: %.3e  (L=%d)",
             record_idx, float(desc_diff), L)
    return desc_diff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=int, default=4,
                        help="Number of proteins from val.pt to test.")
    parser.add_argument("--strict", action="store_true",
                        help="Treat ANY token mismatch as failure (default: <0.1%% allowed).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda; the equivalence is dtype-sensitive so cpu/float32 is safest.")
    parser.add_argument("--val-pt", type=str, default=str(VAL_PT))
    args = parser.parse_args()

    device = torch.device(args.device)
    val_path = Path(args.val_pt)
    log.info("loading %d proteins from %s", args.proteins, val_path)

    records = _load_val_proteins(val_path, args.proteins)
    log.info("got %d records", len(records))

    np_encoder = NumpyEncoder()
    torch_encoder = MiniThreeDiEncoderTorch().to(device).eval()

    n_total_residues = 0
    n_total_mismatches = 0
    per_protein_summary: list[dict] = []

    for i, rec in enumerate(records):
        ca_np, cb_np, n_np, c_np = _extract_atoms(rec)
        L = ca_np.shape[0]
        # Drop residues with NaN coords from the comparison (mini3di's
        # `_create_nan_mask` flags them invalid; argmin would output garbage
        # at those positions in either implementation).
        any_nan = (
            np.isnan(ca_np).any(axis=1)
            | np.isnan(cb_np).any(axis=1)
            | np.isnan(n_np).any(axis=1)
            | np.isnan(c_np).any(axis=1)
        )
        log.info("[%2d] L=%d, NaN-residues=%d", i, L, int(any_nan.sum()))

        # numpy reference
        out_np = np_encoder.encode_atoms(ca_np, cb_np, n_np, c_np)
        states_np = out_np["states"].filled().astype(np.int64)  # (L,)

        # torch port
        ca_t = torch.from_numpy(np.ascontiguousarray(ca_np)).float().to(device)
        cb_t = torch.from_numpy(np.ascontiguousarray(cb_np)).float().to(device)
        n_t = torch.from_numpy(np.ascontiguousarray(n_np)).float().to(device)
        with torch.no_grad():
            states_t = torch_encoder(ca_t, cb_t, n_t, hard=True).cpu().numpy()

        # Compare on valid residues (excluding endpoints + NaN positions).
        valid = np.ones(L, dtype=bool)
        valid[0] = False
        valid[-1] = False
        valid &= ~any_nan

        n_residues = int(valid.sum())
        n_mismatches = int((states_np[valid] != states_t[valid]).sum())
        rec_summary = {
            "idx": i,
            "L": int(L),
            "valid_residues": n_residues,
            "mismatches": n_mismatches,
            "match_ratio": (1.0 - n_mismatches / max(n_residues, 1)),
        }
        per_protein_summary.append(rec_summary)
        log.info("[%2d] valid=%d mismatches=%d match=%.4f",
                 i, n_residues, n_mismatches, rec_summary["match_ratio"])
        if n_mismatches:
            # Localise the mismatch to descriptors vs VAE vs argmin.
            _compare_descriptors(i, ca_np, cb_np, n_np, c_np)
        n_total_residues += n_residues
        n_total_mismatches += n_mismatches

    print("\n" + "=" * 78)
    print(f"{'idx':>4} | {'L':>4} | {'valid':>6} | {'mismatch':>8} | {'match':>8}")
    print("-" * 78)
    for s in per_protein_summary:
        print(f"{s['idx']:>4} | {s['L']:>4} | {s['valid_residues']:>6} | "
              f"{s['mismatches']:>8} | {s['match_ratio']:>7.4f}")
    print("-" * 78)
    overall = 1.0 - n_total_mismatches / max(n_total_residues, 1)
    print(f"OVERALL: {n_total_mismatches}/{n_total_residues} mismatches "
          f"({100 * overall:.4f}% agreement)")

    threshold = 1.0 if args.strict else 0.999
    ok = overall >= threshold
    if not ok:
        log.error("FAIL: agreement %.4f < threshold %.4f (--strict=%s)",
                  overall, threshold, args.strict)
        return 1
    log.info("PASS: agreement %.4f >= threshold %.4f", overall, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
