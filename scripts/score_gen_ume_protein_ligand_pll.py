"""Gen-UME protein-ligand pseudo-likelihood scoring (4-modality AR-MC analog).

Scores per-sample protein-ligand complexes using the protein-ligand checkpoint's
own absorbing-prior discrete-flow training objective as a Monte-Carlo estimator
of the any-order autoregressive log-likelihood, extended from 2 modalities
(protein seq, protein struc) to 4 (+ ligand atom, ligand struc).

For each sample and each modality, we compute:

  P1 (random-t Monte-Carlo, K stratified-t draws):
      score_unif  = mean over K draws of   avg_per_masked_position_CE(t_k)
      score_arllh = mean over K draws of   sum_CE_masked(t_k) / ((1 - t_k) * L_modality)

When scoring one modality, all other 3 are held clean (timesteps_other = 1).

Additionally:
  - joint_protein_score  = seq_score + struc_score                  (additive, protein-only)
  - joint_ligand_score   = lig_atom_score + lig_struc_score         (additive, ligand-only)
  - joint_all_score      = sum across all 4 modalities              (additive)
  - joint_true_4_score   = ONE forward per draw with all 4 masked   (true AO-ARM on unified 2(L+M)-stream)

Scoring proceeds **per target**: for each sample we load source `_protein.pt`
+ `_ligand.pt`, joint-encode through `model.encode_protein_ligand_structure`
to get GT structure tokens (protein + ligand), then run the K stratified-t
draws across modalities. Output CSV one row per sample, joined on `pdb_id`
to existing eval-dir quality CSVs by downstream correlate scripts.

Usage:
    uv run python scripts/score_gen_ume_protein_ligand_pll.py \\
        --source-data-dir /cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_gen_ume_all_latest/last.ckpt \\
        --output pll_scores_pl_all_<ts>.csv \\
        --K 32

The output CSV has one row per `pdb_id`. Use the matching eval-dir's
`forward_folding_results.csv` / `inverse_folding_results.csv` /
`conditioned_gen_results.csv` for the quality metrics; join on `pdb_id`
(or `ligand_id` for conditioned generation) downstream in
`scripts/correlate_pll_with_quality_ligand.py`.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_gen_ume_pl_pll")


_FIXED_T = (0.25, 0.5, 0.75)
_MODALITIES = ("seq", "struc", "lig_atom", "lig_struc")


def _list_targets(source_data_dir: Path) -> list[str]:
    """Return sorted list of `pdb_id`s present in source data dir.

    Each target is identified by a pair `(pdb_id)_protein.pt` + `(pdb_id)_ligand.pt`.
    """
    proteins = sorted(source_data_dir.glob("*_protein.pt"))
    pdb_ids = []
    for pf in proteins:
        pid = pf.name[: -len("_protein.pt")]
        if (source_data_dir / f"{pid}_ligand.pt").exists():
            pdb_ids.append(pid)
    return pdb_ids


def _filter_targets(pdb_ids: list[str], filter_csv: Path | None, id_col: str) -> list[str]:
    if filter_csv is None:
        return pdb_ids
    if not filter_csv.exists():
        logger.warning("--filter-csv %s does not exist; using all targets in source dir", filter_csv)
        return pdb_ids
    keep = set()
    with filter_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or id_col not in reader.fieldnames:
            logger.warning("Column %s not found in %s; using all targets", id_col, filter_csv)
            return pdb_ids
        for r in reader:
            v = (r.get(id_col) or "").strip()
            if v:
                keep.add(v)
    return [p for p in pdb_ids if p in keep]


def _load_target(source_data_dir: Path, pdb_id: str, device: torch.device) -> dict:
    pf = source_data_dir / f"{pdb_id}_protein.pt"
    lf = source_data_dir / f"{pdb_id}_ligand.pt"
    pdata = torch.load(pf, weights_only=False, map_location="cpu")
    ldata = torch.load(lf, weights_only=False, map_location="cpu")

    # Required protein fields
    coords_res = pdata.get("coords_res", pdata.get("coords"))
    sequence = pdata.get("sequence")
    if coords_res is None or sequence is None:
        raise ValueError(f"protein .pt for {pdb_id} missing coords_res or sequence")
    L = int(coords_res.shape[0])
    protein_mask = pdata.get("mask", torch.ones(L))
    protein_indices = pdata.get("indices", torch.arange(L))

    # Required ligand fields
    ligand_coords = ldata.get("atom_coords", ldata.get("coords", ldata.get("ligand_coords")))
    if ligand_coords is None:
        raise ValueError(f"ligand .pt for {pdb_id} missing atom_coords")
    M = int(ligand_coords.shape[0])
    ligand_atom_types = ldata.get("element_indices")
    if ligand_atom_types is None:
        # Fall back to all-carbon (idx 3 in the encoder vocab); rarely used in posebusters
        ligand_atom_types = torch.full((M,), 3, dtype=torch.long)
    ligand_mask = ldata.get("mask", torch.ones(M))
    ligand_indices = ldata.get("atom_indices", ldata.get("indices", torch.arange(M)))
    bond_matrix = ldata.get("bond_matrix")

    return {
        "pdb_id": pdb_id,
        "L": L,
        "M": M,
        "coords_res": coords_res.to(device).float(),
        "sequence": sequence.to(device).long(),  # PDB-integer encoding [L]
        "protein_mask": protein_mask.to(device).float(),
        "protein_indices": protein_indices.to(device).long(),
        "ligand_coords": ligand_coords.to(device).float(),
        "ligand_atom_types": ligand_atom_types.to(device).long(),
        "ligand_mask": ligand_mask.to(device).float(),
        "ligand_indices": ligand_indices.to(device).long(),
        "bond_matrix": bond_matrix.to(device).long() if bond_matrix is not None else None,
    }


def _tokenize_seq_pdbints(aa_transform, sequence_pdbints: torch.Tensor) -> torch.Tensor:
    """Same as protein-only scorer: PDB-int sequence -> model sequence tensor."""
    out = aa_transform({"sequence": sequence_pdbints.cpu()})
    return out["sequence"]


def _encode_gt_tokens(model, target: dict) -> dict:
    """Joint-encode GT (protein, ligand) coords -> GT structure tokens for both."""
    coords = target["coords_res"].unsqueeze(0)            # [1, L, 3, 3]
    pmask = target["protein_mask"].unsqueeze(0)          # [1, L]
    pidx = target["protein_indices"].unsqueeze(0)        # [1, L]
    lcoords = target["ligand_coords"].unsqueeze(0)       # [1, M, 3]
    lmask = target["ligand_mask"].unsqueeze(0)           # [1, M]
    lidx = target["ligand_indices"].unsqueeze(0)         # [1, M]
    latoms = target["ligand_atom_types"].unsqueeze(0)    # [1, M]
    bond = target["bond_matrix"].unsqueeze(0) if target["bond_matrix"] is not None else None

    out = model.encode_protein_ligand_structure(
        protein_coords=coords,
        protein_mask=pmask,
        protein_indices=pidx,
        ligand_coords=lcoords,
        ligand_mask=lmask,
        ligand_indices=lidx,
        ligand_atom_types=latoms,
        bond_matrix=bond,
    )
    return out  # dict with protein_tokens [1, L], ligand_tokens [1, M], etc.


def _build_inputs(model, aa_transform, target: dict) -> dict:
    """Return per-sample tokens / masks ready for the K-batch forward pass."""
    device = target["coords_res"].device
    seq_clean = _tokenize_seq_pdbints(aa_transform, target["sequence"]).to(device).unsqueeze(0)
    L = int(seq_clean.shape[1])
    if L != target["L"]:
        # AminoAcidTokenizerTransform may strip cls/eos; if length differs, error out so we
        # don't silently misalign.
        raise ValueError(f"Tokenized sequence length {L} != coords L {target['L']} for {target['pdb_id']}")

    enc = _encode_gt_tokens(model, target)
    struc_clean = enc["protein_tokens"]                       # [1, L]
    lig_struc_clean = enc["ligand_tokens"]                    # [1, M]

    return {
        "seq_clean": seq_clean,
        "struc_clean": struc_clean,
        "lig_atom_clean": target["ligand_atom_types"].unsqueeze(0),
        "lig_struc_clean": lig_struc_clean,
        "protein_mask": target["protein_mask"].unsqueeze(0),
        "ligand_mask": target["ligand_mask"].unsqueeze(0),
        "residue_index": target["protein_indices"].unsqueeze(0),
        "bond_matrix": target["bond_matrix"].unsqueeze(0) if target["bond_matrix"] is not None else None,
    }


def _expand_clean(t: torch.Tensor, K: int) -> torch.Tensor:
    return t.expand(K, *([-1] * (t.dim() - 1))).contiguous() if t is not None else None


def _absorbing_corrupt(
    clean: torch.Tensor, t_values: torch.Tensor, mask_id: int, valid_mask: torch.Tensor, rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask each valid position independently with prob (1 - t_k).

    Returns (x_t, masked_pos) where masked_pos is True at positions that were masked.
    Padding positions (valid_mask==0) are never masked (kept as the original padding token).
    """
    K = t_values.shape[0]
    rand = torch.rand(*clean.shape, generator=rng, device=clean.device)
    mask_pos = (rand > t_values.view(K, *([1] * (clean.dim() - 1)))) & valid_mask.bool()
    x_t = torch.where(mask_pos, torch.full_like(clean, mask_id), clean)
    return x_t, mask_pos


@torch.no_grad()
def _forward_kbatch(
    model,
    *,
    K: int,
    seq_tokens: torch.Tensor,
    struc_tokens: torch.Tensor,
    lig_atom_tokens: torch.Tensor,
    lig_struc_tokens: torch.Tensor,
    protein_mask: torch.Tensor,
    ligand_mask: torch.Tensor,
    residue_index: torch.Tensor,
    bond_matrix: torch.Tensor | None,
    timesteps: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Single forward pass with K-batched inputs across all modalities."""
    L = seq_tokens.shape[1]
    conditioning = torch.zeros(K, L, 1, device=seq_tokens.device)
    bond_K = bond_matrix.expand(K, *bond_matrix.shape[1:]).contiguous() if bond_matrix is not None else None

    out = model.forward(
        x_t={
            "sequence_tokens": seq_tokens,
            "structure_tokens": struc_tokens,
            "ligand_atom_tokens": lig_atom_tokens,
            "ligand_structure_tokens": lig_struc_tokens,
        },
        mask=protein_mask,
        residue_index=residue_index,
        conditioning_tensor=conditioning,
        timesteps=timesteps,
        ligand_mask=ligand_mask,
        bond_matrix=bond_K,
    )
    return out


def _ce_on_masked(
    logits: torch.Tensor,           # [K, N, V]
    target: torch.Tensor,           # [K, N]
    masked_pos: torch.Tensor,       # [K, N] bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (avg_per_masked_CE[K], sum_CE_masked[K], n_masked[K])."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    ce = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [K, N]
    n_masked = masked_pos.sum(dim=1).to(torch.float32)
    sum_ce = (ce * masked_pos.float()).sum(dim=1)
    avg_ce = sum_ce / n_masked.clamp(min=1.0)
    return avg_ce, sum_ce, n_masked


@torch.no_grad()
def _score_one_modality(
    model,
    *,
    modality: str,
    inputs: dict,
    K: int,
    t_values: torch.Tensor,
    seq_mask_id: int,
    struc_mask_id: int,
    lig_atom_mask_id: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask only `modality`, hold the other three clean. Returns (avg_ce[K], sum_ce[K], n_masked[K])."""
    device = inputs["seq_clean"].device

    seq_K = _expand_clean(inputs["seq_clean"], K)
    struc_K = _expand_clean(inputs["struc_clean"], K)
    lig_atom_K = _expand_clean(inputs["lig_atom_clean"], K)
    lig_struc_K = _expand_clean(inputs["lig_struc_clean"], K)
    pmask_K = _expand_clean(inputs["protein_mask"], K)
    lmask_K = _expand_clean(inputs["ligand_mask"], K)
    residx_K = _expand_clean(inputs["residue_index"], K)

    # Only the timestep for the modality being scored is set to t_k; others held at 1 (clean).
    t_clean = torch.ones(K, device=device)
    timesteps = {
        "sequence_tokens": t_clean,
        "structure_tokens": t_clean,
        "ligand_atom_tokens": t_clean,
        "ligand_structure_tokens": t_clean,
    }

    # Apply absorbing corruption only to the chosen modality.
    if modality == "seq":
        x_t, mask_pos = _absorbing_corrupt(seq_K, t_values, seq_mask_id, pmask_K, rng)
        seq_input = x_t
        timesteps["sequence_tokens"] = t_values
        target = seq_K
    else:
        seq_input = seq_K

    if modality == "struc":
        x_t, mask_pos = _absorbing_corrupt(struc_K, t_values, struc_mask_id, pmask_K, rng)
        struc_input = x_t
        timesteps["structure_tokens"] = t_values
        target = struc_K
    else:
        struc_input = struc_K

    if modality == "lig_atom":
        x_t, mask_pos = _absorbing_corrupt(lig_atom_K, t_values, lig_atom_mask_id, lmask_K, rng)
        lig_atom_input = x_t
        timesteps["ligand_atom_tokens"] = t_values
        target = lig_atom_K
    else:
        lig_atom_input = lig_atom_K

    if modality == "lig_struc":
        x_t, mask_pos = _absorbing_corrupt(lig_struc_K, t_values, struc_mask_id, lmask_K, rng)
        lig_struc_input = x_t
        timesteps["ligand_structure_tokens"] = t_values
        target = lig_struc_K
    else:
        lig_struc_input = lig_struc_K

    out = _forward_kbatch(
        model,
        K=K,
        seq_tokens=seq_input,
        struc_tokens=struc_input,
        lig_atom_tokens=lig_atom_input,
        lig_struc_tokens=lig_struc_input,
        protein_mask=pmask_K,
        ligand_mask=lmask_K,
        residue_index=residx_K,
        bond_matrix=inputs["bond_matrix"],
        timesteps=timesteps,
    )

    logits_key = {
        "seq": "sequence_logits",
        "struc": "structure_logits",
        "lig_atom": "ligand_atom_logits",
        "lig_struc": "ligand_structure_logits",
    }[modality]
    return _ce_on_masked(out[logits_key], target, mask_pos)


@torch.no_grad()
def _score_joint_4(
    model,
    *,
    inputs: dict,
    K: int,
    t_values: torch.Tensor,
    seq_mask_id: int,
    struc_mask_id: int,
    lig_atom_mask_id: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ONE forward per draw with all 4 modalities masked at the same t_k.

    Mask patterns sampled independently per modality per position. Same scalar
    t_k for all 4 modalities per draw (matches the linear MD4 schedule applied
    jointly to a unified 2(L+M)-token stream).
    """
    device = inputs["seq_clean"].device
    seq_K = _expand_clean(inputs["seq_clean"], K)
    struc_K = _expand_clean(inputs["struc_clean"], K)
    lig_atom_K = _expand_clean(inputs["lig_atom_clean"], K)
    lig_struc_K = _expand_clean(inputs["lig_struc_clean"], K)
    pmask_K = _expand_clean(inputs["protein_mask"], K)
    lmask_K = _expand_clean(inputs["ligand_mask"], K)
    residx_K = _expand_clean(inputs["residue_index"], K)

    seq_x, seq_mp = _absorbing_corrupt(seq_K, t_values, seq_mask_id, pmask_K, rng)
    struc_x, struc_mp = _absorbing_corrupt(struc_K, t_values, struc_mask_id, pmask_K, rng)
    lig_atom_x, lig_atom_mp = _absorbing_corrupt(lig_atom_K, t_values, lig_atom_mask_id, lmask_K, rng)
    lig_struc_x, lig_struc_mp = _absorbing_corrupt(lig_struc_K, t_values, struc_mask_id, lmask_K, rng)

    timesteps = {
        "sequence_tokens": t_values,
        "structure_tokens": t_values,
        "ligand_atom_tokens": t_values,
        "ligand_structure_tokens": t_values,
    }

    out = _forward_kbatch(
        model,
        K=K,
        seq_tokens=seq_x,
        struc_tokens=struc_x,
        lig_atom_tokens=lig_atom_x,
        lig_struc_tokens=lig_struc_x,
        protein_mask=pmask_K,
        ligand_mask=lmask_K,
        residue_index=residx_K,
        bond_matrix=inputs["bond_matrix"],
        timesteps=timesteps,
    )

    avg_seq, sum_seq, n_seq = _ce_on_masked(out["sequence_logits"], seq_K, seq_mp)
    avg_struc, sum_struc, n_struc = _ce_on_masked(out["structure_logits"], struc_K, struc_mp)
    avg_la, sum_la, n_la = _ce_on_masked(out["ligand_atom_logits"], lig_atom_K, lig_atom_mp)
    avg_ls, sum_ls, n_ls = _ce_on_masked(out["ligand_structure_logits"], lig_struc_K, lig_struc_mp)

    n_total = n_seq + n_struc + n_la + n_ls
    sum_total = sum_seq + sum_struc + sum_la + sum_ls
    avg_total = sum_total / n_total.clamp(min=1.0)
    return avg_total, sum_total, n_total


@torch.no_grad()
def score_one_sample(
    model,
    *,
    inputs: dict,
    K: int,
    seed: int,
    seq_mask_id: int,
    struc_mask_id: int,
    lig_atom_mask_id: int,
    eps: float = 0.02,
) -> dict[str, float]:
    """Score one sample on all 4 modalities + 4 joint variants."""
    device = inputs["seq_clean"].device
    L = int(inputs["seq_clean"].shape[1])
    M = int(inputs["lig_atom_clean"].shape[1])

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    edges = torch.linspace(eps, 1.0 - eps, steps=K + 1, device=device)
    u = torch.empty(K, device=device).uniform_(0.0, 1.0, generator=rng)
    t_random = edges[:-1] + u * (edges[1:] - edges[:-1])
    t_fixed = torch.tensor(_FIXED_T, device=device, dtype=torch.float32)

    out: dict[str, float] = {}
    modality_lengths = {"seq": L, "struc": L, "lig_atom": M, "lig_struc": M}

    for modality in _MODALITIES:
        avg_ce, sum_ce, n_masked = _score_one_modality(
            model,
            modality=modality,
            inputs=inputs,
            K=K,
            t_values=t_random,
            seq_mask_id=seq_mask_id,
            struc_mask_id=struc_mask_id,
            lig_atom_mask_id=lig_atom_mask_id,
            rng=rng,
        )
        out[f"{modality}_score_unif"] = avg_ce.mean().item()
        Lm = float(modality_lengths[modality])
        weighted = sum_ce / ((1.0 - t_random) * Lm)
        out[f"{modality}_score_arllh"] = weighted.mean().item()
        out[f"{modality}_n_draws"] = float(K)
        out[f"{modality}_mean_n_masked"] = n_masked.mean().item()

        # P2 fixed-t draws (cheap).
        avg_ce_fixed, _, _ = _score_one_modality(
            model,
            modality=modality,
            inputs=inputs,
            K=len(_FIXED_T),
            t_values=t_fixed,
            seq_mask_id=seq_mask_id,
            struc_mask_id=struc_mask_id,
            lig_atom_mask_id=lig_atom_mask_id,
            rng=rng,
        )
        for tv, ce_v in zip(_FIXED_T, avg_ce_fixed.tolist()):
            out[f"{modality}_score_t{tv:g}"] = ce_v

    # Additive joints
    out["joint_protein_score_unif"] = out["seq_score_unif"] + out["struc_score_unif"]
    out["joint_protein_score_arllh"] = out["seq_score_arllh"] + out["struc_score_arllh"]
    out["joint_ligand_score_unif"] = out["lig_atom_score_unif"] + out["lig_struc_score_unif"]
    out["joint_ligand_score_arllh"] = out["lig_atom_score_arllh"] + out["lig_struc_score_arllh"]
    out["joint_all_score_unif"] = (
        out["seq_score_unif"]
        + out["struc_score_unif"]
        + out["lig_atom_score_unif"]
        + out["lig_struc_score_unif"]
    )
    out["joint_all_score_arllh"] = (
        out["seq_score_arllh"]
        + out["struc_score_arllh"]
        + out["lig_atom_score_arllh"]
        + out["lig_struc_score_arllh"]
    )

    # True 4-way joint AR-MC
    avg_j, sum_j, n_j = _score_joint_4(
        model,
        inputs=inputs,
        K=K,
        t_values=t_random,
        seq_mask_id=seq_mask_id,
        struc_mask_id=struc_mask_id,
        lig_atom_mask_id=lig_atom_mask_id,
        rng=rng,
    )
    out["joint_true_4_score_unif"] = avg_j.mean().item()
    weighted_j = sum_j / ((1.0 - t_random) * 2.0 * float(L + M))
    out["joint_true_4_score_arllh"] = weighted_j.mean().item()
    out["joint_true_4_n_draws"] = float(K)
    out["joint_true_4_mean_n_masked"] = n_j.mean().item()

    return out


_OUT_PER_MOD = ["{m}_score_unif", "{m}_score_arllh", "{m}_n_draws", "{m}_mean_n_masked",
                "{m}_score_t0.25", "{m}_score_t0.5", "{m}_score_t0.75"]
_OUT_JOINTS = [
    "joint_protein_score_unif", "joint_protein_score_arllh",
    "joint_ligand_score_unif", "joint_ligand_score_arllh",
    "joint_all_score_unif", "joint_all_score_arllh",
    "joint_true_4_score_unif", "joint_true_4_score_arllh",
    "joint_true_4_n_draws", "joint_true_4_mean_n_masked",
]


def _build_out_columns() -> list[str]:
    cols = ["pdb_id", "L", "M", "scoring_seed"]
    for m in _MODALITIES:
        cols.extend([t.format(m=m) for t in _OUT_PER_MOD])
    cols.extend(_OUT_JOINTS)
    return cols


def _open_writer(path: Path, columns: list[str]):
    fh = path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    return fh, writer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data-dir", required=True, type=Path,
                   help="Directory with <pdb_id>_protein.pt + <pdb_id>_ligand.pt pairs (PoseBusters source).")
    p.add_argument("--ckpt", required=True, type=Path, help="ProteinLigandEncoderLightningModule .ckpt path")
    p.add_argument("--K", type=int, default=32, help="Random-t Monte-Carlo draws per modality")
    p.add_argument("--seed", type=int, default=20260505, help="Per-sample base seed")
    p.add_argument("--max-samples", type=int, default=None, help="Score only first N targets (for smoke testing)")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path; default pll_scores_pl_<ts>.csv in CWD")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-protein-length", type=int, default=512, help="Skip targets longer than this")
    p.add_argument("--filter-csv", type=Path, default=None,
                   help="Optional eval-dir CSV; only score targets whose pdb_id appears in it.")
    p.add_argument("--filter-id-col", default="pdb_id",
                   help="Column name in --filter-csv carrying the target ID (default: pdb_id; use ligand_id for cond-gen)")
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args()

    if not args.source_data_dir.is_dir():
        raise FileNotFoundError(f"--source-data-dir not found: {args.source_data_dir}")

    pdb_ids = _list_targets(args.source_data_dir)
    pdb_ids = _filter_targets(pdb_ids, args.filter_csv, args.filter_id_col)
    if args.max_samples is not None:
        pdb_ids = pdb_ids[: args.max_samples]
    logger.info("Scoring %d targets from %s", len(pdb_ids), args.source_data_dir)

    output_path = args.output or Path.cwd() / f"pll_scores_pl_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing scores to %s", output_path)

    device = torch.device(args.device)

    # Lazy heavy imports
    from lobster.model.gen_ume import ProteinLigandEncoderLightningModule
    from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = ProteinLigandEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)
    # Align interpolant device (some bionemo MOCO ops use this).
    model.interpolant_seq.device = device
    model.interpolant_struc.device = device
    if hasattr(model, "interpolant_ligand_atom"):
        model.interpolant_ligand_atom.device = device
    if hasattr(model, "interpolant_ligand_struc"):
        model.interpolant_ligand_struc.device = device
    logger.info("Model loaded in %.1fs (device=%s)", time.time() - t0, device)

    seq_mask_id = int(getattr(model, "mask_token_id"))
    struc_mask_id = int(getattr(model, "mask_index_struc_tokens"))
    lig_atom_mask_id = int(getattr(model, "ligand_mask_token_id"))
    logger.info(
        "mask ids: seq=%d  struc=%d  lig_atom=%d  vocab(seq=%d, struc=%d, lig_atom=%d)",
        seq_mask_id, struc_mask_id, lig_atom_mask_id,
        int(model.vocab_size), int(model.num_struc_classes), int(model.ligand_atom_vocab_size),
    )

    aa_transform = AminoAcidTokenizerTransform(max_length=args.max_protein_length)

    cols = _build_out_columns()
    fh, writer = _open_writer(output_path, cols)
    try:
        n_done = 0
        n_skipped = 0
        t_start = time.time()
        for row_idx, pdb_id in enumerate(pdb_ids):
            try:
                target = _load_target(args.source_data_dir, pdb_id, device)
            except Exception as e:
                n_skipped += 1
                logger.warning("Skipping %s: load failed: %s", pdb_id, e)
                continue
            if target["L"] > args.max_protein_length:
                n_skipped += 1
                logger.info("Skipping %s: L=%d > max_protein_length=%d",
                            pdb_id, target["L"], args.max_protein_length)
                continue
            try:
                inputs = _build_inputs(model, aa_transform, target)
                # Sanity: clamp out-of-vocab tokens
                if (inputs["struc_clean"] >= int(model.num_struc_classes)).any():
                    n_skipped += 1
                    logger.warning("Skipping %s: struc tokens OOV", pdb_id)
                    continue
                if (inputs["seq_clean"] >= int(model.vocab_size)).any():
                    n_skipped += 1
                    logger.warning("Skipping %s: seq tokens OOV", pdb_id)
                    continue
                if (inputs["lig_atom_clean"] >= int(model.ligand_atom_vocab_size)).any():
                    n_skipped += 1
                    logger.warning("Skipping %s: lig_atom tokens OOV", pdb_id)
                    continue
            except Exception as e:
                n_skipped += 1
                logger.warning("Skipping %s: input prep failed: %s", pdb_id, e)
                continue

            sample_seed = (args.seed * 1_000_003 + row_idx) & 0x7FFFFFFF
            try:
                scores = score_one_sample(
                    model,
                    inputs=inputs,
                    K=args.K,
                    seed=sample_seed,
                    seq_mask_id=seq_mask_id,
                    struc_mask_id=struc_mask_id,
                    lig_atom_mask_id=lig_atom_mask_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skipped += 1
                logger.warning("OOM on %s (L=%d, M=%d); skipping", pdb_id, target["L"], target["M"])
                continue

            row_out = {
                "pdb_id": pdb_id,
                "L": target["L"],
                "M": target["M"],
                "scoring_seed": sample_seed,
                **{k: v for k, v in scores.items() if k in cols},
            }
            for k, v in list(row_out.items()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row_out[k] = ""
            writer.writerow(row_out)
            fh.flush()
            n_done += 1

            if (n_done % args.log_every) == 0 or n_done == 1:
                dt = time.time() - t_start
                logger.info(
                    "[%4d/%d] %s L=%d M=%d  seq=%.3f struc=%.3f lig_atom=%.3f lig_struc=%.3f joint_true_4=%.3f  (%.2fs/sample)",
                    n_done, len(pdb_ids), pdb_id, target["L"], target["M"],
                    scores["seq_score_unif"], scores["struc_score_unif"],
                    scores["lig_atom_score_unif"], scores["lig_struc_score_unif"],
                    scores["joint_true_4_score_unif"],
                    dt / max(1, n_done),
                )
    finally:
        fh.close()

    logger.info("Done. Scored %d targets; skipped %d. Output: %s", n_done, n_skipped, output_path)


if __name__ == "__main__":
    main()
