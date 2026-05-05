"""Inverse-folding best-of-N with 4-modality PLL on PoseBusters (protein+ligand).

For each PoseBusters target, generates `N` independent inverse-fold candidates
conditioned on (GT protein backbone coords, GT ligand atom + structure tokens)
and scores **the predicted protein sequence tokens** with the model's own
4-modality pseudo-likelihood.

Quality metric: AAR (amino-acid recovery vs GT sequence). Headline downstream
metric for IF is sc-TM via cofold (not computed here; pair the candidate CSV
with a downstream cofold step, e.g. Boltz2).

Pickers: random, {seq,struc,lig_atom,lig_struc}_pll, {protein,ligand,all}_joint,
joint_true_4, oracle_aar (ceiling).

Use --N 1 for E0 correlation, --N 30 (or higher) for E2 best-of-N.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gen_ume_protein_ligand_pll import (  # noqa: E402
    _list_targets,
    _load_target,
    _tokenize_seq_pdbints,
    _encode_gt_tokens,
    score_one_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("if_bestofN_pll_ligand")


_PLL_VARIANTS = [
    "seq_score_unif", "seq_score_arllh",
    "struc_score_unif", "struc_score_arllh",
    "lig_atom_score_unif", "lig_atom_score_arllh",
    "lig_struc_score_unif", "lig_struc_score_arllh",
    "joint_protein_score_unif", "joint_protein_score_arllh",
    "joint_ligand_score_unif", "joint_ligand_score_arllh",
    "joint_all_score_unif", "joint_all_score_arllh",
    "joint_true_4_score_unif", "joint_true_4_score_arllh",
]
_CANDIDATE_COLS = [
    "pdb_id", "L", "M", "candidate_idx", "seed", "include_ligand",
    "aar", "predicted_sequence", "gen_seconds", "score_seconds",
] + _PLL_VARIANTS

_PICKERS = [
    ("random_pick", None, None),
    ("seq_pll_pick", "seq_score_unif", "min"),
    ("struc_pll_pick", "struc_score_unif", "min"),
    ("lig_atom_pll_pick", "lig_atom_score_unif", "min"),
    ("lig_struc_pll_pick", "lig_struc_score_unif", "min"),
    ("joint_protein_pll_pick", "joint_protein_score_unif", "min"),
    ("joint_ligand_pll_pick", "joint_ligand_score_unif", "min"),
    ("joint_all_pll_pick", "joint_all_score_unif", "min"),
    ("joint_true_4_pll_pick", "joint_true_4_score_unif", "min"),
    ("oracle_aar_pick", "aar", "max"),
]
_SUMMARY_COLS = (
    ["pdb_id", "L", "M", "include_ligand", "n_candidates",
     "aar_min", "aar_mean", "aar_median", "aar_max", "aar_std"]
    + [f"{p}_{suffix}" for p, _, _ in _PICKERS for suffix in ("idx", "aar")]
)


def _do_picks(rows: list[dict]) -> dict:
    out = {}
    for picker_name, key, direction in _PICKERS:
        if direction is None:
            idx = 0
        elif direction == "max":
            vals = [r.get(key) for r in rows]
            idx = int(max(range(len(rows)),
                          key=lambda i: (vals[i] if vals[i] is not None else float("-inf"))))
        else:
            vals = [r.get(key) for r in rows]
            idx = int(min(range(len(rows)),
                          key=lambda i: (vals[i] if vals[i] is not None else float("inf"))))
        out[f"{picker_name}_idx"] = idx
        out[f"{picker_name}_aar"] = float(rows[idx]["aar"])
    return out


def _compute_aar(target: dict, pred_seq_tokens: torch.Tensor, aa_transform) -> tuple[float, str]:
    """Sequence-identity AAR vs GT, restricted to valid (mask=1) positions.
    Decodes Lobster AA token IDs back to PDB int via inverse of aa_transform if needed.
    """
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv

    valid = target["protein_mask"].bool()
    gt_pdbints = target["sequence"][valid].tolist()
    gt_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in gt_pdbints)

    # Convert predicted Lobster AA tokens back to standard AA letters.
    # Lobster's AminoAcidTokenizerTransform uses a 33-token vocab. Use the model's
    # built-in convert_lobster_aa_tokenization_to_standard_aa to map back to PDB ints,
    # then to letters.
    from lobster.model.latent_generator.utils.residue_constants import convert_lobster_aa_tokenization_to_standard_aa

    pred_seq_logits = torch.nn.functional.one_hot(
        pred_seq_tokens.long(), num_classes=33
    ).float()
    pred_pdbints = convert_lobster_aa_tokenization_to_standard_aa(
        pred_seq_logits, device=pred_seq_tokens.device
    ).squeeze(0)
    pred_pdbints = pred_pdbints[valid]
    pred_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in pred_pdbints.tolist())

    if len(gt_str) != len(pred_str) or len(gt_str) == 0:
        return float("nan"), pred_str
    matches = sum(a == b for a, b in zip(gt_str, pred_str))
    aar = matches / len(gt_str)
    return float(aar), pred_str


def _resolve_schedule(name: str | None):
    if name is None:
        return None
    from bionemo.moco.schedules.inference_time_schedules import (
        LogInferenceSchedule, LinearInferenceSchedule, PowerInferenceSchedule,
    )
    return {
        "LogInferenceSchedule": LogInferenceSchedule,
        "LinearInferenceSchedule": LinearInferenceSchedule,
        "PowerInferenceSchedule": PowerInferenceSchedule,
    }[name]


@torch.no_grad()
def _generate_one(model, target: dict, *, include_ligand: bool, gen_kwargs: dict):
    """One inverse-fold candidate. Returns (pred_seq_tokens [1, L], lig_atom_in, lig_struc_in)."""
    L = target["L"]
    M = target["M"]
    pmask = target["protein_mask"].unsqueeze(0)
    pidx = target["protein_indices"].unsqueeze(0)
    pcoords = target["coords_res"].unsqueeze(0)

    input_lig_atom = None
    input_lig_struc = None
    bond_K = None
    if include_ligand:
        enc = _encode_gt_tokens(model, target)
        input_lig_atom = target["ligand_atom_types"].unsqueeze(0)
        input_lig_struc = enc["ligand_tokens"]
        bond_K = target["bond_matrix"].unsqueeze(0) if target["bond_matrix"] is not None else None

    ligand_is_context = include_ligand and gen_kwargs["ligand_context_mode"] == "structure_tokens"

    result = model.generate_sample(
        length=L, num_samples=1, inverse_folding=True,
        nsteps=gen_kwargs["nsteps"],
        inference_schedule_seq=_resolve_schedule(gen_kwargs["schedule_seq"]),
        inference_schedule_struc=_resolve_schedule(gen_kwargs["schedule_struc"]),
        inference_schedule_ligand_atom=_resolve_schedule(gen_kwargs.get("schedule_lig_atom")),
        inference_schedule_ligand_struc=_resolve_schedule(gen_kwargs.get("schedule_lig_struc")),
        temperature_seq=gen_kwargs["temperature_seq"],
        temperature_struc=gen_kwargs["temperature_struc"],
        stochasticity_seq=gen_kwargs["stochasticity_seq"],
        stochasticity_struc=gen_kwargs["stochasticity_struc"],
        temperature_ligand=gen_kwargs["temperature_ligand"],
        stochasticity_ligand=gen_kwargs["stochasticity_ligand"],
        input_structure_coords=pcoords,
        input_mask=pmask,
        input_indices=pidx,
        generate_ligand=include_ligand,
        num_atoms=M if include_ligand else 0,
        input_ligand_atom_tokens=input_lig_atom,
        input_ligand_structure_tokens=input_lig_struc,
        input_bond_matrix=bond_K,
        ligand_is_context=ligand_is_context,
    )
    pred_seq = result["generated_seq_tokens"]
    return pred_seq, input_lig_atom, input_lig_struc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max-protein-length", type=int, default=512)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=20260505)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-ligand", action="store_true")
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--temperature-seq", type=float, default=0.5)
    p.add_argument("--temperature-struc", type=float, default=0.5)
    p.add_argument("--stochasticity-seq", type=int, default=20)
    p.add_argument("--stochasticity-struc", type=int, default=20)
    p.add_argument("--temperature-ligand", type=float, default=0.5)
    p.add_argument("--stochasticity-ligand", type=int, default=20)
    p.add_argument("--schedule-seq", default="LogInferenceSchedule",
                   choices=["LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--schedule-struc", default="LinearInferenceSchedule",
                   choices=["LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--schedule-lig-atom", default=None,
                   choices=[None, "LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--schedule-lig-struc", default=None,
                   choices=[None, "LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--ligand-context-mode", default="structure_tokens",
                   choices=["structure_tokens", "atom_bond_only"])
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    include_ligand = not args.no_ligand

    pdb_ids = _list_targets(args.source_data_dir)
    if args.max_targets is not None:
        pdb_ids = pdb_ids[: args.max_targets]
    if not pdb_ids:
        raise FileNotFoundError(f"No targets found in {args.source_data_dir}")
    logger.info("Found %d targets", len(pdb_ids))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    cand_path = args.output_dir / f"bestofN_if_lig_candidates_{ts}.csv"
    summ_path = args.output_dir / f"bestofN_if_lig_summary_{ts}.csv"
    cand_fh = cand_path.open("w", newline="")
    summ_fh = summ_path.open("w", newline="")
    cand_writer = csv.DictWriter(cand_fh, fieldnames=_CANDIDATE_COLS, extrasaction="ignore")
    summ_writer = csv.DictWriter(summ_fh, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
    cand_writer.writeheader()
    summ_writer.writeheader()

    from lobster.model.gen_ume import ProteinLigandEncoderLightningModule
    from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = ProteinLigandEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)
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

    aa_transform = AminoAcidTokenizerTransform(max_length=args.max_protein_length)
    gen_kwargs = dict(
        nsteps=args.nsteps,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        temperature_ligand=args.temperature_ligand,
        stochasticity_ligand=args.stochasticity_ligand,
        schedule_seq=args.schedule_seq,
        schedule_struc=args.schedule_struc,
        schedule_lig_atom=args.schedule_lig_atom,
        schedule_lig_struc=args.schedule_lig_struc,
        ligand_context_mode=args.ligand_context_mode,
    )

    n_targets_done = 0
    n_skipped = 0
    t_start = time.time()

    for ti, pdb_id in enumerate(pdb_ids):
        try:
            target = _load_target(args.source_data_dir, pdb_id, device)
        except Exception as e:
            n_skipped += 1
            logger.warning("Skipping %s: load failed: %s", pdb_id, e)
            continue
        if target["L"] > args.max_protein_length:
            n_skipped += 1
            continue

        # GT struc tokens (clean) — same for every candidate.
        try:
            enc_gt = _encode_gt_tokens(model, target)
            gt_struc_tokens = enc_gt["protein_tokens"]
        except Exception as e:
            logger.warning("Skipping %s: encode failed: %s", pdb_id, e)
            n_skipped += 1
            continue

        rows: list[dict] = []
        for ci in range(args.N):
            seed = (args.seed_base + ti * 1_000_003 + ci) & 0x7FFFFFFF
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                pred_seq, lig_atom_in, lig_struc_in = _generate_one(
                    model, target, include_ligand=include_ligand, gen_kwargs=gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM gen %s candidate %d", pdb_id, ci)
                continue
            except Exception as e:
                logger.warning("Gen failed %s candidate %d: %s", pdb_id, ci, e)
                continue
            gen_seconds = time.time() - tg0

            try:
                aar, pred_str = _compute_aar(target, pred_seq, aa_transform)
            except Exception as e:
                logger.warning("AAR failed %s candidate %d: %s", pdb_id, ci, e)
                continue

            inputs = {
                "seq_clean": pred_seq,                                          # [1, L] PREDICTED
                "struc_clean": gt_struc_tokens,                                 # GT
                "lig_atom_clean": (lig_atom_in if include_ligand
                                   else target["ligand_atom_types"].unsqueeze(0)),
                "lig_struc_clean": (lig_struc_in if include_ligand
                                    else _encode_gt_tokens(model, target)["ligand_tokens"]),
                "protein_mask": target["protein_mask"].unsqueeze(0),
                "ligand_mask": target["ligand_mask"].unsqueeze(0),
                "residue_index": target["protein_indices"].unsqueeze(0),
                "bond_matrix": target["bond_matrix"].unsqueeze(0) if target["bond_matrix"] is not None else None,
            }
            ts0 = time.time()
            try:
                scores = score_one_sample(
                    model, inputs=inputs, K=args.K,
                    seed=seed ^ 0xA5A5A5,
                    seq_mask_id=seq_mask_id,
                    struc_mask_id=struc_mask_id,
                    lig_atom_mask_id=lig_atom_mask_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM score %s candidate %d", pdb_id, ci)
                scores = {}
            score_seconds = time.time() - ts0

            row = {
                "pdb_id": pdb_id, "L": target["L"], "M": target["M"],
                "candidate_idx": ci, "seed": seed, "include_ligand": include_ligand,
                "aar": aar, "predicted_sequence": pred_str,
                "gen_seconds": round(gen_seconds, 3),
                "score_seconds": round(score_seconds, 3),
                **{k: v for k, v in scores.items() if k in _CANDIDATE_COLS},
            }
            rows.append(row)
            cand_writer.writerow(row)
            cand_fh.flush()

        if not rows:
            n_skipped += 1
            continue

        aars = [r["aar"] for r in rows]
        mean_aar = sum(aars) / len(aars)
        std_aar = (sum((x - mean_aar) ** 2 for x in aars) / len(aars)) ** 0.5
        picks = _do_picks(rows)
        summ_writer.writerow({
            "pdb_id": pdb_id, "L": target["L"], "M": target["M"],
            "include_ligand": include_ligand, "n_candidates": len(rows),
            "aar_min": min(aars), "aar_mean": mean_aar,
            "aar_median": sorted(aars)[len(aars) // 2], "aar_max": max(aars),
            "aar_std": std_aar,
            **picks,
        })
        summ_fh.flush()
        n_targets_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%4d/%d] %s L=%d M=%d  aar[mean=%.3f, max=%.3f]  pick: r=%.3f sP=%.3f st=%.3f la=%.3f ls=%.3f jpro=%.3f jlig=%.3f jall=%.3f jt4=%.3f or=%.3f  (%.1fs/target)",
                ti + 1, len(pdb_ids), pdb_id, target["L"], target["M"],
                mean_aar, max(aars),
                picks["random_pick_aar"], picks["seq_pll_pick_aar"],
                picks["struc_pll_pick_aar"], picks["lig_atom_pll_pick_aar"],
                picks["lig_struc_pll_pick_aar"], picks["joint_protein_pll_pick_aar"],
                picks["joint_ligand_pll_pick_aar"], picks["joint_all_pll_pick_aar"],
                picks["joint_true_4_pll_pick_aar"], picks["oracle_aar_pick_aar"],
                elapsed / max(1, n_targets_done),
            )

    cand_fh.close()
    summ_fh.close()
    logger.info("Done. %d targets / %d skipped. Outputs: %s | %s",
                n_targets_done, n_skipped, cand_path, summ_path)


if __name__ == "__main__":
    main()
