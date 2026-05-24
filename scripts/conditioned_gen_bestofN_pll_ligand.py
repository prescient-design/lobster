"""Ligand-conditioned protein generation best-of-N with 4-modality PLL.

For each PoseBusters target, generates `N` independent protein samples
conditioned on (GT ligand atom + structure tokens) — both protein sequence
AND structure are sampled simultaneously — and scores **the predicted
(seq, struc) pair** with the model's own 4-modality pseudo-likelihood.

In-loop quality (cheap proxies): pseudo-AAR vs GT (sequence diversity / pocket
recovery proxy), decoded structure self-consistency tm via decoded coords,
ligand-pocket distance from decoded coords. The headline metric (RF3
pass-rate after Boltz2 cofold of (predicted_seq, GT_ligand)) is a downstream
step performed on the candidates CSV.

Use --N 1 for E0 correlation; --N 30 for E3 best-of-N.
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
    _encode_gt_tokens,
    score_one_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cg_bestofN_pll_ligand")


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
    "pdb_id", "L", "M", "candidate_idx", "seed",
    # ----- Production-parity inline metrics (decoded protein + decoded ligand)
    # mirrors `compute_contact_metrics` / `compute_binding_pocket` in
    # `src/lobster/metrics/ligand_conditioned_protein_generation.py`.
    "n_pocket_residues", "n_contacts",
    "n_residues_in_contact", "frac_residues_in_contact",
    "n_ligand_atoms_in_contact", "frac_ligand_atoms_in_contact",
    "min_protein_ligand_dist", "mean_min_dist_per_residue",
    # ----- predicted sequence (kept for the cofold leg, which produces the
    # publishable CG headline cofold_iptm / cofold_both_pass via
    # `merge_cg_cofold_into_candidates.py`).
    "predicted_sequence",
    "gen_seconds", "score_seconds",
] + _PLL_VARIANTS

# CG is de-novo; production headline = cofold_iptm. Inline oracle uses contact
# metrics (geometric pocket coupling) — the only meaningful inline signal
# without GT (de-novo) or cofold.
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
    ("oracle_contact_pick", "n_contacts", "max"),
]
_SUMMARY_COLS = (
    ["pdb_id", "L", "M", "n_candidates",
     "n_pocket_residues_mean", "n_contacts_mean",
     "frac_residues_in_contact_mean", "min_protein_ligand_dist_mean"]
    + [f"{p}_{suffix}" for p, _, _ in _PICKERS for suffix in ("idx", "n_contacts")]
)


def _do_picks(rows: list[dict]) -> dict:
    """For each picker, store (idx, n_contacts of picked row) — the latter is a
    coarse "did this candidate produce a pocket-coupled design?" gauge."""
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
        out[f"{picker_name}_n_contacts"] = int(rows[idx].get("n_contacts", 0) or 0)
    return out


def _decode_pred_sequence(pred_seq_tokens: torch.Tensor, n_valid: int | None = None) -> str:
    """Decode predicted Lobster AA tokens to a standard AA letter string.

    `n_valid` (optional) trims the result to that many leading positions so
    cofold consumers see exactly the L residues that were generated.
    """
    from lobster.model.latent_generator.utils.residue_constants import (
        convert_lobster_aa_tokenization_to_standard_aa,
        restype_order_with_x_inv,
    )
    pred_logits = torch.nn.functional.one_hot(pred_seq_tokens.long(), num_classes=33).float()
    pred_pdbints = convert_lobster_aa_tokenization_to_standard_aa(
        pred_logits, device=pred_seq_tokens.device).squeeze(0)
    s = "".join(restype_order_with_x_inv.get(int(t), "X") for t in pred_pdbints.tolist())
    if n_valid is not None:
        s = s[:n_valid]
    return s


def _contact_metrics(
    pred_coords: torch.Tensor,
    pred_lig_coords: torch.Tensor | None,
    pmask: torch.Tensor,
    *,
    pocket_threshold: float = 5.0,
    contact_threshold: float = 4.5,
) -> dict:
    """Production-parity inline CG metrics on (decoded protein, decoded ligand).

    Mirrors `compute_contact_metrics` and `compute_binding_pocket` in
    `src/lobster/metrics/ligand_conditioned_protein_generation.py` exactly:
      - pocket = decoded CA within `pocket_threshold` Å of any decoded ligand atom
      - contact = decoded CA-ligand atom pair within `contact_threshold` Å
    """
    nan = float("nan")
    out = {
        "n_pocket_residues": 0,
        "n_contacts": 0,
        "n_residues_in_contact": 0,
        "frac_residues_in_contact": nan,
        "n_ligand_atoms_in_contact": 0,
        "frac_ligand_atoms_in_contact": nan,
        "min_protein_ligand_dist": nan,
        "mean_min_dist_per_residue": nan,
    }
    if pred_lig_coords is None:
        return out
    valid = pmask.bool()
    if valid.dim() == 2:
        valid = valid.squeeze(0)                                  # [L]
    ca = pred_coords[0, valid][:, 1, :].detach().cpu().float()    # [L_valid, 3]
    lig = pred_lig_coords[0].detach().cpu().float()               # [M, 3]
    if ca.shape[0] == 0 or lig.shape[0] == 0:
        return out
    dists = torch.cdist(ca, lig)                                  # [L_valid, M]
    min_per_res = dists.min(dim=1).values
    min_per_lig = dists.min(dim=0).values

    n_residues = int(ca.shape[0])
    n_ligand_atoms = int(lig.shape[0])
    out["n_pocket_residues"] = int((min_per_res < pocket_threshold).sum().item())
    out["n_residues_in_contact"] = int((min_per_res < contact_threshold).sum().item())
    out["n_ligand_atoms_in_contact"] = int((min_per_lig < contact_threshold).sum().item())
    out["n_contacts"] = int((dists < contact_threshold).sum().item())
    out["frac_residues_in_contact"] = out["n_residues_in_contact"] / n_residues
    out["frac_ligand_atoms_in_contact"] = out["n_ligand_atoms_in_contact"] / n_ligand_atoms
    out["min_protein_ligand_dist"] = float(dists.min().item())
    out["mean_min_dist_per_residue"] = float(min_per_res.mean().item())
    return out


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
def _generate_one(model, target: dict, *, gen_kwargs: dict):
    M = target["M"]
    L_override = gen_kwargs.get("length_override")
    if L_override is None:
        L = target["L"]
        pmask = target["protein_mask"].unsqueeze(0)
        pidx = target["protein_indices"].unsqueeze(0)
    else:
        device = target["coords_res"].device
        L = int(L_override)
        pmask = torch.ones((1, L), device=device)
        pidx = torch.arange(L, device=device).unsqueeze(0)
    lmask = target["ligand_mask"].unsqueeze(0)

    # Match the production CG path
    # (`src/lobster/metrics/ligand_conditioned_protein_generation.py:_evaluate_single_design`):
    #   - atom tokens + bond matrix from GT (always)
    #   - structure tokens / embeddings ONLY in `structure_tokens` mode, encoded
    #     via `model.encode_ligand_structure` (ligand-only, with continuous
    #     embeddings) on the *centered* ligand
    #   - in `atom_bond_only` mode: pass `None` for structure tokens/embeddings
    #     and let the model sample them from the prior
    input_lig_atom = target["ligand_atom_types"].unsqueeze(0).long()
    bond_K = target["bond_matrix"].unsqueeze(0).long() if target["bond_matrix"] is not None else None
    ligand_is_context = gen_kwargs["ligand_context_mode"] == "structure_tokens"

    input_lig_struc = None
    input_lig_struc_embeds = None
    if ligand_is_context:
        # Center ligand at origin so the generated protein (which starts from
        # noise around the origin) is spatially close to it.
        lc = target["ligand_coords"].float()
        lm_t = target["ligand_mask"].float()
        lidx = target["ligand_indices"].long()
        valid = lm_t.bool()
        if valid.any():
            lc = lc - lc[valid].mean(dim=0, keepdim=True)
        lc_b = lc.unsqueeze(0)
        lm_b = lm_t.unsqueeze(0)
        lidx_b = lidx.unsqueeze(0)
        with torch.no_grad():
            enc_out = model.encode_ligand_structure(lc_b, lm_b, lidx_b, return_continuous=True)
        input_lig_struc = enc_out[0]
        input_lig_struc_embeds = enc_out[2] if len(enc_out) > 2 else None

    result = model.generate_sample(
        length=L, num_samples=1,
        inverse_folding=False, forward_folding=False,
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
        generate_ligand=True,
        num_atoms=M,
        input_ligand_atom_tokens=input_lig_atom,
        input_ligand_structure_tokens=input_lig_struc,
        input_ligand_structure_embeddings=input_lig_struc_embeds,
        input_bond_matrix=bond_K,
        ligand_is_context=ligand_is_context,
    )
    decoded = model.decode_structure(result, pmask, ligand_mask=lmask)
    vit_out = decoded["vit_decoder"]
    if isinstance(vit_out, dict):
        pred_coords = vit_out["protein_coords"]
        pred_lig_coords = vit_out.get("ligand_coords")
    else:
        pred_coords = vit_out
        pred_lig_coords = None

    pred_seq = result["generated_seq_tokens"]
    pred_struc = result.get("generated_struc_tokens")
    if pred_struc is None and "structure_logits" in result:
        pred_struc = result["structure_logits"].argmax(dim=-1)

    # Generated (final) ligand tokens — what we want to PLL-score, regardless of
    # ligand_context_mode. For structure_tokens mode they equal the GT context
    # (since ligand_is_context=True freezes them). For atom_bond_only mode they
    # are the model's generated outputs.
    gen_lig_atom = result.get("generated_ligand_atom_tokens", input_lig_atom)
    gen_lig_struc = result.get("generated_ligand_struc_tokens")
    if gen_lig_struc is None and "ligand_structure_logits" in result:
        gen_lig_struc = result["ligand_structure_logits"].argmax(dim=-1)
    return pred_coords, pred_lig_coords, pred_seq, pred_struc, gen_lig_atom, gen_lig_struc, pmask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max-protein-length", type=int, default=512)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--target-id", type=str, default=None,
                   help="If set, restrict run to this single pdb_id (used to "
                        "shard generation across SLURM array tasks).")
    p.add_argument("--candidate-offset", type=int, default=0,
                   help="Added to candidate_idx and seed within each target. Used "
                        "to shard a per-target N across array tasks without "
                        "seed/idx collisions.")
    p.add_argument("--seed-base", type=int, default=20260505)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Match production "ReST i1 / optimized" hyperparameters from
    # scripts/eval_cg_boltz_checkpoint.py + cmdline argparse defaults.
    p.add_argument("--nsteps", type=int, default=200,
                   help="ReST i1 CG benchmark used 200; cmdline default 100.")
    p.add_argument("--temperature-seq", type=float, default=0.153)
    p.add_argument("--temperature-struc", type=float, default=0.05)
    p.add_argument("--stochasticity-seq", type=int, default=20)
    p.add_argument("--stochasticity-struc", type=int, default=20)
    p.add_argument("--temperature-ligand", type=float, default=0.1)
    p.add_argument("--stochasticity-ligand", type=int, default=5)
    p.add_argument("--schedule-seq", default="LinearInferenceSchedule",
                   choices=["LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--schedule-struc", default="PowerInferenceSchedule",
                   choices=["LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--schedule-lig-atom", default="PowerInferenceSchedule",
                   choices=["LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--schedule-lig-struc", default="LinearInferenceSchedule",
                   choices=["LogInferenceSchedule", "LinearInferenceSchedule", "PowerInferenceSchedule"])
    p.add_argument("--ligand-context-mode", default="atom_bond_only",
                   choices=["structure_tokens", "atom_bond_only"])
    p.add_argument("--length", default="gt",
                   help="'gt' = use each target's GT protein length (allows TM-to-GT correlation); "
                        "or an integer (e.g. 100) to mirror the ReST i1 CG benchmark — in that "
                        "case TM/RMSD/AAR vs GT are skipped (downstream cofold required).")
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    all_pdb_ids = _list_targets(args.source_data_dir)
    pdb_ids = list(all_pdb_ids)
    if args.max_targets is not None:
        pdb_ids = pdb_ids[: args.max_targets]
    if args.target_id is not None:
        if args.target_id not in pdb_ids:
            raise ValueError(
                f"--target-id '{args.target_id}' not found in {args.source_data_dir}; "
                f"available: {pdb_ids}"
            )
        pdb_ids = [args.target_id]
    if not pdb_ids:
        raise FileNotFoundError(f"No targets found in {args.source_data_dir}")
    # Map each surviving pdb_id to its index in the FULL un-filtered list so
    # that seeds are stable & non-colliding across array shards.
    ti_global_of = {pid: all_pdb_ids.index(pid) for pid in pdb_ids}
    logger.info("Found %d targets (run set: %s)", len(pdb_ids), pdb_ids)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    cand_path = args.output_dir / f"bestofN_cg_lig_candidates_{ts}.csv"
    summ_path = args.output_dir / f"bestofN_cg_lig_summary_{ts}.csv"
    cand_fh = cand_path.open("w", newline="")
    summ_fh = summ_path.open("w", newline="")
    cand_writer = csv.DictWriter(cand_fh, fieldnames=_CANDIDATE_COLS, extrasaction="ignore")
    summ_writer = csv.DictWriter(summ_fh, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
    cand_writer.writeheader()
    summ_writer.writeheader()

    from lobster.model.gen_ume import ProteinLigandEncoderLightningModule

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

    length_override = None if args.length == "gt" else int(args.length)
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
        length_override=length_override,
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

        rows: list[dict] = []
        ti_global = ti_global_of[pdb_id]
        for ci_local in range(args.N):
            ci = ci_local + args.candidate_offset
            seed = (args.seed_base + ti_global * 1_000_003 + ci) & 0x7FFFFFFF
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                (pred_coords, pred_lig_coords, pred_seq, pred_struc,
                 lig_atom_in, lig_struc_in, pmask_gen) = _generate_one(
                    model, target, gen_kwargs=gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM gen %s candidate %d", pdb_id, ci)
                continue
            except Exception as e:
                logger.warning("Gen failed %s candidate %d: %s", pdb_id, ci, e)
                continue
            gen_seconds = time.time() - tg0

            length_matches_gt = (gen_kwargs["length_override"] is None
                                 or int(gen_kwargs["length_override"]) == target["L"])

            try:
                contact = _contact_metrics(pred_coords, pred_lig_coords, pmask_gen)
            except Exception as e:
                logger.warning("contact metrics failed %s candidate %d: %s", pdb_id, ci, e)
                contact = _contact_metrics(pred_coords, None, pmask_gen)  # all-NaN fallback

            try:
                n_valid_seq = int(pmask_gen.bool().sum().item()) if length_matches_gt else int(pred_seq.shape[1])
                pred_str = _decode_pred_sequence(pred_seq, n_valid=n_valid_seq)
            except Exception as e:
                logger.warning("seq decode failed %s candidate %d: %s", pdb_id, ci, e)
                pred_str = ""

            # PLL inputs use the generated length (must match seq/struc shapes).
            L_pred = pred_seq.shape[1]
            if length_matches_gt:
                pmask_pll = target["protein_mask"].unsqueeze(0)
                ridx_pll = target["protein_indices"].unsqueeze(0)
            else:
                device_pll = pred_seq.device
                pmask_pll = torch.ones((1, L_pred), device=device_pll)
                ridx_pll = torch.arange(L_pred, device=device_pll).unsqueeze(0)

            # PLL inputs use the GENERATED ligand tokens (returned from the
            # generation loop). For structure_tokens mode they equal the GT
            # context (frozen by ligand_is_context=True); for atom_bond_only
            # mode they are the model's actual outputs. If the model returns no
            # structure tokens (e.g. continuous-only path), fall back to a
            # ligand-only encoding of the GT pose so PLL still has a valid input.
            lig_atom_pll = lig_atom_in
            if lig_struc_in is None:
                with torch.no_grad():
                    enc_fb = model.encode_ligand_structure(
                        target["ligand_coords"].unsqueeze(0),
                        target["ligand_mask"].unsqueeze(0),
                        target["ligand_indices"].unsqueeze(0),
                        return_continuous=False,
                    )
                lig_struc_pll = enc_fb[0]
            else:
                lig_struc_pll = lig_struc_in

            inputs = {
                "seq_clean": pred_seq,                                          # PREDICTED
                "struc_clean": pred_struc,                                      # PREDICTED
                "lig_atom_clean": lig_atom_pll,                                 # GENERATED (or GT if frozen)
                "lig_struc_clean": lig_struc_pll,                               # GENERATED (or fallback)
                "protein_mask": pmask_pll,
                "ligand_mask": target["ligand_mask"].unsqueeze(0),
                "residue_index": ridx_pll,
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
                "candidate_idx": ci, "seed": seed,
                # Production-parity inline metrics (decoded vs decoded)
                **{k: contact[k] for k in (
                    "n_pocket_residues", "n_contacts",
                    "n_residues_in_contact", "frac_residues_in_contact",
                    "n_ligand_atoms_in_contact", "frac_ligand_atoms_in_contact",
                    "min_protein_ligand_dist", "mean_min_dist_per_residue",
                )},
                "predicted_sequence": pred_str,
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

        def _agg(rows: list[dict], key: str) -> float:
            xs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
            xs = [x for x in xs if x == x]  # drop NaNs
            return (sum(xs) / len(xs)) if xs else float("nan")

        picks = _do_picks(rows)
        summ_writer.writerow({
            "pdb_id": pdb_id, "L": target["L"], "M": target["M"],
            "n_candidates": len(rows),
            "n_pocket_residues_mean": _agg(rows, "n_pocket_residues"),
            "n_contacts_mean": _agg(rows, "n_contacts"),
            "frac_residues_in_contact_mean": _agg(rows, "frac_residues_in_contact"),
            "min_protein_ligand_dist_mean": _agg(rows, "min_protein_ligand_dist"),
            **picks,
        })
        summ_fh.flush()
        n_targets_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            n_contacts_mean = _agg(rows, "n_contacts")
            n_pocket_mean = _agg(rows, "n_pocket_residues")
            logger.info(
                "[%4d/%d] %s L=%d M=%d  pocket[mean=%.1f]  contacts[mean=%.1f]  pick(n_contacts): r=%d sP=%d st=%d la=%d ls=%d jpro=%d jlig=%d jall=%d jt4=%d orC=%d  (%.1fs/target)",
                ti + 1, len(pdb_ids), pdb_id, target["L"], target["M"],
                n_pocket_mean, n_contacts_mean,
                picks["random_pick_n_contacts"], picks["seq_pll_pick_n_contacts"],
                picks["struc_pll_pick_n_contacts"], picks["lig_atom_pll_pick_n_contacts"],
                picks["lig_struc_pll_pick_n_contacts"], picks["joint_protein_pll_pick_n_contacts"],
                picks["joint_ligand_pll_pick_n_contacts"], picks["joint_all_pll_pick_n_contacts"],
                picks["joint_true_4_pll_pick_n_contacts"], picks["oracle_contact_pick_n_contacts"],
                elapsed / max(1, n_targets_done),
            )

    cand_fh.close()
    summ_fh.close()
    logger.info("Done. %d targets / %d skipped. Outputs: %s | %s",
                n_targets_done, n_skipped, cand_path, summ_path)


if __name__ == "__main__":
    main()
