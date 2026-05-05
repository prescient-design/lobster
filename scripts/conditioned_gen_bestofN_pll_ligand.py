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
    "pseudo_aar", "predicted_sequence",
    "tm_to_gt", "rmsd_to_gt", "ligand_pocket_min_dist",
    "gen_seconds", "score_seconds",
] + _PLL_VARIANTS

# CG has no ground-truth oracle (de-novo generation). Use (cheap) pseudo-AAR
# and decoded TM-to-GT as proxy oracles to bound the picker headroom.
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
    ("oracle_tm_pick", "tm_to_gt", "max"),
    ("oracle_aar_pick", "pseudo_aar", "max"),
]
_SUMMARY_COLS = (
    ["pdb_id", "L", "M", "n_candidates",
     "tm_min", "tm_mean", "tm_max", "aar_mean", "aar_max"]
    + [f"{p}_{suffix}" for p, _, _ in _PICKERS for suffix in ("idx", "tm")]
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
        out[f"{picker_name}_tm"] = float(rows[idx]["tm_to_gt"])
    return out


def _compute_pseudo_aar(target: dict, pred_seq_tokens: torch.Tensor) -> tuple[float, str]:
    from lobster.model.latent_generator.utils.residue_constants import (
        convert_lobster_aa_tokenization_to_standard_aa,
        restype_order_with_x_inv,
    )
    valid = target["protein_mask"].bool()
    gt_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in target["sequence"][valid].tolist())
    pred_logits = torch.nn.functional.one_hot(pred_seq_tokens.long(), num_classes=33).float()
    pred_pdbints = convert_lobster_aa_tokenization_to_standard_aa(
        pred_logits, device=pred_seq_tokens.device).squeeze(0)[valid]
    pred_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in pred_pdbints.tolist())
    if len(gt_str) != len(pred_str) or not gt_str:
        return float("nan"), pred_str
    return float(sum(a == b for a, b in zip(gt_str, pred_str)) / len(gt_str)), pred_str


def _compute_tm_to_gt(target: dict, pred_coords: torch.Tensor) -> tuple[float, float]:
    from lobster.metrics import align_and_compute_rmsd
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    from tmtools import tm_align
    valid = target["protein_mask"].bool()
    seq_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in target["sequence"][valid].tolist())
    gt = target["coords_res"][valid]
    pred = pred_coords[0, valid]
    tm = tm_align(pred[:, 1, :].cpu().numpy(), gt[:, 1, :].detach().cpu().numpy(), seq_str, seq_str)
    rmsd = align_and_compute_rmsd(coords1=pred, coords2=gt, mask=None,
                                   return_aligned=False, device=pred.device)
    return float(tm.tm_norm_chain1), float(rmsd)


def _ligand_pocket_min_dist(target: dict, pred_coords: torch.Tensor) -> float:
    """Min CA-atom distance to any ligand heavy-atom center (proxy for pocket coupling)."""
    if target["ligand_coords"] is None:
        return float("nan")
    pmask = target["protein_mask"].bool()
    ca = pred_coords[0, pmask][:, 1, :]      # [L_valid, 3]
    lig = target["ligand_coords"]            # [M, 3]
    if ca.shape[0] == 0 or lig.shape[0] == 0:
        return float("nan")
    d = torch.cdist(ca.float(), lig.float())
    return float(d.min().item())


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

    enc = _encode_gt_tokens(model, target)
    input_lig_atom = target["ligand_atom_types"].unsqueeze(0)
    input_lig_struc = enc["ligand_tokens"]
    bond_K = target["bond_matrix"].unsqueeze(0) if target["bond_matrix"] is not None else None
    ligand_is_context = gen_kwargs["ligand_context_mode"] == "structure_tokens"

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
        input_bond_matrix=bond_K,
        ligand_is_context=ligand_is_context,
    )
    decoded = model.decode_structure(result, pmask, ligand_mask=lmask)
    vit_out = decoded["vit_decoder"]
    pred_coords = vit_out["protein_coords"] if isinstance(vit_out, dict) else vit_out

    pred_seq = result["generated_seq_tokens"]
    pred_struc = result.get("generated_struc_tokens")
    if pred_struc is None and "structure_logits" in result:
        pred_struc = result["structure_logits"].argmax(dim=-1)
    return pred_coords, pred_seq, pred_struc, input_lig_atom, input_lig_struc


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

    pdb_ids = _list_targets(args.source_data_dir)
    if args.max_targets is not None:
        pdb_ids = pdb_ids[: args.max_targets]
    if not pdb_ids:
        raise FileNotFoundError(f"No targets found in {args.source_data_dir}")
    logger.info("Found %d targets", len(pdb_ids))

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
        for ci in range(args.N):
            seed = (args.seed_base + ti * 1_000_003 + ci) & 0x7FFFFFFF
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                pred_coords, pred_seq, pred_struc, lig_atom_in, lig_struc_in = _generate_one(
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

            if length_matches_gt:
                try:
                    tm, rmsd = _compute_tm_to_gt(target, pred_coords)
                except Exception as e:
                    logger.warning("TM failed %s candidate %d: %s", pdb_id, ci, e)
                    tm, rmsd = float("nan"), float("nan")
                try:
                    aar, pred_str = _compute_pseudo_aar(target, pred_seq)
                except Exception:
                    aar, pred_str = float("nan"), ""
                try:
                    pocket_d = _ligand_pocket_min_dist(target, pred_coords)
                except Exception:
                    pocket_d = float("nan")
            else:
                tm, rmsd = float("nan"), float("nan")
                aar, pred_str = float("nan"), ""
                # Pocket distance still meaningful: just min CA-to-ligand.
                from torch import cdist
                try:
                    ca = pred_coords[0, :, 1, :]
                    pocket_d = float(cdist(ca.float(), target["ligand_coords"].float()).min().item())
                except Exception:
                    pocket_d = float("nan")

            # PLL inputs use the generated length (must match seq/struc shapes).
            L_pred = pred_seq.shape[1]
            if length_matches_gt:
                pmask_pll = target["protein_mask"].unsqueeze(0)
                ridx_pll = target["protein_indices"].unsqueeze(0)
            else:
                device_pll = pred_seq.device
                pmask_pll = torch.ones((1, L_pred), device=device_pll)
                ridx_pll = torch.arange(L_pred, device=device_pll).unsqueeze(0)

            inputs = {
                "seq_clean": pred_seq,                                          # PREDICTED
                "struc_clean": pred_struc,                                      # PREDICTED
                "lig_atom_clean": lig_atom_in,                                  # GT
                "lig_struc_clean": lig_struc_in,                                # GT
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
                "pseudo_aar": aar, "predicted_sequence": pred_str,
                "tm_to_gt": tm, "rmsd_to_gt": rmsd,
                "ligand_pocket_min_dist": pocket_d,
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

        tms = [r["tm_to_gt"] for r in rows if r["tm_to_gt"] == r["tm_to_gt"]]  # drop NaNs
        aars = [r["pseudo_aar"] for r in rows if r["pseudo_aar"] == r["pseudo_aar"]]
        picks = _do_picks(rows)
        summ_writer.writerow({
            "pdb_id": pdb_id, "L": target["L"], "M": target["M"],
            "n_candidates": len(rows),
            "tm_min": min(tms) if tms else float("nan"),
            "tm_mean": (sum(tms) / len(tms)) if tms else float("nan"),
            "tm_max": max(tms) if tms else float("nan"),
            "aar_mean": (sum(aars) / len(aars)) if aars else float("nan"),
            "aar_max": max(aars) if aars else float("nan"),
            **picks,
        })
        summ_fh.flush()
        n_targets_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            tm_max = max(tms) if tms else float("nan")
            tm_mean = (sum(tms) / len(tms)) if tms else float("nan")
            logger.info(
                "[%4d/%d] %s L=%d M=%d  tm[mean=%.3f max=%.3f]  pick: r=%.3f sP=%.3f st=%.3f la=%.3f ls=%.3f jpro=%.3f jlig=%.3f jall=%.3f jt4=%.3f orTM=%.3f orAAR=%.3f  (%.1fs/target)",
                ti + 1, len(pdb_ids), pdb_id, target["L"], target["M"],
                tm_mean, tm_max,
                picks["random_pick_tm"], picks["seq_pll_pick_tm"],
                picks["struc_pll_pick_tm"], picks["lig_atom_pll_pick_tm"],
                picks["lig_struc_pll_pick_tm"], picks["joint_protein_pll_pick_tm"],
                picks["joint_ligand_pll_pick_tm"], picks["joint_all_pll_pick_tm"],
                picks["joint_true_4_pll_pick_tm"], picks["oracle_tm_pick_tm"],
                picks["oracle_aar_pick_tm"],
                elapsed / max(1, n_targets_done),
            )

    cand_fh.close()
    summ_fh.close()
    logger.info("Done. %d targets / %d skipped. Outputs: %s | %s",
                n_targets_done, n_skipped, cand_path, summ_path)


if __name__ == "__main__":
    main()
