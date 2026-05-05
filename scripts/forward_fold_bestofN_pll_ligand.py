"""Forward-folding best-of-N with 4-modality PLL on PoseBusters (protein+ligand).

For each PoseBusters target (`<pdb_id>_protein.pt` + `<pdb_id>_ligand.pt`),
generates `N` independent forward-fold candidates and scores **the predicted
protein structure tokens** with the model's own 4-modality pseudo-likelihood
(seq + struc + lig_atom + lig_struc, plus joints). Quality metrics are
computed from the decoded predicted coords (TM-score, RMSD vs GT). One row
per (target, candidate) is written to a candidates CSV; one row per target
to a summary CSV with picker indices.

Pickers compared:
  random_pick (= candidate 0 = single-shot baseline),
  seq_pll_pick, struc_pll_pick, lig_atom_pll_pick, lig_struc_pll_pick,
  joint_protein_pll_pick (= seq + struc),
  joint_ligand_pll_pick  (= lig_atom + lig_struc),
  joint_all_pll_pick     (additive 4),
  joint_true_4_pll_pick  (true 4-way joint),
  oracle_tm_pick (argmax tm_score, ceiling).

Usage:
    uv run python scripts/forward_fold_bestofN_pll_ligand.py \\
        --source-data-dir /cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_gen_ume_all_latest/last.ckpt \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_pl_bestofN_ff_all \\
        --N 10 --K 32

For E0 (per-prediction correlation), use --N 1.
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
logger = logging.getLogger("ff_bestofN_pll_ligand")


# Quality + per-PLL columns for candidates CSV.
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
    "tm_score", "rmsd", "ligand_rmsd", "ligand_centroid_distance",
    "gen_seconds", "score_seconds",
] + _PLL_VARIANTS


# (picker_name, score_col, direction)  — direction = "min" for NLL, "max" for oracle TM
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
    ("oracle_tm_pick", "tm_score", "max"),
]
_SUMMARY_COLS = (
    ["pdb_id", "L", "M", "include_ligand", "n_candidates",
     "tm_min", "tm_mean", "tm_median", "tm_max", "tm_std"]
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
        out[f"{picker_name}_tm"] = float(rows[idx]["tm_score"])
    return out


def _compute_tm_rmsd(target: dict, pred_coords: torch.Tensor) -> tuple[float, float]:
    """TM-score (tm_align CA) and CA-RMSD (Kabsch) between predicted and GT protein."""
    from lobster.metrics import align_and_compute_rmsd
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    from tmtools import tm_align

    valid = target["protein_mask"].bool()
    seq_str = "".join(restype_order_with_x_inv.get(int(s), "X") for s in target["sequence"][valid].tolist())
    gt = target["coords_res"][valid]                # [L_valid, 3, 3]
    pred = pred_coords[0, valid]                    # [L_valid, 3, 3]

    tm = tm_align(
        pred[:, 1, :].cpu().numpy(),
        gt[:, 1, :].detach().cpu().numpy(),
        seq_str, seq_str,
    )
    rmsd = align_and_compute_rmsd(coords1=pred, coords2=gt, mask=None,
                                   return_aligned=False, device=pred.device)
    return float(tm.tm_norm_chain1), float(rmsd)


def _kabsch_rmsd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Atom-wise RMSD after rigid-body alignment (Kabsch). Inputs: [M, 3]."""
    pred = pred.detach().cpu().double()
    gt = gt.detach().cpu().double()
    pc = pred.mean(0)
    gc = gt.mean(0)
    P = pred - pc
    G = gt - gc
    H = P.T @ G
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=H.dtype))
    R = Vt.T @ D @ U.T
    P_aligned = P @ R.T
    return float(torch.sqrt(((P_aligned - G) ** 2).sum(-1).mean()).item())


def _ligand_rmsd(target: dict, pred_lig_coords: torch.Tensor | None) -> tuple[float, float]:
    """Per-atom RMSD (Kabsch) + centroid distance for ligand atoms."""
    if pred_lig_coords is None or target["ligand_coords"] is None:
        return float("nan"), float("nan")
    gt = target["ligand_coords"]              # [M, 3]
    pred = pred_lig_coords[0]                 # [M, 3]
    if gt.shape[0] != pred.shape[0] or gt.shape[0] < 3:
        return float("nan"), float("nan")
    try:
        rmsd = _kabsch_rmsd(pred, gt)
    except Exception:
        rmsd = float("nan")
    centroid = float(torch.linalg.norm(pred.mean(0).cpu().double() - gt.mean(0).cpu().double()).item())
    return rmsd, centroid


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
def _generate_one(
    model,
    target: dict,
    *,
    aa_transform,
    include_ligand: bool,
    gen_kwargs: dict,
):
    """One forward-fold candidate. Returns (predicted_coords, predicted_lig_coords, predicted_struc_tokens, gt_lig_struc_tokens)."""
    device = target["coords_res"].device
    L = target["L"]
    M = target["M"]
    seq_pdbints = target["sequence"]
    seq_tokenized = _tokenize_seq_pdbints(aa_transform, seq_pdbints).to(device)
    if int(seq_tokenized.shape[0]) != L:
        raise ValueError(f"Seq tokenization length {int(seq_tokenized.shape[0])} != L={L}")
    input_seq = seq_tokenized.unsqueeze(0)       # [1, L]
    pmask = target["protein_mask"].unsqueeze(0)  # [1, L]
    pidx = target["protein_indices"].unsqueeze(0)
    lmask = target["ligand_mask"].unsqueeze(0) if include_ligand else None

    # Encode GT ligand structure tokens (used as fixed context).
    input_lig_atom = None
    input_lig_struc = None
    bond_K = None
    if include_ligand:
        enc = _encode_gt_tokens(model, target)
        input_lig_atom = target["ligand_atom_types"].unsqueeze(0)            # [1, M]
        input_lig_struc = enc["ligand_tokens"]                                 # [1, M]
        bond_K = target["bond_matrix"].unsqueeze(0) if target["bond_matrix"] is not None else None

    ligand_is_context = include_ligand and gen_kwargs["ligand_context_mode"] == "structure_tokens"

    result = model.generate_sample(
        length=L,
        num_samples=1,
        forward_folding=True,
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
        input_sequence_tokens=input_seq,
        input_mask=pmask,
        input_indices=pidx,
        generate_ligand=include_ligand,
        num_atoms=M if include_ligand else 0,
        input_ligand_atom_tokens=input_lig_atom,
        input_ligand_structure_tokens=input_lig_struc,
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

    pred_struc_tokens = result.get("generated_struc_tokens")
    if pred_struc_tokens is None and "structure_logits" in result:
        pred_struc_tokens = result["structure_logits"].argmax(dim=-1)
    return pred_coords, pred_lig_coords, pred_struc_tokens, input_lig_atom, input_lig_struc, input_seq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--N", type=int, default=10, help="Candidates per target (use 1 for E0 correlation only)")
    p.add_argument("--K", type=int, default=32, help="PLL Monte-Carlo draws per modality")
    p.add_argument("--max-protein-length", type=int, default=512)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=20260505)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-ligand", action="store_true",
                   help="Disable ligand context (forward fold protein only). Default: include ligand.")
    # Match production hyperparameters from evaluate_protein_ligand_forward_folding.py.
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
    cand_path = args.output_dir / f"bestofN_ff_lig_candidates_{ts}.csv"
    summ_path = args.output_dir / f"bestofN_ff_lig_summary_{ts}.csv"
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

        rows: list[dict] = []
        for ci in range(args.N):
            seed = (args.seed_base + ti * 1_000_003 + ci) & 0x7FFFFFFF
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                pred_coords, pred_lig_coords, pred_struc, lig_atom_in, lig_struc_in, seq_in = _generate_one(
                    model, target, aa_transform=aa_transform,
                    include_ligand=include_ligand, gen_kwargs=gen_kwargs,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM gen %s candidate %d; skipping", pdb_id, ci)
                continue
            except Exception as e:
                logger.warning("Gen failed for %s candidate %d: %s", pdb_id, ci, e)
                continue
            gen_seconds = time.time() - tg0

            try:
                tm, rmsd = _compute_tm_rmsd(target, pred_coords)
            except Exception as e:
                logger.warning("TM/RMSD failed %s candidate %d: %s", pdb_id, ci, e)
                continue
            lig_rmsd, lig_centroid = _ligand_rmsd(target, pred_lig_coords) if include_ligand else (float("nan"), float("nan"))

            # Build PLL inputs: GT seq, PREDICTED struc tokens, GT lig (atom + struc).
            inputs = {
                "seq_clean": seq_in,                                           # [1, L]
                "struc_clean": pred_struc,                                     # [1, L]  PREDICTED
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
                "candidate_idx": ci, "seed": seed,
                "include_ligand": include_ligand,
                "tm_score": tm, "rmsd": rmsd,
                "ligand_rmsd": lig_rmsd, "ligand_centroid_distance": lig_centroid,
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

        tms = [r["tm_score"] for r in rows]
        mean_tm = sum(tms) / len(tms)
        std_tm = (sum((x - mean_tm) ** 2 for x in tms) / len(tms)) ** 0.5
        picks = _do_picks(rows)
        summ_writer.writerow({
            "pdb_id": pdb_id, "L": target["L"], "M": target["M"],
            "include_ligand": include_ligand, "n_candidates": len(rows),
            "tm_min": min(tms), "tm_mean": mean_tm,
            "tm_median": sorted(tms)[len(tms) // 2], "tm_max": max(tms),
            "tm_std": std_tm,
            **picks,
        })
        summ_fh.flush()
        n_targets_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%4d/%d] %s L=%d M=%d  tm[mean=%.3f, max=%.3f]  pick: r=%.3f sP=%.3f st=%.3f la=%.3f ls=%.3f jpro=%.3f jlig=%.3f jall=%.3f jt4=%.3f or=%.3f  (%.1fs/target)",
                ti + 1, len(pdb_ids), pdb_id, target["L"], target["M"],
                mean_tm, max(tms),
                picks["random_pick_tm"], picks["seq_pll_pick_tm"],
                picks["struc_pll_pick_tm"], picks["lig_atom_pll_pick_tm"],
                picks["lig_struc_pll_pick_tm"], picks["joint_protein_pll_pick_tm"],
                picks["joint_ligand_pll_pick_tm"], picks["joint_all_pll_pick_tm"],
                picks["joint_true_4_pll_pick_tm"], picks["oracle_tm_pick_tm"],
                elapsed / max(1, n_targets_done),
            )

    cand_fh.close()
    summ_fh.close()
    logger.info("Done. %d targets / %d skipped. Outputs: %s | %s",
                n_targets_done, n_skipped, cand_path, summ_path)


if __name__ == "__main__":
    main()
