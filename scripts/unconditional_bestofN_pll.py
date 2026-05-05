"""Unconditional best-of-N: PLL-driven candidate selection on LEFLUR-P-VAL.

Mirrors `inverse_fold_bestofN_pll.py` but for *unconditional* generation: per
(length, slot) we generate N independent candidates with the LEFLUR-P-VAL
hyperparameters (TED-val25-base, seq20/struc60, biasV=1.0, steps=25, log/power
inference schedules), score each with the in-model PLL, and ESMFold the
generated sequence to compute self-consistency metrics (sc-TM and sc-RMSD vs
the model's own decoded backbone, plus pLDDT).

Per-slot pickers:
    random_pick         : candidate 0 (= single-shot baseline)
    seq_pll_pick        : argmin seq_score_unif
    struc_pll_pick      : argmin struc_score_unif
    joint_pll_pick      : argmin joint_score_unif         (sum-of-conditionals)
    joint_true_pll_pick : argmin joint_true_score_unif    (true joint masking)
    oracle_tm_pick      : argmax esmfold_tm_score (= sc-TM upper bound)

Outputs:
  - bestofN_uc_candidates_<ts>.csv : one row per (length, slot, candidate)
  - bestofN_uc_summary_<ts>.csv    : one row per (length, slot) with picks

Usage:
    uv run python scripts/unconditional_bestofN_pll.py \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional \\
        --N 30 --slots 10 --lengths 100,200,300,400,500
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
from score_gen_ume_pll import score_one_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unconditional_bestofN_pll")


_CANDIDATE_COLS = [
    "length",
    "slot",
    "candidate_idx",
    "seed",
    "esmfold_plddt",
    "esmfold_tm_score",
    "esmfold_rmsd",
    "esmfold_pae",
    "seq_score_unif",
    "seq_score_arllh",
    "struc_score_unif",
    "struc_score_arllh",
    "joint_score_unif",
    "joint_score_arllh",
    "joint_true_score_unif",
    "joint_true_score_arllh",
    "seq_score_t0.25",
    "seq_score_t0.5",
    "seq_score_t0.75",
    "struc_score_t0.25",
    "struc_score_t0.5",
    "struc_score_t0.75",
    "gen_seconds",
    "score_seconds",
    "esmfold_seconds",
]


_PICKERS = [
    ("random_pick", None, "fixed_zero"),
    ("seq_pll_pick", "seq_score_unif", "argmin"),
    ("struc_pll_pick", "struc_score_unif", "argmin"),
    ("joint_pll_pick", "joint_score_unif", "argmin"),
    ("joint_true_pll_pick", "joint_true_score_unif", "argmin"),
    ("oracle_tm_pick", "esmfold_tm_score", "argmax"),
]


def _summary_cols() -> list[str]:
    cols = [
        "length",
        "slot",
        "n_candidates",
        "esmfold_tm_mean",
        "esmfold_tm_min",
        "esmfold_tm_max",
        "esmfold_rmsd_mean",
        "esmfold_plddt_mean",
    ]
    for picker_name, _, _ in _PICKERS:
        cols.append(f"{picker_name}_idx")
        for metric in ("esmfold_tm_score", "esmfold_rmsd", "esmfold_plddt"):
            cols.append(f"{picker_name}_{metric}")
    return cols


_SUMMARY_COLS = _summary_cols()


def _get_inference_schedule_class(name: str):
    import bionemo.moco.schedules.inference_time_schedules as sched

    return getattr(sched, name)


def _build_logit_bias(bias_cfg: dict | None, device: torch.device):
    if not bias_cfg:
        return None
    from lobster.tokenization._amino_acid import AA_VOCAB

    bias = torch.zeros(len(AA_VOCAB), device=device)
    for aa, val in bias_cfg.items():
        if aa in AA_VOCAB:
            bias[AA_VOCAB[aa]] = float(val)
    return bias


@torch.no_grad()
def _generate_one_candidate(model, length: int, gen_kwargs):
    """One unconditional (seq, struc) candidate.

    Returns (seq_std[1,L], seq_lobster[1,L], struc_argmax[1,L], coords[1,L,3,3], mask[1,L]).

    `seq_std` is in standard 0..20 vocab (for ESMFold + AAR), `seq_lobster` is in
    the model's input 33-vocab (for PLL scoring — argmax of sequence_logits).
    """
    from lobster.model.latent_generator.utils.residue_constants import (
        convert_lobster_aa_tokenization_to_standard_aa,
    )

    gen_sample = model.generate_sample(
        length=length,
        num_samples=1,
        nsteps=gen_kwargs["nsteps"],
        temperature_seq=gen_kwargs["temperature_seq"],
        temperature_struc=gen_kwargs["temperature_struc"],
        stochasticity_seq=gen_kwargs["stochasticity_seq"],
        stochasticity_struc=gen_kwargs["stochasticity_struc"],
        inference_schedule_seq=gen_kwargs["inference_schedule_seq"],
        inference_schedule_struc=gen_kwargs["inference_schedule_struc"],
        asynchronous_sampling=gen_kwargs.get("asynchronous_sampling", False),
        sequence_logit_bias=gen_kwargs.get("sequence_logit_bias"),
        sequence_logit_bias_steps=gen_kwargs.get("sequence_logit_bias_steps", 0),
    )

    device = gen_sample["sequence_logits"].device
    mask = torch.ones((1, length), device=device)

    seq_lobster = gen_sample["sequence_logits"].argmax(dim=-1)  # [1, L] in 33-vocab

    if gen_sample["sequence_logits"].shape[-1] == 33:
        seq_std = convert_lobster_aa_tokenization_to_standard_aa(
            gen_sample["sequence_logits"], device=device
        )
    else:
        seq_std = seq_lobster.clone()
        seq_std[seq_std > 21] = 20

    decoded = model.decode_structure(gen_sample, mask)
    x_recon_xyz = decoded["vit_decoder"]  # [1, L, 3, 3]
    struc_argmax = gen_sample["structure_logits"].argmax(dim=-1)  # [1, L]
    return seq_std, seq_lobster, struc_argmax, x_recon_xyz, mask


@torch.no_grad()
def _esmfold_one_candidate(plm_fold, seq_std: torch.Tensor, x_recon_xyz: torch.Tensor, device: torch.device):
    """sc-TM / sc-RMSD vs the model's own decoded backbone, plus pLDDT, PAE."""
    from lobster.metrics import align_and_compute_rmsd
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    from tmtools import tm_align

    L = seq_std.shape[1]
    seq_chars = [restype_order_with_x_inv.get(int(t), "X") for t in seq_std[0].tolist()]
    sequence_str = "".join(seq_chars)
    sequence_str_clean = sequence_str.replace("X", "A")

    if not sequence_str_clean:
        return float("nan"), float("nan"), float("nan"), float("nan")

    tokenized = plm_fold.tokenizer.encode_plus(
        sequence_str_clean,
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    outputs = plm_fold.model(tokenized)

    pred_coords = outputs["positions"][-1, 0, :, :3, :]  # [L, 3, 3]
    plddt = float(outputs["plddt"].mean().item())
    pae = float(outputs["predicted_aligned_error"].mean().item())

    if pred_coords.shape[0] != L:
        return plddt, float("nan"), float("nan"), pae

    gen = x_recon_xyz[0]

    try:
        tm_out = tm_align(
            pred_coords[:, 1, :].cpu().numpy(),
            gen[:, 1, :].detach().cpu().numpy(),
            sequence_str,
            sequence_str,
        )
        tm = float(tm_out.tm_norm_chain1)
    except Exception:
        tm = float("nan")

    try:
        rmsd = float(
            align_and_compute_rmsd(
                coords1=pred_coords,
                coords2=gen,
                mask=None,
                return_aligned=False,
                device=pred_coords.device,
            )
        )
    except Exception:
        rmsd = float("nan")

    return plddt, tm, rmsd, pae


def _do_picks(rows: list[dict]) -> dict:
    out: dict = {}
    for picker_name, key, mode in _PICKERS:
        if mode == "fixed_zero":
            idx = 0
        else:
            vals = [r.get(key) for r in rows]

            def _safe(v, mode_inner=mode):
                if v is None:
                    return float("inf") if mode_inner == "argmin" else float("-inf")
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return float("inf") if mode_inner == "argmin" else float("-inf")
                if fv != fv:
                    return float("inf") if mode_inner == "argmin" else float("-inf")
                return fv

            scored = [_safe(v) for v in vals]
            if mode == "argmin":
                idx = int(min(range(len(rows)), key=lambda i: scored[i]))
            else:
                idx = int(max(range(len(rows)), key=lambda i: scored[i]))
        out[f"{picker_name}_idx"] = idx
        for metric in ("esmfold_tm_score", "esmfold_rmsd", "esmfold_plddt"):
            v = rows[idx].get(metric)
            out[f"{picker_name}_{metric}"] = float(v) if v is not None else float("nan")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--N", type=int, default=30, help="Candidates per slot")
    p.add_argument("--slots", type=int, default=10, help="Number of design slots per length")
    p.add_argument("--K", type=int, default=32, help="PLL Monte-Carlo draws per modality")
    p.add_argument(
        "--lengths",
        type=str,
        default="100,200,300,400,500",
        help="Comma-separated list of design lengths",
    )
    p.add_argument("--seed-base", type=int, default=20260503)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # LEFLUR-P-VAL (TED-val25-base) hyperparameters
    p.add_argument("--nsteps", type=int, default=400)
    p.add_argument("--temperature-seq", type=float, default=0.27315634404739075)
    p.add_argument("--temperature-struc", type=float, default=0.31640411575109995)
    p.add_argument("--stochasticity-seq", type=int, default=20)
    p.add_argument("--stochasticity-struc", type=int, default=60)
    p.add_argument("--inference-schedule-seq", default="LogInferenceSchedule")
    p.add_argument("--inference-schedule-struc", default="PowerInferenceSchedule")
    p.add_argument("--bias-V", type=float, default=1.0, help="Logit bias on Valine")
    p.add_argument("--bias-steps", type=int, default=25, help="Logit bias schedule steps")
    p.add_argument("--no-esmfold", action="store_true")
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    logger.info("Lengths: %s, slots/length: %d, candidates/slot: %d", lengths, args.slots, args.N)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    cand_path = args.output_dir / f"bestofN_uc_candidates_{ts}.csv"
    summ_path = args.output_dir / f"bestofN_uc_summary_{ts}.csv"
    cand_fh = cand_path.open("w", newline="")
    summ_fh = summ_path.open("w", newline="")
    cand_writer = csv.DictWriter(cand_fh, fieldnames=_CANDIDATE_COLS, extrasaction="ignore")
    summ_writer = csv.DictWriter(summ_fh, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
    cand_writer.writeheader()
    summ_writer.writeheader()
    cand_fh.flush()
    summ_fh.flush()

    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = UMESequenceStructureEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)
    model.interpolant_seq.device = device
    model.interpolant_struc.device = device
    logger.info("Model loaded in %.1fs (device=%s)", time.time() - t0, device)

    seq_mask_id = int(model.mask_token_id)
    struc_mask_id = int(model.mask_index_struc_tokens)
    logger.info("seq_mask_id=%d  struc_mask_id=%d", seq_mask_id, struc_mask_id)

    plm_fold = None
    if not args.no_esmfold:
        from lobster.model import LobsterPLMFold

        logger.info("Loading ESMFold...")
        t0 = time.time()
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=max(lengths))
        plm_fold.to(device)
        plm_fold.model.eval()
        logger.info("ESMFold loaded in %.1fs", time.time() - t0)

    sched_seq_cls = _get_inference_schedule_class(args.inference_schedule_seq)
    sched_struc_cls = _get_inference_schedule_class(args.inference_schedule_struc)
    bias_cfg = {"V": args.bias_V} if args.bias_V is not None else None
    bias_tensor = _build_logit_bias(bias_cfg, device)

    gen_kwargs = dict(
        nsteps=args.nsteps,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        inference_schedule_seq=sched_seq_cls,
        inference_schedule_struc=sched_struc_cls,
        asynchronous_sampling=False,
        sequence_logit_bias=bias_tensor,
        sequence_logit_bias_steps=args.bias_steps,
    )

    n_slots_done = 0
    t_start = time.time()

    for li, length in enumerate(lengths):
        for slot in range(args.slots):
            rows: list[dict] = []
            for ci in range(args.N):
                seed = args.seed_base + li * 10_000_019 + slot * 1_000_003 + ci
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                tg0 = time.time()
                try:
                    seq_std, seq_lobster, struc_argmax, x_recon_xyz, mask = _generate_one_candidate(
                        model, length, gen_kwargs
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning("OOM gen L=%d slot=%d cand=%d", length, slot, ci)
                    continue
                except Exception as e:
                    logger.warning("Gen failed L=%d slot=%d cand=%d: %s", length, slot, ci, e)
                    continue
                gen_seconds = time.time() - tg0

                te0 = time.time()
                if plm_fold is not None:
                    try:
                        plddt, tm, rmsd, pae = _esmfold_one_candidate(plm_fold, seq_std, x_recon_xyz, device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        plddt = tm = rmsd = pae = float("nan")
                    except Exception as e:
                        logger.warning("ESMFold failed L=%d slot=%d cand=%d: %s", length, slot, ci, e)
                        plddt = tm = rmsd = pae = float("nan")
                else:
                    plddt = tm = rmsd = pae = float("nan")
                esmfold_seconds = time.time() - te0

                seq_clean = seq_lobster[:, :length]
                struc_clean = struc_argmax[:, :length]
                mask_t = mask[:, :length]
                residue_index = torch.arange(length, device=device).unsqueeze(0)

                ts0 = time.time()
                try:
                    scores = score_one_sample(
                        model,
                        seq_clean=seq_clean,
                        struc_clean=struc_clean,
                        mask=mask_t,
                        residue_index=residue_index,
                        K=args.K,
                        seed=seed ^ 0xA5A5A5,
                        seq_mask_id=seq_mask_id,
                        struc_mask_id=struc_mask_id,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    scores = {}
                except Exception as e:
                    logger.warning("PLL failed L=%d slot=%d cand=%d: %s", length, slot, ci, e)
                    scores = {}
                score_seconds = time.time() - ts0

                row = {
                    "length": length,
                    "slot": slot,
                    "candidate_idx": ci,
                    "seed": seed,
                    "esmfold_plddt": plddt,
                    "esmfold_tm_score": tm,
                    "esmfold_rmsd": rmsd,
                    "esmfold_pae": pae,
                    "gen_seconds": round(gen_seconds, 3),
                    "score_seconds": round(score_seconds, 3),
                    "esmfold_seconds": round(esmfold_seconds, 3),
                    **{k: v for k, v in scores.items() if k in _CANDIDATE_COLS},
                }
                rows.append(row)
                cand_writer.writerow(row)
                cand_fh.flush()

            if not rows:
                logger.warning("No usable candidates for L=%d slot=%d", length, slot)
                continue

            tms = [r["esmfold_tm_score"] for r in rows if r["esmfold_tm_score"] == r["esmfold_tm_score"]]
            rmsds = [r["esmfold_rmsd"] for r in rows if r["esmfold_rmsd"] == r["esmfold_rmsd"]]
            plddts = [r["esmfold_plddt"] for r in rows if r["esmfold_plddt"] == r["esmfold_plddt"]]
            picks = _do_picks(rows)

            def _mean(xs):
                return sum(xs) / len(xs) if xs else float("nan")

            summary = {
                "length": length,
                "slot": slot,
                "n_candidates": len(rows),
                "esmfold_tm_mean": _mean(tms),
                "esmfold_tm_min": min(tms) if tms else float("nan"),
                "esmfold_tm_max": max(tms) if tms else float("nan"),
                "esmfold_rmsd_mean": _mean(rmsds),
                "esmfold_plddt_mean": _mean(plddts),
                **picks,
            }
            summ_writer.writerow(summary)
            summ_fh.flush()
            n_slots_done += 1

            if (n_slots_done % args.log_every) == 0:
                elapsed = time.time() - t_start
                logger.info(
                    "[%3d/%d] L=%3d slot=%2d  tm[mean=%.3f max=%.3f]  pick: rand=%.3f seq=%.3f struc=%.3f joint=%.3f jt=%.3f orac=%.3f  (%.1fs/slot)",
                    n_slots_done,
                    len(lengths) * args.slots,
                    length,
                    slot,
                    summary["esmfold_tm_mean"],
                    summary["esmfold_tm_max"],
                    summary["random_pick_esmfold_tm_score"],
                    summary["seq_pll_pick_esmfold_tm_score"],
                    summary["struc_pll_pick_esmfold_tm_score"],
                    summary["joint_pll_pick_esmfold_tm_score"],
                    summary["joint_true_pll_pick_esmfold_tm_score"],
                    summary["oracle_tm_pick_esmfold_tm_score"],
                    elapsed / max(1, n_slots_done),
                )

    cand_fh.close()
    summ_fh.close()
    logger.info(
        "Done. %d slots processed. Outputs:\n  %s\n  %s",
        n_slots_done,
        cand_path,
        summ_path,
    )


if __name__ == "__main__":
    main()
