"""Self-reflection for inverse folding.

Mirrors the existing unconditional self-reflection loop, but for IF: per CAMEO
target we repeatedly generate a sequence given the GT backbone and accept the
first one whose model-internal forward-fold lands within `--min-tm-score` TM of
the target backbone (default 0.833, same as the production unconditional SR
gate). After acceptance, we run ESMFold on the accepted sequence to score
designability (sc-TM / RMSD / pLDDT vs GT) for direct comparison with the
single-shot baseline and the best-of-N PLL pickers.

Key difference from `inverse_fold_bestofN_pll.py`:
- best-of-N: generate N candidates, score all with PLL, pick argmin
- self-reflection: generate one at a time, accept first that passes a *cheap
  internal forward-fold consistency check*; only ESMFold the accepted design

The internal check is the model's own forward-fold (no external folding model
needed). ESMFold is only used to compute the final designability metrics, not
as part of the gate.

Outputs:
  - if_sr_attempts_<ts>.csv  : one row per (target, attempt) with internal TM,
                                accepted flag, ESMFold metrics if accepted
  - if_sr_summary_<ts>.csv   : one row per target with attempts_used, accepted,
                                accepted ESMFold metrics
  - if_sr_report.md          : markdown summary

Usage:
    uv run python scripts/inverse_fold_self_reflection.py \\
        --inputs '/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt' \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_if_self_reflection \\
        --max-attempts 30 --min-tm-score 0.833
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("if_self_reflection")


_ATTEMPT_COLS = [
    "target",
    "length",
    "attempt_idx",
    "seed",
    "internal_tm",
    "internal_rmsd",
    "accepted",
    "aar",
    "esmfold_plddt",
    "esmfold_tm_score",
    "esmfold_rmsd",
    "esmfold_pae",
    "gen_seconds",
    "fwd_seconds",
    "esmfold_seconds",
]

_SUMMARY_COLS = [
    "target",
    "length",
    "max_attempts",
    "min_tm_score",
    "attempts_used",
    "accepted",
    "accepted_internal_tm",
    "accepted_aar",
    "accepted_esmfold_tm",
    "accepted_esmfold_rmsd",
    "accepted_esmfold_plddt",
    "accepted_esmfold_pae",
    "best_internal_tm",       # if no accept, the best across all attempts
    "best_attempt_idx",
    "fallback_used",          # 1 if no attempt passed and we returned best-by-internal-TM
    "wall_seconds",
]


def _resolve_inputs(spec: str) -> list[Path]:
    if "*" in spec or "?" in spec:
        return [Path(p) for p in sorted(glob.glob(spec))]
    p = Path(spec)
    if p.is_file():
        return [p]
    if p.is_dir():
        return [Path(x) for x in sorted(glob.glob(str(p / "*.pt")))]
    raise FileNotFoundError(f"Input spec does not match anything: {spec}")


def _load_target(path: Path, device: torch.device, max_length: int):
    from lobster.transforms._structure_transforms import StructureBackboneTransform

    transform = StructureBackboneTransform(max_length=max_length)
    raw = torch.load(path, map_location="cpu")
    raw = transform(raw)
    L = int(raw["coords_res"].shape[0])
    if L < 30 or L > max_length:
        return None
    pct20 = (raw["sequence"] == 20).float().mean().item()
    if pct20 > 0.1:
        return None

    coords_res = raw["coords_res"].to(device)
    seq_int = raw["sequence"]
    if seq_int.dim() > 1:
        seq_int = seq_int.squeeze()
    mask = raw["mask"].to(device).float()
    indices = raw["indices"].to(device).long()
    nan_idx = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
    mask[nan_idx] = 0.0
    coords_res[nan_idx] = 0.0
    return {
        "coords_res": coords_res,
        "sequence_int": seq_int,
        "mask": mask,
        "indices": indices,
        "L": L,
    }


@torch.no_grad()
def _generate_inverse_candidate(model, target, gen_kwargs):
    """Generate one IF candidate. Returns (seq_std[1,L], seq_lobster[1,L], gen_logits)."""
    from lobster.model.latent_generator.utils.residue_constants import (
        convert_lobster_aa_tokenization_to_standard_aa,
    )

    L = target["L"]
    coords_res = target["coords_res"].unsqueeze(0)
    mask = target["mask"][:L].unsqueeze(0)
    indices = target["indices"][:L].unsqueeze(0)

    gen_sample = model.generate_sample(
        length=L,
        num_samples=1,
        inverse_folding=True,
        input_structure_coords=coords_res,
        input_mask=mask,
        input_indices=indices,
        **gen_kwargs,
    )

    seq_lobster = gen_sample["sequence_logits"].argmax(dim=-1)
    if gen_sample["sequence_logits"].shape[-1] == 33:
        seq_std = convert_lobster_aa_tokenization_to_standard_aa(
            gen_sample["sequence_logits"], device=coords_res.device
        )
    else:
        seq_std = seq_lobster.clone()
        seq_std[seq_std > 21] = 20

    return seq_std, seq_lobster


@torch.no_grad()
def _internal_forward_fold(model, seq_lobster, target, fwd_kwargs):
    """Forward-fold the generated sequence with the model and compute TM/RMSD vs GT backbone."""
    from lobster.metrics import align_and_compute_rmsd
    from tmtools import tm_align

    L = target["L"]
    mask = target["mask"][:L].unsqueeze(0)
    indices = target["indices"][:L].unsqueeze(0)

    fwd_sample = model.generate_sample(
        length=L,
        num_samples=1,
        forward_folding=True,
        input_sequence_tokens=seq_lobster,
        input_mask=mask,
        input_indices=indices,
        **fwd_kwargs,
    )
    decoded = model.decode_structure(fwd_sample, mask)
    fwd_xyz = decoded["vit_decoder"][0]  # [L, 3, 3]

    valid = target["mask"][:L].bool()
    fwd_valid = fwd_xyz[valid]
    gt_valid = target["coords_res"][:L][valid]
    if fwd_valid.shape[0] == 0:
        return float("nan"), float("nan")

    # Use a placeholder sequence for TM-align (sequence string only used for length)
    seq_str = "A" * fwd_valid.shape[0]
    try:
        tm_out = tm_align(
            fwd_valid[:, 1, :].cpu().numpy(),
            gt_valid[:, 1, :].detach().cpu().numpy(),
            seq_str,
            seq_str,
        )
        tm = float(tm_out.tm_norm_chain1)
    except Exception:
        tm = float("nan")

    try:
        rmsd = float(
            align_and_compute_rmsd(
                coords1=fwd_valid,
                coords2=gt_valid,
                mask=None,
                return_aligned=False,
                device=fwd_valid.device,
            )
        )
    except Exception:
        rmsd = float("nan")

    return tm, rmsd


@torch.no_grad()
def _esmfold_score(plm_fold, seq_std, target, device):
    """ESMFold the sequence and compute (plddt, tm vs GT, rmsd vs GT, pae)."""
    from lobster.metrics import align_and_compute_rmsd
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    from tmtools import tm_align

    L = target["L"]
    valid = target["mask"][:L].bool()
    cand = seq_std[0, :L][valid].cpu().tolist()
    candidate_str = "".join(restype_order_with_x_inv.get(int(t), "X") for t in cand)
    candidate_str_clean = candidate_str.replace("X", "A")

    if not candidate_str_clean:
        return float("nan"), float("nan"), float("nan"), float("nan")

    tokenized = plm_fold.tokenizer.encode_plus(
        candidate_str_clean,
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    outputs = plm_fold.model(tokenized)
    pred_coords = outputs["positions"][-1, 0, :, :3, :]
    plddt = float(outputs["plddt"].mean().item())
    pae = float(outputs["predicted_aligned_error"].mean().item())

    orig = target["coords_res"][:L][valid]
    if pred_coords.shape[0] != orig.shape[0]:
        return plddt, float("nan"), float("nan"), pae

    try:
        tm_out = tm_align(
            pred_coords[:, 1, :].cpu().numpy(),
            orig[:, 1, :].detach().cpu().numpy(),
            candidate_str,
            candidate_str,
        )
        tm = float(tm_out.tm_norm_chain1)
    except Exception:
        tm = float("nan")

    try:
        rmsd = float(
            align_and_compute_rmsd(
                coords1=pred_coords,
                coords2=orig,
                mask=None,
                return_aligned=False,
                device=pred_coords.device,
            )
        )
    except Exception:
        rmsd = float("nan")

    return plddt, tm, rmsd, pae


def _compute_aar(seq_std, target):
    L = target["L"]
    valid = target["mask"][:L].bool()
    cand = seq_std[0, :L][valid]
    gt = target["sequence_int"][:L].to(cand.device)[valid]
    if cand.numel() == 0:
        return float("nan")
    return float((cand == gt).float().mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--inputs",
        default="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt",
    )
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-attempts", type=int, default=30,
                   help="Stop after this many retries even if no attempt passes")
    p.add_argument("--min-tm-score", type=float, default=0.833,
                   help="Internal forward-fold TM threshold to accept (default mirrors unconditional SR)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=20260503)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # IF generation hyperparameters (mirror generate_inverse_folding_denovo_cameo.yaml)
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--temperature-seq", type=float, default=0.14995423473740457)
    p.add_argument("--temperature-struc", type=float, default=0.4178150796307539)
    p.add_argument("--stochasticity-seq", type=int, default=10)
    p.add_argument("--stochasticity-struc", type=int, default=50)
    # Internal FF hyperparameters (mirror generate_forward_folding_denovo_cameo.yaml)
    p.add_argument("--fwd-nsteps", type=int, default=200)
    p.add_argument("--fwd-temperature-seq", type=float, default=0.36126569346108364)
    p.add_argument("--fwd-temperature-struc", type=float, default=0.21962552487521867)
    p.add_argument("--fwd-stochasticity-seq", type=int, default=10)
    p.add_argument("--fwd-stochasticity-struc", type=int, default=20)
    p.add_argument("--no-esmfold", action="store_true")
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    targets = _resolve_inputs(args.inputs)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if not targets:
        raise FileNotFoundError(f"No targets found at {args.inputs}")
    logger.info("Found %d target structure files", len(targets))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    att_path = args.output_dir / f"if_sr_attempts_{ts}.csv"
    summ_path = args.output_dir / f"if_sr_summary_{ts}.csv"
    att_fh = att_path.open("w", newline="")
    summ_fh = summ_path.open("w", newline="")
    att_writer = csv.DictWriter(att_fh, fieldnames=_ATTEMPT_COLS, extrasaction="ignore")
    summ_writer = csv.DictWriter(summ_fh, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
    att_writer.writeheader()
    summ_writer.writeheader()
    att_fh.flush()
    summ_fh.flush()

    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = UMESequenceStructureEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)
    model.interpolant_seq.device = device
    model.interpolant_struc.device = device
    logger.info("Model loaded in %.1fs", time.time() - t0)

    plm_fold = None
    if not args.no_esmfold:
        from lobster.model import LobsterPLMFold

        logger.info("Loading ESMFold...")
        t0 = time.time()
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=args.max_length)
        plm_fold.to(device)
        plm_fold.model.eval()
        logger.info("ESMFold loaded in %.1fs", time.time() - t0)

    gen_kwargs = dict(
        nsteps=args.nsteps,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        asynchronous_sampling=False,
    )
    fwd_kwargs = dict(
        nsteps=args.fwd_nsteps,
        temperature_seq=args.fwd_temperature_seq,
        temperature_struc=args.fwd_temperature_struc,
        stochasticity_seq=args.fwd_stochasticity_seq,
        stochasticity_struc=args.fwd_stochasticity_struc,
        asynchronous_sampling=False,
    )

    n_done = 0
    n_skipped = 0
    n_pass = 0
    t_start = time.time()

    for ti, target_path in enumerate(targets):
        try:
            target = _load_target(target_path, device, args.max_length)
        except Exception as e:
            logger.warning("Failed to load %s: %s", target_path.name, e)
            n_skipped += 1
            continue
        if target is None:
            n_skipped += 1
            continue

        L = target["L"]
        target_name = target_path.stem
        wall_t0 = time.time()

        accepted_attempt = None
        best_internal_tm = -1.0
        best_attempt_data = None
        attempts_data = []

        for attempt in range(args.max_attempts):
            seed = args.seed_base + ti * 1_000_003 + attempt
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                seq_std, seq_lobster = _generate_inverse_candidate(model, target, gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM gen %s attempt %d", target_name, attempt)
                continue
            except Exception as e:
                logger.warning("Gen failed %s attempt %d: %s", target_name, attempt, e)
                continue
            gen_seconds = time.time() - tg0

            tf0 = time.time()
            try:
                int_tm, int_rmsd = _internal_forward_fold(model, seq_lobster, target, fwd_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                int_tm = int_rmsd = float("nan")
            except Exception as e:
                logger.warning("FwdFold failed %s attempt %d: %s", target_name, attempt, e)
                int_tm = int_rmsd = float("nan")
            fwd_seconds = time.time() - tf0

            passed = (int_tm == int_tm) and (int_tm >= args.min_tm_score)
            attempt_row = {
                "target": target_name,
                "length": L,
                "attempt_idx": attempt,
                "seed": seed,
                "internal_tm": int_tm,
                "internal_rmsd": int_rmsd,
                "accepted": int(passed) if attempt == 0 or passed else 0,
                "gen_seconds": round(gen_seconds, 3),
                "fwd_seconds": round(fwd_seconds, 3),
                "esmfold_seconds": 0.0,
            }

            # Track best-by-internal-TM regardless
            if int_tm == int_tm and int_tm > best_internal_tm:
                best_internal_tm = int_tm
                best_attempt_data = (attempt, seq_std, seq_lobster, int_tm, int_rmsd, attempt_row.copy())

            if passed:
                # ESMFold-score the accepted candidate
                aar = _compute_aar(seq_std, target)
                te0 = time.time()
                if plm_fold is not None:
                    try:
                        plddt, tm, rmsd, pae = _esmfold_score(plm_fold, seq_std, target, device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        plddt = tm = rmsd = pae = float("nan")
                    except Exception as e:
                        logger.warning("ESMFold failed %s attempt %d: %s", target_name, attempt, e)
                        plddt = tm = rmsd = pae = float("nan")
                else:
                    plddt = tm = rmsd = pae = float("nan")
                attempt_row.update(
                    {
                        "aar": aar,
                        "esmfold_plddt": plddt,
                        "esmfold_tm_score": tm,
                        "esmfold_rmsd": rmsd,
                        "esmfold_pae": pae,
                        "esmfold_seconds": round(time.time() - te0, 3),
                    }
                )
                attempts_data.append(attempt_row)
                att_writer.writerow(attempt_row)
                att_fh.flush()
                accepted_attempt = (attempt, seq_std, attempt_row)
                break

            attempts_data.append(attempt_row)
            att_writer.writerow(attempt_row)
            att_fh.flush()

        # No attempt passed -> fall back to best-by-internal-TM
        fallback_used = 0
        if accepted_attempt is None:
            fallback_used = 1
            if best_attempt_data is None:
                # Every attempt failed catastrophically
                logger.warning("All attempts failed catastrophically for %s; skipping", target_name)
                summary = {
                    "target": target_name, "length": L, "max_attempts": args.max_attempts,
                    "min_tm_score": args.min_tm_score, "attempts_used": args.max_attempts,
                    "accepted": 0, "best_internal_tm": float("nan"), "best_attempt_idx": -1,
                    "fallback_used": 1, "wall_seconds": round(time.time() - wall_t0, 1),
                }
                summ_writer.writerow(summary)
                summ_fh.flush()
                n_skipped += 1
                continue

            attempt_idx, seq_std, _seq_lob, int_tm, int_rmsd, _row = best_attempt_data
            aar = _compute_aar(seq_std, target)
            te0 = time.time()
            if plm_fold is not None:
                try:
                    plddt, tm, rmsd, pae = _esmfold_score(plm_fold, seq_std, target, device)
                except Exception as e:
                    logger.warning("ESMFold (fallback) failed %s: %s", target_name, e)
                    plddt = tm = rmsd = pae = float("nan")
            else:
                plddt = tm = rmsd = pae = float("nan")
            accepted_row = {
                "target": target_name,
                "length": L,
                "attempt_idx": attempt_idx,
                "seed": args.seed_base + ti * 1_000_003 + attempt_idx,
                "internal_tm": int_tm,
                "internal_rmsd": int_rmsd,
                "accepted": 0,  # Did not actually pass
                "aar": aar,
                "esmfold_plddt": plddt,
                "esmfold_tm_score": tm,
                "esmfold_rmsd": rmsd,
                "esmfold_pae": pae,
                "esmfold_seconds": round(time.time() - te0, 3),
                "gen_seconds": 0.0,
                "fwd_seconds": 0.0,
            }
            accepted_attempt = (attempt_idx, seq_std, accepted_row)
            # Don't double-write; the original attempt_row was already in the CSV

        attempt_idx, seq_std, accepted_row = accepted_attempt
        attempts_used = len(attempts_data)
        if not fallback_used:
            n_pass += 1
        summary = {
            "target": target_name,
            "length": L,
            "max_attempts": args.max_attempts,
            "min_tm_score": args.min_tm_score,
            "attempts_used": attempts_used,
            "accepted": int(not fallback_used),
            "accepted_internal_tm": accepted_row["internal_tm"],
            "accepted_aar": accepted_row.get("aar", float("nan")),
            "accepted_esmfold_tm": accepted_row.get("esmfold_tm_score", float("nan")),
            "accepted_esmfold_rmsd": accepted_row.get("esmfold_rmsd", float("nan")),
            "accepted_esmfold_plddt": accepted_row.get("esmfold_plddt", float("nan")),
            "accepted_esmfold_pae": accepted_row.get("esmfold_pae", float("nan")),
            "best_internal_tm": best_internal_tm,
            "best_attempt_idx": attempt_idx,
            "fallback_used": fallback_used,
            "wall_seconds": round(time.time() - wall_t0, 1),
        }
        summ_writer.writerow(summary)
        summ_fh.flush()
        n_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%3d/%d] %-20s L=%3d  attempts=%2d %s  best_int_tm=%.3f  esmTM=%.3f  RMSD=%.2f  AAR=%.2f%%  pass_rate=%.0f%% (%.1fs/target)",
                ti + 1,
                len(targets),
                target_name,
                L,
                attempts_used,
                "ACCEPT" if not fallback_used else "FALLBK",
                best_internal_tm if best_internal_tm > -1 else float("nan"),
                summary["accepted_esmfold_tm"] if summary["accepted_esmfold_tm"] == summary["accepted_esmfold_tm"] else float("nan"),
                summary["accepted_esmfold_rmsd"] if summary["accepted_esmfold_rmsd"] == summary["accepted_esmfold_rmsd"] else float("nan"),
                100.0 * (summary["accepted_aar"] if summary["accepted_aar"] == summary["accepted_aar"] else 0.0),
                100.0 * n_pass / max(1, n_done),
                elapsed / max(1, n_done),
            )

    att_fh.close()
    summ_fh.close()

    logger.info(
        "Done. %d targets processed (%d skipped). %d / %d passed SR gate (%.1f%%). Outputs:\n  %s\n  %s",
        n_done, n_skipped, n_pass, n_done, 100.0 * n_pass / max(1, n_done), att_path, summ_path,
    )


if __name__ == "__main__":
    main()
