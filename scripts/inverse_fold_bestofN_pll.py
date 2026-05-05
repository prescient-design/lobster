"""Inverse-folding best-of-N: PLL-driven sequence selection on CAMEO.

For each CAMEO target (a `.pt` file), generates N independent candidate sequences
conditioned on the GT backbone, scores each with the in-model PLL, runs ESMFold on
each candidate sequence to get designability metrics (pLDDT / TM-score vs GT /
RMSD vs GT), and writes:

  - `bestofN_if_candidates_<ts>.csv`  one row per (target, candidate) with AAR,
                                     ESMFold (plddt / tm_score / rmsd / pae),
                                     and all PLL variants
                                     (`seq_score_unif`, `struc_score_unif`,
                                      `joint_score_unif`, `joint_true_score_unif`,
                                      `_arllh`, fixed-`t`).
  - `bestofN_if_summary_<ts>.csv`    one row per target with the picks each
                                     ranker makes
                                     (random / seq_pll / struc_pll / joint_pll /
                                      joint_true_pll / oracle_tm / oracle_aar)
                                     and the resulting AAR / TM / RMSD / pLDDT.

Reuses `score_one_sample` from `scripts/score_gen_ume_pll.py`. ESMFold is invoked
directly via `LobsterPLMFold` for single-chain protein evaluation. No edits to
`generate.py`.

Usage:
    uv run python scripts/inverse_fold_bestofN_pll.py \\
        --inputs '/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt' \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse \\
        --N 30
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gen_ume_pll import score_one_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inverse_fold_bestofN_pll")


_CANDIDATE_COLS = [
    "target",
    "length",
    "candidate_idx",
    "seed",
    "aar",
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


# (picker_name, key, mode) where mode ∈ {"argmin", "argmax", "fixed_zero"}.
_PICKERS = [
    ("random_pick", None, "fixed_zero"),
    ("seq_pll_pick", "seq_score_unif", "argmin"),
    ("struc_pll_pick", "struc_score_unif", "argmin"),
    ("joint_pll_pick", "joint_score_unif", "argmin"),
    ("joint_true_pll_pick", "joint_true_score_unif", "argmin"),
    ("oracle_tm_pick", "esmfold_tm_score", "argmax"),
    ("oracle_aar_pick", "aar", "argmax"),
]


def _summary_cols() -> list[str]:
    cols = [
        "target",
        "length",
        "n_candidates",
        "aar_mean",
        "aar_min",
        "aar_max",
        "esmfold_tm_mean",
        "esmfold_tm_min",
        "esmfold_tm_max",
        "esmfold_rmsd_mean",
        "esmfold_plddt_mean",
    ]
    for picker_name, _, _ in _PICKERS:
        cols.append(f"{picker_name}_idx")
        for metric in ("aar", "esmfold_tm_score", "esmfold_rmsd", "esmfold_plddt"):
            cols.append(f"{picker_name}_{metric}")
    return cols


_SUMMARY_COLS = _summary_cols()


def _resolve_inputs(spec: str) -> list[Path]:
    if "*" in spec or "?" in spec:
        paths = sorted(glob.glob(spec))
    else:
        p = Path(spec)
        if p.is_file():
            paths = [str(p)]
        elif p.is_dir():
            paths = sorted(glob.glob(str(p / "*.pt")))
        else:
            raise FileNotFoundError(f"Input spec does not match anything: {spec}")
    return [Path(p) for p in paths]


def _load_target(path: Path, device: torch.device, max_length: int):
    from lobster.transforms._structure_transforms import StructureBackboneTransform

    transform = StructureBackboneTransform(max_length=max_length)
    raw = torch.load(path, map_location="cpu")
    raw = transform(raw)
    L = int(raw["coords_res"].shape[0])
    if L < 30:
        return None
    if L > max_length:
        return None
    pct20 = (raw["sequence"] == 20).float().mean().item()
    if pct20 > 0.1:
        return None

    coords_res = raw["coords_res"].to(device)  # [L, 3, 3]
    seq_int = raw["sequence"]
    if seq_int.dim() > 1:
        seq_int = seq_int.squeeze()
    mask = raw["mask"].to(device).float()  # [L]
    indices = raw["indices"].to(device).long()  # [L]
    nan_idx = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
    mask[nan_idx] = 0.0
    coords_res[nan_idx] = 0.0
    chains = raw.get("chains")
    if chains is not None:
        chains = chains.to(device)
    return {
        "coords_res": coords_res,
        "sequence_int": seq_int,
        "mask": mask,
        "indices": indices,
        "chains": chains,
        "L": L,
    }


def _tokenize_input_sequence(aa_transform, seq_int_tensor: torch.Tensor) -> torch.Tensor:
    out = aa_transform({"sequence": seq_int_tensor})
    return out["sequence"]


@torch.no_grad()
def _encode_gt_structure_tokens(model, target, device: torch.device) -> torch.Tensor:
    """Encode GT backbone coords -> discrete structure tokens [1, L].

    Mirrors `score_gen_ume_pll_failed_attempts._encode_pdb_to_struc_tokens`:
    `model.encode_structure` returns a 3-tuple `(x_quant, x_quant_emb, mask_out)`
    where `x_quant_emb` is `[B, L, V_struc]` soft assignments. We argmax to get
    discrete token indices.
    """
    L = target["L"]
    coords_res = target["coords_res"].unsqueeze(0)  # [1, L, 3, 3]
    mask = target["mask"][:L].unsqueeze(0)  # [1, L]
    residue_index = target["indices"][:L].unsqueeze(0)
    coords_res = torch.nan_to_num(coords_res, nan=0.0)
    _x_quant, x_quant_emb, _mask_out = model.encode_structure(coords_res, mask, residue_index)
    if x_quant_emb.dim() != 3:
        raise RuntimeError(f"Unexpected encode_structure output shape: {x_quant_emb.shape}")
    struc_tokens = x_quant_emb.argmax(dim=-1)  # [1, L]
    return struc_tokens


@torch.no_grad()
def _generate_one_candidate(model, target, gen_kwargs):
    """One inverse-folding candidate. Returns the candidate sequence tensor [1, L]."""
    L = target["L"]
    coords_res = target["coords_res"].unsqueeze(0)  # [1, L, 3, 3]
    mask = target["mask"][:L].unsqueeze(0)  # [1, L]
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

    if gen_sample["sequence_logits"].shape[-1] == 33:
        from lobster.model.latent_generator.utils.residue_constants import (
            convert_lobster_aa_tokenization_to_standard_aa,
        )

        seq = convert_lobster_aa_tokenization_to_standard_aa(
            gen_sample["sequence_logits"], device=coords_res.device
        )
    else:
        seq = gen_sample["sequence_logits"].argmax(dim=-1)
        seq[seq > 21] = 20

    return seq  # [1, L] in standard 0..20 vocab (20 = X)


def _build_pll_inputs(seq_lobster_tokens, struc_clean_tokens, target):
    L = target["L"]
    seq_clean = seq_lobster_tokens[:, :L]
    struc_clean = struc_clean_tokens[:, :L]
    mask_t = target["mask"][:L].unsqueeze(0)
    residue_index = target["indices"][:L].unsqueeze(0)
    return seq_clean, struc_clean, mask_t, residue_index


def _compute_aar(candidate_std: torch.Tensor, gt_seq_int: torch.Tensor, mask: torch.Tensor) -> float:
    """% identity in standard 0..19 (20=X) vocab on masked positions.

    `candidate_std`: [1, L] standard AA indices.
    `gt_seq_int`: [L] standard AA indices.
    """
    valid = mask.bool()
    cand = candidate_std[0, : gt_seq_int.shape[0]][valid]
    gt = gt_seq_int.to(cand.device)[valid]
    if cand.numel() == 0:
        return float("nan")
    return float((cand == gt).float().mean().item())


@torch.no_grad()
def _esmfold_one_candidate(
    plm_fold,
    candidate_std: torch.Tensor,
    target,
    device: torch.device,
):
    """ESMFold the candidate sequence; return (plddt, tm, rmsd, pae) or NaNs."""
    from lobster.metrics import align_and_compute_rmsd
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    from tmtools import tm_align

    L = target["L"]
    valid = target["mask"][:L].bool()
    cand = candidate_std[0, :L][valid].cpu().tolist()
    gt = target["sequence_int"][:L].to(candidate_std.device)[valid].cpu().tolist()

    candidate_str = "".join(restype_order_with_x_inv.get(int(t), "X") for t in cand)
    gt_str = "".join(restype_order_with_x_inv.get(int(t), "X") for t in gt)
    candidate_str_clean = candidate_str.replace("X", "A")  # ESMFold cannot tokenise X

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

    pred_coords = outputs["positions"][-1, 0, :, :3, :]  # [L, 3, 3]
    plddt = float(outputs["plddt"].mean().item())
    pae = float(outputs["predicted_aligned_error"].mean().item())

    orig = target["coords_res"][:L][valid]
    if pred_coords.shape[0] != orig.shape[0]:
        return plddt, float("nan"), float("nan"), pae

    try:
        tm_out = tm_align(
            pred_coords[:, 1, :].cpu().numpy(),
            orig[:, 1, :].detach().cpu().numpy(),
            gt_str,
            gt_str,
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


def _do_picks(rows: list[dict]) -> dict:
    """Compute pick indices per picker; return both index and the recorded metrics."""
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
                if fv != fv:  # NaN
                    return float("inf") if mode_inner == "argmin" else float("-inf")
                return fv

            scored = [_safe(v) for v in vals]
            if mode == "argmin":
                idx = int(min(range(len(rows)), key=lambda i: scored[i]))
            else:
                idx = int(max(range(len(rows)), key=lambda i: scored[i]))
        out[f"{picker_name}_idx"] = idx
        for metric in ("aar", "esmfold_tm_score", "esmfold_rmsd", "esmfold_plddt"):
            v = rows[idx].get(metric)
            out[f"{picker_name}_{metric}"] = float(v) if v is not None else float("nan")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--inputs",
        default="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt",
        help="Glob, dir, or file path for input structures",
    )
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--N", type=int, default=30, help="Candidates per target")
    p.add_argument("--K", type=int, default=32, help="PLL Monte-Carlo draws per modality")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=20260503)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Mirror generate_inverse_folding_denovo_cameo.yaml defaults
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--temperature-seq", type=float, default=0.14995423473740457)
    p.add_argument("--temperature-struc", type=float, default=0.4178150796307539)
    p.add_argument("--stochasticity-seq", type=int, default=10)
    p.add_argument("--stochasticity-struc", type=int, default=50)
    p.add_argument("--no-esmfold", action="store_true", help="Skip ESMFold (debug only)")
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
    cand_path = args.output_dir / f"bestofN_if_candidates_{ts}.csv"
    summ_path = args.output_dir / f"bestofN_if_summary_{ts}.csv"
    cand_fh = cand_path.open("w", newline="")
    summ_fh = summ_path.open("w", newline="")
    cand_writer = csv.DictWriter(cand_fh, fieldnames=_CANDIDATE_COLS, extrasaction="ignore")
    summ_writer = csv.DictWriter(summ_fh, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
    cand_writer.writeheader()
    summ_writer.writeheader()
    cand_fh.flush()
    summ_fh.flush()

    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule
    from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

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

    aa_transform = AminoAcidTokenizerTransform(max_length=args.max_length)

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

    n_targets_done = 0
    n_skipped = 0
    t_start = time.time()

    for ti, target_path in enumerate(targets):
        try:
            target = _load_target(target_path, device, args.max_length)
        except Exception as e:
            logger.warning("Failed to load %s: %s", target_path.name, e)
            n_skipped += 1
            continue
        if target is None:
            logger.info("Skipping %s (filter rules)", target_path.name)
            n_skipped += 1
            continue

        L = target["L"]
        target_name = target_path.stem

        try:
            struc_clean_tokens = _encode_gt_structure_tokens(model, target, device)  # [1, L]
        except Exception as e:
            logger.warning("Failed to encode GT structure tokens for %s: %s", target_name, e)
            n_skipped += 1
            continue

        rows: list[dict] = []
        for ci in range(args.N):
            seed = args.seed_base + ti * 1_000_003 + ci
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                candidate_std = _generate_one_candidate(model, target, gen_kwargs)  # [1, L] in standard vocab
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM generating %s candidate %d; skipping", target_name, ci)
                continue
            except Exception as e:
                logger.warning("Generation failed for %s candidate %d: %s", target_name, ci, e)
                continue
            gen_seconds = time.time() - tg0

            aar = _compute_aar(candidate_std, target["sequence_int"], target["mask"])

            te0 = time.time()
            if plm_fold is not None:
                try:
                    plddt, tm, rmsd, pae = _esmfold_one_candidate(plm_fold, candidate_std, target, device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning("OOM ESMFold %s candidate %d", target_name, ci)
                    plddt = tm = rmsd = pae = float("nan")
                except Exception as e:
                    logger.warning("ESMFold failed %s candidate %d: %s", target_name, ci, e)
                    plddt = tm = rmsd = pae = float("nan")
            else:
                plddt = tm = rmsd = pae = float("nan")
            esmfold_seconds = time.time() - te0

            seq_lobster = aa_transform({"sequence": candidate_std[0].cpu()})["sequence"].to(device)
            seq_lobster_tokens = seq_lobster.unsqueeze(0)

            seq_clean, struc_clean, mask_t, residue_index = _build_pll_inputs(
                seq_lobster_tokens, struc_clean_tokens, target
            )

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
                logger.warning("OOM scoring %s candidate %d", target_name, ci)
                scores = {}
            except Exception as e:
                logger.warning("PLL scoring failed %s candidate %d: %s", target_name, ci, e)
                scores = {}
            score_seconds = time.time() - ts0

            row = {
                "target": target_name,
                "length": L,
                "candidate_idx": ci,
                "seed": seed,
                "aar": aar,
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
            logger.warning("No usable candidates for %s; skipping summary", target_name)
            n_skipped += 1
            continue

        aars = [r["aar"] for r in rows]
        tms = [r["esmfold_tm_score"] for r in rows if r["esmfold_tm_score"] == r["esmfold_tm_score"]]
        rmsds = [r["esmfold_rmsd"] for r in rows if r["esmfold_rmsd"] == r["esmfold_rmsd"]]
        plddts = [r["esmfold_plddt"] for r in rows if r["esmfold_plddt"] == r["esmfold_plddt"]]
        picks = _do_picks(rows)

        def _mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        summary = {
            "target": target_name,
            "length": L,
            "n_candidates": len(rows),
            "aar_mean": _mean(aars),
            "aar_min": min(aars) if aars else float("nan"),
            "aar_max": max(aars) if aars else float("nan"),
            "esmfold_tm_mean": _mean(tms),
            "esmfold_tm_min": min(tms) if tms else float("nan"),
            "esmfold_tm_max": max(tms) if tms else float("nan"),
            "esmfold_rmsd_mean": _mean(rmsds),
            "esmfold_plddt_mean": _mean(plddts),
            **picks,
        }
        summ_writer.writerow(summary)
        summ_fh.flush()
        n_targets_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%3d/%d] %-20s L=%3d  aar[mean=%.3f] tm[mean=%.3f max=%.3f]  "
                "pick: rand_tm=%.3f seq_tm=%.3f struc_tm=%.3f joint_tm=%.3f jt_tm=%.3f orac_tm=%.3f  "
                "(%.1fs/target)",
                ti + 1,
                len(targets),
                target_name,
                L,
                summary["aar_mean"],
                summary["esmfold_tm_mean"],
                summary["esmfold_tm_max"],
                summary["random_pick_esmfold_tm_score"],
                summary["seq_pll_pick_esmfold_tm_score"],
                summary["struc_pll_pick_esmfold_tm_score"],
                summary["joint_pll_pick_esmfold_tm_score"],
                summary["joint_true_pll_pick_esmfold_tm_score"],
                summary["oracle_tm_pick_esmfold_tm_score"],
                elapsed / max(1, n_targets_done),
            )

    cand_fh.close()
    summ_fh.close()
    logger.info(
        "Done. %d targets processed (%d skipped). Outputs:\n  %s\n  %s",
        n_targets_done,
        n_skipped,
        cand_path,
        summ_path,
    )


if __name__ == "__main__":
    main()
