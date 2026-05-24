"""Forward-folding best-of-N: PLL-driven candidate selection on CAMEO.

For each CAMEO target (a `.pt` file), generates N independent forward-fold candidates,
scores each with the in-model PLL (sequence / structure / joint heads), and writes:

  - `bestofN_ff_candidates_<ts>.csv`  one row per (target, candidate)
                                      with TM, RMSD, and all PLL variants.
  - `bestofN_ff_summary_<ts>.csv`     one row per target with the picks each ranker
                                      makes (random/seq_pll/struc_pll/joint_pll/oracle)
                                      and the resulting TM-scores.

Reuses `_generate_forward_folding`'s candidate-generation contract via direct calls
to `model.generate_sample(forward_folding=True, ...)`, and the `score_one_sample`
PLL function from `scripts/score_gen_ume_pll.py`. No edits to `generate.py`.

Usage:
    uv run python scripts/forward_fold_bestofN_pll.py \\
        --inputs '/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt' \\
        --ckpt /cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_cameo_bestofN_pll \\
        --N 10
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

# Import score_one_sample from the existing scorer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gen_ume_pll import score_one_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forward_fold_bestofN_pll")


_CANDIDATE_COLS = [
    "target",
    "length",
    "candidate_idx",
    "seed",
    "tm_score",
    "rmsd",
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
]


_PICKERS = [
    ("random_pick", None),  # always candidate 0 = single-shot baseline
    ("seq_pll_pick", "seq_score_unif"),
    ("struc_pll_pick", "struc_score_unif"),
    ("joint_pll_pick", "joint_score_unif"),
    ("joint_true_pll_pick", "joint_true_score_unif"),
    ("oracle_pick", "tm_score"),  # argmax instead of argmin
]


_SUMMARY_COLS = [
    "target",
    "length",
    "n_candidates",
    "tm_min",
    "tm_mean",
    "tm_median",
    "tm_max",
    "tm_std",
    "random_pick_idx",
    "random_pick_tm",
    "seq_pll_pick_idx",
    "seq_pll_pick_tm",
    "struc_pll_pick_idx",
    "struc_pll_pick_tm",
    "joint_pll_pick_idx",
    "joint_pll_pick_tm",
    "joint_true_pll_pick_idx",
    "joint_true_pll_pick_tm",
    "oracle_pick_idx",
    "oracle_pick_tm",
]


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
    return {
        "coords_res": coords_res,
        "sequence_int": seq_int,
        "mask": mask,
        "indices": indices,
        "L": L,
    }


def _tokenize_input_sequence(aa_transform, seq_int_tensor: torch.Tensor) -> torch.Tensor:
    out = aa_transform({"sequence": seq_int_tensor})
    return out["sequence"]


@torch.no_grad()
def _generate_one_candidate(model, target, padded_seq_tokens, gen_kwargs):
    """One candidate forward-fold for a single target. Returns (gen_sample, recon_xyz, struc_argmax)."""
    L = target["L"]
    mask = target["mask"].unsqueeze(0)  # [1, L]
    indices = target["indices"].unsqueeze(0)  # [1, L]

    gen_sample = model.generate_sample(
        length=L,
        num_samples=1,
        forward_folding=True,
        input_sequence_tokens=padded_seq_tokens,  # [1, L]
        input_mask=mask,
        input_indices=indices,
        **gen_kwargs,
    )
    decoded = model.decode_structure(gen_sample, mask)
    x_recon_xyz = decoded["vit_decoder"]  # [1, L, 3, 3]
    struc_argmax = gen_sample["structure_logits"].argmax(dim=-1)  # [1, L]
    return gen_sample, x_recon_xyz, struc_argmax


def _compute_tm_rmsd(target, x_recon_xyz):
    """TM-score (via tm_align) and CA-RMSD (Kabsch) between generated and GT structures."""
    from lobster.metrics import align_and_compute_rmsd
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    from tmtools import tm_align

    valid = target["mask"].bool()
    seq_i = target["sequence_int"][valid.cpu().numpy()].tolist()
    sequence_str = "".join(restype_order_with_x_inv[int(j)] for j in seq_i)

    orig = target["coords_res"][valid]
    gen = x_recon_xyz[0, valid]

    tm = tm_align(
        gen[:, 1, :].cpu().numpy(),
        orig[:, 1, :].detach().cpu().numpy(),
        sequence_str,
        sequence_str,
    )
    rmsd = align_and_compute_rmsd(
        coords1=gen,
        coords2=orig,
        mask=None,
        return_aligned=False,
        device=gen.device,
    )
    return float(tm.tm_norm_chain1), float(rmsd)


def _build_pll_inputs(seq_tokens_padded, struc_argmax, target):
    L = target["L"]
    valid = target["mask"].bool()
    seq_clean = seq_tokens_padded[0, :L].unsqueeze(0)
    struc_clean = struc_argmax[0, :L].unsqueeze(0)
    mask_t = target["mask"][:L].unsqueeze(0)  # [1, L]
    residue_index = target["indices"][:L].unsqueeze(0)
    return seq_clean, struc_clean, mask_t, residue_index, valid


def _do_picks(rows, candidate_idx_offset=0):
    """Compute pick indices for each picker on the candidates of one target."""
    out = {}
    for picker_name, key in _PICKERS:
        if picker_name == "random_pick":
            idx = 0  # first candidate in this pool (shard-local for array chunks)
        elif picker_name == "oracle_pick":
            tms = [r["tm_score"] for r in rows]
            idx = int(max(range(len(rows)), key=lambda i: (tms[i] if tms[i] is not None else float("-inf"))))
        else:
            vals = [r[key] for r in rows]
            idx = int(min(range(len(rows)), key=lambda i: (vals[i] if vals[i] is not None else float("inf"))))
        out[picker_name + "_idx"] = idx
        out[picker_name + "_tm"] = float(rows[idx]["tm_score"])
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
    p.add_argument("--N", type=int, default=10, help="Candidates per target")
    p.add_argument("--K", type=int, default=32, help="PLL Monte-Carlo draws per modality")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-targets", type=int, default=None, help="(debug) limit number of targets")
    p.add_argument(
        "--target-id",
        type=str,
        default=None,
        help="If set, run only this target stem (e.g. 7dz2.C). Used by SLURM array workers.",
    )
    p.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="Added to candidate_idx and seed. Shards N across array tasks without collisions.",
    )
    p.add_argument("--seed-base", type=int, default=20260430)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Mirror generate_forward_folding_denovo_cameo.yaml defaults:
    p.add_argument("--nsteps", type=int, default=200)
    p.add_argument("--temperature-seq", type=float, default=0.3610371899835548)
    p.add_argument("--temperature-struc", type=float, default=0.2195534567490864)
    p.add_argument("--stochasticity-seq", type=int, default=1)
    p.add_argument("--stochasticity-struc", type=int, default=20)
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    all_targets = _resolve_inputs(args.inputs)
    targets = list(all_targets)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if args.target_id is not None:
        want = args.target_id
        matched = [p for p in targets if p.stem == want]
        if not matched:
            raise FileNotFoundError(
                f"--target-id '{want}' not found under {args.inputs}; "
                f"have {len(targets)} targets"
            )
        targets = matched
    if not targets:
        raise FileNotFoundError(f"No targets found at {args.inputs}")
    ti_global_of = {p.stem: all_targets.index(p) for p in targets}
    logger.info("Found %d target structure files (running %d)", len(all_targets), len(targets))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    cand_path = args.output_dir / f"bestofN_ff_candidates_{ts}.csv"
    summ_path = args.output_dir / f"bestofN_ff_summary_{ts}.csv"
    cand_fh = cand_path.open("w", newline="")
    summ_fh = summ_path.open("w", newline="")
    cand_writer = csv.DictWriter(cand_fh, fieldnames=_CANDIDATE_COLS, extrasaction="ignore")
    summ_writer = csv.DictWriter(summ_fh, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
    cand_writer.writeheader()
    summ_writer.writeheader()
    cand_fh.flush()
    summ_fh.flush()

    # Lazy heavy imports
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
        ti_global = ti_global_of[target_path.stem]
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

        seq_tokenized = _tokenize_input_sequence(aa_transform, target["sequence_int"])
        padded_seq_tokens = torch.zeros(1, L, device=device, dtype=torch.long)
        seq_len = min(int(seq_tokenized.shape[0]), L)
        padded_seq_tokens[0, :seq_len] = seq_tokenized[:seq_len].to(device)

        rows: list[dict] = []
        for ci in range(args.N):
            cand_idx = args.candidate_offset + ci
            seed = args.seed_base + ti_global * 1_000_003 + cand_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            tg0 = time.time()
            try:
                _gen_sample, x_recon_xyz, struc_argmax = _generate_one_candidate(
                    model, target, padded_seq_tokens, gen_kwargs
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM generating %s candidate %d; skipping candidate", target_name, ci)
                continue
            gen_seconds = time.time() - tg0

            try:
                tm, rmsd = _compute_tm_rmsd(target, x_recon_xyz)
            except Exception as e:
                logger.warning("TM/RMSD failed for %s candidate %d: %s", target_name, ci, e)
                continue

            seq_clean, struc_clean, mask_t, residue_index, _valid = _build_pll_inputs(
                padded_seq_tokens, struc_argmax, target
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
                logger.warning("OOM scoring %s candidate %d; recording without scores", target_name, ci)
                scores = {}
            score_seconds = time.time() - ts0

            row = {
                "target": target_name,
                "length": L,
                "candidate_idx": cand_idx,
                "seed": seed,
                "tm_score": tm,
                "rmsd": rmsd,
                "gen_seconds": round(gen_seconds, 3),
                "score_seconds": round(score_seconds, 3),
                **{k: v for k, v in scores.items() if k in _CANDIDATE_COLS},
            }
            rows.append(row)
            cand_writer.writerow(row)
            cand_fh.flush()

        if not rows:
            logger.warning("No usable candidates for %s; skipping summary", target_name)
            n_skipped += 1
            continue

        tms = [r["tm_score"] for r in rows]
        picks = _do_picks(rows, candidate_idx_offset=args.candidate_offset)
        summary = {
            "target": target_name,
            "length": L,
            "n_candidates": len(rows),
            "tm_min": min(tms),
            "tm_mean": sum(tms) / len(tms),
            "tm_median": sorted(tms)[len(tms) // 2],
            "tm_max": max(tms),
            "tm_std": (sum((x - sum(tms) / len(tms)) ** 2 for x in tms) / len(tms)) ** 0.5,
            **picks,
        }
        summ_writer.writerow(summary)
        summ_fh.flush()
        n_targets_done += 1

        if (ti + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%3d/%d] %-20s L=%3d  tm[mean=%.3f min=%.3f max=%.3f]  pick: rand=%.3f seq=%.3f struc=%.3f joint=%.3f orac=%.3f  (%.1fs/target)",
                ti + 1,
                len(targets),
                target_name,
                L,
                summary["tm_mean"],
                summary["tm_min"],
                summary["tm_max"],
                summary["random_pick_tm"],
                summary["seq_pll_pick_tm"],
                summary["struc_pll_pick_tm"],
                summary["joint_pll_pick_tm"],
                summary["joint_true_pll_pick_tm"],
                summary["oracle_pick_tm"],
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
