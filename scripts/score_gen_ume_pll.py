"""Gen-UME pseudo-likelihood scoring (ProteinMPNN analog).

Scores per-sample (sequence, structure) pairs from an existing eval directory using
the model's own absorbing-prior discrete-flow training objective as a Monte-Carlo
estimator of the any-order autoregressive log-likelihood.

For each sample and each modality (sequence and structure), we compute:

  P1 (random-t Monte-Carlo, K draws):
      score_unif  = mean over K draws of   avg_per_masked_position_CE(t_k)
      score_arllh = mean over K draws of   sum_CE_masked(t_k) / ((1 - t_k) * L)
                    (equivalent in expectation; different finite-K variance)

  P2 (fixed-t draws at t in {0.25, 0.5, 0.75}):
      score_t<x> = avg_per_masked_position_CE(t = x)

When scoring the sequence head, the structure head is held clean (timesteps_struc = 1).
When scoring the structure head, the sequence head is held clean (timesteps_seq = 1).

Inputs are read from the eval directory's `sequences_*.csv` (sequence string in
column `sequence` or `original_sequence` for forward folding; structure tokens
as comma-separated ints in column `latent_generator_tokens`).

Output: pll_scores_<timestamp>.csv next to the input CSV.

Usage (typical):
    uv run python scripts/score_gen_ume_pll.py \\
        --eval-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_cameo_forward_folding \\
        --ckpt /cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt \\
        --task forward_folding \\
        --K 16
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_gen_ume_pll")


_TASK_TO_PREFIX = {
    "forward_folding": "sequences_forward_folding_",
    "inverse_folding": "sequences_inverse_folding_",
    "unconditional": "sequences_unconditional_",
}

_FIXED_T = (0.25, 0.5, 0.75)


def _detect_task(eval_dir: Path) -> str:
    for task, prefix in _TASK_TO_PREFIX.items():
        if list(eval_dir.glob(f"{prefix}*.csv")):
            return task
    raise FileNotFoundError(
        f"No sequences_(forward_folding|inverse_folding|unconditional)_*.csv found under {eval_dir}"
    )


def _find_sequences_csv(eval_dir: Path, task: str) -> Path:
    matches = sorted(eval_dir.glob(f"{_TASK_TO_PREFIX[task]}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {_TASK_TO_PREFIX[task]}*.csv under {eval_dir}")
    if len(matches) > 1:
        logger.warning("Multiple sequences CSVs found; using newest: %s", matches[-1].name)
    return matches[-1]


def _seq_source_column(task: str) -> str:
    """Which column contains the sequence we want to score.

    Forward folding: model was given the GT sequence as input (`original_sequence`)
                     and produced a structure; we score the (GT seq, generated struc) pair.
    Inverse folding: model was given the GT structure and produced a sequence; we score
                     (generated seq, GT struc).
    Unconditional:   both seq and struc are generated; we score (generated seq, generated struc).
    """
    return "original_sequence" if task == "forward_folding" else "sequence"


def _load_rows(csv_path: Path, seq_col: str, max_samples: int | None):
    rows = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            seq = (r.get(seq_col) or "").strip()
            tokens_str = (r.get("latent_generator_tokens") or "").strip()
            if not seq or not tokens_str:
                continue
            try:
                struc_tokens = [int(t) for t in tokens_str.split(",") if t != ""]
            except ValueError:
                continue
            if len(struc_tokens) != len(seq):
                logger.debug(
                    "Length mismatch run_id=%s seq_len=%d struc_len=%d; skipping",
                    r.get("run_id", "?"),
                    len(seq),
                    len(struc_tokens),
                )
                continue
            rows.append(
                {
                    "run_id": r.get("run_id", ""),
                    "iteration": r.get("iteration", ""),
                    "sample_idx": r.get("sample_idx", ""),
                    "length": len(struc_tokens),
                    "input_structure": r.get("input_structure", ""),
                    "trial_selected": r.get("trial_selected", ""),
                    "sequence_type": r.get("sequence_type", ""),
                    "_seq_str": seq,
                    "_struc_tokens": struc_tokens,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _tokenize_sequence(aa_transform, seq_str: str) -> torch.Tensor:
    """Tokenize an AA string using the exact same AminoAcidTokenizerTransform that generate.py
    feeds to the model (PDB int -> letter via restype_order_with_x_inv -> AminoAcidTokenizerFast
    with cls/eos -> strip cls/eos). We invert the first step (string -> PDB int tensor) so the
    transform's __call__ contract is satisfied; the resulting `sequence` tensor is bit-identical
    to what generate.py builds for `padded_sequences`.
    """
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x

    pdb_ints = torch.tensor([restype_order_with_x.get(c, 20) for c in seq_str], dtype=torch.long)
    out = aa_transform({"sequence": pdb_ints})
    return out["sequence"]


def _build_inputs(
    seq_str: str,
    struc_tokens_list: list[int],
    aa_transform,
    device: torch.device,
):
    """Return (seq_clean[1,L], struc_clean[1,L], mask[1,L], residue_index[1,L])."""
    seq_tensor = _tokenize_sequence(aa_transform, seq_str)
    L = int(seq_tensor.shape[0])
    if len(struc_tokens_list) != L:
        raise ValueError(f"Length mismatch after tokenization: seq {L} vs struc {len(struc_tokens_list)}")
    seq_clean = seq_tensor.to(device).unsqueeze(0)
    struc_clean = torch.tensor(struc_tokens_list, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones(1, L, device=device)
    residue_index = torch.arange(L, device=device).unsqueeze(0)
    return seq_clean, struc_clean, mask, residue_index


@torch.no_grad()
def _score_modality(
    model,
    *,
    modality: str,  # "seq" or "struc"
    seq_clean: torch.Tensor,
    struc_clean: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    t_values: torch.Tensor,  # [K] in (0, 1)
    seq_mask_id: int,
    struc_mask_id: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward on K replicas of (seq, struc) at given t_values; return per-draw stats.

    Returns (avg_per_masked_CE[K], sum_CE_masked[K], n_masked[K]) for the chosen modality.
    """
    K = t_values.shape[0]
    L = seq_clean.shape[1]
    device = seq_clean.device

    seq_batch = seq_clean.expand(K, -1).contiguous()
    struc_batch = struc_clean.expand(K, -1).contiguous()
    mask_batch = mask.expand(K, -1).contiguous()
    residue_batch = residue_index.expand(K, -1).contiguous()

    if modality == "seq":
        # Mask sequence with prob (1 - t_k) per position; struc held clean.
        rand = torch.rand(K, L, generator=rng, device=device)
        mask_pos = rand > t_values.unsqueeze(1)  # True where masked
        x_t_seq = torch.where(mask_pos, torch.full_like(seq_batch, seq_mask_id), seq_batch)
        x_t_struc = struc_batch
        timesteps = {
            "sequence_tokens": t_values.to(device),
            "structure_tokens": torch.ones(K, device=device),
        }
        target = seq_batch
    elif modality == "struc":
        rand = torch.rand(K, L, generator=rng, device=device)
        mask_pos = rand > t_values.unsqueeze(1)
        x_t_struc = torch.where(mask_pos, torch.full_like(struc_batch, struc_mask_id), struc_batch)
        x_t_seq = seq_batch
        timesteps = {
            "sequence_tokens": torch.ones(K, device=device),
            "structure_tokens": t_values.to(device),
        }
        target = struc_batch
    else:
        raise ValueError(f"unknown modality {modality}")

    conditioning = torch.zeros(K, L, 1, device=device)

    out = model.forward(
        {"sequence_tokens": x_t_seq, "structure_tokens": x_t_struc},
        mask_batch,
        residue_batch,
        conditioning,
        timesteps=timesteps,
    )

    logits_key = "sequence_logits" if modality == "seq" else "structure_logits"
    logits = out[logits_key]  # [K, L, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    ce_per_pos = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [K, L]

    masked = mask_pos & mask_batch.bool()
    n_masked = masked.sum(dim=1).to(torch.float32)  # [K]
    sum_ce = (ce_per_pos * masked.float()).sum(dim=1)  # [K]
    avg_ce = sum_ce / n_masked.clamp(min=1.0)
    return avg_ce, sum_ce, n_masked


@torch.no_grad()
def _score_joint(
    model,
    *,
    seq_clean: torch.Tensor,        # [1, L]
    struc_clean: torch.Tensor,      # [1, L]
    mask: torch.Tensor,             # [1, L]
    residue_index: torch.Tensor,
    t_values: torch.Tensor,         # [K]
    seq_mask_id: int,
    struc_mask_id: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass per draw with BOTH modalities masked at rate (1 - t_k).

    Implements a Monte-Carlo estimator of the true joint AO-ARM log-likelihood
    over the unified (seq, struc) 2L-token stream. The same scalar t_k is used
    for both modalities per draw (matching the linear MD4 schedule applied
    jointly), but mask patterns are sampled independently across positions for
    each modality.

    Returns (avg_per_masked_CE[K], sum_CE_total[K], n_masked_total[K]) where
    numerator/denominator sum across both modalities.
    """
    K = t_values.shape[0]
    L = seq_clean.shape[1]
    device = seq_clean.device

    seq_batch = seq_clean.expand(K, -1).contiguous()
    struc_batch = struc_clean.expand(K, -1).contiguous()
    mask_batch = mask.expand(K, -1).contiguous()
    residue_batch = residue_index.expand(K, -1).contiguous()

    rand_seq = torch.rand(K, L, generator=rng, device=device)
    rand_struc = torch.rand(K, L, generator=rng, device=device)
    mask_seq_pos = rand_seq > t_values.unsqueeze(1)
    mask_struc_pos = rand_struc > t_values.unsqueeze(1)

    x_t_seq = torch.where(mask_seq_pos, torch.full_like(seq_batch, seq_mask_id), seq_batch)
    x_t_struc = torch.where(
        mask_struc_pos, torch.full_like(struc_batch, struc_mask_id), struc_batch
    )

    timesteps = {
        "sequence_tokens": t_values.to(device),
        "structure_tokens": t_values.to(device),
    }
    conditioning = torch.zeros(K, L, 1, device=device)

    out = model.forward(
        {"sequence_tokens": x_t_seq, "structure_tokens": x_t_struc},
        mask_batch,
        residue_batch,
        conditioning,
        timesteps=timesteps,
    )

    seq_log_probs = torch.log_softmax(out["sequence_logits"].float(), dim=-1)
    struc_log_probs = torch.log_softmax(out["structure_logits"].float(), dim=-1)
    seq_ce = -seq_log_probs.gather(-1, seq_batch.unsqueeze(-1)).squeeze(-1)
    struc_ce = -struc_log_probs.gather(-1, struc_batch.unsqueeze(-1)).squeeze(-1)

    seq_masked = mask_seq_pos & mask_batch.bool()
    struc_masked = mask_struc_pos & mask_batch.bool()

    n_masked_seq = seq_masked.sum(dim=1).to(torch.float32)
    n_masked_struc = struc_masked.sum(dim=1).to(torch.float32)
    sum_ce_seq = (seq_ce * seq_masked.float()).sum(dim=1)
    sum_ce_struc = (struc_ce * struc_masked.float()).sum(dim=1)

    n_masked_total = n_masked_seq + n_masked_struc
    sum_ce_total = sum_ce_seq + sum_ce_struc
    avg_ce = sum_ce_total / n_masked_total.clamp(min=1.0)

    return avg_ce, sum_ce_total, n_masked_total


@torch.no_grad()
def score_one_sample(
    model,
    *,
    seq_clean: torch.Tensor,
    struc_clean: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    K: int,
    seed: int,
    seq_mask_id: int,
    struc_mask_id: int,
    eps: float = 0.02,
) -> dict[str, float]:
    """Score one sample on both modalities. Returns flat dict of scalars."""
    device = seq_clean.device
    L = seq_clean.shape[1]

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Stratified t draws: K equal-width strata in (eps, 1-eps), one uniform sample per stratum.
    # Eliminates the t-selection variance that dominates at small K (the dominant source of
    # cross-seed variance in pilot runs); only mask-pattern noise remains.
    edges = torch.linspace(eps, 1.0 - eps, steps=K + 1, device=device)
    u = torch.empty(K, device=device).uniform_(0.0, 1.0, generator=rng)
    t_random = edges[:-1] + u * (edges[1:] - edges[:-1])
    t_fixed = torch.tensor(_FIXED_T, device=device, dtype=torch.float32)

    out: dict[str, float] = {}
    for modality, prefix in (("seq", "seq"), ("struc", "struc")):
        # Random-t draws (P1)
        avg_ce, sum_ce, n_masked = _score_modality(
            model,
            modality=modality,
            seq_clean=seq_clean,
            struc_clean=struc_clean,
            mask=mask,
            residue_index=residue_index,
            t_values=t_random,
            seq_mask_id=seq_mask_id,
            struc_mask_id=struc_mask_id,
            rng=rng,
        )
        score_unif = avg_ce.mean().item()
        # ELBO-style (1-t)-reweighted summed CE / L
        weighted = sum_ce / ((1.0 - t_random) * float(L))
        score_arllh = weighted.mean().item()
        out[f"{prefix}_score_unif"] = score_unif
        out[f"{prefix}_score_arllh"] = score_arllh
        out[f"{prefix}_n_draws"] = float(K)
        out[f"{prefix}_mean_n_masked"] = n_masked.mean().item()

        # Fixed-t draws (P2)
        avg_ce_fixed, _, _ = _score_modality(
            model,
            modality=modality,
            seq_clean=seq_clean,
            struc_clean=struc_clean,
            mask=mask,
            residue_index=residue_index,
            t_values=t_fixed,
            seq_mask_id=seq_mask_id,
            struc_mask_id=struc_mask_id,
            rng=rng,
        )
        for t_val, ce_val in zip(_FIXED_T, avg_ce_fixed.tolist()):
            out[f"{prefix}_score_t{t_val:g}"] = ce_val

    out["joint_score_unif"] = out["seq_score_unif"] + out["struc_score_unif"]
    out["joint_score_arllh"] = out["seq_score_arllh"] + out["struc_score_arllh"]

    # True joint AR-MC: one forward pass per draw with both modalities masked
    # at rate (1 - t_k) on the unified 2L-token stream. K random-t draws
    # (reuse the same stratified-t schedule for apples-to-apples).
    avg_ce_joint, sum_ce_joint, n_masked_joint = _score_joint(
        model,
        seq_clean=seq_clean,
        struc_clean=struc_clean,
        mask=mask,
        residue_index=residue_index,
        t_values=t_random,
        seq_mask_id=seq_mask_id,
        struc_mask_id=struc_mask_id,
        rng=rng,
    )
    out["joint_true_score_unif"] = avg_ce_joint.mean().item()
    weighted_joint = sum_ce_joint / ((1.0 - t_random) * 2.0 * float(L))
    out["joint_true_score_arllh"] = weighted_joint.mean().item()
    out["joint_true_n_draws"] = float(K)
    out["joint_true_mean_n_masked"] = n_masked_joint.mean().item()
    return out


_OUT_COLUMNS = [
    "run_id",
    "iteration",
    "sample_idx",
    "length",
    "input_structure",
    "trial_selected",
    "sequence_type",
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
    "seq_n_draws",
    "seq_mean_n_masked",
    "struc_n_draws",
    "struc_mean_n_masked",
    "joint_true_n_draws",
    "joint_true_mean_n_masked",
    "scoring_seed",
]


def _open_writer(path: Path):
    fh = path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=_OUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    return fh, writer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, type=Path, help="Eval directory containing sequences_*.csv")
    p.add_argument("--ckpt", required=True, type=Path, help="Gen-UME checkpoint path")
    p.add_argument(
        "--task",
        choices=("auto", "forward_folding", "inverse_folding", "unconditional"),
        default="auto",
        help="Eval task; if 'auto' inferred from sequences CSV filename",
    )
    p.add_argument("--K", type=int, default=16, help="Random-t Monte-Carlo draws per modality")
    p.add_argument("--seed", type=int, default=20260430, help="Per-sample base seed")
    p.add_argument("--max-samples", type=int, default=None, help="Score only first N rows (for smoke testing)")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path; default pll_scores_<ts>.csv in --eval-dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=512, help="Skip rows longer than this")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    eval_dir: Path = args.eval_dir
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"--eval-dir does not exist: {eval_dir}")

    task = args.task if args.task != "auto" else _detect_task(eval_dir)
    seq_col = _seq_source_column(task)
    csv_path = _find_sequences_csv(eval_dir, task)
    logger.info("Task=%s; reading %s; sequence column='%s'", task, csv_path.name, seq_col)

    rows = _load_rows(csv_path, seq_col, args.max_samples)
    if not rows:
        raise RuntimeError(f"No usable rows found in {csv_path}")
    rows = [r for r in rows if r["length"] <= args.max_length]
    logger.info("Loaded %d rows (max_length filter %d)", len(rows), args.max_length)

    output_path = args.output or (eval_dir / f"pll_scores_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv")
    logger.info("Writing scores to %s", output_path)

    device = torch.device(args.device)

    # Lazy imports (heavy)
    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule
    from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = UMESequenceStructureEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)
    # Make sure interpolant device tracking aligns (used internally by some MOCO ops).
    model.interpolant_seq.device = device
    model.interpolant_struc.device = device
    logger.info("Model loaded in %.1fs (device=%s)", time.time() - t0, device)

    seq_mask_id = int(getattr(model, "mask_token_id"))
    struc_mask_id = int(getattr(model, "mask_index_struc_tokens"))
    logger.info("seq_mask_id=%d  struc_mask_id=%d  vocab_seq=%d  vocab_struc=%d",
                seq_mask_id, struc_mask_id, int(model.vocab_size), int(model.num_struc_classes))

    aa_transform = AminoAcidTokenizerTransform(max_length=args.max_length)
    inner_tok = aa_transform.tokenizer_transform.tokenizer
    if int(inner_tok.mask_token_id) != seq_mask_id:
        logger.warning(
            "Tokenizer mask_token_id=%d != model.mask_token_id=%d",
            inner_tok.mask_token_id,
            seq_mask_id,
        )

    fh, writer = _open_writer(output_path)
    try:
        n_done = 0
        n_skipped = 0
        t_start = time.time()
        for row_idx, row in enumerate(rows):
            try:
                seq_clean, struc_clean, mask_t, residue_index = _build_inputs(
                    row["_seq_str"], row["_struc_tokens"], aa_transform, device
                )
                # Sanity: clamp out-of-vocab struc tokens to avoid OOB CE; report and skip if any.
                if (struc_clean >= int(model.num_struc_classes)).any() or (struc_clean < 0).any():
                    n_skipped += 1
                    logger.warning("Skipping row %s: struc tokens out of vocab", row.get("run_id"))
                    continue
                if (seq_clean >= int(model.vocab_size)).any() or (seq_clean < 0).any():
                    n_skipped += 1
                    logger.warning("Skipping row %s: seq tokens out of vocab", row.get("run_id"))
                    continue
            except Exception as e:
                n_skipped += 1
                logger.warning("Skipping row %s: %s", row.get("run_id"), e)
                continue

            sample_seed = (args.seed * 1_000_003 + row_idx) & 0x7FFFFFFF
            try:
                scores = score_one_sample(
                    model,
                    seq_clean=seq_clean,
                    struc_clean=struc_clean,
                    mask=mask_t,
                    residue_index=residue_index,
                    K=args.K,
                    seed=sample_seed,
                    seq_mask_id=seq_mask_id,
                    struc_mask_id=struc_mask_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skipped += 1
                logger.warning("OOM on row %s (L=%d); skipping", row.get("run_id"), row["length"])
                continue

            out_row = {
                "run_id": row["run_id"],
                "iteration": row["iteration"],
                "sample_idx": row["sample_idx"],
                "length": row["length"],
                "input_structure": row["input_structure"],
                "trial_selected": row["trial_selected"],
                "sequence_type": row["sequence_type"],
                "scoring_seed": sample_seed,
                **{k: v for k, v in scores.items() if k in _OUT_COLUMNS},
            }
            # Make sure all numeric scores serialize cleanly
            for k, v in list(out_row.items()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out_row[k] = ""
            writer.writerow(out_row)
            fh.flush()
            n_done += 1

            if (n_done % args.log_every) == 0:
                dt = time.time() - t_start
                logger.info(
                    "[%4d/%d] L=%d  seq_unif=%.3f arllh=%.3f  struc_unif=%.3f arllh=%.3f  (%.2fs/sample)",
                    n_done,
                    len(rows),
                    row["length"],
                    scores["seq_score_unif"],
                    scores["seq_score_arllh"],
                    scores["struc_score_unif"],
                    scores["struc_score_arllh"],
                    dt / max(1, n_done),
                )
    finally:
        fh.close()

    logger.info("Done. Scored %d rows; skipped %d. Output: %s", n_done, n_skipped, output_path)


if __name__ == "__main__":
    main()
