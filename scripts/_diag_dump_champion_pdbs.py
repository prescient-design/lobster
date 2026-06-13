"""Sample + dump PDBs from the current Kabsch-champion checkpoint.

Reads the champion materialised by Phase 6/7's `--save-best-kab-dir` flag:

    <champion_dir>/champion.json            -- meta: run_name, ckpt_tag, etc.
    <champion_dir>/champion.ckpt            -- the copied Lightning ckpt
    <champion_dir>/champion_model_cfg.yaml  -- resolved cfg.model

Composes the matching experiment cfg by mapping the champion's run_name
to ``train_latent_generator_3di_input_<run_name>`` and loads
`champion.ckpt` into it -- same code path as the eval phases. Then runs
the Phase 3c SDE winner config (``init_temperature=1.0, min_t=0.0``)
on the 30-protein PDB val batch, Kabsch-aligns each sample to GT, and
writes one PDB per protein:

    <out_dir>/_gt/gt_NN_lenLLL.pdb           GT coords + INPUT 3Di tokens
    <out_dir>/champion/sample_NN_lenLLL.pdb  sample coords (Kabsch-aligned)
                                             + the model's re-encoded 3Di
    <out_dir>/manifest.json                  per-protein Kab + 3DR + kwargs

Inference is chunked over ``--batch-size`` proteins so the 110M-param
``base`` ckpt fits on a 22 GB A10G (chunk=4 covers 30 proteins safely).

    uv run python scripts/_diag_dump_champion_pdbs.py \\
        --champion-dir /cv/scratch/u/lisanzas/champion_3di_flow_velocity \\
        --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from compare_3di_input_runs import (
    _build_val_loader,
    _center,
    _compose_cfg,
    _kabsch_align,
    _load_model,
    _rmsd,
)
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.mini3di import Encoder, calculate_cb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dump_champion")

SEED = 0

WINNER_KWARGS: dict = dict(
    n_steps=400,
    sampling_mode="sde",
    schedule_type="power",
    schedule_exponent=2.0,
    guidance_scale=1.0,
    sc_scale_noise=0.3,
    sc_scale_score=1.0,
    gt_mode="us",
    gt_p=1.0,
    t_lim_ode=0.99,
    init_temperature=1.0,
    min_t=0.0,
)

# 3Di alphabet -> num2aa lookup, identical to `_diag_inference_sweep_velocity_dump_best.py`.
_THREE_DI_LETTERS = "ACDEFGHIKLMNPQRSTVWYX"
_NUM2AA_BY_LETTER = {
    "A": 0, "C": 4, "D": 3, "E": 6, "F": 13, "G": 7, "H": 8, "I": 9,
    "K": 11, "L": 10, "M": 12, "N": 2, "P": 14, "Q": 5, "R": 1, "S": 15,
    "T": 16, "V": 19, "W": 17, "Y": 18, "X": 20,
}
_THREE_DI_TO_NUM2AA = torch.tensor(
    [_NUM2AA_BY_LETTER[c] for c in _THREE_DI_LETTERS], dtype=torch.long
)
_3DI_ENCODER = Encoder()


def _three_di_to_num2aa(states: np.ndarray) -> torch.Tensor:
    states_t = torch.from_numpy(np.asarray(states, dtype=np.int64))
    states_t = torch.clamp(states_t, 0, 20)
    return _THREE_DI_TO_NUM2AA[states_t]


def _encode_3di(coords: torch.Tensor) -> np.ndarray:
    Ca, Cb, N, C = calculate_cb({"coords_res": coords})
    enc = _3DI_ENCODER.encode_atoms(Ca, Cb, N, C)
    return enc["states"].filled().astype(np.int64)


@torch.no_grad()
def _dump_gt_pdb(out_dir: Path, idx: int, length: int, xyz_full: torch.Tensor,
                 mask_i: torch.Tensor, gt_3di_full: torch.Tensor) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = xyz_full[mask_i].cpu()
    tokens_3di = gt_3di_full[mask_i].cpu().numpy().astype(np.int64)
    seq = _three_di_to_num2aa(tokens_3di)
    writepdb(str(out_dir / f"gt_{idx:02d}_len{length:03d}.pdb"), xyz, seq)


@torch.no_grad()
def _dump_sample_pdb(out_dir: Path, idx: int, length: int,
                     coords_full: torch.Tensor, mask_i: torch.Tensor) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    if length < 2:
        return np.array([], dtype=np.int64)
    xyz = coords_full[mask_i].detach().cpu().to(torch.float32)
    try:
        tokens_3di = _encode_3di(xyz)
    except Exception as exc:
        log.debug("3Di encode failed at sample %d: %s", idx, exc)
        tokens_3di = np.full(length, 20, dtype=np.int64)
    seq = _three_di_to_num2aa(tokens_3di)
    writepdb(str(out_dir / f"sample_{idx:02d}_len{length:03d}.pdb"), xyz, seq)
    return tokens_3di


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion-dir", required=True, type=str,
        help="Directory containing champion.{ckpt,json,model_cfg.yaml}",
    )
    parser.add_argument(
        "--out-dir", default=None, type=str,
        help=("Where to write the PDB dump. Default: "
              "<champion_dir>/sample_pdbs/."),
    )
    parser.add_argument(
        "--batch-size", default=4, type=int,
        help=("Sampling chunk size (proteins per model.sample call). "
              "Default 4 is safe for 110M-param `base` on a 22 GB A10G; "
              "raise on bigger GPUs or smaller models for speed."),
    )
    parser.add_argument(
        "--n-samples", default=1, type=int,
        help=("Best-of-N sampling: per protein, draw N independent samples "
              "(seed = SEED + sample_idx), keep the one with lowest "
              "Kabsch RMSD vs GT. Wall-clock scales N x; memory unchanged "
              "(samples drawn sequentially within each chunk). Default 1 "
              "(single-sample, same as before)."),
    )
    args = parser.parse_args()

    champion_dir = Path(args.champion_dir)
    meta = json.loads((champion_dir / "champion.json").read_text())
    log.info("champion meta: %s", meta)

    out_dir = Path(args.out_dir) if args.out_dir else (champion_dir / "sample_pdbs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map run_name -> experiment cfg name (same convention used everywhere).
    run_name = meta["run_name"]
    exp = f"train_latent_generator_3di_input_{run_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s out=%s exp=%s", device, out_dir, exp)
    log.info("WINNER_KWARGS=%s", WINNER_KWARGS)
    torch.manual_seed(SEED)

    val_pt = "/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/validation.pt"
    overrides = [
        "data.batch_size=32",  # loader returns all 30 proteins in one batch
        "data.num_workers=0",
        f"data.path_to_datasets=[{val_pt},{val_pt},{val_pt}]",
        "data.cluster_file=null",
    ]
    cfg = _compose_cfg(exp, overrides)
    loader = _build_val_loader(cfg)
    batch = next(iter(loader))
    raw = {k: batch[k] for k in ["3di_states", "mask", "indices", "coords_res"]}
    lengths = raw["mask"].sum(-1).long().tolist()
    N = len(lengths)
    log.info("val batch: %d proteins, len min/med/max=%d/%d/%d",
             N, min(lengths), int(np.median(lengths)), max(lengths))

    ckpt = champion_dir / "champion.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no ckpt at {ckpt}")
    model = _load_model(cfg, ckpt, device)

    # Dump GT PDBs once. Use the centered GT (same coords the loss/eval
    # operate on); this matches what Kabsch-aligned samples are compared to.
    gt_raw = raw["coords_res"].to(device).to(torch.get_default_dtype())
    gt_seq_mask = raw["mask"].to(device).float()
    gt_c = _center(gt_raw, gt_seq_mask)
    gt_dir = out_dir / "_gt"
    log.info("dumping GT PDBs -> %s", gt_dir)
    for i in range(N):
        m = raw["mask"][i].bool()
        _dump_gt_pdb(gt_dir, i, int(lengths[i]),
                     gt_c[i].cpu(), m, raw["3di_states"][i])

    # Sample in chunks of args.batch_size; for each chunk do best-of-N
    # sampling (N=args.n_samples). Per protein keep the lowest-Kabsch
    # sample's coords, predicted 3Di tokens, and the corresponding 3DR.
    chunk = int(args.batch_size)
    n_samples = int(args.n_samples)
    sample_dir = out_dir / "champion"
    log.info("sampling %d proteins in chunks of %d, best-of-%d -> %s",
             N, chunk, n_samples, sample_dir)
    # Final per-protein records (length N, ordered by global protein index).
    best_kab_per: list[float] = []          # lowest Kab seen for each protein
    best_rec_per: list[float] = []          # 3DR of THAT lowest-Kab sample
    pred_3di_per: list[list[int]] = []      # predicted 3Di tokens of THAT sample
    kab_samples_per: list[list[float]] = []  # per-protein per-sample Kabs
    rec_samples_per: list[list[float]] = []  # per-protein per-sample 3DR
    input_3di_cpu = raw["3di_states"]
    mask_full = raw["mask"]

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        n_in_chunk = end - start
        raw_chunk = {k: v[start:end] for k, v in raw.items()}
        states, seq_mask, residue_index = model.featurize(
            {k: raw_chunk[k].to(device) for k in ["3di_states", "mask", "indices"]}
        )
        gt_c_chunk = _center(raw_chunk["coords_res"].to(device).to(torch.get_default_dtype()), seq_mask)

        # Per-chunk accumulators across the N samples. We track per-sample
        # Kabsch (cheap, vectorised) for every draw, but only encode 3Di
        # ONCE per protein -- on the per-protein-best-Kab aligned coords --
        # since `mini3di.Encoder.encode_atoms` is a slow Python pipeline.
        chunk_best_kab = np.full(n_in_chunk, np.inf, dtype=np.float64)
        chunk_best_aligned = torch.zeros_like(gt_c_chunk)
        chunk_kab_samples = np.full((n_in_chunk, n_samples), np.nan, dtype=np.float64)

        for s in range(n_samples):
            # Vary seed per sample so the N draws are independent. Same seed
            # offset across chunks so cross-protein samples remain comparable.
            torch.manual_seed(SEED + s)
            samp = model.sample(
                states=states, seq_mask=seq_mask, residue_index=residue_index,
                **WINNER_KWARGS,
            )
            if not torch.isfinite(samp).all().item():
                log.warning("non-finite sample at chunk start=%d sample=%d -- skipping",
                            start, s)
                continue
            aligned = _kabsch_align(_center(samp, seq_mask), gt_c_chunk, seq_mask)
            kab = _rmsd(aligned, gt_c_chunk, seq_mask).cpu().numpy()
            chunk_kab_samples[:, s] = kab
            for j in range(n_in_chunk):
                if np.isfinite(kab[j]) and kab[j] < chunk_best_kab[j]:
                    chunk_best_kab[j] = float(kab[j])
                    chunk_best_aligned[j] = aligned[j]
            log.debug("  chunk start=%d sample=%d kab_mean=%.2f", start, s, float(np.nanmean(kab)))

        # Write the per-protein-BEST sample PDB and gather record fields.
        for j in range(n_in_chunk):
            global_i = start + j
            L = int(lengths[global_i])
            m = mask_full[global_i].bool()
            # Use the helper to write the best aligned coords + re-encode 3Di
            # (matches the single-sample path's PDB convention).
            tokens_3di = _dump_sample_pdb(sample_dir, global_i, L,
                                          chunk_best_aligned[j].cpu(), m)
            pred_3di_per.append([int(x) for x in tokens_3di])
            best_kab_per.append(float(chunk_best_kab[j]))
            gt_3di = input_3di_cpu[global_i][m].numpy().astype(np.int64)
            rec = (float((tokens_3di == gt_3di).mean())
                   if tokens_3di.size == gt_3di.size and tokens_3di.size > 0
                   else float("nan"))
            best_rec_per.append(rec)
            kab_samples_per.append([None if not np.isfinite(v) else float(v)
                                    for v in chunk_kab_samples[j].tolist()])
            # Per-sample 3DR is not computed (mini3di encode is slow); we
            # only encode 3Di on the best-Kab sample. Keep the field for
            # schema parity but fill with None.
            rec_samples_per.append([None] * n_samples)
            samples_kab_str = (
                f" mean={np.nanmean(chunk_kab_samples[j]):.2f}/std={np.nanstd(chunk_kab_samples[j]):.2f}"
                if n_samples > 1 else ""
            )
            log.info("  [%2d/%d] L=%d best_kab=%.2f A 3DR=%s%s",
                     global_i + 1, N, L, float(chunk_best_kab[j]),
                     "nan" if np.isnan(rec) else f"{100*rec:.1f}%",
                     samples_kab_str)

        # Tee out per-sample chunk-level summary so progress is visible.
        if n_samples > 1:
            mean_per_sample = np.nanmean(chunk_kab_samples, axis=0)
            best_per_sample = np.nanmin(chunk_kab_samples, axis=0)
            log.info("    chunk %d-%d kab_mean_per_sample=%s best_per_sample=%s",
                     start, end - 1,
                     [f"{v:.2f}" for v in mean_per_sample],
                     [f"{v:.2f}" for v in best_per_sample])
    # Aggregate over proteins. With n_samples=1 these collapse to the
    # single-sample numbers (`best_*` == `*`), so this code path is the
    # same shape as before for backwards-compat.
    valid_best_rec = [v for v in best_rec_per if not np.isnan(v)]
    valid_best_kab = [v for v in best_kab_per if not np.isnan(v)]
    mean_best_rec = float(np.mean(valid_best_rec)) if valid_best_rec else float("nan")
    mean_best_kab = float(np.mean(valid_best_kab)) if valid_best_kab else float("nan")

    # First-sample ("single-sample baseline") Kab so we can read off the
    # best-of-N improvement directly from logs. (Per-sample 3DR is not
    # computed -- it would require encoding 3Di on N x N_proteins coords;
    # 3DR is reported only for the per-protein-best-Kab sample.)
    first_sample_kabs = [s[0] if (s and s[0] is not None) else None for s in kab_samples_per]
    valid_first_kabs = [v for v in first_sample_kabs if v is not None]
    mean_first_kab = float(np.mean(valid_first_kabs)) if valid_first_kabs else float("nan")

    if n_samples > 1:
        log.info(
            "CHAMPION best-of-%d: kab=%.2f A  3DR=%.1f%%   (single-sample baseline kab=%.2f A; improvement %.2f A)",
            n_samples, mean_best_kab, 100.0 * mean_best_rec,
            mean_first_kab, mean_first_kab - mean_best_kab,
        )
    else:
        log.info("CHAMPION (re-sampled): kab=%.2f A  3DR=%.1f%%",
                 mean_best_kab, 100.0 * mean_best_rec)

    manifest = {
        "champion_meta": meta,
        "kwargs": WINNER_KWARGS,
        "seed": SEED,
        "chunk_size": chunk,
        "n_samples": n_samples,
        "n_proteins": N,
        "summary": {
            "best_of_n_kabsch_mean": mean_best_kab,
            "best_of_n_three_di_recovery_mean": mean_best_rec,
            "single_sample_kabsch_mean": mean_first_kab,
            # legacy keys (alias to best-of-N, which is the dumped PDB):
            "kabsch_mean": mean_best_kab,
            "three_di_recovery_mean": mean_best_rec,
        },
        "per_protein": [
            {
                "idx": i,
                "length": int(lengths[i]),
                "kabsch_rmsd": best_kab_per[i],          # best-of-N
                "three_di_recovery": (
                    None if np.isnan(best_rec_per[i]) else best_rec_per[i]
                ),
                "kab_per_sample": kab_samples_per[i],    # length n_samples
                "three_di_recovery_per_sample": rec_samples_per[i],
                "input_3di": input_3di_cpu[i][mask_full[i].bool()].numpy().astype(int).tolist(),
                "predicted_3di": pred_3di_per[i],        # 3Di of the best-Kab sample
            }
            for i in range(N)
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest -> %s/manifest.json", out_dir)
    log.info("Done. PDBs at %s/champion/ ; GT at %s/_gt/", out_dir, out_dir)


if __name__ == "__main__":
    main()
