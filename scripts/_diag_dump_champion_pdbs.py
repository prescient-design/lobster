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

    # Sample in chunks of args.batch_size.
    chunk = int(args.batch_size)
    sample_dir = out_dir / "champion"
    log.info("sampling %d proteins in chunks of %d -> %s", N, chunk, sample_dir)
    kab_per: list[float] = []
    rec_per: list[float] = []
    pred_3di_per: list[list[int]] = []
    input_3di_cpu = raw["3di_states"]
    mask_full = raw["mask"]

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        raw_chunk = {k: v[start:end] for k, v in raw.items()}
        states, seq_mask, residue_index = model.featurize(
            {k: raw_chunk[k].to(device) for k in ["3di_states", "mask", "indices"]}
        )
        gt_c_chunk = _center(raw_chunk["coords_res"].to(device).to(torch.get_default_dtype()), seq_mask)
        torch.manual_seed(SEED)  # reset per-chunk for cross-script reproducibility
        samp = model.sample(
            states=states, seq_mask=seq_mask, residue_index=residue_index,
            **WINNER_KWARGS,
        )
        if not torch.isfinite(samp).all().item():
            raise RuntimeError(f"non-finite sample at chunk start={start}")
        aligned = _kabsch_align(_center(samp, seq_mask), gt_c_chunk, seq_mask)
        kab = _rmsd(aligned, gt_c_chunk, seq_mask).cpu().numpy()

        for j in range(end - start):
            global_i = start + j
            L = int(lengths[global_i])
            m = mask_full[global_i].bool()
            tokens_3di = _dump_sample_pdb(sample_dir, global_i, L, aligned[j].cpu(), m)
            pred_3di_per.append([int(x) for x in tokens_3di])
            kab_per.append(float(kab[j]))
            gt_3di = input_3di_cpu[global_i][m].numpy().astype(np.int64)
            rec = float((tokens_3di == gt_3di).mean()) if tokens_3di.size == gt_3di.size and tokens_3di.size > 0 else float("nan")
            rec_per.append(rec)
            log.info("  [%2d/%d] L=%d kab=%.2f A 3DR=%s",
                     global_i + 1, N, L, kab[j],
                     "nan" if np.isnan(rec) else f"{100*rec:.1f}%")

    valid_rec = [v for v in rec_per if not np.isnan(v)]
    valid_kab = [v for v in kab_per if not np.isnan(v)]
    mean_rec = float(np.mean(valid_rec)) if valid_rec else float("nan")
    mean_kab = float(np.mean(valid_kab)) if valid_kab else float("nan")
    log.info("CHAMPION (re-sampled): kab=%.2f A  3DR=%.1f%%", mean_kab, 100.0 * mean_rec)

    manifest = {
        "champion_meta": meta,
        "kwargs": WINNER_KWARGS,
        "seed": SEED,
        "chunk_size": chunk,
        "n_proteins": N,
        "summary": {"kabsch_mean": mean_kab, "three_di_recovery_mean": mean_rec},
        "per_protein": [
            {
                "idx": i,
                "length": int(lengths[i]),
                "kabsch_rmsd": kab_per[i],
                "three_di_recovery": (None if np.isnan(rec_per[i]) else rec_per[i]),
                "input_3di": input_3di_cpu[i][mask_full[i].bool()].numpy().astype(int).tolist(),
                "predicted_3di": pred_3di_per[i],
            }
            for i in range(N)
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest -> %s/manifest.json", out_dir)
    log.info("Done. PDBs at %s/champion/ ; GT at %s/_gt/", out_dir, out_dir)


if __name__ == "__main__":
    main()
