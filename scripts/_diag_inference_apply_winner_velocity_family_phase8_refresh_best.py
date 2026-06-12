"""Phase 8 -- refresh Phase 6 numbers for the runs whose val-best ckpt has
moved since Phase 6 ran (~14 h ago).

Three runs have a new lower-val-loss ckpt since Phase 6:
    selfcond    e1503 v=0.5378  ->  e1832 v=0.4772
    mask3di     e1236 v=0.5920  ->  e2209 v=0.5540
    base_selfcond e109 v=0.7502 ->  e200  v=0.6743

The other three (`flow_nokabsch`, `flow_nokabsch_velocity`,
`flow_nokabsch_velocity_base`) are unchanged on disk -- their Phase 6
numbers are still the correct val-best read.

Same Phase 3c SDE winner config as Phase 6/7 (init=1.0, min_t=0.0) so
results are directly comparable.

Output: `.compare_runs/inference_sweep_velocity/phase8_refresh_best_val.json`

    uv run python scripts/_diag_inference_apply_winner_velocity_family_phase8_refresh_best.py
"""

from __future__ import annotations

import json
import logging
import re
import time
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
from lobster.model.latent_generator.utils.mini3di import Encoder, calculate_cb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("apply_winner_phase8")

# Only the runs whose val-best ckpt has moved since Phase 6.
RUNS = [
    ("flow_nokabsch_velocity_selfcond",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond",
     "latent_generator_3di_input_flow_nokabsch_velocity_selfcond/runs/2026-06-08T19-40-54"),
    ("flow_nokabsch_velocity_mask3di",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_mask3di",
     "latent_generator_3di_input_flow_nokabsch_velocity_mask3di/runs/2026-06-09T00-06-30"),
    ("flow_nokabsch_velocity_base_selfcond",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base_selfcond",
     "latent_generator_3di_input_flow_nokabsch_velocity_base_selfcond/runs/2026-06-10T11-58-16"),
]
SCRATCH = Path("/cv/scratch/u/lisanzas")
OUT_DIR = Path("/cv/home/lisanzas/lobster/.compare_runs/inference_sweep_velocity")
SEED = 0
_3DI_ENCODER = Encoder()

WINNER = dict(
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

_VAL_LOSS_RE = re.compile(r"val_loss=([0-9]+\.?[0-9]*)\.ckpt$")


def _best_val_loss_ckpt(run_dir: Path) -> tuple[Path | None, float | None]:
    best: tuple[Path | None, float | None] = (None, None)
    for p in run_dir.glob("epoch=*-step=*-val_loss=*.ckpt"):
        m = _VAL_LOSS_RE.search(p.name)
        if m is None:
            continue
        v = float(m.group(1))
        if best[0] is None or v < best[1]:
            best = (p, v)
    return best


def _three_di_recovery(samples: torch.Tensor, input_3di: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    B = samples.shape[0]
    out = np.zeros(B, dtype=np.float32)
    samples_cpu = samples.detach().cpu().to(torch.float32)
    input_3di_cpu = input_3di.detach().cpu().to(torch.long).numpy()
    mask_cpu = mask.detach().cpu().bool().numpy()
    for i in range(B):
        m = mask_cpu[i]
        if m.sum() < 2:
            out[i] = float("nan"); continue
        coords_i = samples_cpu[i, m]
        if not torch.isfinite(coords_i).all():
            out[i] = float("nan"); continue
        gt_3di_i = input_3di_cpu[i, m]
        Ca, Cb, N, C = calculate_cb({"coords_res": coords_i})
        try:
            enc = _3DI_ENCODER.encode_atoms(Ca, Cb, N, C)
            pred_states = enc["states"].filled().astype(np.int64)
        except Exception:
            out[i] = float("nan"); continue
        out[i] = float((pred_states == gt_3di_i).mean())
    return out


@torch.no_grad()
def _eval_one(model, raw, states, seq_mask, residue_index, gt_c, *, label, kwargs) -> dict:
    torch.manual_seed(SEED)
    t0 = time.time()
    samp = model.sample(states=states, seq_mask=seq_mask, residue_index=residue_index, **kwargs)
    finite = torch.isfinite(samp).all().item()
    if not finite:
        kab = np.full(samp.shape[0], np.nan, dtype=np.float32)
        rec = np.full(samp.shape[0], np.nan, dtype=np.float32)
    else:
        aligned = _kabsch_align(_center(samp, seq_mask), gt_c, seq_mask)
        kab = _rmsd(aligned, gt_c, seq_mask).cpu().numpy()
        rec = _three_di_recovery(aligned, raw["3di_states"], seq_mask)
    elapsed = time.time() - t0
    rec_f = rec[~np.isnan(rec)]
    kab_f = kab[~np.isnan(kab)]
    rec_mean = float(rec_f.mean()) if len(rec_f) else float("nan")
    kab_mean = float(kab_f.mean()) if len(kab_f) else float("nan")
    log.info("[%s]  kab=%.2f A  3DR=%.1f%%  (%.1fs)", label, kab_mean, 100.0 * rec_mean, elapsed)
    return {
        "label": label,
        "kabsch_mean": kab_mean,
        "three_di_recovery_mean": rec_mean,
        "kabsch_per_protein": [None if np.isnan(v) else float(v) for v in kab.tolist()],
        "three_di_recovery_per_protein": [None if np.isnan(v) else float(v) for v in rec.tolist()],
        "elapsed_seconds": elapsed,
    }


@torch.no_grad()
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s out=%s", device, OUT_DIR)
    log.info("WINNER (Phase 3c, no Phase 4): %s", WINNER)
    torch.manual_seed(SEED)

    val_pt = "/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/validation.pt"
    overrides = [
        "data.batch_size=32", "data.num_workers=0",
        f"data.path_to_datasets=[{val_pt},{val_pt},{val_pt}]",
        "data.cluster_file=null",
    ]

    base_cfg = _compose_cfg(RUNS[0][1], overrides)
    loader = _build_val_loader(base_cfg)
    batch = next(iter(loader))
    raw = {k: batch[k] for k in ["3di_states", "mask", "indices", "coords_res"]}
    lengths = raw["mask"].sum(-1).long().tolist()
    log.info("shared batch=%d lengths min/med/max=%d/%d/%d",
             len(lengths), min(lengths), int(np.median(lengths)), max(lengths))

    results: list[dict] = []
    for name, exp, rd in RUNS:
        run_dir = SCRATCH / rd
        best_ckpt, best_val = _best_val_loss_ckpt(run_dir)
        if best_ckpt is None or not best_ckpt.exists():
            log.warning("[%s] no val-best ckpt -- skipping", name)
            continue
        log.info("[%s] best=%s (val_loss=%s)", name, best_ckpt.name, best_val)

        cfg = _compose_cfg(exp, overrides)
        try:
            model = _load_model(cfg, best_ckpt, device)
        except Exception as e:
            log.warning("[%s] failed to load: %s", name, e)
            continue
        states, seq_mask, residue_index = model.featurize(
            {k: raw[k].to(device) for k in ["3di_states", "mask", "indices"]}
        )
        gt_c = _center(raw["coords_res"].to(device).to(torch.get_default_dtype()), seq_mask)

        res = _eval_one(model, raw, states, seq_mask, residue_index, gt_c,
                        label=f"{name}/best_val", kwargs=dict(WINNER))
        res["run"] = name
        res["ckpt_path"] = str(best_ckpt)
        res["ckpt_val_loss"] = best_val
        res["ckpt_name"] = best_ckpt.name
        results.append(res)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "winner_config": WINNER,
        "batch_size": int(len(lengths)),
        "lengths_min_med_max": [int(min(lengths)), int(np.median(lengths)), int(max(lengths))],
        "results": results,
    }
    out_path = OUT_DIR / "phase8_refresh_best_val.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_path)

    print("\n" + "=" * 92)
    print("Phase 8 -- refresh val-best for runs that moved since Phase 6 (init=1.0, min_t=0.0)")
    print("=" * 92)
    print(f"{'run':>40} | {'ckpt (epoch=N-step=M)':>28} | {'val_loss':>9} | {'Kab (A)':>8} | {'3DR (%)':>7}")
    print("-" * 110)
    for r in results:
        kab = "nan" if np.isnan(r["kabsch_mean"]) else f"{r['kabsch_mean']:.2f}"
        rec = "nan" if np.isnan(r["three_di_recovery_mean"]) else f"{100*r['three_di_recovery_mean']:.1f}%"
        vl = "-" if r["ckpt_val_loss"] is None else f"{r['ckpt_val_loss']:.4f}"
        ckpt_short = r["ckpt_name"].replace("-val_loss=", "/v=").replace(".ckpt", "")
        print(f"{r['run']:>40} | {ckpt_short:>28} | {vl:>9} | {kab:>8} | {rec:>7}")


if __name__ == "__main__":
    main()
