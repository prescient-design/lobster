"""Phase 7 -- Phase 3c SDE winner on each run's `last.ckpt` (companion to Phase 6).

Phase 6 evaluated each run's lowest-val-loss ckpt at the Phase 3c SDE
winner config (init=1.0, min_t=0.0). This script does the same eval at
each run's `last.ckpt` -- the most-recent training step -- so we can
read off (best, last) pairs and see whether each run is still
improving past its val minimum, has plateaued, or has overshot.

For `flow_nokabsch_velocity_base` the latest run dir on disk is the
RESUME run launched at 2026-06-11T02-54-55 (lr=8e-5, callback-driven
LR override on top of the val-best snapshot). Its `last.ckpt` therefore
reflects ~13 h of post-resume training -- a useful read on whether
8e-5 is moving the model in the right direction.

Output: `.compare_runs/inference_sweep_velocity/phase7_last_ckpt.json`

    uv run python scripts/_diag_inference_apply_winner_velocity_family_phase7_last.py
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

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
log = logging.getLogger("apply_winner_phase7_last")

# (display_name, experiment_name) -- the run-dir per `name` is auto-resolved
# to the most-recently-modified `runs/<timestamp>/` directory. For `base`
# that resolves to the resume run (2026-06-11T02-54-55), whose `last.ckpt`
# is the lr=8e-5 post-resume tip.
#
# `selfcond_distill` resumed from `selfcond` val-best e1503 onto the wider
# UME mix; its in-run val_loss is on AFDB CAMEO, but here we evaluate on
# the same PDB val batch as the rest of the family by composing the
# `selfcond` experiment cfg (the model architecture is identical so the
# distill ckpt loads cleanly). Run `--only` to filter to a subset.
RUNS = [
    ("flow_nokabsch",
     "train_latent_generator_3di_input_flow_nokabsch"),
    ("flow_nokabsch_velocity",
     "train_latent_generator_3di_input_flow_nokabsch_velocity"),
    ("flow_nokabsch_velocity_selfcond",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond"),
    ("flow_nokabsch_velocity_mask3di",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_mask3di"),
    ("flow_nokabsch_velocity_base",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base"),
    ("flow_nokabsch_velocity_base_selfcond",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base_selfcond"),
    ("flow_nokabsch_velocity_selfcond_distill",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond"),
    # Distogram + distogram_3di have extra heads (pair head, 3Di-token head)
    # so they MUST use their own experiment cfgs -- the selfcond cfg has no
    # `enable_distogram` / `enable_3di_head` flags and a strict ckpt load
    # would fail on the missing keys.
    ("flow_nokabsch_velocity_selfcond_distogram",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram"),
    ("flow_nokabsch_velocity_selfcond_distogram_3di",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di"),
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


class _KabChampion:
    """Track the lowest-Kabsch checkpoint seen across phase 6/7 runs.

    State on disk in ``save_dir``:
      - ``champion.json``       small meta blob (kabsch_mean, run, ckpt name, ...)
      - ``champion.ckpt``       the actual Lightning ckpt (copied from source)
      - ``champion_model_cfg.yaml``  resolved Hydra ``cfg.model`` so an inference
        script can rebuild the matching model class without the experiment cfg
      - ``champion_eval_record.json``  full eval payload (per-protein arrays,
        elapsed time, chunk_size, n_proteins, etc.) for later inspection

    Persistence semantics: on ``__init__`` we read ``champion.json`` if it
    exists and seed the in-memory ``best_kab`` from it; subsequent calls
    to ``maybe_save`` only overwrite when the new Kab is strictly lower.
    Because both phase 6 (best_val) and phase 7 (last) can share the same
    save_dir, the champion across an entire eval session is the lowest
    Kab over BOTH phases combined.
    """

    def __init__(self, save_dir: Path | None) -> None:
        self.save_dir = save_dir
        self.best_kab: float = float("inf")
        self.meta: dict | None = None
        if save_dir is None:
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        meta_path = save_dir / "champion.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
                v = self.meta.get("kabsch_mean")
                if isinstance(v, (int, float)) and not math.isnan(v):
                    self.best_kab = float(v)
                    log.info(
                        "[champion] existing champion at %s: %s/%s kab=%.4f",
                        save_dir, self.meta.get("run_name"),
                        self.meta.get("ckpt_tag"), self.best_kab,
                    )
            except Exception as e:
                log.warning("[champion] could not parse %s: %s -- starting fresh",
                            meta_path, e)

    def maybe_save(
        self,
        *,
        ckpt_src: Path,
        cfg,
        run_name: str,
        ckpt_tag: str,
        eval_record: dict,
        run_dir_name: str | None = None,
        ckpt_val_loss: float | None = None,
    ) -> bool:
        if self.save_dir is None:
            return False
        kab = eval_record.get("kabsch_mean")
        if not isinstance(kab, (int, float)) or math.isnan(kab):
            return False
        if kab >= self.best_kab:
            return False
        log.info(
            "[champion] NEW: %s/%s kab=%.4f (prev best %.4f) -> %s",
            run_name, ckpt_tag, kab, self.best_kab, self.save_dir,
        )
        # Copy ckpt (resolves symlinks).
        shutil.copy(Path(ckpt_src).resolve(), self.save_dir / "champion.ckpt")
        # Save the model cfg so inference can reconstruct the network without
        # needing the experiment overrides. `cfg.model` is the resolved
        # `Tokenizer3diInputFlow` config (target + decoder_factory + ...).
        OmegaConf.save(cfg.model, self.save_dir / "champion_model_cfg.yaml")
        # Compact meta.
        meta = {
            "run_name": run_name,
            "ckpt_tag": ckpt_tag,
            "kabsch_mean": float(kab),
            "three_di_recovery_mean": float(eval_record.get("three_di_recovery_mean", float("nan"))),
            "ckpt_source_path": str(Path(ckpt_src).resolve()),
            "ckpt_source_name": Path(ckpt_src).name,
            "ckpt_val_loss": ckpt_val_loss,
            "run_dir": run_dir_name,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_size": eval_record.get("chunk_size"),
            "n_proteins": eval_record.get("n_proteins"),
        }
        (self.save_dir / "champion.json").write_text(json.dumps(meta, indent=2))
        (self.save_dir / "champion_eval_record.json").write_text(
            json.dumps(eval_record, indent=2)
        )
        self.best_kab = float(kab)
        self.meta = meta
        return True


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
def _eval_chunk(model, raw_chunk, *, kwargs, device) -> tuple[np.ndarray, np.ndarray]:
    """One chunk forward: featurize -> sample -> Kabsch-align -> per-protein
    Kabsch RMSD + 3Di-recovery arrays. Returns ``(kab, rec)`` of shape ``(B,)``.
    """
    states, seq_mask, residue_index = model.featurize(
        {k: raw_chunk[k].to(device) for k in ["3di_states", "mask", "indices"]}
    )
    gt_c = _center(raw_chunk["coords_res"].to(device).to(torch.get_default_dtype()), seq_mask)
    samp = model.sample(states=states, seq_mask=seq_mask, residue_index=residue_index, **kwargs)
    if not torch.isfinite(samp).all().item():
        kab = np.full(samp.shape[0], np.nan, dtype=np.float32)
        rec = np.full(samp.shape[0], np.nan, dtype=np.float32)
    else:
        aligned = _kabsch_align(_center(samp, seq_mask), gt_c, seq_mask)
        kab = _rmsd(aligned, gt_c, seq_mask).cpu().numpy()
        rec = _three_di_recovery(aligned, raw_chunk["3di_states"], seq_mask)
    return kab, rec


@torch.no_grad()
def _eval_one(model, raw, *, label, kwargs, device, chunk_size: int) -> dict:
    """Iterate ``raw`` (the full 30-protein val batch) in chunks of
    ``chunk_size`` proteins, call the model on each chunk, and aggregate
    per-protein Kabsch / 3Di-recovery arrays.

    The seed is reset to ``SEED`` at the start of each chunk so that
    chunk-i's noise is determined only by ``chunk_size`` (not by which
    chunk_idx within the run); this keeps cross-variant numbers
    noise-comparable for the SAME ``chunk_size``. Numbers from different
    chunk_size values are NOT byte-comparable -- different RNG draws
    drive sampling. Variants needing bs<30 (distogram heads) thus must
    be compared against bs<30 numbers from the rest of the family.
    """
    N = raw["mask"].shape[0]
    kab_all: list[np.ndarray] = []
    rec_all: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        raw_chunk = {k: v[start:end] for k, v in raw.items()}
        torch.manual_seed(SEED)
        kab, rec = _eval_chunk(model, raw_chunk, kwargs=kwargs, device=device)
        kab_all.append(kab)
        rec_all.append(rec)
    kab = np.concatenate(kab_all) if kab_all else np.zeros(0, dtype=np.float32)
    rec = np.concatenate(rec_all) if rec_all else np.zeros(0, dtype=np.float32)
    elapsed = time.time() - t0
    rec_f = rec[~np.isnan(rec)]
    kab_f = kab[~np.isnan(kab)]
    rec_mean = float(rec_f.mean()) if len(rec_f) else float("nan")
    kab_mean = float(kab_f.mean()) if len(kab_f) else float("nan")
    n_chunks = (N + chunk_size - 1) // chunk_size
    log.info(
        "[%s]  kab=%.2f A  3DR=%.1f%%  (%.1fs over %d proteins x %d chunks of %d)",
        label, kab_mean, 100.0 * rec_mean, elapsed, N, n_chunks, chunk_size,
    )
    return {
        "label": label,
        "kabsch_mean": kab_mean,
        "three_di_recovery_mean": rec_mean,
        "kabsch_per_protein": [None if np.isnan(v) else float(v) for v in kab.tolist()],
        "three_di_recovery_per_protein": [None if np.isnan(v) else float(v) for v in rec.tolist()],
        "elapsed_seconds": elapsed,
        "chunk_size": int(chunk_size),
        "n_proteins": int(N),
    }


@torch.no_grad()
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=("Comma-separated list of run names to evaluate (filters RUNS). "
              "Names are matched as substrings, so `--only base,distill` runs "
              "the two `_base*` rows plus the `_distill` row. Default: all."),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=("Eval-time batch size (`data.batch_size` override). Lower this "
              "for variants whose decoder builds large pair tensors (e.g. "
              "the distogram-head variants need (B, L, L, 2D) which is "
              "32 GB at B=32, L=512, D=512 -- use 4 or 8 on a 22 GB A10G)."),
    )
    parser.add_argument(
        "--save-best-kab-dir",
        type=str,
        default=None,
        help=("If set, copy the lowest-Kabsch ckpt seen during this eval "
              "(across all variants in RUNS) into this directory along "
              "with the resolved model cfg and a `champion.json` meta blob. "
              "State persists across runs: re-running the script with the "
              "same dir reads the existing champion.json and only "
              "overwrites if a new variant beats it on Kab. Phase 6 and "
              "Phase 7 can share the same dir to track the global champion "
              "across both `last` and `best_val` checkpoints."),
    )
    args = parser.parse_args()
    champion = _KabChampion(Path(args.save_best_kab_dir) if args.save_best_kab_dir else None)

    if args.only is not None:
        keep = [s.strip() for s in args.only.split(",") if s.strip()]
        runs = [(n, e) for (n, e) in RUNS if any(k in n for k in keep)]
        if not runs:
            log.error("--only=%s matched no runs in RUNS; valid: %s",
                      args.only, [n for n, _ in RUNS])
            return
        log.info("--only=%s -> %d run(s): %s", args.only, len(runs), [n for n, _ in runs])
    else:
        runs = list(RUNS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s out=%s", device, OUT_DIR)
    log.info("WINNER (Phase 3c, no Phase 4): %s", WINNER)
    torch.manual_seed(SEED)

    val_pt = "/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/validation.pt"
    # Keep loader at 32 (val.pt has 30 proteins; this returns them all in
    # one batch). `args.batch_size` is the MODEL-side chunk inside
    # `_eval_one` -- separate knob, only constrains GPU memory.
    overrides = [
        "data.batch_size=32", "data.num_workers=0",
        f"data.path_to_datasets=[{val_pt},{val_pt},{val_pt}]",
        "data.cluster_file=null",
    ]

    base_cfg = _compose_cfg(runs[0][1], overrides)
    loader = _build_val_loader(base_cfg)
    batch = next(iter(loader))
    raw = {k: batch[k] for k in ["3di_states", "mask", "indices", "coords_res"]}
    lengths = raw["mask"].sum(-1).long().tolist()
    log.info("shared batch=%d lengths min/med/max=%d/%d/%d",
             len(lengths), min(lengths), int(np.median(lengths)), max(lengths))

    results: list[dict] = []
    for name, exp in runs:
        # Auto-resolve to the most-recently-modified runs/<timestamp>/.
        # For the base run this picks up the resume run dir.
        runs_root = SCRATCH / f"latent_generator_3di_input_{name}/runs"
        try:
            run_dir = max((p for p in runs_root.iterdir() if p.is_dir()),
                          key=lambda p: p.stat().st_mtime)
        except (FileNotFoundError, ValueError):
            log.warning("[%s] no runs/ dir -- skipping", name)
            continue

        last_ckpt = run_dir / "last.ckpt"
        if not last_ckpt.exists():
            log.warning("[%s] %s/last.ckpt missing -- skipping", name, run_dir.name)
            continue
        log.info("[%s] run_dir=%s last.ckpt mtime=%s",
                 name, run_dir.name, time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime(last_ckpt.stat().st_mtime)))

        cfg = _compose_cfg(exp, overrides)
        try:
            model = _load_model(cfg, last_ckpt, device)
        except Exception as e:
            log.warning("[%s] failed to load: %s", name, e)
            continue

        try:
            res = _eval_one(
                model, raw,
                label=f"{name}/last",
                kwargs=dict(WINNER),
                device=device,
                chunk_size=int(args.batch_size),
            )
        except torch.cuda.OutOfMemoryError as e:
            log.warning("[%s] OOM at chunk_size=%d -- skipping (try smaller --batch-size)",
                        name, int(args.batch_size))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        res["run"] = name
        res["ckpt_path"] = str(last_ckpt)
        res["ckpt_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(last_ckpt.stat().st_mtime))
        res["run_dir"] = run_dir.name
        results.append(res)

        champion.maybe_save(
            ckpt_src=last_ckpt, cfg=cfg,
            run_name=name, ckpt_tag="last",
            eval_record=res, run_dir_name=run_dir.name,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "winner_config": WINNER,
        "batch_size": int(len(lengths)),
        "lengths_min_med_max": [int(min(lengths)), int(np.median(lengths)), int(max(lengths))],
        "results": results,
    }
    out_path = OUT_DIR / "phase7_last_ckpt.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_path)

    print("\n" + "=" * 92)
    print("Phase 7 -- Phase 3c SDE winner (init=1.0, min_t=0.0) on `last.ckpt` per run")
    print("=" * 92)
    print(f"{'run':>40} | {'run_dir':>20} | {'last.ckpt mtime':>19} | {'Kab (A)':>8} | {'3DR (%)':>7}")
    print("-" * 110)
    for r in results:
        kab = "nan" if np.isnan(r["kabsch_mean"]) else f"{r['kabsch_mean']:.2f}"
        rec = "nan" if np.isnan(r["three_di_recovery_mean"]) else f"{100*r['three_di_recovery_mean']:.1f}%"
        print(f"{r['run']:>40} | {r['run_dir']:>20} | {r['ckpt_mtime']:>19} | {kab:>8} | {rec:>7}")


if __name__ == "__main__":
    main()
