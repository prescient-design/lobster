"""Phase 6 -- re-apply Phase 3c SDE winner to LATEST val-best ckpts.

Phase 5b (yesterday) ran the Phase 3c winner (init=1.0, min_t=0.0) across
the velocity family on both `last.ckpt` and the lowest-val-loss ckpt.
That established `best_val` as the right read across the family
(`base/best_val` recovered from `last`'s 20.49 A to 14.22 A).

Phase 6 re-runs the SAME config on whatever val-best ckpt is on disk
*now* -- the small-net runs have been training for another ~24 h and
should have a new minimum, e.g.:

    flow_nokabsch                e1577 v=0.6376  ->  e2092 v=0.6336
    flow_nokabsch_velocity       e1916 v=0.5384  ->  e2241 v=0.5373
    flow_nokabsch_velocity_selfcond e1410 v=0.5556 -> e1661 v=0.5442
    flow_nokabsch_velocity_mask3di  e1057 v=0.6166 -> e1236 v=0.6045
    flow_nokabsch_velocity_base_selfcond e51  v=?  -> e136  v=0.7755

The base run's val-best is unchanged from Phase 5b (the resume
relaunch hasn't passed a val cycle yet); it's included only as a sanity
check that the harness reproduces the 14.22 A number.

Skipped: re-sweeping init x min_t -- Phase 4's (0.5, 0.1) tune
transferred to the small-net family but regressed `base` (+3.6 A in
Phase 5), so we treat that pair as non-portable and stick with the
Phase 3c defaults.

Output: `.compare_runs/inference_sweep_velocity/phase6_latest_best_val.json`

    uv run python scripts/_diag_inference_apply_winner_velocity_family_phase6.py
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
log = logging.getLogger("apply_winner_phase6")

RUNS = [
    ("flow_nokabsch",
     "train_latent_generator_3di_input_flow_nokabsch",
     "latent_generator_3di_input_flow_nokabsch/runs/2026-06-07T20-44-24"),
    ("flow_nokabsch_velocity",
     "train_latent_generator_3di_input_flow_nokabsch_velocity",
     "latent_generator_3di_input_flow_nokabsch_velocity/runs/2026-06-07T20-54-26"),
    ("flow_nokabsch_velocity_selfcond",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond",
     "latent_generator_3di_input_flow_nokabsch_velocity_selfcond/runs/2026-06-08T19-40-54"),
    ("flow_nokabsch_velocity_mask3di",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_mask3di",
     "latent_generator_3di_input_flow_nokabsch_velocity_mask3di/runs/2026-06-09T00-06-30"),
    ("flow_nokabsch_velocity_base",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base",
     "latent_generator_3di_input_flow_nokabsch_velocity_base/runs/2026-06-08T19-25-48"),
    ("flow_nokabsch_velocity_base_selfcond",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base_selfcond",
     "latent_generator_3di_input_flow_nokabsch_velocity_base_selfcond/runs/2026-06-10T11-58-16"),
    # Distill resumed from selfcond e1503 onto wider UME mix; its in-run
    # val_loss is on AFDB CAMEO (NOT comparable to PDB val_loss column),
    # but we use the selfcond experiment cfg below so the val LOADER is
    # the same PDB batch as the rest of the family.
    ("flow_nokabsch_velocity_selfcond_distill",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond",
     "latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill/runs/2026-06-11T03-53-16"),
    # Distogram + distogram_3di need their own experiment cfgs (extra heads
    # in the model arch). For these we want the run_dir to auto-resolve to
    # the latest training run; we don't hardcode it here. Phase 6 picks
    # `_best_val_loss_ckpt(SCRATCH/rd)`, so we pass the latest run dir.
    ("flow_nokabsch_velocity_selfcond_distogram",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram",
     None),  # rd resolved in main() below to most-recent runs/<ts>
    ("flow_nokabsch_velocity_selfcond_distogram_3di",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di",
     None),
    # 3Di-CE-from-coords variants (Step Y/Y2): own experiment cfg required
    # because the ckpt state_dict carries `mini3di_torch.*` buffers when
    # `aux_3di_coord_ce_weight>0`.
    ("flow_nokabsch_velocity_selfcond_3di_coord_ce",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce",
     None),
    ("flow_nokabsch_velocity_base_3di_coord_ce",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce",
     None),
    # Step Y2-W: weight=0.5 sibling of base_3di_coord_ce. Same arch.
    ("flow_nokabsch_velocity_base_3di_coord_ce_w0p5",
     "train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0p5",
     None),
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
      - ``champion_eval_record.json``  full eval payload (per-protein arrays etc.)

    Persistence: on ``__init__`` we read ``champion.json`` if it exists and
    seed ``best_kab`` from it; ``maybe_save`` only overwrites when the new
    Kab is strictly lower. Phase 6 (best_val) and phase 7 (last) can share
    one ``save_dir`` to track the global champion across both ckpt tags.
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
        shutil.copy(Path(ckpt_src).resolve(), self.save_dir / "champion.ckpt")
        OmegaConf.save(cfg.model, self.save_dir / "champion_model_cfg.yaml")
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
    """Iterate ``raw`` in chunks of ``chunk_size`` proteins, call the model
    on each chunk, aggregate per-protein Kabsch / 3Di-recovery arrays.

    Seed is reset to ``SEED`` at the start of each chunk so cross-variant
    numbers are noise-comparable for the SAME ``chunk_size``. Numbers from
    different ``chunk_size`` values are NOT byte-comparable across the
    family (different RNG draws drive sampling).
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
              "State persists across runs and across Phase 6/7 if both "
              "scripts share the same dir -- only overwrites when a new "
              "ckpt strictly beats the existing champion's Kabsch."),
    )
    args = parser.parse_args()
    champion = _KabChampion(Path(args.save_best_kab_dir) if args.save_best_kab_dir else None)

    if args.only is not None:
        keep = [s.strip() for s in args.only.split(",") if s.strip()]
        runs = [r for r in RUNS if any(k in r[0] for k in keep)]
        if not runs:
            log.error("--only=%s matched no runs in RUNS; valid: %s",
                      args.only, [r[0] for r in RUNS])
            return
        log.info("--only=%s -> %d run(s): %s", args.only, len(runs), [r[0] for r in runs])
    else:
        runs = list(RUNS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s out=%s", device, OUT_DIR)
    log.info("WINNER (Phase 3c, no Phase 4): %s", WINNER)
    torch.manual_seed(SEED)

    val_pt = "/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/validation.pt"
    # Loader gives all 30 val proteins in one batch; chunking happens
    # inside `_eval_one` based on `--batch-size` (the model-side size).
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
    for name, exp, rd in runs:
        if rd is None:
            # Auto-resolve to the most-recently-modified runs/<timestamp>/
            # under the conventional scratch directory for this run name.
            runs_root = SCRATCH / f"latent_generator_3di_input_{name}/runs"
            try:
                run_dir = max((p for p in runs_root.iterdir() if p.is_dir()),
                              key=lambda p: p.stat().st_mtime)
            except (FileNotFoundError, ValueError):
                log.warning("[%s] no runs/ dir under %s -- skipping", name, runs_root)
                continue
        else:
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

        try:
            res = _eval_one(
                model, raw,
                label=f"{name}/best_val",
                kwargs=dict(WINNER),
                device=device,
                chunk_size=int(args.batch_size),
            )
        except torch.cuda.OutOfMemoryError:
            log.warning("[%s] OOM at chunk_size=%d -- skipping (try smaller --batch-size)",
                        name, int(args.batch_size))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        res["run"] = name
        res["ckpt_path"] = str(best_ckpt)
        res["ckpt_val_loss"] = best_val
        res["ckpt_name"] = best_ckpt.name
        results.append(res)

        champion.maybe_save(
            ckpt_src=best_ckpt, cfg=cfg,
            run_name=name, ckpt_tag="best_val",
            eval_record=res, run_dir_name=run_dir.name,
            ckpt_val_loss=best_val,
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
    out_path = OUT_DIR / "phase6_latest_best_val.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_path)

    print("\n" + "=" * 92)
    print("Phase 6 -- Phase 3c SDE winner (init=1.0, min_t=0.0) on LATEST val-best ckpt per run")
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
