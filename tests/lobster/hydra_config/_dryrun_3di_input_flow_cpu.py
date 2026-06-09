"""CPU dry-run for the flow-matching 3Di-input training experiment.

Parallels ``_dryrun_3di_input_cpu.py`` for the regression variant.
Composes ``experiment=train_latent_generator_3di_input_flow`` with the PDB
data overlay swapped to use the tiny ``validation.pt`` (~3 MB) for all
three splits, instantiates the datamodule + flow model, pulls one batch,
runs:

- a single ``training_step`` (asserts the ``1/(1-t)^2``-reweighted loss is finite)
- ``sample(n_steps=2, sampling_mode='ode')`` end-to-end
- ``sample(n_steps=2, sampling_mode='sde', sc_scale_noise=0.1)`` end-to-end

Not part of pytest collection -- invoke directly:

    .venv/bin/python tests/lobster/hydra_config/_dryrun_3di_input_flow_cpu.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_ROOT = REPO_ROOT / "src" / "lobster" / "hydra_config"

DRY_DIR = REPO_ROOT / ".dryrun_3di_input_data"
SMALL_TRAIN = str(DRY_DIR / "train.pt")
SMALL_VAL = str(DRY_DIR / "val.pt")
SMALL_TEST = str(DRY_DIR / "test.pt")


def main() -> int:
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(HYDRA_ROOT),
            job_name="dryrun_3di_input_flow_cpu",
        ):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=train_latent_generator_3di_input_flow",
                    (f"data.path_to_datasets=[{SMALL_TRAIN},{SMALL_VAL},{SMALL_TEST}]"),
                    "data.batch_size=2",
                    "data.num_workers=0",
                ],
            )

        print(f"[compose] model._target_     = {cfg.model._target_}")
        print(f"[compose] data._target_      = {cfg.data._target_}")
        print(f"[compose] decoder._target_   = {cfg.model.decoder_factory.decoder_mapping.vit_decoder._target_}")
        print(f"[compose] interpolant        = {cfg.model.interpolant._target_}")
        print(f"[compose] sampling_mode      = {cfg.model.sampling_mode}")
        print(f"[compose] num_registers      = {cfg.model.decoder_factory.decoder_mapping.vit_decoder.num_registers}")

        print("[instantiate] datamodule ...")
        dm = instantiate(cfg.data)
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"[batch] keys = {sorted(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"[batch]   {k:>20s}: {tuple(v.shape)}  {v.dtype}")

        assert "3di_states" in batch
        assert "coords_res" in batch
        assert "mask" in batch
        assert "indices" in batch

        print("[instantiate] flow model ...")
        model = instantiate(cfg.model)

        # Tiny override so the CPU dry-run doesn't try a 50-step integration.
        model.n_sampling_steps = 2

        print("[training_step] dry-run on CPU ...")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad()
        out = model.training_step(batch, batch_idx=0)
        loss = out["loss"]
        x_recon = out["x_recon"]
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert "vit_decoder" in x_recon
        assert x_recon["vit_decoder"].shape == batch["coords_res"].shape, (
            f"x_recon shape {x_recon['vit_decoder'].shape} != coords_res {batch['coords_res'].shape}"
        )
        loss.backward()
        opt.step()
        print(f"[training_step] loss = {float(loss):.6f}, x_recon shape = {tuple(x_recon['vit_decoder'].shape)}")

        print("[sample] ODE, n_steps=2 ...")
        model.eval()
        states, seq_mask, residue_index = model.featurize(batch)
        x_samp_ode = model.sample(
            states=states,
            seq_mask=seq_mask,
            residue_index=residue_index,
            n_steps=2,
            sampling_mode="ode",
        )
        assert x_samp_ode.shape == batch["coords_res"].shape
        assert torch.isfinite(x_samp_ode).all(), "ODE sample is not finite"
        print(f"[sample/ode]  shape = {tuple(x_samp_ode.shape)}, finite=OK")

        # Verify CoM is approximately zero post-step
        m = seq_mask[..., None, None]
        com = (x_samp_ode * m).sum(dim=(1, 2)) / (seq_mask.sum(dim=-1).clamp_min(1.0) * x_samp_ode.shape[-2]).unsqueeze(
            -1
        )
        max_com = float(com.abs().max().item())
        print(f"[sample/ode]  max |CoM| per sample = {max_com:.2e}  (should be ~0)")

        # SDE path (delta #4): moco's step_score_stochastic refuses to run
        # when the interpolant has an augmentation_type set (it cannot convert
        # the vector field to a score when the noise has been Kabsch-rotated).
        # The dispatch hook is still tested -- we temporarily clear
        # augmentation_type so the SDE branch exercises end-to-end.
        print("[sample] SDE, n_steps=2, sc_noise=0.1 (augmentation_type temporarily=None) ...")
        saved_aug = model.flow_matcher.augmentation_type
        model.flow_matcher.augmentation_type = None
        try:
            x_samp_sde = model.sample(
                states=states,
                seq_mask=seq_mask,
                residue_index=residue_index,
                n_steps=2,
                sampling_mode="sde",
                sc_scale_noise=0.1,
                sc_scale_score=0.0,
            )
        finally:
            model.flow_matcher.augmentation_type = saved_aug
        assert x_samp_sde.shape == batch["coords_res"].shape
        assert torch.isfinite(x_samp_sde).all(), "SDE sample is not finite"
        print(f"[sample/sde]  shape = {tuple(x_samp_sde.shape)}, finite=OK")

        print("\nOK -- 3Di-input flow dry-run passed on CPU.")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    sys.exit(main())
