"""CPU dry-run for the self-conditioned 3Di-input flow experiment (Step F).

Parallels ``_dryrun_3di_input_flow_cpu.py`` but exercises the self-cond
pathway:

- training_step runs many iterations to almost-certainly hit the
  ``selfcond_train_prob=0.5`` warm-forward branch at least once;
- a sample() call confirms that ``x_selfcond`` is carried across the
  ODE steps without dtype / shape / device drift;
- a second training_step with ``model.selfcond_train_prob = 1.0``
  forces the warm-forward branch to fire deterministically, isolating
  the self-cond gradient path so a future regression there is caught.

Not part of pytest collection -- invoke directly:

    .venv/bin/python tests/lobster/hydra_config/_dryrun_3di_input_flow_selfcond_cpu.py
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
            job_name="dryrun_3di_input_flow_selfcond_cpu",
        ):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=train_latent_generator_3di_input_flow_selfcond",
                    f"data.path_to_datasets=[{SMALL_TRAIN},{SMALL_VAL},{SMALL_TEST}]",
                    "data.batch_size=2",
                    "data.num_workers=0",
                ],
            )

        decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
        print(f"[compose] model._target_              = {cfg.model._target_}")
        print(f"[compose] decoder._target_            = {decoder_cfg._target_}")
        print(f"[compose] decoder.use_self_conditioning = {decoder_cfg.use_self_conditioning}")
        print(f"[compose] model.use_self_conditioning   = {cfg.model.use_self_conditioning}")
        print(f"[compose] model.selfcond_train_prob     = {cfg.model.selfcond_train_prob}")

        assert decoder_cfg.use_self_conditioning is True
        assert cfg.model.use_self_conditioning is True

        print("[instantiate] datamodule ...")
        dm = instantiate(cfg.data)
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"[batch] keys = {sorted(batch.keys())}")

        print("[instantiate] flow model (selfcond=True) ...")
        model = instantiate(cfg.model)
        model.n_sampling_steps = 2

        # Defensive: confirm the second zero-init projection is built and
        # actually starts at zero (so the first forward is mathematically
        # equivalent to a non-self-cond model -- key safety property).
        # Snapshot weights with `.clone().detach()`; `nn.Parameter.weight`
        # is a live reference and `opt.step()` would otherwise mutate any
        # plain alias in place, masking gradient regressions.
        _, decoder = model._decoder
        assert decoder.coord_in_proj_selfcond is not None, (
            "decoder.coord_in_proj_selfcond not built despite use_self_conditioning=True"
        )
        sc_w0 = decoder.coord_in_proj_selfcond.weight.detach().clone()
        sc_b0 = decoder.coord_in_proj_selfcond.bias.detach().clone()
        assert torch.all(sc_w0 == 0.0) and torch.all(sc_b0 == 0.0), (
            "coord_in_proj_selfcond must be zero-init for the safety property; "
            f"weight max abs = {sc_w0.abs().max():.3e}, bias max abs = {sc_b0.abs().max():.3e}"
        )
        print("[init] coord_in_proj_selfcond verified zero-init.")

        print("[training_step] dry-run on CPU (selfcond_train_prob = 0.5) ...")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # Run a handful of training steps. With p=0.5 over 8 iterations,
        # the probability that the warm-forward branch never fires is
        # 0.5**8 ~= 0.4%. Seeding the loop pins this to a stable trace.
        torch.manual_seed(0)
        for it in range(8):
            opt.zero_grad()
            out = model.training_step(batch, batch_idx=it)
            loss = out["loss"]
            assert torch.isfinite(loss), f"loss not finite at iter {it}: {loss}"
            loss.backward()
            opt.step()
            print(f"  [it={it}] loss = {float(loss):.6f}")

        # Now force the self-cond branch on every step and check the
        # gradient path is connected. We assert two independent things:
        #
        # (1) on the first forced step, `coord_in_proj_selfcond.weight.grad`
        #     comes out non-zero right after `backward()` (proves gradient
        #     reaches the parameter at all);
        # (2) after `opt.step()` runs across the whole forced loop the
        #     parameter has actually drifted from its zero-init snapshot
        #     (proves the optimizer is also wired correctly).
        print("[training_step] forced self-cond (selfcond_train_prob = 1.0) ...")
        model.selfcond_train_prob = 1.0
        torch.manual_seed(1)
        for it in range(5):
            opt.zero_grad()
            out = model.training_step(batch, batch_idx=100 + it)
            loss = out["loss"]
            assert torch.isfinite(loss), f"loss not finite (forced) at iter {it}: {loss}"
            loss.backward()
            if it == 0:
                # (1) gradient-reaches-parameter check, before the
                # optimizer has had a chance to write anything.
                g = decoder.coord_in_proj_selfcond.weight.grad
                gb = decoder.coord_in_proj_selfcond.bias.grad
                assert g is not None and gb is not None, (
                    "coord_in_proj_selfcond grads are None after backward(); "
                    "the parameter is detached from the computation graph."
                )
                grad_max = float(g.abs().max().item())
                grad_b_max = float(gb.abs().max().item())
                print(f"  [grad/it0] |dW| max = {grad_max:.3e}, |db| max = {grad_b_max:.3e}")
                assert grad_max > 0.0, (
                    "coord_in_proj_selfcond.weight.grad is identically zero on "
                    "the first forced self-cond step. Likely cause: x_selfcond "
                    "ended up all-zero (e.g. mask/padding bug) or the second "
                    "projection's output is being clobbered."
                )
            opt.step()
            print(f"  [forced it={it}] loss = {float(loss):.6f}")
        # (2) parameter actually moved.
        sc_w_after = decoder.coord_in_proj_selfcond.weight.detach()
        moved = float((sc_w_after - sc_w0).abs().max().item())
        print(f"[opt] coord_in_proj_selfcond.weight |delta| max vs zero-init = {moved:.3e}")
        assert moved > 0.0, (
            "coord_in_proj_selfcond.weight matches its zero-init snapshot after "
            "5 forced self-cond training steps -- the optimizer is not stepping "
            "this parameter."
        )

        print("[sample] ODE, n_steps=2, self-cond chain on ...")
        model.eval()
        states, seq_mask, residue_index = model.featurize(batch)
        x_samp = model.sample(
            states=states,
            seq_mask=seq_mask,
            residue_index=residue_index,
            n_steps=2,
            sampling_mode="ode",
        )
        assert x_samp.shape == batch["coords_res"].shape
        assert torch.isfinite(x_samp).all(), "ODE sample not finite"
        m = seq_mask[..., None, None]
        com = (x_samp * m).sum(dim=(1, 2)) / (seq_mask.sum(dim=-1).clamp_min(1.0) * x_samp.shape[-2]).unsqueeze(-1)
        max_com = float(com.abs().max().item())
        print(f"[sample/ode]  shape = {tuple(x_samp.shape)}, max |CoM| = {max_com:.2e}")

        print("\nOK -- 3Di-input flow self-cond dry-run passed on CPU.")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    sys.exit(main())
