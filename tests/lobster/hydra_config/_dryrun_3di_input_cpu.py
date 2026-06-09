"""CPU dry-run for the 3Di-input training experiment.

Composes ``experiment=train_latent_generator_3di_input`` with the PDB data
overlay swapped to use the tiny ``validation.pt`` (~3 MB) for all three
splits, instantiates the datamodule + model, pulls one batch through
``Structure3diTransform`` + collate, and runs a single
``training_step`` on CPU.

This catches:
  * data overlay path resolution + ``StructureDataset`` schema parsing,
  * the ``Structure3diTransform`` chain emitting ``3di_states``,
  * the collate function producing the keys ``Tokenizer3diInput.featurize``
    expects,
  * the ``ViTDecoder`` ``(indexed=True, struc_token_codebook_size=21)``
    integer-ID embedding path,
  * the loss factory wiring (``L2Loss`` + ``PairWiseL2Loss``).

Not part of the pytest collection — invoke directly:

    .venv/bin/python tests/lobster/hydra_config/_dryrun_3di_input_cpu.py

The slurm launcher should still be preceded by a real single-GPU
``lobster_train ... trainer.max_steps=1`` on a GPU node, since some
ops (e.g. flash-attn paths) silently fall back on CPU.
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

# Small (~3 MB) copies of ``validation.pt`` renamed ``train.pt``/``val.pt``/
# ``test.pt`` so ``StructureLightningDataModule.setup`` takes the precomputed-
# splits code path (the iid-split path produces ``Subset`` objects that
# ``train_dataloader`` cannot handle). Copies live in the workspace because
# ``torch_geometric.data.Dataset.__init__`` rewrites ``pre_transform.pt``
# next to the input, which is rejected on the read-only ``/cv/data`` mount.
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
            job_name="dryrun_3di_input_cpu",
        ):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=train_latent_generator_3di_input",
                    (f"data.path_to_datasets=[{SMALL_TRAIN},{SMALL_VAL},{SMALL_TEST}]"),
                    "data.batch_size=2",
                    "data.num_workers=0",
                ],
            )

        print(f"[compose] model._target_     = {cfg.model._target_}")
        print(f"[compose] data._target_      = {cfg.data._target_}")
        print(f"[compose] data.path_to_datasets[0] = {cfg.data.path_to_datasets[0]}")
        print(f"[compose] data.batch_size    = {cfg.data.batch_size}")

        print("[instantiate] datamodule ...")
        dm = instantiate(cfg.data)
        print(f"[dm] transform_fn = {dm._transform_fn}")
        if hasattr(dm._transform_fn, "transforms"):
            for t in dm._transform_fn.transforms:
                print(f"[dm]   - {type(t).__name__}")
        dm.setup("fit")
        train_loader = dm.train_dataloader()

        # Peek at what the underlying StructureDataset actually returns
        # PRE-collate (so we can localise where 3di_states gets lost).
        inner = dm._train_dataset
        while hasattr(inner, "datasets"):
            inner = inner.datasets[0]
        while hasattr(inner, "dataset") and not hasattr(inner, "transform"):
            inner = inner.dataset
        print(f"[dataset] type = {type(inner).__name__}")
        print(f"[dataset] transform = {inner.transform}")
        for idx in range(5):
            try:
                raw = inner[idx]
                keys = list(raw.keys()) if hasattr(raw, "keys") else list(raw.index)
                n = raw["coords_res"].shape[0] if "coords_res" in keys else "?"
                print(f"[dataset[{idx}]] n={n} has_3di={'3di_states' in keys} keys={keys}")
            except Exception as e:
                print(f"[dataset[{idx}]] FAILED: {type(e).__name__}: {e}")

        # Pull a batch list directly from the dataset (no DataLoader) and feed
        # it to the collate_fn the dataloader would use, so we localise where
        # 3di_states is dropped.
        sample_list = [inner[i] for i in range(2)]
        print(
            f"[sample_list[0]] has_3di? "
            f"{'3di_states' in (sample_list[0].keys() if hasattr(sample_list[0], 'keys') else sample_list[0].index)}"
        )
        from lobster.data._collate_structure import collate_fn_backbone_binder_target

        manual_batch = collate_fn_backbone_binder_target(sample_list)
        print(f"[manual_collate] keys = {sorted(manual_batch.keys())}")

        batch = next(iter(train_loader))
        print(f"[batch] keys = {sorted(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"[batch]   {k:>20s}: {tuple(v.shape)}  {v.dtype}")

        assert "3di_states" in batch, "Structure3diTransform did not emit 3di_states"
        assert "coords_res" in batch, "StructureBackboneTransform did not emit coords_res"
        assert "mask" in batch, "StructureBackboneTransform did not emit mask"
        assert "indices" in batch, "StructureBackboneTransform did not emit indices"

        print("[instantiate] model ...")
        model = instantiate(cfg.model)
        model.eval()

        print("[forward] running model.forward(batch) ...")
        with torch.no_grad():
            coords_pred = model(batch)
        print(f"[forward] coords_pred shape = {tuple(coords_pred.shape)}")

        print("[training_step] dry-run on CPU ...")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad()
        loss = model.training_step(batch, batch_idx=0)
        if isinstance(loss, dict):
            loss = loss["loss"]
        loss.backward()
        opt.step()
        print(f"[training_step] loss = {float(loss):.6f}")

        print("\nOK — 3Di-input dry-run passed on CPU.")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    sys.exit(main())
