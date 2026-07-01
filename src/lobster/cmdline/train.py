import logging

import dotenv
import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

import wandb
from lobster.cmdline._utils import instantiate_callbacks

log = logging.getLogger(__name__)

dotenv.load_dotenv(".env")


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def train(cfg: DictConfig) -> tuple[pl.LightningModule, pl.LightningDataModule, list[pl.Callback]]:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    datamodule = hydra.utils.instantiate(cfg.data)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    if cfg.compile:
        model.compile()

    # Fresh-finetune from a pretrained checkpoint (separate from Lightning's
    # `ckpt_path` resume path). Loads model weights only — optimizer state,
    # global step, and LR schedule all start fresh. Use this when you want
    # to take an existing checkpoint's representations into a new training
    # objective / data mix (e.g. leflur-base -> Pinder + epitope conditioning).
    #
    # Resolution: short names like "leflur-base" go through
    # `lobster.model.leflur.checkpoints.resolve_checkpoint` (which knows
    # the HF registry); anything else is treated as a literal path.
    # Mutually exclusive with `cfg.model.ckpt_path` (which is the Lightning
    # full-resume path; both set => error to avoid silent precedence).
    pretrained_ckpt = cfg.model.get("pretrained_ckpt") if hasattr(cfg, "model") else None
    if pretrained_ckpt is not None:
        if cfg.model.get("ckpt_path") is not None:
            raise ValueError(
                "Set EITHER `model.pretrained_ckpt` (load weights, fresh "
                "optimizer/step) OR `model.ckpt_path` (full Lightning resume), "
                "not both."
            )
        from lobster.model.leflur.checkpoints import resolve_checkpoint

        resolved = resolve_checkpoint(pretrained_ckpt)
        log.info(f"Loading pretrained weights from {pretrained_ckpt!r} -> {resolved}")
        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        result = model.load_state_dict(ckpt["state_dict"], strict=False)
        log.info(f"Pretrained load: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
        if result.missing_keys:
            log.info(f"  first 5 missing: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            log.info(f"  first 5 unexpected: {result.unexpected_keys[:5]}")

        # Optionally zero out the encoder's conditioning_embedding on load.
        # Rationale: leflur-base trained with `cond_percentage=0` so the
        # all-zero conditioning input never gradient-shaped this layer; its
        # weights are the init (~N(0, 0.5)). Feeding non-zero epitope
        # tensors into that random direction injects a large arbitrary
        # signal that the model has to learn to ignore before it can learn
        # what to use. Zeroing the layer on load makes step 0 behave
        # identically to leflur-base; gradient flow then carves a
        # meaningful direction. Set false to keep the pretrained init for
        # an A/B comparison.
        if cfg.model.get("zero_init_conditioning_on_load", False):
            if hasattr(model, "encoder") and hasattr(model.encoder, "conditioning_embedding"):
                with torch.no_grad():
                    model.encoder.conditioning_embedding.weight.zero_()
                    if model.encoder.conditioning_embedding.bias is not None:
                        model.encoder.conditioning_embedding.bias.zero_()
                log.info("Zero-initialized encoder.conditioning_embedding on pretrained load")
            else:
                log.warning(
                    "zero_init_conditioning_on_load=True but model has no encoder.conditioning_embedding; skipping."
                )

        # Zero-init the per-residue chain-id embedding (Commit C). Same logic
        # as zero_init_conditioning_on_load: leflur-base was trained without
        # this layer, so the default random init injects an arbitrary direction
        # whenever chain ids != 0 fire. Zeroing here makes step 0 behave
        # identically to leflur-base; gradient flow then carves chain-1 / chain-2
        # rows. The `padding_idx=0` row stays frozen at zero automatically.
        if cfg.model.get("zero_init_chain_embedding_on_load", False):
            if (
                hasattr(model, "encoder")
                and hasattr(model.encoder, "chain_embedding")
                and model.encoder.chain_embedding is not None
            ):
                with torch.no_grad():
                    model.encoder.chain_embedding.weight.zero_()
                log.info("Zero-initialized encoder.chain_embedding on pretrained load")
            else:
                log.warning(
                    "zero_init_chain_embedding_on_load=True but model has no encoder.chain_embedding; skipping."
                )

    if not cfg.dryrun and rank_zero_only.rank == 0:
        logger = hydra.utils.instantiate(cfg.logger)

        if isinstance(logger, WandbLogger):
            wandb.init(
                config=log_cfg,  # type: ignore[arg-type]
                project=cfg.logger.project,
                entity=cfg.logger.entity,
                group=cfg.logger.group,
                notes=cfg.logger.notes,
                tags=cfg.logger.tags,
                name=cfg.logger.get("name"),
                resume=cfg.logger.get("resume"),
                id=cfg.logger.get("id"),
                allow_val_change=cfg.logger.get("allow_val_change"),
            )
    else:
        logger = None

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=cfg.logger.get("allow_val_change"))

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)

        if cfg.run_test:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        wandb.finish()

    return model, datamodule, callbacks
