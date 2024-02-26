import hydra
import lightning.pytorch as pl
import torch
import wandb
import os
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from prescient.transforms import Normalize

from prescient_plm.cmdline._utils import instantiate_callbacks


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def train(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # wandb_api_key = os.getenv("WANDB_API_KEY")
    # if wandb_api_key is not None:
    wandb.login(host="https://genentech.wandb.io/")
            # key=wandb_api_key)
    wandb.init(
        config=log_cfg,  # type: ignore[arg-type]
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=cfg.logger.group,
        notes=cfg.logger.notes,
        tags=cfg.logger.tags,
        name=cfg.logger.get("name"),
    )

    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        logger = None

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    # plugins = instantiate_plugins(cfg.get("plugins"))

    # trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, pl.loggers.WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg})

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)

        if cfg.run_test:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()
    return True
