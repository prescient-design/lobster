import dotenv
import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

import wandb
from lobster.cmdline._utils import instantiate_callbacks

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
