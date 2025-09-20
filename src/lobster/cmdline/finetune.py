import dotenv
import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb


@hydra.main(version_base=None, config_path="../hydra_config", config_name="finetune")
def finetune(cfg: DictConfig) -> tuple[pl.LightningModule, pl.LightningDataModule, list[pl.Callback]]:
    dotenv.load_dotenv(".env")

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    datamodule = hydra.utils.instantiate(cfg.data)

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Missing 'model' section in config")

    # Generic Hydra instantiation of the model (and nested encoder/config)
    model = hydra.utils.instantiate(model_cfg)

    # Compile if requested
    if cfg.compile and hasattr(model, "compile"):
        model.compile()

    # Logger
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

    # Callbacks and trainer
    callbacks_cfg = cfg.get("callbacks")
    callbacks = None
    if callbacks_cfg:
        if isinstance(callbacks_cfg, DictConfig):
            callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
        elif isinstance(callbacks_cfg, (list, ListConfig)):
            callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg]
        else:
            callbacks = [hydra.utils.instantiate(callbacks_cfg)]

    # Instantiate Trainer from config only; attach Python objects afterward to avoid OmegaConf errors
    trainer = hydra.utils.instantiate(cfg.trainer)
    if callbacks:
        # extend preserves any callbacks defined via config
        trainer.callbacks.extend(callbacks)
    if logger is not None:
        trainer.logger = logger

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=cfg.logger.get("allow_val_change"))

    # Fit/Test
    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.model.get("ckpt_path"))
        if cfg.run_test:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        wandb.finish()

    return model, datamodule, callbacks  # type: ignore[return-value]


if __name__ == "__main__":
    finetune()
