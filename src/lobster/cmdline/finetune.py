import dotenv
import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb
from lobster.model._ume_property_regression import UMEPropertyRegression, UMEPropertyRegressionConfig
from lobster.model._ume import UME
from lobster.post_train.unfreezing import set_unfrozen_layers


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def finetune(cfg: DictConfig) -> tuple[pl.LightningModule, pl.LightningDataModule, list[pl.Callback]]:
    dotenv.load_dotenv(".env")

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # Instantiate arbitrary pre-setup (seed, plugins, etc.)
    hydra.utils.instantiate(cfg.setup)

    # Data module (expecting our gRED finetune DataModule config)
    datamodule = hydra.utils.instantiate(cfg.data)

    # Model: construct UME and head via Hydra targets while enforcing required params
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Missing 'model' section in config")

    ume_cfg = model_cfg.get("ume")
    if ume_cfg is None:
        raise ValueError("Missing 'model.ume' section in config")

    # Require model_name and use_flash_attn with no defaults
    if "model_name" not in ume_cfg or ume_cfg.get("model_name") in (None, ""):
        raise ValueError("'model.ume.model_name' is required and must be non-empty")
    if "use_flash_attn" not in ume_cfg or ume_cfg.get("use_flash_attn") is None:
        raise ValueError("'model.ume.use_flash_attn' is required and must be set to true/false")

    model_name = ume_cfg.get("model_name")
    use_flash_attn = bool(ume_cfg.get("use_flash_attn"))
    cache_dir = ume_cfg.get("cache_dir")

    # Instantiate encoder
    ume = UME.from_pretrained(model_name=model_name, use_flash_attn=use_flash_attn, cache_dir=cache_dir)

    # Instantiate head config via Hydra to keep YAML declarative and swappable
    cfg_src = model_cfg.get("config")
    if cfg_src is None or "_target_" not in cfg_src:
        raise ValueError("'model.config' must be provided with a Hydra _target_ for the head config class")
    head_cfg = hydra.utils.instantiate(cfg_src)

    # Instantiate model class via Hydra target, injecting built ume and head_cfg
    class_spec = model_cfg.get("class")
    if class_spec is None or "_target_" not in class_spec:
        raise ValueError("'model.class' must be provided with a Hydra _target_ for the model class")
    target_str = class_spec.get("_target_")
    model_cls = hydra.utils.get_class(target_str)
    model = model_cls(ume=ume, config=head_cfg)

    # Optional: configurable unfreezing via single parameter
    if isinstance(model, UMEPropertyRegression):
        unfreeze_cfg = cfg.get("unfreezing")
        if unfreeze_cfg is not None:
            num_layers = unfreeze_cfg.get("num_layers")
            if num_layers is not None:
                set_unfrozen_layers(model.ume, int(num_layers))

    # Compile if requested
    if cfg.compile:
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


