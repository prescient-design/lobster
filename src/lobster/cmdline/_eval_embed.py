import hydra
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from lobster.cmdline._utils import instantiate_callbacks

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def eval_embed(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.to(device)

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

    logger = hydra.utils.instantiate(cfg.logger)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    print(trainer)

    if cfg.model.get("pca_components") is not None:
        print("Fitting PCA on embeddings from training set...")
        model.fit_pca(datamodule.train_dataloader())

    trainer.fit(model, datamodule=datamodule)

    if cfg.run_test:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()
    return True
