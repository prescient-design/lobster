import hydra
import lightning.pytorch as pl
import pandas as pd
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../hydra_config", config_name="embed")
def predict(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="predict")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_from_checkpoint(cfg.model.ckpt_path)  # might be model._target_
    model.to(device)

    trainer = pl.Trainer(logger=False, accelerator=model.device.type)

    with torch.inference_mode():
        preds = trainer.predict(model, datamodule.predict_dataloader())
    embedding_df = pd.concat(preds, ignore_index=True, axis=0)

    # convert column dtypes for json encoding
    embedding_df.columns = embedding_df.columns.astype(str)

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        notes=f"predictions from {cfg.data.source}",
        config=vars(cfg),
    )

    wandb.log({"predictions": wandb.Table(dataframe=embedding_df)})

    wandb.finish()
