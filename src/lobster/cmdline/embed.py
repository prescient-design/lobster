import hydra
import lightning.pytorch as pl
import pandas as pd
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from lobster.model._utils import model_typer

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../hydra_config", config_name="embed")
def embed(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="predict")
    model = model_typer[cfg.model_type].load_from_checkpoint(cfg.model.ckpt_path)
    model.to(device)

    trainer = pl.Trainer(logger=False, accelerator=model.device.type)

    with torch.inference_mode():
        preds = trainer.predict(model, datamodule.predict_dataloader())
    preds_flat = torch.hstack(preds)
    embedding_df = pd.DataFrame(preds_flat.numpy(), columns=["preds"])
    # embedding_df = pd.concat(preds, ignore_index=True, axis=0)

    # convert column dtypes for json encoding
    embedding_df.columns = embedding_df.columns.astype(str)
    embedding_df.to_csv(cfg.pred_output, index=False)

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        notes=f"embedding sequences from {cfg.data.source}",
        config=vars(cfg),
    )

    wandb.log({"embeddings": wandb.Table(dataframe=embedding_df)})

    wandb.finish()
