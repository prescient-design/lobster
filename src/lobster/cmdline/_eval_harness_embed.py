import hydra
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

device = "cuda" if torch.cuda.is_available() else "cpu"


def instantiate_model() -> LightningModule:
    pass


def get_embeddings(model: LightningModule, batch: None) -> torch.Tensor:
    pass


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def eval_embed(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    pass
