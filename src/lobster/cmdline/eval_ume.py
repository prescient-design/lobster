import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf

from lobster.evaluation import evaluate_ume_models


@hydra.main(config_path="../hydra_configs", config_name="evaluate_ume")
def eval_ume(cfg: DictConfig) -> None:
    """Run Ume model evaluation from Hydra config."""
    print(OmegaConf.to_yaml(cfg))

    evaluate_ume_models(
        checkpoints=cfg.models,
        output_dir=hydra.utils.to_absolute_path(cfg.output_dir),
        batch_size=cfg.batch_size,
        max_samples=cfg.get("max_samples"),
        find_closest_checkpoints=cfg.get("find_closest_checkpoints", False),
        find_evenly_spaced=cfg.get("find_evenly_spaced", False),
    )
