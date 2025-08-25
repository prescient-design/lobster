import logging

import hydra
import hydra.utils
import torch
from omegaconf import DictConfig, OmegaConf

from lobster.evaluation import evaluate_model_with_callbacks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../hydra_config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """Run model evaluation with specified callbacks."""

    logger.info("Starting model evaluation with callbacks")
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))

    logger.info("Instantiating callbacks...")
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]

    logger.info("Instantiating model...")
    if hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        logger.info(f"✓ Loading model from checkpoint: {cfg.model.ckpt_path}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        model = model_cls.load_from_checkpoint(cfg.model.ckpt_path)
        logger.info("✓ Successfully loaded model from checkpoint")
    else:
        logger.info("⚠️  No checkpoint path found, instantiating fresh model")
        model = hydra.utils.instantiate(cfg.model)
        logger.info("⚠️  Using randomly initialized model weights")

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

    evaluate_model_with_callbacks(
        callbacks=callbacks,
        model=model,
        dataloader=dataloader,
        output_dir=cfg.output_dir,
        metadata=OmegaConf.to_container(cfg, resolve=True),
    )
