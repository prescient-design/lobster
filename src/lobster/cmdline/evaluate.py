import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf

from lobster.evaluation import evaluate_model_with_callbacks


@hydra.main(config_path="../hydra_configs", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """Run model evaluation with specified callbacks."""
    print(OmegaConf.to_yaml(cfg))

    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]

    if cfg.model_path is not None:
        model_cls = hydra.utils.get_class(cfg.model)
        model = model_cls.load_from_checkpoint(cfg.model_path)
    else:
        model = hydra.utils.instantiate(cfg.model)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

    evaluate_model_with_callbacks(
        callbacks=callbacks,
        model=model,
        dataloader=dataloader,
        output_dir=cfg.output_dir,
    )
