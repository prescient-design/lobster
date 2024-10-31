import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        print("[instantiate_callbacks] No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("[instantiate_callbacks] Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"[instantiate_callbacks] Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_plugins(plugins_cfg: DictConfig) -> list[Callback]:
    """Instantiates plugins from config."""
    plugins: list[PrecisionPlugin] = []

    if not plugins_cfg:
        print("[instantiate_plugins] No plugin configs found! Skipping..")
        return plugins

    if not isinstance(plugins_cfg, DictConfig):
        raise TypeError("[instantiate_plugins] Plugins config must be a DictConfig!")

    for _, cb_conf in plugins_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"[instantiate_plugins] Instantiating plugin <{cb_conf._target_}>")
            plugins.append(hydra.utils.instantiate(cb_conf))

    return plugins
