from dagster import Config, PermissiveConfig
from omegaconf import DictConfig, OmegaConf


def dagster_to_omega_config(dagster_config: Config | PermissiveConfig) -> DictConfig:

    config = dagster_config._convert_to_config_dictionary()
    omega_conf = OmegaConf.create(config)

    return omega_conf
