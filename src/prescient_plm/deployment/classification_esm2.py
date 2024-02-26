import importlib.resources
import os
from uuid import uuid4

import pandas as pd
from dagster import Config, get_dagster_logger
from omegaconf import OmegaConf
from prescient.io import load_omega_config

from prescient_plm.affinity_experiments.main import main
from prescient_plm.deployment.utils import (
    dagster_to_omega_config,
    load_target_sequences_dict,
    resolve_s3_checkpoint_path,
)


class PredictConfig(Config):
    """Configuration class for the predict function."""

    _dir = "s3://prescient-data-dev/model_deployment/classification_esm2"

    device: str = "cuda"
    cfg: dict = OmegaConf.to_container(
        load_omega_config(
            config_dir=str(
                importlib.resources.files("prescient_plm")
                / "affinity_experiments/config"
            ),
            config_name="config",
            compose=True,
            overrides=["experiment=classification_esm2_150M_round_inference"],
        ),
    )

    cfg["data_folder"] = f"{_dir}/data/inputs/"
    cfg["single_run_dir"] = f"{_dir}/runs"
    cfg["use_gpu"] = False if device == "cpu" else True
    cfg["use_multi_gpu"] = False
    cfg[
        "checkpoint_path"
    ] = "s3://prescient-data-dev/model_deployment/classification_esm2/checkpoints/dummy.pth"


def predict(dataframe: pd.DataFrame, config: PredictConfig) -> pd.DataFrame:
    """Predict function template.

    Please fill in
        - model
        - transform_fn
        - output_transform_fn

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe
    config : PredictConfig
        Configuration object

    Returns
    -------
    pd.DataFrame
        Output dataframe with predictions

    """
    logger = get_dagster_logger()

    # Prepare config
    config = dagster_to_omega_config(config)

    logger.info(OmegaConf.to_yaml(config, resolve=True))

    # Add target sequence
    logger.info("Add target sequence to input dataframe")

    target_seq_dict = load_target_sequences_dict()
    if any([target not in target_seq_dict for target in dataframe.target.unique()]):
        raise ValueError("Target sequence not found for one or more targets")

    dataframe[config.cfg.target_seq] = dataframe.target.apply(target_seq_dict.get)

    # Save input dataframe to S3 which will be used by the main function
    filename = f"{uuid4().hex[:8]}.csv"

    logger.info(
        f"Uploading input dataframe to {os.path.join(config.cfg.data_folder, filename)}"
    )

    dataframe.to_csv(os.path.join(config.cfg.data_folder, filename), index=False)
    config.cfg.test_data = filename

    # Download checkpoint from S3
    logger.info(f"Downloading checkpoint from {config.cfg.checkpoint_path}")
    config.cfg.checkpoint_path = resolve_s3_checkpoint_path(
        config.cfg.checkpoint_path, logger=logger
    )
    logger.info(f"Local checkpoint: {config.cfg.checkpoint_path}")

    # Run main function
    logger.info("Start predictions...")
    output_dataframe = main(config.cfg)

    logger.info(
        f"Finished predictions. Number of rows: {len(output_dataframe)}. Columns: {output_dataframe.columns}"
    )

    return output_dataframe


if __name__ == "__main__":
    print(PredictConfig())
