import importlib.resources
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lobster.data import DataFrameDatasetInMemory
from lobster.model._utils import model_typer
from lobster.tokenization import PmlmTokenizerTransform

try:
    from dagster import get_dagster_logger

    DAGSTER_IMPORT_ERROR = False
except ImportError:
    DAGSTER_IMPORT_ERROR = True


@dataclass
class PredictConfig:
    model_type: str = "PrescientPMLM"
    model_name: str | None = None
    checkpoint_path: str | None = (
        "s3://prescient-pcluster-data/freyn6/models/pmlm/prod/2023-10-30T15-23-25.795635/last.ckpt"
    )
    device: Literal["cpu", "cuda"] = "cpu"
    tokenizer: str = "pmlm_tokenizer"
    max_length: int = 512
    mlm: bool = True


def predict(dataframe: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Predict function template.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe
    config : DictConfig
        OmegaConf's DictConfig object with your config fields

    Returns
    -------
    pd.DataFrame
        Output dataframe with predictions

    """

    if DAGSTER_IMPORT_ERROR:
        raise ImportError(
            "dagster` is not installed. This function is not meant to be used inside a Dagster pipeline"
        )

    logger = get_dagster_logger()

    dataframe = dataframe.reset_index(drop=True)

    # Load model
    model_class = model_typer[config.model_type]

    if config.checkpoint_path is not None:
        logger.info(f"Loading model from {config.checkpoint_path}")
        model = model_class.load_from_checkpoint(config.checkpoint_path)
    else:
        if config.model_name is None:
            raise ValueError("Model name or checkpoint path must be provided")

        logger.info(f"No checkpoint path provided, using model {config.model_name} directly")
        model = model_class(model_name=config.model_name)

    model.to(config.device)

    # Transform fn - tokenization
    tokenizer_path = importlib.resources.files("lobster") / "assets" / config.tokenizer

    transform_fn = PmlmTokenizerTransform(
        tokenizer_path,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        mlm=config.mlm,
    )

    # Prepare dataloader
    dataloader = DataLoader(
        DataFrameDatasetInMemory(data=dataframe, transform_fn=transform_fn, columns=["sequence"]),
        batch_size=64,
        num_workers=1,
    )

    # Trainer
    trainer = Trainer(logger=False, accelerator=model.device.type)

    # Predict
    logger.info("Predicting")

    with torch.inference_mode():
        embedding_df = trainer.predict(model, dataloader)

    # Format output dataframe
    embedding_df = pd.concat(embedding_df).reset_index(drop=True)
    embedding_df.columns = embedding_df.columns.astype(str)

    # Add columns from the original dataset to the output
    embedding_df = pd.merge(
        dataframe,
        embedding_df,
        left_index=True,
        right_index=True,
    )

    logger.info(f"Predictions: {embedding_df.head()}")

    return embedding_df
