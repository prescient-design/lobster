import logging
import os
from functools import partial
from typing import List, Union

import datatrove.io
import litdata
from upath import UPath

from ._convert import convert_parquet_to_lightning_data
from ._download import download

logger = logging.getLogger(__name__)


def convert_huggingface_dataset_to_lightning(
    data_path: str,
    raw_output_uri: str | UPath,
    processed_output_uri: str | UPath,
    keys: List[str] = None,
    num_workers: int = None,
    chunk_bytes: str = "128MB",
    compression: Union[str, None] = None,
) -> None:
    """
    Download and process a dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset in format 'hf://datasets/owner/dataset'
    split_id : str
        The split of the dataset to process (e.g., 'train', 'test', 'validation')
    raw_output_uri : str
        The URI where the raw downloaded files will be stored
    processed_output_uri : str
        The URI where the processed Lightning Data will be stored
    keys : list, optional
        Specific columns to extract from the dataset
    num_workers : int, optional
        Number of workers for parallel processing. Defaults to CPU count.
    chunk_bytes : str, default="64MB"
        Size of chunks for Lightning Data optimization
    compression : str or None, default=None
        Compression algorithm for Lightning Data
    """
    raw_output_uri = str(raw_output_uri)
    processed_output_uri = str(processed_output_uri)

    logger.info(f"Downloading and processing dataset from {data_path}.")

    if num_workers is None:
        num_workers = os.cpu_count()

    # Get all HF dataset files
    data_folder = datatrove.io.get_datafolder(data_path)

    filepaths = data_folder.get_shard(0, 1, recursive=True)

    data = []
    for filepath in filepaths:
        with data_folder.open(filepath, "rb") as f:
            data.append([filepath, str(f.url())])

    logger.info(f"Downloading {len(data)} files from {data_path} to {raw_output_uri}.")

    # Download the data
    litdata.map(
        fn=download,
        inputs=data,
        output_dir=raw_output_uri,
        num_workers=num_workers,
    )

    logger.info("Converting parquet files to Lightning Data format...")

    parquet_files = [str(f) for f in UPath(raw_output_uri).rglob("*.parquet")]

    # Convert to Lightning data format
    litdata.optimize(
        partial(convert_parquet_to_lightning_data, keys=keys) if keys else convert_parquet_to_lightning_data,
        parquet_files,
        processed_output_uri,
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
        compression=compression,
    )
