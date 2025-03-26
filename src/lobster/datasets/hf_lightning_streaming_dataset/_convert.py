from typing import Any, Iterator, List

import pyarrow.parquet as pq


def convert_parquet_to_lightning_data(
    parquet_file: str, keys: List[str] = None, batch_size: int = 128
) -> Iterator[List[Any]]:
    """
    Convert a parquet file to Lightning Data format.

    Parameters
    ----------
    parquet_file : str
        Path to the parquet file
    keys : list, optional
        List of columns to extract from the parquet file. If None, all columns are used.

    Yields
    ------
    list
        Values from each row in the parquet file
    """
    parquet_file = pq.ParquetFile(parquet_file)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()

        if keys:
            df = df[keys]

        for _, row in df.iterrows():
            if row.isnull().values.any():
                continue

            yield row.to_dict()
