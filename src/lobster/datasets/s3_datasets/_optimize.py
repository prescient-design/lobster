import os

import pandas as pd
from litdata import StreamingDataset, optimize
from upath import UPath

from lobster.constants import Split

from .amplify import AMPLIFY
from .atomica import Atomica
from .calm import Calm
from .m320m import M320M
from .peptide_atlas import PeptideAtlas
from .zinc import ZINC


def convert(filepath: str):
    df = pd.read_parquet(filepath)

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict = {k: v for k, v in row_dict.items() if isinstance(v, str)}

        yield row_dict


if __name__ == "__main__":
    for dataset in [Atomica, M320M, AMPLIFY, ZINC, PeptideAtlas, Calm]:
        dataset_name = dataset.__name__
        for split in [
            # Split.TEST,
            Split.TRAIN,
            Split.VALIDATION,
        ]:
            input_dir = dataset.SPLITS[split]
            output_dir = dataset.OPTIMIZED_SPLITS[split]

            files = [str(file) for file in UPath(input_dir).rglob("*.parquet")]

            print(f"Optimizing {dataset_name} {split} dataset")
            print(f"Input directory: {input_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Files: {len(files)}")
            print("\n\n")

            optimize(
                convert,
                files,
                output_dir,
                num_workers=min(os.cpu_count(), len(files)),
                chunk_bytes="64MB",
                mode="overwrite",
            )
            ds = StreamingDataset(
                output_dir,
            )
            print(f"\nLength of {split.value} dataset: {len(ds)}\n\n\n")
