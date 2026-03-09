from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import torch
from litdata import optimize

from lobster.constants import Modality, Split
from lobster.data._ume_datamodule import collate_with_modality
from lobster.datasets.s3_datasets.base import UMEStreamingDataset


class _SequenceConverter:
    def __call__(self, sequence: str) -> Iterator[dict[str, str]]:
        yield {"sequence": sequence}


def _make_optimized_dataset(tmp_path, sequences: list[str], subdir: str = "optimized") -> str:
    output_dir = str(tmp_path / subdir)
    optimize(
        _SequenceConverter(),
        sequences,
        output_dir,
        num_workers=1,
        chunk_bytes="64MB",
        mode="overwrite",
    )
    return output_dir


def _make_raw_parquet_dataset(tmp_path, sequences: list[str], subdir: str = "parquet") -> str:
    output_dir = tmp_path / subdir
    output_dir.mkdir()
    pd.DataFrame({"sequence": sequences}).to_parquet(output_dir / "data.parquet", index=False)
    return str(output_dir)


def _make_local_dataset(raw_dir: str, optimized_dir: str):
    class LocalAminoDataset(UMEStreamingDataset):
        MODALITY = Modality.AMINO_ACID
        SEQUENCE_KEY = "sequence"
        SPLITS = {
            Split.TRAIN: raw_dir,
            Split.VALIDATION: raw_dir,
        }
        OPTIMIZED_SPLITS = {
            Split.TRAIN: optimized_dir,
            Split.VALIDATION: optimized_dir,
        }

    return LocalAminoDataset


class TestUMEStreamingDataset:
    def test_raw_parquet_dataset(self, tmp_path):
        raw_dir = _make_raw_parquet_dataset(tmp_path, ["EVQL", "QVQL"], subdir="raw")
        optimized_dir = _make_optimized_dataset(tmp_path, ["EVQL", "QVQL"], subdir="optimized")
        dataset_cls = _make_local_dataset(raw_dir, optimized_dir)

        dataset = dataset_cls(split=Split.VALIDATION, max_length=64, use_optimized=False)
        item = next(iter(dataset))

        assert item["sequence"] == "EVQL"
        assert item["modality"] == Modality.AMINO_ACID.value
        assert item["dataset"] == "LocalAminoDataset"
        assert item["input_ids"].shape[-1] == 64

    def test_optimized_dataset(self, tmp_path):
        raw_dir = _make_raw_parquet_dataset(tmp_path, ["EVQL", "QVQL"], subdir="raw")
        optimized_dir = _make_optimized_dataset(tmp_path, ["EVQL", "QVQL"], subdir="optimized")
        dataset_cls = _make_local_dataset(raw_dir, optimized_dir)

        dataset = dataset_cls(split=Split.VALIDATION, max_length=64, use_optimized=True)
        item = next(iter(dataset))

        assert item["sequence"] == "EVQL"
        assert item["input_ids"].shape[-1] == 64

    def test_transform_fn_skip_behavior(self, tmp_path):
        raw_dir = _make_raw_parquet_dataset(tmp_path, ["BAD", "GOOD"], subdir="raw")
        optimized_dir = _make_optimized_dataset(tmp_path, ["BAD", "GOOD"], subdir="optimized")
        dataset_cls = _make_local_dataset(raw_dir, optimized_dir)

        dataset = dataset_cls(
            split=Split.VALIDATION,
            max_length=64,
            use_optimized=False,
            transform_fn=lambda sequence: None if sequence == "BAD" else sequence,
        )
        item = next(iter(dataset))

        assert item["sequence"] == "GOOD"

    def test_extra_transform_fns_passthrough(self, tmp_path):
        raw_dir = _make_raw_parquet_dataset(tmp_path, ["EVQL"], subdir="raw")
        optimized_dir = _make_optimized_dataset(tmp_path, ["EVQL"], subdir="optimized")
        dataset_cls = _make_local_dataset(raw_dir, optimized_dir)

        dataset = dataset_cls(
            split=Split.VALIDATION,
            max_length=64,
            use_optimized=False,
            extra_transform_fns={"length": len},
        )
        item = next(iter(dataset))

        assert item["length"] == 4

    def test_tokenize_false(self, tmp_path):
        raw_dir = _make_raw_parquet_dataset(tmp_path, ["EVQL"], subdir="raw")
        optimized_dir = _make_optimized_dataset(tmp_path, ["EVQL"], subdir="optimized")
        dataset_cls = _make_local_dataset(raw_dir, optimized_dir)

        dataset = dataset_cls(split=Split.VALIDATION, max_length=64, use_optimized=False, tokenize=False)
        item = next(iter(dataset))

        assert item["input_ids"] is None
        assert item["attention_mask"] is None
        assert item["sequence"] == "EVQL"


class TestCollateWithModality:
    def test_interaction_branch(self):
        batch = [
            {
                "input_ids1": torch.ones(1, 8),
                "attention_mask1": torch.ones(1, 8),
                "input_ids2": torch.zeros(1, 8),
                "attention_mask2": torch.zeros(1, 8),
                "modality1": Modality.AMINO_ACID,
                "modality2": Modality.SMILES,
                "sequence1": "AAA",
                "sequence2": "BBB",
                "dataset": "Atomica",
            }
        ]

        collated = collate_with_modality(batch)

        assert collated["dataset"] == ["Atomica"]
        assert collated["modality1"] == [Modality.AMINO_ACID]
        assert collated["modality2"] == [Modality.SMILES]
        assert collated["sequence1"] == ["AAA"]
        assert collated["sequence2"] == ["BBB"]
