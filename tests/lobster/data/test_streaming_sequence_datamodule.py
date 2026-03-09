"""Tests for StreamingSequenceLightningDataModule."""

from __future__ import annotations

import torch
from litdata import optimize

from lobster.data._streaming_sequence_datamodule import (
    StreamingSequenceLightningDataModule,
    _clm_collate_fn,
)


class _SequenceConverter:
    """Picklable callable that treats each input as a raw sequence string."""

    def __call__(self, sequence: str):
        yield {"sequence": sequence}


def _make_optimized_dataset(tmp_path, sequences: list[str] | None = None, subdir: str = "data"):
    """Create a local LitData-optimized dataset with protein sequences."""
    if sequences is None:
        sequences = [
            "EVQLVESGG", "QVQLVQSGA", "DIVMTQSPL", "MKLLVLLFGA",
            "DIQMTQSPS", "EVQLLESGG", "QVQLQQSGA", "DIVLTQSPL",
        ]

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


class TestClmCollateFn:
    """Tests for the _clm_collate_fn collation function."""

    def test_returns_tuple(self):
        batch = [
            {
                "input_ids": torch.ones(1, 8),
                "labels": torch.ones(1, 8),
                "attention_mask": torch.ones(1, 8),
                "sequence": "EVQL",
            },
            {
                "input_ids": torch.zeros(1, 8),
                "labels": torch.zeros(1, 8),
                "attention_mask": torch.zeros(1, 8),
                "sequence": "QVQL",
            },
        ]

        result = _clm_collate_fn(batch)
        assert isinstance(result, tuple)
        assert len(result) == 2

        tensor_dict, sequences = result
        assert isinstance(tensor_dict, dict)
        assert isinstance(sequences, list)

    def test_tensor_shapes(self):
        batch = [
            {
                "input_ids": torch.ones(1, 8),
                "labels": torch.ones(1, 8),
                "attention_mask": torch.ones(1, 8),
                "sequence": "EVQL",
            }
        ] * 4

        tensor_dict, _ = _clm_collate_fn(batch)
        assert tensor_dict["input_ids"].shape == (4, 8)
        assert tensor_dict["labels"].shape == (4, 8)
        assert tensor_dict["attention_mask"].shape == (4, 8)

    def test_sequences_collected(self):
        batch = [
            {
                "input_ids": torch.ones(1, 8),
                "labels": torch.ones(1, 8),
                "attention_mask": torch.ones(1, 8),
                "sequence": "AAA",
            },
            {
                "input_ids": torch.ones(1, 8),
                "labels": torch.ones(1, 8),
                "attention_mask": torch.ones(1, 8),
                "sequence": "BBB",
            },
        ]

        _, sequences = _clm_collate_fn(batch)
        assert sequences == ["AAA", "BBB"]


class TestStreamingSequenceLightningDataModule:
    """Tests for StreamingSequenceLightningDataModule."""

    def test_setup_creates_train_dataset(self, tmp_path):
        train_dir = _make_optimized_dataset(tmp_path, subdir="train")

        dm = StreamingSequenceLightningDataModule(
            train_input_dir=train_dir,
            max_length=64,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert dm._train_dataset is not None

    def test_train_dataloader_returns_batches(self, tmp_path):
        train_dir = _make_optimized_dataset(tmp_path, subdir="train")

        dm = StreamingSequenceLightningDataModule(
            train_input_dir=train_dir,
            max_length=64,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        tensor_dict, sequences = batch
        assert "input_ids" in tensor_dict
        assert "labels" in tensor_dict
        assert "attention_mask" in tensor_dict
        assert len(sequences) == 2

    def test_val_dataloader_none_when_no_val_dir(self, tmp_path):
        train_dir = _make_optimized_dataset(tmp_path, subdir="train")

        dm = StreamingSequenceLightningDataModule(
            train_input_dir=train_dir,
            max_length=64,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert dm.val_dataloader() is None

    def test_val_dataloader_when_val_dir_provided(self, tmp_path):
        train_dir = _make_optimized_dataset(tmp_path, subdir="train")
        val_dir = _make_optimized_dataset(
            tmp_path,
            sequences=["EVQL", "QVQL", "DIVM", "MKLL"],
            subdir="val",
        )

        dm = StreamingSequenceLightningDataModule(
            train_input_dir=train_dir,
            val_input_dir=val_dir,
            max_length=64,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        val_loader = dm.val_dataloader()
        assert val_loader is not None

        batch = next(iter(val_loader))
        tensor_dict, sequences = batch
        assert tensor_dict["input_ids"].shape[0] == 2

    def test_batch_compatible_with_lobster_pclm(self, tmp_path):
        """Verify the batch format matches what LobsterPCLM._compute_loss expects."""
        train_dir = _make_optimized_dataset(tmp_path, subdir="train")

        dm = StreamingSequenceLightningDataModule(
            train_input_dir=train_dir,
            max_length=64,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        # LobsterPCLM._compute_loss does: batch, _targets = batch
        tensor_dict, targets = batch

        # Then accesses: batch["input_ids"].squeeze(), etc.
        input_ids = tensor_dict["input_ids"].squeeze()
        labels = tensor_dict["labels"].squeeze()
        attention_mask = tensor_dict["attention_mask"].squeeze()

        assert input_ids.dim() == 2
        assert labels.dim() == 2
        assert attention_mask.dim() == 2
        assert input_ids.shape[0] == 2  # batch_size
