"""Tests for StreamingSequenceDataset."""

from __future__ import annotations

from litdata import optimize

from lobster.datasets._streaming_sequence_dataset import StreamingSequenceDataset


class _SequenceConverter:
    """Picklable callable that treats each input as a raw sequence string."""

    def __call__(self, sequence: str):
        yield {"sequence": sequence}


def _make_optimized_dataset(tmp_path, sequences: list[str] | None = None):
    """Create a local LitData-optimized dataset with protein sequences."""
    if sequences is None:
        sequences = ["EVQLVESGG", "QVQLVQSGA", "DIVMTQSPL", "MKLLVLLFGA"]

    output_dir = str(tmp_path / "optimized")

    optimize(
        _SequenceConverter(),
        sequences,
        output_dir,
        num_workers=1,
        chunk_bytes="64MB",
        mode="overwrite",
    )
    return output_dir


class TestStreamingSequenceDataset:
    """Tests for StreamingSequenceDataset."""

    def test_dataset_len(self, tmp_path):
        sequences = ["EVQL", "QVQL", "DIVM"]
        output_dir = _make_optimized_dataset(tmp_path, sequences)

        ds = StreamingSequenceDataset(input_dir=output_dir, max_length=64, shuffle=False)
        assert len(ds) == 3

    def test_item_has_expected_keys(self, tmp_path):
        output_dir = _make_optimized_dataset(tmp_path, ["EVQLVESGG"])
        ds = StreamingSequenceDataset(input_dir=output_dir, max_length=64, shuffle=False)

        item = next(iter(ds))
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert "sequence" in item

    def test_item_tensors_shape(self, tmp_path):
        max_length = 32
        output_dir = _make_optimized_dataset(tmp_path, ["EVQLVESGG"])
        ds = StreamingSequenceDataset(
            input_dir=output_dir, max_length=max_length, shuffle=False
        )

        item = next(iter(ds))
        # PmlmTokenizerTransform returns tensors of shape (1, max_length)
        assert item["input_ids"].shape[-1] == max_length
        assert item["labels"].shape[-1] == max_length
        assert item["attention_mask"].shape[-1] == max_length

    def test_labels_ignore_padding(self, tmp_path):
        output_dir = _make_optimized_dataset(tmp_path, ["EVQL"])
        ds = StreamingSequenceDataset(
            input_dir=output_dir, max_length=64, shuffle=False
        )

        item = next(iter(ds))
        labels = item["labels"].squeeze()
        # Labels should have -100 where there is padding
        assert (labels == -100).any(), "Padding positions should be set to -100"

    def test_sequence_preserved(self, tmp_path):
        seq = "EVQLVESGG"
        output_dir = _make_optimized_dataset(tmp_path, [seq])
        ds = StreamingSequenceDataset(
            input_dir=output_dir, max_length=64, shuffle=False
        )

        item = next(iter(ds))
        assert item["sequence"] == seq

    def test_transform_fn_applied(self, tmp_path):
        output_dir = _make_optimized_dataset(tmp_path, ["evqlvesgg"])
        ds = StreamingSequenceDataset(
            input_dir=output_dir,
            max_length=64,
            shuffle=False,
            transform_fn=lambda s: s.upper(),
        )

        item = next(iter(ds))
        assert item["sequence"] == "EVQLVESGG"
