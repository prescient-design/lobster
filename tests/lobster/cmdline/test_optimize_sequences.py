"""Tests for the OAS sequence optimization CLI."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from lobster.cmdline.optimize_sequences import (
    CollectionProgress,
    _convert_oas_csv,
    _convert_oas_parquet,
    _file_passes_filters,
    _parse_oas_metadata,
    _read_oas_file,
    _sequence_is_val,
    _sort_files_by_size,
    optimize_parquet_sequences,
    optimize_sequences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oas_csv(
    tmp_path: Path,
    filename: str = "test.csv",
    metadata: dict | None = None,
    sequences: list[str] | None = None,
) -> str:
    """Write a minimal OAS-format CSV file and return its path."""
    if metadata is None:
        metadata = {
            "Run": "SRR0000001",
            "Species": "human",
            "Chain": "Heavy",
            "Disease": "None",
            "Vaccine": "None",
            "Isotype": "IGHG",
        }

    if sequences is None:
        sequences = ["EVQLVESGG", "QVQLVQSGA", "DIVMTQSPL"]

    meta_line = json.dumps(metadata)
    header = "sequence_alignment_aa,other_col"
    rows = [f"{seq},val" for seq in sequences]

    content = "\n".join([meta_line, header, *rows])
    filepath = tmp_path / filename

    if filename.endswith(".gz"):
        filepath.write_bytes(gzip.compress(content.encode("utf-8")))
    else:
        filepath.write_text(content)

    return str(filepath)


def _make_oas_parquet(
    tmp_path: Path,
    filename: str = "data.parquet",
    sequences: list[str] | None = None,
    species: str = "human",
    chain: str = "Heavy",
    isotype: str = "IGHG",
) -> str:
    """Write a minimal OAS-style parquet file and return its path."""
    import pandas as pd

    if sequences is None:
        sequences = ["EVQLVESGG", "QVQLVQSGA", "DIVMTQSPL"]

    df = pd.DataFrame({
        "sequence_alignment_aa": sequences,
        "Species": [species] * len(sequences),
        "Chain": [chain] * len(sequences),
        "Isotype": [isotype] * len(sequences),
    })
    filepath = tmp_path / filename
    df.to_parquet(filepath, index=False)
    return str(filepath)


# ---------------------------------------------------------------------------
# _sequence_is_val (hash-based split)
# ---------------------------------------------------------------------------


class TestSequenceIsVal:
    """Tests for the deterministic hash-based split function."""

    def test_deterministic(self):
        """Same sequence + seed always gives same result."""
        results = [_sequence_is_val("EVQLVESGG", 0.1, seed=42) for _ in range(100)]
        assert len(set(results)) == 1

    def test_different_seeds_differ(self):
        """Different seeds can produce different assignments for the same seq."""
        # With enough seeds, at least one should differ
        results = {_sequence_is_val("EVQLVESGG", 0.5, seed=s) for s in range(50)}
        assert len(results) == 2  # both True and False should appear

    def test_fraction_zero_always_train(self):
        seqs = [f"SEQ{i}" for i in range(1000)]
        assert not any(_sequence_is_val(s, 0.0, seed=42) for s in seqs)

    def test_fraction_approximate(self):
        """With many sequences, the val fraction should be approximately correct."""
        seqs = [f"SEQ{i}" for i in range(10_000)]
        n_val = sum(_sequence_is_val(s, 0.1, seed=42) for s in seqs)
        # Should be roughly 10% ± 2%
        assert 800 < n_val < 1200


# ---------------------------------------------------------------------------
# _parse_oas_metadata
# ---------------------------------------------------------------------------


class TestParseOasMetadata:
    """Tests for _parse_oas_metadata."""

    def test_plain_json(self):
        line = '{"Species": "human", "Chain": "Heavy"}'
        result = _parse_oas_metadata(line)
        assert result == {"Species": "human", "Chain": "Heavy"}

    def test_csv_escaped_json(self):
        line = '"{""Species"": ""human"", ""Chain"": ""Heavy""}"'
        result = _parse_oas_metadata(line)
        assert result == {"Species": "human", "Chain": "Heavy"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse OAS metadata"):
            _parse_oas_metadata("this is not json")

    def test_whitespace_stripped(self):
        line = '  {"Species": "mouse"}  \n'
        result = _parse_oas_metadata(line)
        assert result["Species"] == "mouse"


# ---------------------------------------------------------------------------
# _file_passes_filters
# ---------------------------------------------------------------------------


class TestFilePassesFilters:
    """Tests for _file_passes_filters."""

    def test_no_filters(self):
        assert _file_passes_filters({"Species": "human"}, {})

    def test_matching_single_filter(self):
        metadata = {"Species": "human", "Chain": "Heavy"}
        filters = {"Species": ["human", "mouse"]}
        assert _file_passes_filters(metadata, filters)

    def test_non_matching_filter(self):
        metadata = {"Species": "camel", "Chain": "Heavy"}
        filters = {"Species": ["human", "mouse"]}
        assert not _file_passes_filters(metadata, filters)

    def test_missing_column_fails(self):
        metadata = {"Chain": "Heavy"}
        filters = {"Species": ["human"]}
        assert not _file_passes_filters(metadata, filters)

    def test_multiple_filters_all_must_match(self):
        metadata = {"Species": "human", "Chain": "Heavy", "Isotype": "IGHG"}
        filters = {"Species": ["human"], "Chain": ["Heavy"]}
        assert _file_passes_filters(metadata, filters)

        filters_fail = {"Species": ["human"], "Chain": ["Light"]}
        assert not _file_passes_filters(metadata, filters_fail)


# ---------------------------------------------------------------------------
# _read_oas_file
# ---------------------------------------------------------------------------


class TestReadOasFile:
    """Tests for _read_oas_file with plain and gzipped files."""

    def test_reads_plain_csv(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "plain.csv", sequences=["AAA"])
        text = _read_oas_file(filepath)
        assert "AAA" in text

    def test_reads_gzipped_csv(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "compressed.csv.gz", sequences=["BBB"])
        text = _read_oas_file(filepath)
        assert "BBB" in text


# ---------------------------------------------------------------------------
# _convert_oas_csv
# ---------------------------------------------------------------------------


class TestConvertOasCsv:
    """Tests for _convert_oas_csv."""

    def test_yields_sequences(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, sequences=["AAA", "BBB", "CCC"])
        results = list(_convert_oas_csv(filepath))
        assert len(results) == 3
        assert results[0] == {"sequence": "AAA"}
        assert results[2] == {"sequence": "CCC"}

    def test_skips_empty_sequences(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, sequences=["AAA", "", "CCC"])
        results = list(_convert_oas_csv(filepath))
        assert len(results) == 2

    def test_filters_exclude_file(self, tmp_path):
        filepath = _make_oas_csv(
            tmp_path,
            metadata={"Species": "camel", "Chain": "Heavy"},
            sequences=["AAA"],
        )
        results = list(_convert_oas_csv(filepath, filters={"Species": ["human"]}))
        assert len(results) == 0

    def test_filters_include_file(self, tmp_path):
        filepath = _make_oas_csv(
            tmp_path,
            metadata={"Species": "human", "Chain": "Heavy"},
            sequences=["AAA", "BBB"],
        )
        results = list(_convert_oas_csv(filepath, filters={"Species": ["human"]}))
        assert len(results) == 2

    def test_missing_column_skips_file(self, tmp_path):
        metadata = {"Species": "human"}
        meta_line = json.dumps(metadata)
        content = f"{meta_line}\nwrong_col\nAAA"
        filepath = tmp_path / "bad.csv"
        filepath.write_text(content)

        results = list(_convert_oas_csv(str(filepath)))
        assert len(results) == 0

    def test_custom_sequence_column(self, tmp_path):
        metadata = {"Species": "human"}
        meta_line = json.dumps(metadata)
        content = f"{meta_line}\nmy_seq,other\nGGG,x\nHHH,y"
        filepath = tmp_path / "custom.csv"
        filepath.write_text(content)

        results = list(_convert_oas_csv(str(filepath), sequence_column="my_seq"))
        assert len(results) == 2
        assert results[0] == {"sequence": "GGG"}

    def test_yields_sequences_from_gzipped(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "data.csv.gz", sequences=["XX", "YY"])
        results = list(_convert_oas_csv(filepath))
        assert len(results) == 2
        assert results[0] == {"sequence": "XX"}
        assert results[1] == {"sequence": "YY"}


# ---------------------------------------------------------------------------
# optimize_sequences (CSV integration)
# ---------------------------------------------------------------------------


class TestOptimizeSequences:
    """Integration tests for optimize_sequences using local filesystem."""

    def test_optimize_no_split(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_csv(input_dir, "a.csv", sequences=["EVQL", "QVQL"])
        _make_oas_csv(input_dir, "b.csv", sequences=["DIVM", "DVQL"])

        optimize_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            num_workers=1,
        )

        assert (output_dir / "index.json").exists()
        from litdata import StreamingDataset
        ds = StreamingDataset(str(output_dir))
        assert len(ds) == 4

    def test_optimize_with_val_split_iid(self, tmp_path):
        """Val split is iid across sequences via deterministic hash."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Many sequences across two differently-sized files
        _make_oas_csv(input_dir, "big.csv", sequences=[f"BIG{i}" for i in range(100)])
        _make_oas_csv(input_dir, "small.csv", sequences=["TINY"])

        optimize_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            val_fraction=0.2,
            num_workers=1,
            seed=42,
        )

        assert (output_dir / "train" / "index.json").exists()
        assert (output_dir / "val" / "index.json").exists()

        from litdata import StreamingDataset
        train_ds = StreamingDataset(str(output_dir / "train"))
        val_ds = StreamingDataset(str(output_dir / "val"))

        # Total across both splits should equal total sequences (101)
        assert len(train_ds) + len(val_ds) == 101
        # Val should be roughly 20%
        assert 10 < len(val_ds) < 30

    def test_optimize_with_filters(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_csv(
            input_dir, "human.csv",
            metadata={"Species": "human", "Chain": "Heavy"},
            sequences=["EVQL", "QVQL"],
        )
        _make_oas_csv(
            input_dir, "mouse.csv",
            metadata={"Species": "mouse", "Chain": "Heavy"},
            sequences=["MSEQ"],
        )

        optimize_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            filters={"Species": ["human"]},
            num_workers=1,
        )

        from litdata import StreamingDataset
        ds = StreamingDataset(str(output_dir))
        assert len(ds) == 2

    def test_val_fraction_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="val_fraction must be in"):
            optimize_sequences(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                val_fraction=1.5,
            )

    def test_optimize_gzipped_files(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_csv(input_dir, "a.csv.gz", sequences=["EVQL", "QVQL"])
        _make_oas_csv(input_dir, "b.csv", sequences=["DIVM"])

        optimize_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            num_workers=1,
        )

        from litdata import StreamingDataset
        ds = StreamingDataset(str(output_dir))
        assert len(ds) == 3

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files matching"):
            optimize_sequences(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
            )


# ---------------------------------------------------------------------------
# _convert_oas_parquet
# ---------------------------------------------------------------------------


class TestConvertOasParquet:
    """Tests for _convert_oas_parquet."""

    def test_yields_sequences(self, tmp_path):
        filepath = _make_oas_parquet(tmp_path, sequences=["AAA", "BBB", "CCC"])
        results = list(_convert_oas_parquet(filepath))
        assert len(results) == 3
        assert results[0] == {"sequence": "AAA"}
        assert results[2] == {"sequence": "CCC"}

    def test_filters_include_matching_rows(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({
            "sequence_alignment_aa": ["AAA", "BBB", "CCC"],
            "Species": ["human", "mouse", "human"],
            "Chain": ["Heavy", "Heavy", "Light"],
        })
        filepath = tmp_path / "mixed.parquet"
        df.to_parquet(filepath, index=False)

        results = list(_convert_oas_parquet(str(filepath), filters={"Species": ["human"]}))
        assert len(results) == 2
        seqs = [r["sequence"] for r in results]
        assert "AAA" in seqs
        assert "CCC" in seqs

    def test_filters_multiple_columns(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({
            "sequence_alignment_aa": ["AAA", "BBB", "CCC"],
            "Species": ["human", "human", "human"],
            "Chain": ["Heavy", "Light", "Heavy"],
        })
        filepath = tmp_path / "multi.parquet"
        df.to_parquet(filepath, index=False)

        results = list(_convert_oas_parquet(
            str(filepath), filters={"Species": ["human"], "Chain": ["Heavy"]}
        ))
        assert len(results) == 2

    def test_filters_exclude_all_rows(self, tmp_path):
        filepath = _make_oas_parquet(tmp_path, species="camel", sequences=["ZZZ"])
        results = list(_convert_oas_parquet(filepath, filters={"Species": ["human"]}))
        assert len(results) == 0

    def test_missing_sequence_column(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"wrong_col": ["AAA"]})
        filepath = tmp_path / "bad.parquet"
        df.to_parquet(filepath, index=False)

        results = list(_convert_oas_parquet(str(filepath)))
        assert len(results) == 0

    def test_skips_nan_sequences(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({
            "sequence_alignment_aa": ["AAA", None, "CCC"],
            "Species": ["human"] * 3,
        })
        filepath = tmp_path / "nans.parquet"
        df.to_parquet(filepath, index=False)

        results = list(_convert_oas_parquet(str(filepath)))
        assert len(results) == 2

    def test_filter_column_not_in_file(self, tmp_path):
        filepath = _make_oas_parquet(tmp_path, sequences=["AAA", "BBB"])
        results = list(_convert_oas_parquet(filepath, filters={"Vaccine": ["None"]}))
        assert len(results) == 2

    def test_custom_sequence_column(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"my_seq": ["GGG", "HHH"], "Species": ["human"] * 2})
        filepath = tmp_path / "custom.parquet"
        df.to_parquet(filepath, index=False)

        results = list(_convert_oas_parquet(str(filepath), sequence_column="my_seq"))
        assert len(results) == 2
        assert results[0] == {"sequence": "GGG"}


# ---------------------------------------------------------------------------
# optimize_parquet_sequences (integration)
# ---------------------------------------------------------------------------


class TestOptimizeParquetSequences:
    """Integration tests for optimize_parquet_sequences using local filesystem."""

    def test_optimize_no_split(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_parquet(input_dir, "a.parquet", sequences=["EVQL", "QVQL"])
        _make_oas_parquet(input_dir, "b.parquet", sequences=["DIVM", "DVQL"])

        optimize_parquet_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            num_workers=1,
        )

        assert (output_dir / "index.json").exists()
        from litdata import StreamingDataset
        ds = StreamingDataset(str(output_dir))
        assert len(ds) == 4

    def test_optimize_with_val_split_iid(self, tmp_path):
        """Val split is iid across sequences via deterministic hash."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_parquet(input_dir, "big.parquet", sequences=[f"BIG{i}" for i in range(100)])
        _make_oas_parquet(input_dir, "small.parquet", sequences=["TINY"])

        optimize_parquet_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            val_fraction=0.2,
            num_workers=1,
            seed=42,
        )

        assert (output_dir / "train" / "index.json").exists()
        assert (output_dir / "val" / "index.json").exists()

        from litdata import StreamingDataset
        train_ds = StreamingDataset(str(output_dir / "train"))
        val_ds = StreamingDataset(str(output_dir / "val"))

        assert len(train_ds) + len(val_ds) == 101
        assert 10 < len(val_ds) < 30

    def test_optimize_with_row_filters(self, tmp_path):
        import pandas as pd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        df = pd.DataFrame({
            "sequence_alignment_aa": ["HUMAN1", "MOUSE1", "HUMAN2"],
            "Species": ["human", "mouse", "human"],
            "Chain": ["Heavy", "Heavy", "Light"],
        })
        df.to_parquet(input_dir / "mixed.parquet", index=False)

        optimize_parquet_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            filters={"Species": ["human"]},
            num_workers=1,
        )

        from litdata import StreamingDataset
        ds = StreamingDataset(str(output_dir))
        assert len(ds) == 2

    def test_optimize_hive_partitioned(self, tmp_path):
        input_dir = tmp_path / "input"
        part1 = input_dir / "file_id=SRR001_Heavy_IGHG"
        part2 = input_dir / "file_id=SRR002_Light_IGKC"
        part1.mkdir(parents=True)
        part2.mkdir(parents=True)
        output_dir = tmp_path / "output"

        _make_oas_parquet(part1, "part-0.parquet", sequences=["EVQL"])
        _make_oas_parquet(part2, "part-0.parquet", sequences=["DIVM", "QVQL"])

        optimize_parquet_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            num_workers=1,
        )

        from litdata import StreamingDataset
        ds = StreamingDataset(str(output_dir))
        assert len(ds) == 3

    def test_val_fraction_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="val_fraction must be in"):
            optimize_parquet_sequences(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                val_fraction=1.5,
            )

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .parquet files found"):
            optimize_parquet_sequences(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
            )


# ---------------------------------------------------------------------------
# _sort_files_by_size
# ---------------------------------------------------------------------------


class TestSortFilesBySize:
    """Tests for _sort_files_by_size."""

    def test_sorts_smallest_first(self, tmp_path):
        small = tmp_path / "small.txt"
        medium = tmp_path / "medium.txt"
        large = tmp_path / "large.txt"

        small.write_text("a")
        medium.write_text("a" * 100)
        large.write_text("a" * 10000)

        files = [str(large), str(small), str(medium)]
        sorted_files = _sort_files_by_size(files)

        assert sorted_files == [str(small), str(medium), str(large)]

    def test_empty_list(self):
        assert _sort_files_by_size([]) == []


# ---------------------------------------------------------------------------
# CollectionProgress
# ---------------------------------------------------------------------------


class TestCollectionProgress:
    """Tests for CollectionProgress."""

    def test_fresh_start(self, tmp_path):
        progress = CollectionProgress(tmp_path / "progress")
        assert progress.done_files == set()
        assert progress.sequence_count == 0

    def test_record_and_resume(self, tmp_path):
        pdir = tmp_path / "progress"
        progress = CollectionProgress(pdir, split_name="train")

        progress.record_file("file1.csv", num_sequences=10)
        progress.record_file("file2.csv", num_sequences=5)

        assert "file1.csv" in progress.done_files
        assert "file2.csv" in progress.done_files

        # Simulate restart
        progress2 = CollectionProgress(pdir, split_name="train")
        assert progress2.done_files == {"file1.csv", "file2.csv"}
        assert progress2.sequence_count == 15

    def test_filter_remaining(self, tmp_path):
        pdir = tmp_path / "progress"
        progress = CollectionProgress(pdir)
        progress.record_file("done.csv", num_sequences=1)

        remaining = progress.filter_remaining(["done.csv", "new.csv"])
        assert remaining == ["new.csv"]

    def test_clear(self, tmp_path):
        pdir = tmp_path / "progress"
        progress = CollectionProgress(pdir, split_name="train")
        progress.record_file("file.csv", num_sequences=1)
        progress.clear()

        assert not (pdir / "progress_train.json").exists()

    def test_independent_split_tracking(self, tmp_path):
        """Train and val progress are tracked independently."""
        pdir = tmp_path / "progress"
        train_progress = CollectionProgress(pdir, split_name="train")
        val_progress = CollectionProgress(pdir, split_name="val")

        train_progress.record_file("file1.csv", num_sequences=10)

        assert "file1.csv" in train_progress.done_files
        assert "file1.csv" not in val_progress.done_files
