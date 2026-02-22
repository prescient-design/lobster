"""Tests for the OAS sequence optimization CLI."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from lobster.cmdline.optimize_sequences import (
    _convert_oas_csv,
    _file_passes_filters,
    _parse_oas_metadata,
    _read_oas_file,
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
        results = list(
            _convert_oas_csv(filepath, filters={"Species": ["human"]})
        )
        assert len(results) == 0

    def test_filters_include_file(self, tmp_path):
        filepath = _make_oas_csv(
            tmp_path,
            metadata={"Species": "human", "Chain": "Heavy"},
            sequences=["AAA", "BBB"],
        )
        results = list(
            _convert_oas_csv(filepath, filters={"Species": ["human"]})
        )
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

        results = list(
            _convert_oas_csv(str(filepath), sequence_column="my_seq")
        )
        assert len(results) == 2
        assert results[0] == {"sequence": "GGG"}

    def test_yields_sequences_from_gzipped(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "data.csv.gz", sequences=["XX", "YY"])
        results = list(_convert_oas_csv(filepath))
        assert len(results) == 2
        assert results[0] == {"sequence": "XX"}
        assert results[1] == {"sequence": "YY"}


# ---------------------------------------------------------------------------
# optimize_sequences (integration)
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

    def test_optimize_with_val_split(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Need enough files for a meaningful split
        for i in range(10):
            _make_oas_csv(input_dir, f"file_{i}.csv", sequences=[f"SEQ{i}"])

        optimize_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            val_fraction=0.3,
            num_workers=1,
        )

        assert (output_dir / "train" / "index.json").exists()
        assert (output_dir / "val" / "index.json").exists()

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
        # Only the 2 human sequences should be included
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
