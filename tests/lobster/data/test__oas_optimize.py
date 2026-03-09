"""Tests for OAS optimization helpers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from lobster.data._oas_optimize import (
    convert_oas_csv,
    convert_oas_parquet,
    file_passes_filters,
    optimize_oas_csv_sequences,
    optimize_oas_parquet_sequences,
    parse_oas_metadata,
    read_oas_file,
)


def _make_oas_csv(
    tmp_path: Path,
    filename: str = "test.csv",
    metadata: dict | None = None,
    sequences: list[str] | None = None,
) -> str:
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

    content = "\n".join(
        [
            json.dumps(metadata),
            "sequence_alignment_aa,other_col",
            *[f"{sequence},val" for sequence in sequences],
        ]
    )
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
    import pandas as pd

    if sequences is None:
        sequences = ["EVQLVESGG", "QVQLVQSGA", "DIVMTQSPL"]

    dataframe = pd.DataFrame(
        {
            "sequence_alignment_aa": sequences,
            "Species": [species] * len(sequences),
            "Chain": [chain] * len(sequences),
            "Isotype": [isotype] * len(sequences),
        }
    )
    filepath = tmp_path / filename
    dataframe.to_parquet(filepath, index=False)
    return str(filepath)


class TestParseOasMetadata:
    def test_plain_json(self):
        assert parse_oas_metadata('{"Species": "human", "Chain": "Heavy"}') == {
            "Species": "human",
            "Chain": "Heavy",
        }

    def test_csv_escaped_json(self):
        assert parse_oas_metadata('"{""Species"": ""human"", ""Chain"": ""Heavy""}"') == {
            "Species": "human",
            "Chain": "Heavy",
        }

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse OAS metadata"):
            parse_oas_metadata("this is not json")

    def test_whitespace_stripped(self):
        assert parse_oas_metadata('  {"Species": "mouse"}  \n')["Species"] == "mouse"


class TestFilePassesFilters:
    def test_no_filters(self):
        assert file_passes_filters({"Species": "human"}, {})

    def test_matching_single_filter(self):
        metadata = {"Species": "human", "Chain": "Heavy"}
        filters = {"Species": ["human", "mouse"]}
        assert file_passes_filters(metadata, filters)

    def test_non_matching_filter(self):
        metadata = {"Species": "camel", "Chain": "Heavy"}
        filters = {"Species": ["human", "mouse"]}
        assert not file_passes_filters(metadata, filters)

    def test_missing_column_fails(self):
        assert not file_passes_filters({"Chain": "Heavy"}, {"Species": ["human"]})

    def test_multiple_filters_all_must_match(self):
        metadata = {"Species": "human", "Chain": "Heavy", "Isotype": "IGHG"}
        assert file_passes_filters(metadata, {"Species": ["human"], "Chain": ["Heavy"]})
        assert not file_passes_filters(metadata, {"Species": ["human"], "Chain": ["Light"]})


class TestReadOasFile:
    def test_reads_plain_csv(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "plain.csv", sequences=["AAA"])
        assert "AAA" in read_oas_file(filepath)

    def test_reads_gzipped_csv(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "compressed.csv.gz", sequences=["BBB"])
        assert "BBB" in read_oas_file(filepath)


class TestConvertOasCsv:
    def test_yields_sequences(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, sequences=["AAA", "BBB", "CCC"])
        assert list(convert_oas_csv(filepath)) == [{"sequence": "AAA"}, {"sequence": "BBB"}, {"sequence": "CCC"}]

    def test_skips_empty_sequences(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, sequences=["AAA", "", "CCC"])
        assert len(list(convert_oas_csv(filepath))) == 2

    def test_filters_exclude_file(self, tmp_path):
        filepath = _make_oas_csv(
            tmp_path,
            metadata={"Species": "camel", "Chain": "Heavy"},
            sequences=["AAA"],
        )
        assert list(convert_oas_csv(filepath, filters={"Species": ["human"]})) == []

    def test_filters_include_file(self, tmp_path):
        filepath = _make_oas_csv(
            tmp_path,
            metadata={"Species": "human", "Chain": "Heavy"},
            sequences=["AAA", "BBB"],
        )
        assert len(list(convert_oas_csv(filepath, filters={"Species": ["human"]}))) == 2

    def test_missing_column_skips_file(self, tmp_path):
        filepath = tmp_path / "bad.csv"
        filepath.write_text(f'{json.dumps({"Species": "human"})}\nwrong_col\nAAA')
        assert list(convert_oas_csv(str(filepath))) == []

    def test_custom_sequence_column(self, tmp_path):
        filepath = tmp_path / "custom.csv"
        filepath.write_text(f'{json.dumps({"Species": "human"})}\nmy_seq,other\nGGG,x\nHHH,y')
        assert list(convert_oas_csv(str(filepath), sequence_column="my_seq")) == [
            {"sequence": "GGG"},
            {"sequence": "HHH"},
        ]

    def test_yields_sequences_from_gzipped(self, tmp_path):
        filepath = _make_oas_csv(tmp_path, "data.csv.gz", sequences=["XX", "YY"])
        assert list(convert_oas_csv(filepath)) == [{"sequence": "XX"}, {"sequence": "YY"}]


class TestOptimizeOasCsvSequences:
    def test_optimize_no_split(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_csv(input_dir, "a.csv", sequences=["EVQL", "QVQL"])
        _make_oas_csv(input_dir, "b.csv", sequences=["DIVM", "DVQL"])

        optimize_oas_csv_sequences(input_dir=str(input_dir), output_dir=str(output_dir), num_workers=1)

        from litdata import StreamingDataset

        assert (output_dir / "index.json").exists()
        assert len(StreamingDataset(str(output_dir))) == 4

    def test_optimize_with_val_split_iid(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_csv(input_dir, "big.csv", sequences=[f"BIG{index}" for index in range(100)])
        _make_oas_csv(input_dir, "small.csv", sequences=["TINY"])

        optimize_oas_csv_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            val_fraction=0.2,
            num_workers=1,
            seed=42,
        )

        from litdata import StreamingDataset

        train_dataset = StreamingDataset(str(output_dir / "train"))
        val_dataset = StreamingDataset(str(output_dir / "val"))
        assert len(train_dataset) + len(val_dataset) == 101
        assert 10 < len(val_dataset) < 30

    def test_optimize_with_filters(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_csv(
            input_dir,
            "human.csv",
            metadata={"Species": "human", "Chain": "Heavy"},
            sequences=["EVQL", "QVQL"],
        )
        _make_oas_csv(
            input_dir,
            "mouse.csv",
            metadata={"Species": "mouse", "Chain": "Heavy"},
            sequences=["MSEQ"],
        )

        optimize_oas_csv_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            filters={"Species": ["human"]},
            num_workers=1,
        )

        from litdata import StreamingDataset

        assert len(StreamingDataset(str(output_dir))) == 2

    def test_val_fraction_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="val_fraction must be in"):
            optimize_oas_csv_sequences(
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

        optimize_oas_csv_sequences(input_dir=str(input_dir), output_dir=str(output_dir), num_workers=1)

        from litdata import StreamingDataset

        assert len(StreamingDataset(str(output_dir))) == 3

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files matching"):
            optimize_oas_csv_sequences(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))


class TestConvertOasParquet:
    def test_yields_sequences(self, tmp_path):
        filepath = _make_oas_parquet(tmp_path, sequences=["AAA", "BBB", "CCC"])
        assert list(convert_oas_parquet(filepath)) == [{"sequence": "AAA"}, {"sequence": "BBB"}, {"sequence": "CCC"}]

    def test_filters_include_matching_rows(self, tmp_path):
        import pandas as pd

        dataframe = pd.DataFrame(
            {
                "sequence_alignment_aa": ["AAA", "BBB", "CCC"],
                "Species": ["human", "mouse", "human"],
                "Chain": ["Heavy", "Heavy", "Light"],
            }
        )
        filepath = tmp_path / "mixed.parquet"
        dataframe.to_parquet(filepath, index=False)

        results = list(convert_oas_parquet(str(filepath), filters={"Species": ["human"]}))
        assert [result["sequence"] for result in results] == ["AAA", "CCC"]

    def test_filters_multiple_columns(self, tmp_path):
        import pandas as pd

        dataframe = pd.DataFrame(
            {
                "sequence_alignment_aa": ["AAA", "BBB", "CCC"],
                "Species": ["human", "human", "human"],
                "Chain": ["Heavy", "Light", "Heavy"],
            }
        )
        filepath = tmp_path / "multi.parquet"
        dataframe.to_parquet(filepath, index=False)

        assert len(list(convert_oas_parquet(str(filepath), filters={"Species": ["human"], "Chain": ["Heavy"]}))) == 2

    def test_filters_exclude_all_rows(self, tmp_path):
        filepath = _make_oas_parquet(tmp_path, species="camel", sequences=["ZZZ"])
        assert list(convert_oas_parquet(filepath, filters={"Species": ["human"]})) == []

    def test_missing_sequence_column(self, tmp_path):
        import pandas as pd

        filepath = tmp_path / "bad.parquet"
        pd.DataFrame({"wrong_col": ["AAA"]}).to_parquet(filepath, index=False)
        assert list(convert_oas_parquet(str(filepath))) == []

    def test_skips_nan_sequences(self, tmp_path):
        import pandas as pd

        filepath = tmp_path / "nans.parquet"
        pd.DataFrame({"sequence_alignment_aa": ["AAA", None, "CCC"], "Species": ["human"] * 3}).to_parquet(
            filepath, index=False
        )
        assert list(convert_oas_parquet(str(filepath))) == [{"sequence": "AAA"}, {"sequence": "CCC"}]

    def test_filter_column_not_in_file(self, tmp_path):
        filepath = _make_oas_parquet(tmp_path, sequences=["AAA", "BBB"])
        assert len(list(convert_oas_parquet(filepath, filters={"Vaccine": ["None"]}))) == 2

    def test_custom_sequence_column(self, tmp_path):
        import pandas as pd

        filepath = tmp_path / "custom.parquet"
        pd.DataFrame({"my_seq": ["GGG", "HHH"], "Species": ["human"] * 2}).to_parquet(filepath, index=False)
        assert list(convert_oas_parquet(str(filepath), sequence_column="my_seq")) == [
            {"sequence": "GGG"},
            {"sequence": "HHH"},
        ]


class TestOptimizeOasParquetSequences:
    def test_optimize_no_split(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_parquet(input_dir, "a.parquet", sequences=["EVQL", "QVQL"])
        _make_oas_parquet(input_dir, "b.parquet", sequences=["DIVM", "DVQL"])

        optimize_oas_parquet_sequences(input_dir=str(input_dir), output_dir=str(output_dir), num_workers=1)

        from litdata import StreamingDataset

        assert (output_dir / "index.json").exists()
        assert len(StreamingDataset(str(output_dir))) == 4

    def test_optimize_with_val_split_iid(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _make_oas_parquet(input_dir, "big.parquet", sequences=[f"BIG{index}" for index in range(100)])
        _make_oas_parquet(input_dir, "small.parquet", sequences=["TINY"])

        optimize_oas_parquet_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            val_fraction=0.2,
            num_workers=1,
            seed=42,
        )

        from litdata import StreamingDataset

        train_dataset = StreamingDataset(str(output_dir / "train"))
        val_dataset = StreamingDataset(str(output_dir / "val"))
        assert len(train_dataset) + len(val_dataset) == 101
        assert 10 < len(val_dataset) < 30

    def test_optimize_with_row_filters(self, tmp_path):
        import pandas as pd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        pd.DataFrame(
            {
                "sequence_alignment_aa": ["HUMAN1", "MOUSE1", "HUMAN2"],
                "Species": ["human", "mouse", "human"],
                "Chain": ["Heavy", "Heavy", "Light"],
            }
        ).to_parquet(input_dir / "mixed.parquet", index=False)

        optimize_oas_parquet_sequences(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            filters={"Species": ["human"]},
            num_workers=1,
        )

        from litdata import StreamingDataset

        assert len(StreamingDataset(str(output_dir))) == 2

    def test_optimize_hive_partitioned(self, tmp_path):
        input_dir = tmp_path / "input"
        part1 = input_dir / "file_id=SRR001_Heavy_IGHG"
        part2 = input_dir / "file_id=SRR002_Light_IGKC"
        part1.mkdir(parents=True)
        part2.mkdir(parents=True)
        output_dir = tmp_path / "output"

        _make_oas_parquet(part1, "part-0.parquet", sequences=["EVQL"])
        _make_oas_parquet(part2, "part-0.parquet", sequences=["DIVM", "QVQL"])

        optimize_oas_parquet_sequences(input_dir=str(input_dir), output_dir=str(output_dir), num_workers=1)

        from litdata import StreamingDataset

        assert len(StreamingDataset(str(output_dir))) == 3

    def test_val_fraction_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="val_fraction must be in"):
            optimize_oas_parquet_sequences(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                val_fraction=1.5,
            )

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .parquet files found"):
            optimize_oas_parquet_sequences(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
