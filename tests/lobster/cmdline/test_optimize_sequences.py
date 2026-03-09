from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

import lobster.cmdline.optimize_sequences as optimize_sequences_cli


class TestOptimizeSequencesCli:
    def test_parse_filter_arg(self):
        assert optimize_sequences_cli._parse_filter_arg("human, mouse,,rat") == ["human", "mouse", "rat"]

    def test_build_filters(self):
        args = Namespace(
            species="human,mouse",
            vaccine=None,
            disease="None",
            chain="Heavy",
            isotype=None,
        )
        assert optimize_sequences_cli._build_filters(args) == {
            "Species": ["human", "mouse"],
            "Disease": ["None"],
            "Chain": ["Heavy"],
        }

    def test_main_dispatches_csv(self):
        argv = [
            "lobster_optimize_sequences",
            "--input_dir",
            "/tmp/input",
            "--output_dir",
            "/tmp/output",
            "--species",
            "human",
        ]
        with patch("sys.argv", argv), patch.object(optimize_sequences_cli, "optimize_sequences") as optimize_csv:
            optimize_sequences_cli.main()

        optimize_csv.assert_called_once_with(
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            val_fraction=0.0,
            chunk_bytes="64MB",
            num_workers=None,
            seed=42,
            file_glob=["*.csv", "*.csv.gz"],
            sequence_column="sequence_alignment_aa",
            filters={"Species": ["human"]},
            progress_dir=None,
        )

    def test_main_dispatches_parquet(self):
        argv = [
            "lobster_optimize_sequences",
            "--input_dir",
            "/tmp/input",
            "--output_dir",
            "/tmp/output",
            "--input_format",
            "parquet",
            "--chain",
            "Heavy,Light",
        ]
        with (
            patch("sys.argv", argv),
            patch.object(optimize_sequences_cli, "optimize_parquet_sequences") as optimize_parquet,
        ):
            optimize_sequences_cli.main()

        optimize_parquet.assert_called_once_with(
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            val_fraction=0.0,
            chunk_bytes="64MB",
            num_workers=None,
            seed=42,
            sequence_column="sequence_alignment_aa",
            filters={"Chain": ["Heavy", "Light"]},
            progress_dir=None,
        )
