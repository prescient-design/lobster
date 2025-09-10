import unittest.mock

import pandas as pd

from lobster.datasets import PTMDataset


class TestPTMDataset:
    @unittest.mock.patch("pooch.retrieve")
    @unittest.mock.patch("pandas.read_csv")
    def test__init__(self, mock_read_csv, mock_retrieve, tmp_path):
        """Test basic initialization."""

        mock_df = pd.DataFrame(
            {
                "AC_ID": ["Q8VBW9", "Q60885"],
                "pos": [4, 4],
                "label": ["N-linked asparagine", "N-linked asparagine"],
                "ori_seq": ["MPGQNYSTIS...", "MGDDNDTDIT..."],
                "token": ["<N-linked>...", "<N-linked>..."],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_retrieve.return_value = tmp_path / "ptm_labels.csv"

        dataset = PTMDataset(root=tmp_path)

        assert len(dataset) == 2
        assert dataset.columns == ["AC_ID", "pos", "label", "ori_seq", "token"]

    @unittest.mock.patch("pooch.retrieve")
    @unittest.mock.patch("pandas.read_csv")
    def test__getitem__(self, mock_read_csv, mock_retrieve, tmp_path):
        """Test data retrieval."""
        mock_df = pd.DataFrame(
            {
                "AC_ID": ["Q8VBW9"],
                "pos": [4],
                "label": ["N-linked asparagine"],
                "ori_seq": ["MPGQNYS"],
                "token": ["<N-linked>..."],
            }
        )
        mock_read_csv.return_value = mock_df
        mock_retrieve.return_value = tmp_path / "ptm_labels.csv"

        dataset = PTMDataset(root=tmp_path)
        sample = dataset[0]

        assert sample == ("Q8VBW9", 4, "N-linked asparagine", "MPGQNYS", "<N-linked>...")

    @unittest.mock.patch("pooch.retrieve")
    @unittest.mock.patch("pandas.read_csv")
    def test_single_column(self, mock_read_csv, mock_retrieve, tmp_path):
        """Test single column returns string."""
        mock_df = pd.DataFrame({"ori_seq": ["MPGQNYS"]})
        mock_read_csv.return_value = mock_df
        mock_retrieve.return_value = tmp_path / "ptm_labels.csv"

        dataset = PTMDataset(root=tmp_path, columns=["ori_seq"])
        sample = dataset[0]

        assert sample == "MPGQNYS"
        assert isinstance(sample, str)
