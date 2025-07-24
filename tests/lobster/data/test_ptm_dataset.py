import unittest.mock
from pathlib import Path
import pandas as pd
import pytest

from lobster.datasets import PTMDataset


class TestPTMDataset:
    @unittest.mock.patch("pooch.retrieve")
    @unittest.mock.patch("pandas.read_csv")
    def test__init__(self, mock_read_csv, mock_retrieve, tmp_path):
        """Test basic initialization."""
        
        mock_df = pd.DataFrame({
            "protein_id": ["Q8VBW9", "Q60885"],
            "position": [4, 4],
            "ptm_type": ["N-linked asparagine", "N-linked asparagine"],
            "sequence": ["MPGQNYSTIS...", "MGDDNDTDIT..."]
        })
        mock_read_csv.return_value = mock_df
        
        
        mock_retrieve.return_value = tmp_path / "ptm_labels.csv"
        
        dataset = PTMDataset(root=tmp_path)
        
        assert len(dataset) == 2
        assert dataset.columns == ["protein_id", "position", "ptm_type", "sequence"]

    @unittest.mock.patch("pooch.retrieve")
    @unittest.mock.patch("pandas.read_csv")
    def test__getitem__(self, mock_read_csv, mock_retrieve, tmp_path):
        """Test data retrieval."""
        mock_df = pd.DataFrame({
            "protein_id": ["Q8VBW9"],
            "position": [4],
            "ptm_type": ["N-linked asparagine"],
            "sequence": ["MPGQNYS"]
        })
        mock_read_csv.return_value = mock_df
        mock_retrieve.return_value = tmp_path / "ptm_labels.csv"
        
        dataset = PTMDataset(root=tmp_path)
        sample = dataset[0]
        
        assert sample == ("Q8VBW9", 4, "N-linked asparagine", "MPGQNYS")

    @unittest.mock.patch("pooch.retrieve")
    @unittest.mock.patch("pandas.read_csv")
    def test_single_column(self, mock_read_csv, mock_retrieve, tmp_path):
        """Test single column returns string."""
        mock_df = pd.DataFrame({"sequence": ["MPGQNYS"]})
        mock_read_csv.return_value = mock_df
        mock_retrieve.return_value = tmp_path / "ptm_labels.csv"
        
        dataset = PTMDataset(root=tmp_path, columns=["sequence"])
        sample = dataset[0]
        
        assert sample == "MPGQNYS"
        assert isinstance(sample, str)