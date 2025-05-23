from unittest.mock import patch

import pandas as pd
import pytest

from lobster.datasets import AtomicaDataset


class TestAtomicaDataset:
    def test__init__basic(self):
        """Test basic initialization with mocked data."""
        with patch("pandas.read_parquet") as mock_read_parquet:
            mock_df = pd.DataFrame(
                {
                    "sequence1": ["SEQ1"],
                    "modality1": ["amino_acid"],
                    "sequence2": ["SEQ2"],
                    "modality2": ["nucleotide"],
                    "sequence3": [None],
                    "interaction_type": ["protein-dna"],
                }
            )
            mock_read_parquet.return_value = mock_df

            dataset = AtomicaDataset()

            assert dataset.split == "train"
            assert dataset.columns == ["sequence1", "modality1", "sequence2", "modality2"]
            assert len(dataset) == 1

    def test__init__with_modalities(self):
        """Test filtering by modalities."""
        with patch("pandas.read_parquet") as mock_read_parquet:
            mock_df = pd.DataFrame(
                {
                    "sequence1": ["SEQ1", "SEQ2"],
                    "modality1": ["amino_acid", "amino_acid"],
                    "sequence2": ["SEQ3", "SEQ4"],
                    "modality2": ["nucleotide", "amino_acid"],
                    "sequence3": [None, None],
                    "interaction_type": ["protein-dna", "protein-protein"],
                }
            )
            mock_read_parquet.return_value = mock_df

            dataset = AtomicaDataset(modalities=["protein-dna"])

            assert len(dataset) == 1
            assert dataset.data["interaction_type"].iloc[0] == "protein-dna"

    def test__init__invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            AtomicaDataset(max_modalities=1)

        with pytest.raises(ValueError):
            AtomicaDataset(split="invalid")

        with pytest.raises(ValueError):
            AtomicaDataset(modalities=["invalid-modality"])

    def test___getitem__(self):
        """Test data retrieval."""
        with patch("pandas.read_parquet") as mock_read_parquet:
            mock_df = pd.DataFrame(
                {
                    "sequence1": ["SEQ1"],
                    "modality1": ["amino_acid"],
                    "sequence2": ["SEQ2"],
                    "modality2": ["nucleotide"],
                    "sequence3": [None],
                    "interaction_type": ["protein-dna"],
                }
            )
            mock_read_parquet.return_value = mock_df

            dataset = AtomicaDataset()
            dataset._x = [("SEQ1", "amino_acid", "SEQ2", "nucleotide")]

            assert dataset[0] == ("SEQ1", "amino_acid", "SEQ2", "nucleotide")

    def test___getitem__with_transform(self):
        """Test transform application."""
        with patch("pandas.read_parquet") as mock_read_parquet:
            mock_df = pd.DataFrame(
                {
                    "sequence1": ["seq1"],
                    "modality1": ["amino_acid"],
                    "sequence2": ["seq2"],
                    "modality2": ["nucleotide"],
                    "sequence3": [None],
                    "interaction_type": ["protein-dna"],
                }
            )
            mock_read_parquet.return_value = mock_df

            def transform(x):
                return tuple(item.upper() if isinstance(item, str) else item for item in x)

            dataset = AtomicaDataset(transform=transform)
            dataset._x = [("seq1", "amino_acid", "seq2", "nucleotide")]

            assert dataset[0] == ("SEQ1", "AMINO_ACID", "SEQ2", "NUCLEOTIDE")
