"""Tests for MoleculeNetDataset."""

from unittest.mock import patch

import pandas as pd
import pytest
import torch

from lobster.constants import MOLECULENET_TASKS
from lobster.datasets._moleculenet_dataset import MoleculeNetDataset


class TestMoleculeNetDataset:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            MoleculeNetDataset(task="not_a_task", download=False)

    @patch("lobster.datasets._moleculenet_dataset.pooch.retrieve")
    @patch("lobster.datasets._moleculenet_dataset.pandas.read_csv")
    def test_loads_and_filters_nan(self, mock_read_csv, mock_retrieve, tmp_path):
        mock_retrieve.return_value = str(tmp_path / "file.csv")
        mock_read_csv.return_value = pd.DataFrame(
            {
                "smiles": ["CCO", None, "CCC", "CCCC"],
                "p_np": [1, 0, None, 0],
            }
        )

        dataset = MoleculeNetDataset(root=tmp_path, task="BBBP", download=True)

        assert len(dataset) == 2
        smiles, y = dataset[0]
        assert isinstance(smiles, str)
        assert y.dtype == torch.long
        assert y.shape == (1,)

        task_type, *_ = MOLECULENET_TASKS["BBBP"]
        assert dataset.task_type == task_type
        mock_retrieve.assert_called_once()

    @patch("lobster.datasets._moleculenet_dataset.pooch.retrieve")
    @patch("lobster.datasets._moleculenet_dataset.pandas.read_csv")
    def test_regression_dtype(self, mock_read_csv, mock_retrieve, tmp_path):
        mock_retrieve.return_value = str(tmp_path / "file.csv")
        mock_read_csv.return_value = pd.DataFrame(
            {
                "smiles": ["CCO", "CCC"],
                "measured log solubility in mols per litre": [-0.5, 1.2],
            }
        )

        dataset = MoleculeNetDataset(root=tmp_path, task="ESOL", download=True)
        _, y = dataset[0]
        assert y.dtype == torch.float32
        assert dataset.task_type == "regression"
