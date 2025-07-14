import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.stats

from lobster.transforms.functional._rdkit_descs import (
    smiles_to_normalized_rdkit_descs,
    smiles_to_rdkit_descs,
)

VALID_SMILES = "CCO"  # Ethanol
INVALID_SMILES = "invalid"


@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
@pytest.mark.parametrize(
    ["smiles", "expected"],
    [
        ("foo", None),
        ("CCO", [1.0, 2.0]),
    ],
)
def test_smiles_to_rdkit_descs(mock_calc, smiles, expected):
    mock_calc.return_value = {"desc1": 1.0, "desc2": 2.0}
    result = smiles_to_rdkit_descs(smiles)

    assert result is None if expected is None else result == expected


@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_rdkit_descs_with_nan(mock_calc):
    mock_calc.return_value = {"desc1": 1.0, "desc2": np.nan}
    result = smiles_to_rdkit_descs(VALID_SMILES)

    assert result is not None
    assert result[0] == 1.0
    assert math.isnan(result[1])


def test_smiles_to_normalized_rdkit_descs_invalid_smiles():
    assert smiles_to_normalized_rdkit_descs("foo") is None


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_disjoint_descriptors(mock_calc):
    mock_calc.return_value = {"desc2": 2.0}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES)
    assert result == []


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (-1, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_no_invert(mock_calc):
    mock_calc.return_value = {"desc1": 0.5}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=False)

    assert result is not None
    assert len(result) == 1
    assert 0.0 <= result[0] <= 1.0
    assert np.isclose(result[0], scipy.stats.norm(0, 1).cdf(0.5))


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_clipping(mock_calc):
    # Test clipping below min
    mock_calc.return_value = {"desc1": -10}
    result_low = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=False)
    assert result_low is not None
    assert np.isclose(result_low[0], scipy.stats.norm(0, 1).cdf(0))

    # Test clipping above max
    mock_calc.return_value = {"desc1": 10}
    result_high = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=False)
    assert result_high is not None
    assert np.isclose(result_high[0], scipy.stats.norm(0, 1).cdf(1))


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (-1, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_invert(mock_calc):
    mock_calc.return_value = {"desc1": 0.5}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=True)

    assert result is not None
    assert len(result) == 1
    assert np.isclose(result[0], scipy.stats.norm.ppf(scipy.stats.norm(0, 1).cdf(0.5)))


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_invert_inf(mock_calc):
    # CDF will be 0.5, ppf(0.5) is ~0
    mock_calc.return_value = {"desc1": -10}
    result_neg_inf = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=True)
    assert result_neg_inf is not None
    assert result_neg_inf[0] == pytest.approx(0)

    # CDF will be 1, ppf(1) is ~1
    mock_calc.return_value = {"desc1": 10}
    result_pos_inf = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=True)
    assert result_pos_inf is not None
    assert result_pos_inf[0] == pytest.approx(1)


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_nan_handling(mock_calc):
    mock_calc.return_value = {"desc1": np.nan}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES)

    assert result is not None
    assert len(result) == 1
    assert math.isnan(result[0])


@patch(
    "lobster.transforms.functional._rdkit_descs.DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (MagicMock(), (0, 1, 0, 0)), "desc2": (MagicMock(), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs.Descriptors.CalcMolDescriptors")
def test_smiles_to_normalized_rdkit_descs_output_length(mock_calc):
    mock_calc.return_value = {"desc1": 0.5, "desc2": 0.6}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES)

    assert result is not None
    assert len(result) == 2
