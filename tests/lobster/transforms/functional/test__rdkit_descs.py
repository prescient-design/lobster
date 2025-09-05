import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.stats
import torch

from lobster.transforms.functional._rdkit_descs import (
    _validate_descriptors_available,
    _validate_descriptors_have_distributions,
    smiles_to_normalized_rdkit_descs,
    smiles_to_rdkit_descs,
)

VALID_SMILES = "CCO"  # Ethanol
INVALID_SMILES = "invalid"


@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
@pytest.mark.parametrize(
    ["smiles", "expected"],
    [
        ("foo", None),
        ("CCO", [1.0, 2.0]),
    ],
)
def test_smiles_to_rdkit_descs(mock_smiles_to_desc, smiles, expected):
    if smiles == "foo":
        mock_smiles_to_desc.return_value = None
    else:
        mock_smiles_to_desc.return_value = {"desc1": 1.0, "desc2": 2.0}
    result = smiles_to_rdkit_descs(smiles)

    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert torch.allclose(result, torch.tensor(expected))


@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_rdkit_descs_with_nan(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = {"desc1": 1.0, "desc2": np.nan}
    result = smiles_to_rdkit_descs(VALID_SMILES)

    assert result is not None
    assert result[0] == 1.0
    assert math.isnan(result[1])


@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_invalid_smiles(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = None
    assert smiles_to_normalized_rdkit_descs("foo") is None


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_disjoint_descriptors(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = {"desc2": 2.0}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES)
    assert torch.equal(result, torch.tensor([]))


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (-1, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_no_invert(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = {"desc1": 0.5}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=False)

    assert result is not None
    assert len(result) == 1
    assert 0.0 <= result[0] <= 1.0
    assert np.isclose(result[0], scipy.stats.norm(0, 1).cdf(0.5))


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_clipping(mock_smiles_to_desc):
    # Test clipping below min
    mock_smiles_to_desc.return_value = {"desc1": -10}
    result_low = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=False)
    assert result_low is not None
    assert np.isclose(result_low[0], scipy.stats.norm(0, 1).cdf(0))

    # Test clipping above max
    mock_smiles_to_desc.return_value = {"desc1": 10}
    result_high = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=False)
    assert result_high is not None
    assert np.isclose(result_high[0], scipy.stats.norm(0, 1).cdf(1))


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (-1, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_invert(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = {"desc1": 0.5}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=True)

    assert result is not None
    assert len(result) == 1
    assert np.isclose(result[0], scipy.stats.norm.ppf(scipy.stats.norm(0, 1).cdf(0.5)))


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_invert_inf(mock_smiles_to_desc):
    # CDF will be 0.5, ppf(0.5) is ~0
    mock_smiles_to_desc.return_value = {"desc1": -10}
    result_neg_inf = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=True)
    assert result_neg_inf is not None
    assert result_neg_inf[0] == pytest.approx(0)

    # CDF will be 1, ppf(1) is ~1
    mock_smiles_to_desc.return_value = {"desc1": 10}
    result_pos_inf = smiles_to_normalized_rdkit_descs(VALID_SMILES, invert=True)
    assert result_pos_inf is not None
    assert result_pos_inf[0] == pytest.approx(1)


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_nan_handling(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = {"desc1": np.nan}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES)

    assert result is not None
    assert len(result) == 1
    assert math.isnan(result[0])


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (MagicMock(), (0, 1, 0, 0)), "desc2": (MagicMock(), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_output_length(mock_smiles_to_desc):
    mock_smiles_to_desc.return_value = {"desc1": 0.5, "desc2": 0.6}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES)

    assert result is not None
    assert len(result) == 2


def test_validate_descriptors_available_valid():
    """Test validation passes with valid descriptors."""
    available_descriptors = {"desc1", "desc2", "desc3"}
    descriptor_list = ["desc1", "desc2"]
    # Should not raise
    _validate_descriptors_available(descriptor_list, available_descriptors)


def test_validate_descriptors_available_invalid():
    """Test validation raises with invalid descriptors."""
    available_descriptors = {"desc1", "desc2"}
    descriptor_list = ["desc1", "desc3", "desc4"]

    with pytest.raises(ValueError, match="The following descriptors are not available in RDKit"):
        _validate_descriptors_available(descriptor_list, available_descriptors)


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0)), "desc2": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
def test_validate_descriptors_have_distributions_valid():
    """Test validation passes with descriptors that have distributions."""
    descriptor_list = ["desc1", "desc2"]
    # Should not raise
    _validate_descriptors_have_distributions(descriptor_list)


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
def test_validate_descriptors_have_distributions_invalid():
    """Test validation raises with descriptors that don't have distributions."""
    descriptor_list = ["desc1", "desc2", "desc3"]

    with pytest.raises(ValueError, match="The following descriptors do not have normalization distributions"):
        _validate_descriptors_have_distributions(descriptor_list)


@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_rdkit_descs_with_specific_descriptors(mock_smiles_to_desc):
    """Test smiles_to_rdkit_descs with specific descriptor list."""
    mock_smiles_to_desc.return_value = {"desc1": 1.0, "desc2": 2.0, "desc3": 3.0}
    result = smiles_to_rdkit_descs(VALID_SMILES, descriptor_list=["desc1", "desc3"])

    assert result is not None
    assert len(result) == 2
    assert torch.allclose(result, torch.tensor([1.0, 3.0]))


@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_rdkit_descs_invalid_descriptor_list(mock_smiles_to_desc):
    """Test smiles_to_rdkit_descs raises with invalid descriptor list."""
    mock_smiles_to_desc.return_value = {"desc1": 1.0, "desc2": 2.0}

    with pytest.raises(ValueError, match="The following descriptors are not available in RDKit"):
        smiles_to_rdkit_descs(VALID_SMILES, descriptor_list=["desc1", "desc3"])


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0)), "desc2": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_with_specific_descriptors(mock_smiles_to_desc):
    """Test smiles_to_normalized_rdkit_descs with specific descriptor list."""
    mock_smiles_to_desc.return_value = {"desc1": 0.5, "desc2": 0.6, "desc3": 0.7}
    result = smiles_to_normalized_rdkit_descs(VALID_SMILES, descriptor_list=["desc1", "desc2"])

    assert result is not None
    assert len(result) == 2


@patch(
    "lobster.transforms.functional._rdkit_descs.RDKIT_DESCRIPTOR_DISTRIBUTIONS",
    {"desc1": (scipy.stats.norm(0, 1), (0, 1, 0, 0))},
)
@patch("lobster.transforms.functional._rdkit_descs._smiles_to_descriptors")
def test_smiles_to_normalized_rdkit_descs_invalid_descriptor_list(mock_smiles_to_desc):
    """Test smiles_to_normalized_rdkit_descs raises with descriptor without distribution."""
    mock_smiles_to_desc.return_value = {"desc1": 0.5, "desc2": 0.6}

    with pytest.raises(ValueError, match="The following descriptors do not have normalization distributions"):
        smiles_to_normalized_rdkit_descs(VALID_SMILES, descriptor_list=["desc1", "desc2"])
