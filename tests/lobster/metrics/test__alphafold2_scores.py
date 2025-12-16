from pathlib import Path
from unittest.mock import MagicMock, patch

from lobster.metrics._alphafold2_scores import (
    alphafold2_complex_scores,
    alphafold2_binder_scores,
    _compute_mean_scores,
    download_alphafold2_weights,
)


def test_compute_mean_scores():
    scores = {1: {"pLDDT": 85.5, "pTM": 0.75, "pAE": 3.2}, 2: {"pLDDT": 88.3, "pTM": 0.82, "pAE": 2.8}}

    result = _compute_mean_scores(scores)

    assert result == {"pLDDT": 86.9, "pTM": 0.78, "pAE": 3.0}
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


@patch("lobster.metrics._alphafold2_scores.Path")
def test_download_alphafold2_weights(mock_path_class):
    mock_path = MagicMock()
    mock_path.glob.return_value = [Path("params_model_1.npz"), Path("params_model_2.npz")]
    mock_path_class.return_value = mock_path

    result = download_alphafold2_weights("/fake/weights/dir")

    assert result == "/fake/weights/dir"
    mock_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_path.glob.assert_called_once_with("params_model_*.npz")


@patch("lobster.metrics._alphafold2_scores.download_alphafold2_weights")
@patch("lobster.metrics._alphafold2_scores.copy_dict")
@patch("lobster.metrics._alphafold2_scores.mk_afdesign_model")
@patch("lobster.metrics._alphafold2_scores.Path")
def test_alphafold2_binder_scores(mock_path_class, mock_mk_model, mock_copy_dict, mock_download):
    mock_download.return_value = "/fake/weights"

    mock_model = MagicMock()
    mock_model.aux = {"log": {"plddt": 85.0, "ptm": 0.75, "pae": 3.0}}
    mock_mk_model.return_value = mock_model

    mock_copy_dict.return_value = {"plddt": 85.0, "ptm": 0.75, "pae": 3.0}

    mock_path = MagicMock()
    mock_path_class.return_value = mock_path

    result = alphafold2_binder_scores(binder_sequence="ACDEFGHIKLMNPQRSTVWY", alphafold_weights_dir="/fake/weights")

    assert isinstance(result, dict)
    assert "pLDDT" in result
    assert "pTM" in result
    assert "pAE" in result
    assert result["pLDDT"] == 85.0
    assert result["pTM"] == 0.75
    assert result["pAE"] == 3.0

    mock_mk_model.assert_called_once()
    mock_model.prep_inputs.assert_called_once_with(length=20)
    mock_model.set_seq.assert_called_once_with("ACDEFGHIKLMNPQRSTVWY")
    assert mock_model.predict.call_count == 2


@patch("lobster.metrics._alphafold2_scores.download_alphafold2_weights")
@patch("lobster.metrics._alphafold2_scores.copy_dict")
@patch("lobster.metrics._alphafold2_scores.mk_afdesign_model")
@patch("lobster.metrics._alphafold2_scores.Path")
def test_alphafold2_complex_scores(mock_path_class, mock_mk_model, mock_copy_dict, mock_download):
    mock_download.return_value = "/fake/weights"

    mock_model = MagicMock()
    mock_model.aux = {"log": {"plddt": 87.5, "ptm": 0.80, "i_ptm": 0.72, "pae": 2.5, "i_pae": 3.1}}
    mock_mk_model.return_value = mock_model

    mock_copy_dict.return_value = {"plddt": 87.5, "ptm": 0.80, "i_ptm": 0.72, "pae": 2.5, "i_pae": 3.1}

    mock_path = MagicMock()
    mock_path_class.return_value = mock_path

    result = alphafold2_complex_scores(
        target_pdb="/fake/target.pdb",
        target_chain="A",
        binder_sequence="ACDEFGHIKLMNPQRSTVWY",
        alphafold_weights_dir="/fake/weights",
    )

    assert isinstance(result, dict)
    assert "pLDDT" in result
    assert "pTM" in result
    assert "i_pTM" in result
    assert "pAE" in result
    assert "i_pAE" in result
    assert result["pLDDT"] == 87.5
    assert result["pTM"] == 0.80
    assert result["i_pTM"] == 0.72
    assert result["pAE"] == 2.5
    assert result["i_pAE"] == 3.1

    mock_mk_model.assert_called_once()
    mock_model.prep_inputs.assert_called_once()
    assert mock_model.predict.call_count == 2
