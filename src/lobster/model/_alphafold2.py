import re
import logging
from pathlib import Path
import tarfile
import pooch

try:
    from colabdesign import mk_afdesign_model
    from colabdesign.shared.utils import copy_dict
except ImportError:
    pass

from lobster.constants import DEFAULT_AF2_PREDICTION_MODELS, DEFAULT_AF2_WEIGHTS_DIR

logger = logging.getLogger(__name__)


def predict_alphafold2_complex(
    target_pdb: str,
    target_chain: str,
    binder_sequence: str,
    num_recycles: int = 3,
    alphafold_weights_dir: str | None = None,
    use_multimer: bool = False,
    prediction_models: list[int] | None = None,
    mask_template_sequence: bool = True,
    mask_template_sidechains: bool = False,
) -> dict[int, dict]:
    """
    Predict binder-target complex structure using AlphaFold2.

    Parameters
    ----------
    target_pdb : str
        Path to target PDB file
    target_chain : str
        Chain ID of target protein
    binder_sequence : str
        Amino acid sequence of binder
    num_recycles : int
        Number of AF2 recycles
    alphafold_weights_dir : str | None
        Path to AlphaFold2 parameters directory
        AlphaFold2 weights will be downloaded there if not provided.
    use_multimer : bool
        Use AlphaFold-Multimer (False = monomer)
    prediction_models : list[int] | None
        Which AF2 model params to use (0-4). If None, uses [0, 1]
    mask_template_sequence : bool
        Mask template sequence
    mask_template_sidechains : bool
        Mask template sidechains

    Returns
    -------
    dict[int, dict]
        Dictionary keyed by model number (1-5), each containing:
        - 'aux': auxiliary data with metrics (plddt, ptm, i_ptm, pae, i_pae)
        - 'model': the AF2 model object (for saving PDB or accessing coordinates)
        - 'coordinates': numpy array of shape (N_residues, N_atoms, 3) with xyz coordinates

    Examples
    --------
    >>> pdb_path = "test_data/4N5T.pdb"
    >>> target_chain = "A"
    >>> binder_sequence = "LTFEYWAQLSAA"
    >>> predictions = predict_alphafold2_complex(
    ...     target_pdb=pdb_path,
    ...     target_chain=target_chain,
    ...     binder_sequence=binder_sequence,
    ...     alphafold_weights_dir="data2/alphafold2/weights"
    ... )
    >>> scores = predictions[1]['aux']['plddt']
    >>> coordinates = predictions[1]['coordinates']
    """
    logger.info(f"Running AlphaFold2 complex prediction for target {target_pdb}, chain {target_chain}")

    prediction_models = DEFAULT_AF2_PREDICTION_MODELS if prediction_models is None else prediction_models
    alphafold_weights_dir = download_alphafold2_weights(alphafold_weights_dir)

    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    binder_length = len(binder_sequence)

    complex_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=num_recycles,
        data_dir=alphafold_weights_dir,
        use_multimer=use_multimer,
        use_initial_guess=False,
        use_initial_atom_pos=False,
    )

    complex_model.prep_inputs(
        pdb_filename=target_pdb,
        chain=target_chain,
        binder_len=binder_length,
        rm_target_seq=mask_template_sequence,
        rm_target_sc=mask_template_sidechains,
    )

    predictions = {}

    for model_num in prediction_models:
        logger.info(f"Predicting complex model {model_num + 1}")

        complex_model.predict(seq=binder_sequence, models=[model_num], num_recycles=num_recycles, verbose=False)

        aux_data = copy_dict(complex_model.aux["log"])

        coordinates = complex_model._xyz if hasattr(complex_model, "_xyz") else None
        if coordinates is None and hasattr(complex_model, "_pos"):
            coordinates = complex_model._pos

        predictions[model_num + 1] = {
            "aux": aux_data,
            "model": complex_model,
            "coordinates": coordinates,
        }

    return predictions


def predict_alphafold2_binder(
    binder_sequence: str,
    num_recycles: int = 3,
    alphafold_weights_dir: str | None = None,
    use_multimer: bool = False,
    prediction_models: list[int] | None = None,
) -> dict[int, dict]:
    """
    Predict binder structure alone using AlphaFold2.

    Parameters
    ----------
    binder_sequence : str
        Amino acid sequence of binder
    num_recycles : int
        Number of AF2 recycles
    alphafold_weights_dir : str | None
        Path to AlphaFold2 parameters directory
        AlphaFold2 weights will be downloaded there if not provided.
    use_multimer : bool
        Use AlphaFold-Multimer (False = monomer)
    prediction_models : list[int] | None
        Which AF2 model params to use (0-4). If None, uses [0, 1]

    Returns
    -------
    dict[int, dict]
        Dictionary keyed by model number (1-5), each containing:
        - 'aux': auxiliary data with metrics (plddt, ptm, pae)
        - 'model': the AF2 model object (for saving PDB or accessing coordinates)
        - 'coordinates': numpy array of shape (N_residues, N_atoms, 3) with xyz coordinates

    Examples
    --------
    >>> peptide_sequence = "LTFEYWAQLSAA"
    >>> predictions = predict_alphafold2_binder(
    ...     binder_sequence=peptide_sequence,
    ...     alphafold_weights_dir="data/alphafold2/weights"
    ... )
    >>> scores = predictions[1]['aux']['plddt']
    >>> coordinates = predictions[1]['coordinates']
    """
    logger.info(f"Running AlphaFold2 binder prediction for sequence: {binder_sequence}")

    prediction_models = DEFAULT_AF2_PREDICTION_MODELS if prediction_models is None else prediction_models
    alphafold_weights_dir = download_alphafold2_weights(alphafold_weights_dir)

    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    binder_length = len(binder_sequence)

    binder_model = mk_afdesign_model(
        protocol="hallucination",
        use_templates=False,
        initial_guess=False,
        use_initial_atom_pos=False,
        num_recycles=num_recycles,
        data_dir=alphafold_weights_dir,
        use_multimer=use_multimer,
    )

    binder_model.prep_inputs(length=binder_length)
    binder_model.set_seq(binder_sequence)

    predictions = {}

    for model_num in prediction_models:
        logger.info(f"Predicting binder model {model_num + 1}")

        try:
            binder_model.predict(models=[model_num], num_recycles=num_recycles, verbose=False)
        except IndexError as e:
            raise IndexError(
                f"Model {model_num + 1} not found in AlphaFold2 weights. "
                f"Please check weight directory: {alphafold_weights_dir} contains weights."
            ) from e

        aux_data = copy_dict(binder_model.aux["log"])

        coordinates = binder_model._xyz if hasattr(binder_model, "_xyz") else None
        if coordinates is None and hasattr(binder_model, "_pos"):
            coordinates = binder_model._pos

        predictions[model_num + 1] = {
            "aux": aux_data,
            "model": binder_model,
            "coordinates": coordinates,
        }

    return predictions


def save_alphafold2_predictions(
    predictions: dict[int, dict],
    output_dir: str,
    prefix: str = "structure",
) -> None:
    """
    Save AlphaFold2 predicted structures to PDB files.

    Parameters
    ----------
    predictions : dict[int, dict]
        Predictions from predict_alphafold2_complex() or predict_alphafold2_binder()
    output_dir : str
        Directory to save PDB files
    prefix : str
        Prefix for output filenames

    Examples
    --------
    >>> predictions = predict_alphafold2_binder("LTFEYWAQLSAA")
    >>> save_alphafold2_predictions(predictions, "output/structures", prefix="binder")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_num, pred in predictions.items():
        output_pdb = output_path / f"{prefix}_model{model_num}.pdb"
        pred["model"].save_pdb(str(output_pdb))
        logger.info(f"Saved model {model_num} to {output_pdb}")


def download_alphafold2_weights(weights_dir: str | None = None) -> str:
    """
    Download and extract AlphaFold2 weights if not already present.

    Parameters
    ----------
    weights_dir : str | None
        Path to AlphaFold2 parameters directory. If None, uses default directory.

    Returns
    -------
    str
        Path to weights directory
    """
    weights_dir = DEFAULT_AF2_WEIGHTS_DIR if weights_dir is None else weights_dir

    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    param_files = list(weights_path.glob("params_model_*.npz"))

    if param_files:
        logger.info(f"AlphaFold2 weights found in {weights_dir}")
        return weights_dir

    logger.info(f"Downloading AlphaFold2 weights to {weights_dir}")
    url = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

    tar_file = pooch.retrieve(
        url=url,
        known_hash=None,
        path=str(weights_path),
    )

    logger.info(f"Extracting weights to {weights_dir}")

    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=weights_path)

    Path(tar_file).unlink()
    logger.info(f"AlphaFold2 weights extracted to {weights_dir}")

    return weights_dir
