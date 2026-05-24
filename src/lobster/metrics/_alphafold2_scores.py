import re
import logging
import tarfile
from pathlib import Path
import pooch


try:
    from colabdesign import mk_afdesign_model
    from colabdesign.shared.utils import copy_dict

except ImportError:
    pass

from lobster.constants import DEFAULT_AF2_PREDICTION_MODELS, DEFAULT_AF2_WEIGHTS_DIR

logger = logging.getLogger(__name__)


def alphafold2_complex_scores(
    target_pdb: str,
    target_chain: str,
    binder_sequence: str,
    output_dir: str | None = None,
    num_recycles: int = 3,
    alphafold_weights_dir: str | None = None,
    use_multimer: bool = False,
    prediction_models: list[int] | None = None,
    mask_template_sequence: bool = True,
    mask_template_sidechains: bool = False,
) -> dict[str, float]:
    """
    Run AF2 prediction for binder-target complex.
    Needs target PDB file and binder sequence.

    Parameters
    ----------
    target_pdb : str
        Path to target PDB file
    target_chain : str
        Chain ID of target protein
    binder_sequence : str
        Amino acid sequence of binder
    output_dir : str | None
        Directory to save predicted structures. If provided,
        will save predicted structures there.
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
    dict[str, float]
        Dictionary containing mean scores across models:
        - pLDDT: overall confidence
        - pTM: predicted TM-score
        - i_pTM: interface TM-score
        - pAE: predicted aligned error
        - i_pAE: interface PAE

    Examples
    --------
    >>> pdb_path = "test_data/4N5T.pdb"
    >>> target_chain = "A"
    >>> binder_sequence = "LTFEYWAQLSAA"
    >>> complex_scores = alphafold2_complex_scores(
    ...     target_pdb=pdb_path,
    ...     target_chain=target_chain,
    ...     binder_sequence=binder_sequence,
    ...     output_dir="output/af2_complex",
    ...     alphafold_weights_dir="data2/alphafold2/weights"
    ... )
    >>> print(f"Complex scores: {complex_scores}")
    """
    logger.info(f"Running AlphaFold2 complex scores for target {target_pdb}, chain {target_chain}")

    prediction_models = DEFAULT_AF2_PREDICTION_MODELS if prediction_models is None else prediction_models
    download_alphafold2_weights(alphafold_weights_dir)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

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

    complex_stats = {}

    for model_num in prediction_models:
        logger.info(f"Predicting complex model {model_num + 1}")

        complex_model.predict(seq=binder_sequence, models=[model_num], num_recycles=num_recycles, verbose=False)

        if output_dir is not None:
            target_name = Path(target_pdb).stem
            output_pdb = Path(output_dir) / f"{target_name}_{target_chain}_complex_model{model_num + 1}.pdb"
            complex_model.save_pdb(str(output_pdb))
            logger.info(f"Saved complex model {model_num + 1} to {output_pdb}")

        metrics = copy_dict(complex_model.aux["log"])

        stats = {
            "pLDDT": metrics["plddt"],
            "pTM": metrics["ptm"],
            "i_pTM": metrics["i_ptm"],
            "pAE": metrics["pae"],
            "i_pAE": metrics["i_pae"],
        }
        complex_stats[model_num + 1] = stats

    mean_scores = _compute_mean_scores(complex_stats)

    logger.info(f"Mean scores: {mean_scores}")

    return mean_scores


def alphafold2_binder_scores(
    binder_sequence: str,
    output_dir: str | None = None,
    num_recycles: int = 3,
    alphafold_weights_dir: str | None = None,
    use_multimer: bool = False,
    prediction_models: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Run AF2 prediction for binder alone without PyRosetta relaxation.

    Parameters
    ----------
    binder_sequence : str
        Amino acid sequence of binder
    output_dir : str | None
        Directory to save predicted structures. If provided,
        will save predicted structures there.
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
    dict[str, float]
        Dictionary containing mean scores across models:
        - pLDDT: overall confidence
        - pTM: predicted TM-score
        - pAE: predicted aligned error

    Examples
    --------
    >>> peptide_sequence = "LTFEYWAQLSAA"
    >>> binder_scores = alphafold2_binder_scores(
    ...     binder_sequence=peptide_sequence,
    ...     output_dir="output/af2_binder",
    ...     alphafold_weights_dir="data/alphafold2/weights"
    ... )
    >>> print(f"Binder scores: {binder_scores}")
    """
    logger.info(f"Running AlphaFold2 binder scores for sequence: {binder_sequence}")

    prediction_models = DEFAULT_AF2_PREDICTION_MODELS if prediction_models is None else prediction_models
    download_alphafold2_weights(alphafold_weights_dir)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

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

    binder_stats = {}

    for model_num in prediction_models:
        logger.info(f"Predicting binder model {model_num + 1}")

        try:
            binder_model.predict(models=[model_num], num_recycles=num_recycles, verbose=False)
        except IndexError as e:
            raise IndexError(
                f"Model {model_num + 1} not found in AlphaFold2 weights. \
             Please check weight directory: {alphafold_weights_dir} contains weights."
            ) from e

        if output_dir is not None:
            output_pdb = Path(output_dir) / f"binder_{model_num + 1}.pdb"
            binder_model.save_pdb(str(output_pdb))
            logger.info(f"Saved binder model {model_num + 1} to {output_pdb}")

        metrics = copy_dict(binder_model.aux["log"])

        stats = {"pLDDT": metrics["plddt"], "pTM": metrics["ptm"], "pAE": metrics["pae"]}
        binder_stats[model_num + 1] = stats

    mean_scores = _compute_mean_scores(binder_stats)

    logger.info(f"Mean scores: {mean_scores}")

    return mean_scores


def _compute_mean_scores(scores: dict[int, dict[str, float]]) -> dict[str, float]:
    mean_scores = {}
    first_score = next(iter(scores.values()))
    for score_type in first_score:
        mean_scores[score_type] = round(sum(score[score_type] for score in scores.values()) / len(scores), 2)

    return mean_scores


def download_alphafold2_weights(weights_dir: str | None = None) -> str:
    """Download and extract AlphaFold2 weights if not already present."""
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
        known_hash="36d4b0220f3c735f3296d301152b738c9776d16981d054845a68a1370b26cfe3",
        path=str(weights_path),
    )

    logger.info(f"Extracting weights to {weights_dir}")

    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=weights_path)

    Path(tar_file).unlink()
    logger.info(f"AlphaFold2 weights extracted to {weights_dir}")

    return weights_dir
