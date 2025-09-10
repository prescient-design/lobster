import logging
from typing import Any
from collections.abc import Sequence

from torch import Tensor

from lobster.constants import BIOPYTHON_FEATURES
from lobster.transforms._transform import Transform
from lobster.transforms.functional import get_biopython_features

logger = logging.getLogger(__name__)


class ProteinToBioPythonFeaturesTransform(Transform):
    """Transforms a protein sequence string to its BioPython features as a tensor or dict."""

    def __init__(self, feature_list: Sequence[str] | None = None, return_dict: bool = False) -> None:
        """
        Parameters
        ----------
        feature_list : Sequence[str] | None
            Sequence of specific BioPython features to extract. If None, extracts all available features.
            Available features: sequence_length, molecular_weight, aromaticity_index,
            instability_index, isoelectric_point, alpha_helix_fraction, turn_structure_fraction,
            beta_sheet_fraction, molar_extinction_coefficient_reduced_cysteines,
            molar_extinction_coefficient_oxidized_cysteines, grand_average_hydropathy_index,
            net_charge_at_ph_6, net_charge_at_ph_7.
        return_dict : bool
            If True, returns features as a dictionary. If False, returns as a tensor.
            Defaults to False.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        if feature_list is not None:
            # Validate feature list
            invalid_features = set(feature_list) - BIOPYTHON_FEATURES
            if invalid_features:
                raise ValueError(
                    f"Invalid features requested: {invalid_features}. Available features: {BIOPYTHON_FEATURES}"
                )

        self._feature_list = feature_list
        self._return_dict = return_dict

        logger.info(
            f"Transform {self.__class__.__name__} initialized with features: {feature_list or 'all'}, "
            f"return_dict: {return_dict}"
        )

    def _check_inputs(self, inputs: list[Any]) -> None:
        if not inputs:
            raise ValueError(f"{type(self).__name__} expects one string input, got none.")
        if len(inputs) > 1:
            raise ValueError(
                f"{type(self).__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise TypeError(f"{type(self).__name__} expects a string input, but got type {type(inputs[0])}.")

    def _transform(self, protein_sequence: str, parameters: dict[str, Any]) -> Tensor | dict[str, float]:
        """Convert a protein sequence to its BioPython features.

        Parameters
        ----------
        protein_sequence : str
            The protein sequence (single-letter amino acid codes) to convert.
        parameters : dict[str, Any]
            Not used in this transform but part of the interface.

        Returns
        -------
        Tensor | dict[str, float]
            The BioPython features for the input protein sequence as a tensor or dictionary,
            depending on the return_dict parameter.
        """
        return get_biopython_features(protein_sequence, self._feature_list, return_as_tensor=not self._return_dict)

    @property
    def available_features(self) -> set[str]:
        """Get all available BioPython features."""
        return BIOPYTHON_FEATURES.copy()
