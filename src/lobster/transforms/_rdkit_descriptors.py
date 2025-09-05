import logging
from typing import Any

from torch import Tensor

from lobster.transforms._transform import Transform
from lobster.transforms.functional import smiles_to_normalized_rdkit_descs, smiles_to_rdkit_descs

logger = logging.getLogger(__name__)


class SmilesToRDKitDescriptorsTransform(Transform):
    """Transforms a SMILES string to its RDKit descriptors or ``None`` if the conversion fails."""

    def __init__(self, normalize: bool = True, invert: bool = False, descriptor_list: list[str] | None = None) -> None:
        """
        Parameters
        ----------
        normalize : bool
            If ``True``, the output RDKit descriptors will be normalized. If ``False``, the RDKit descriptors will be
            returned as is.
        invert : bool
            If ``True``, the normalized RDKit descriptors will be inverted into a Normal distribution. If ``False``,
            the normalized RDKit descriptors will be returned as is.

            .. note::
                This is only applicable if ``normalize`` is ``True``.
        descriptor_list : list[str] | None, optional
            List of specific descriptor names to calculate. If None, calculates all available descriptors.
            For normalized descriptors, only descriptors with available distributions will be used.
            Defaults to None.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._normalize = normalize
        self._invert = invert
        self._descriptor_list = descriptor_list

        logger.info(
            f"Transform {self.__class__.__name__} initialized with normalize: {normalize}, invert: {invert}, descriptor list: {descriptor_list}"
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

    def _transform(self, input: str, parameters: dict[str, Any]) -> Tensor | None:
        """Convert a SMILES string to its RDKit descriptors.

        Parameters
        ----------
        input : str
            The SMILES string to convert.
        parameters : dict[str, Any]
            Not used in this transform but part of the interface.

        Returns
        -------
        list[float] | None
            The RDKit descriptors for the input SMILES string or ``None`` if the input is invalid.
        """
        if self._normalize:
            return smiles_to_normalized_rdkit_descs(input, self._invert, self._descriptor_list)

        return smiles_to_rdkit_descs(input, self._descriptor_list)
