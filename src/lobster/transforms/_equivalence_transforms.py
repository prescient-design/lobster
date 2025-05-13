from typing import Any

from lobster.transforms._convert_seqs import convert_aa_to_smiles, convert_nt_to_smiles
from lobster.transforms._transform import Transform


class PeptideToSmilesPairTransform(Transform):
    """
    Transforms a peptide sequence string into a pair of (peptide_sequence, SMILES).
    If the conversion to SMILES fails, the SMILES string will be None.
    """

    def __init__(self, randomize_smiles: bool = False) -> None:
        """
        Parameters
        ----------
        randomize_smiles : bool
            If True, the SMILES string will be randomized (non-canonical).
        """
        super().__init__()
        # This transform expects a single string input
        self._transformed_types = (str,)
        self._randomize_smiles = randomize_smiles

    def _check_inputs(self, inputs: list[Any]) -> None:
        if not inputs:
            raise ValueError(f"{self.__class__.__name__} expects one string input, got none.")
        if len(inputs) > 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise TypeError(f"{self.__class__.__name__} expects a string input, but got type {type(inputs[0])}.")

    def _transform(self, input: str, parameters: dict[str, Any]) -> tuple[str, str | None]:
        """
        Converts a peptide sequence to a SMILES string.

        Parameters
        ----------
        input : str
            The peptide sequence string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        tuple[str, str | None]
            A tuple containing the original peptide sequence and the converted SMILES string (or None if conversion failed).
        """
        smiles_sequence = convert_aa_to_smiles(input, randomize_smiles=self._randomize_smiles)
        return input, smiles_sequence


class NucleotideToSmilesPairTransform(Transform):
    """
    Transforms a nucleotide sequence string into a pair of (nucleotide_sequence, SMILES).
    If the conversion to SMILES fails, the SMILES string will be None.
    """

    def __init__(self, randomize_smiles: bool = False) -> None:
        super().__init__()
        # This transform expects a single string input
        self._transformed_types = (str,)
        self._randomize_smiles = randomize_smiles

    def _check_inputs(self, inputs: list[Any]) -> None:
        if not inputs:
            raise ValueError(f"{self.__class__.__name__} expects one string input, got none.")
        if len(inputs) > 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise TypeError(f"{self.__class__.__name__} expects a string input, but got type {type(inputs[0])}.")

    def _transform(self, input: str, parameters: dict[str, Any]) -> tuple[str, str | None]:
        """
        Converts a nucleotide sequence to a SMILES string directly.

        Parameters
        ----------
        input : str
            The nucleotide sequence string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        tuple[str, str | None]
            A tuple containing the original nucleotide sequence and the converted SMILES string (or None if conversion failed).
        """
        smiles_sequence = convert_nt_to_smiles(
            input.upper(), randomize_smiles=self._randomize_smiles
        )  # Canonicalize to upper for RDKit
        return input, smiles_sequence
