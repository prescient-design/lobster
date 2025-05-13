import random
from typing import Any

from lobster.transforms._convert_seqs import convert_aa_to_smiles, convert_nt_to_smiles, convert_smiles_to_smiles
from lobster.transforms._transform import Transform


class SmilesToSmilesPairTransform(Transform):
    """
    Transforms a SMILES string to its canonical form or a randomized equivalent SMILES string.
    If the conversion fails, the output SMILES string will be None.
    """

    def __init__(self, randomize_smiles: bool = False) -> None:
        """
        Parameters
        ----------
        randomize_smiles : bool
            If True, the output SMILES string will be a randomized (non-canonical)
            equivalent of the input. If False (default), the canonical SMILES string
            will be returned.
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
        Converts a SMILES string to its canonical or a randomized form.

        Parameters
        ----------
        input : str
            The SMILES string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        tuple[str, str | None]
            A tuple containing the original SMILES string and the converted SMILES string (or None if conversion failed).
        """
        return input, convert_smiles_to_smiles(input, randomize_smiles=self._randomize_smiles)


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

    def __init__(self, randomize_smiles: bool = False, randomize_cap: bool = False) -> None:
        """
        Parameters
        ----------
        randomize_smiles : bool
            If True, the SMILES string will be randomized (non-canonical).
        randomize_cap : bool
            If True, the cap of the SMILES string will be randomized.
            Randomization means that phosphate caps might be added to the 5' or 3' end.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._randomize_smiles = randomize_smiles
        self._randomize_cap = randomize_cap

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
        Converts a nucleotide sequence to a SMILES string.

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
        # Canonicalize to upper
        input = input.upper()

        # Randomize cap if requested
        if self._randomize_cap:
            cap = random.choice(["5'", "3'", "both", None])
        else:
            cap = None

        smiles_sequence = convert_nt_to_smiles(input, cap=cap, randomize_smiles=self._randomize_smiles)
        return input, smiles_sequence
