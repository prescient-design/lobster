from typing import Any, Dict, List, Optional, Set, Tuple

from lobster.transforms._convert_seqs import (
    convert_aa_to_smiles,
    convert_smiles_to_aa,
)
from lobster.transforms._transform import Transform

# Default set of allowed amino acids, can be configured
DEFAULT_ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY<unk>")


class SmilesToPeptidePairTransform(Transform):
    """
    Transforms a SMILES string into a pair of (SMILES, peptide_sequence).
    If the conversion to peptide fails, the peptide sequence will be None.
    """

    def __init__(self, allowed_aa: Optional[Set[str]] = None):
        super().__init__()
        self.allowed_aa = allowed_aa if allowed_aa is not None else DEFAULT_ALLOWED_AA
        # This transform expects a single string input
        self._transformed_types = (str,)

    def _check_inputs(self, inputs: List[Any]):
        if not inputs:
            raise ValueError(f"{self.__class__.__name__} expects one string input, got none.")
        if len(inputs) > 1:
            # This could be refined if the transform should operate on the first of many strings
            raise ValueError(
                f"{self.__class__.__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise ValueError(f"{self.__class__.__name__} expects a string input, but got type {type(inputs[0])}.")

    def _transform(self, input: str, parameters: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Converts a SMILES string to a peptide sequence.

        Parameters
        ----------
        input : str
            The SMILES string to convert.
        parameters : Dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        Tuple[str, Optional[str]]
            A tuple containing the original SMILES string and the converted peptide sequence (or None if conversion failed).
        """
        peptide_sequence = convert_smiles_to_aa(input)
        return input, peptide_sequence

    def extra_repr(self) -> str:
        return f"allowed_aa={''.join(sorted(list(self.allowed_aa)))}"


class PeptideToSmilesPairTransform(Transform):
    """
    Transforms a peptide sequence string into a pair of (peptide_sequence, SMILES).
    If the conversion to SMILES fails, the SMILES string will be None.
    """

    def __init__(self, allowed_aa: Optional[Set[str]] = None):
        super().__init__()
        self.allowed_aa = allowed_aa if allowed_aa is not None else DEFAULT_ALLOWED_AA
        # This transform expects a single string input
        self._transformed_types = (str,)

    def _check_inputs(self, inputs: List[Any]):
        if not inputs:
            raise ValueError(f"{self.__class__.__name__} expects one string input, got none.")
        if len(inputs) > 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise ValueError(f"{self.__class__.__name__} expects a string input, but got type {type(inputs[0])}.")
        # Additional check: ensure input looks like a peptide
        # This is a basic check; more sophisticated validation might be needed
        if not all(c in self.allowed_aa for c in inputs[0].upper() if c != "<" and c != ">"):  # crude check
            pass  # Decided to let conversion fail rather than raise here for now.
            # Or one might raise ValueError(f"Input '{inputs[0]}' does not look like a valid peptide sequence for {self.__class__.__name__}")

    def _transform(self, input: str, parameters: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Converts a peptide sequence to a SMILES string.

        Parameters
        ----------
        input : str
            The peptide sequence string to convert.
        parameters : Dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        Tuple[str, Optional[str]]
            A tuple containing the original peptide sequence and the converted SMILES string (or None if conversion failed).
        """
        smiles_sequence = convert_aa_to_smiles(input, self.allowed_aa)
        return input, smiles_sequence

    def extra_repr(self) -> str:
        return f"allowed_aa={''.join(sorted(list(self.allowed_aa)))}"


# __all__ = ["SmilesToPeptidePairTransform", "PeptideToSmilesPairTransform"]
