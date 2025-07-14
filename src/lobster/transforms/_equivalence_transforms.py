import logging
import random
from typing import Any

from lobster.constants import CODON_TABLE_PATH, CODON_TABLE_PATH_VENDOR, Modality
from lobster.transforms._transform import Transform
from lobster.transforms.functional import (
    convert_aa_to_nt_probabilistic,
    convert_aa_to_smiles,
    convert_nt_to_aa,
    convert_nt_to_smiles,
    convert_smiles_to_smiles,
)
from lobster.transforms.functional._utils import invert_residue_to_codon_mapping

from .functional._utils import json_load

logger = logging.getLogger(__name__)


class SmilesToSmilesPairTransform(Transform):
    """
    Transforms a SMILES string to its canonical form or a randomized equivalent SMILES string.
    If the conversion fails, the output SMILES string will be None.
    """

    input_modality = Modality.SMILES
    output_modalities = (
        Modality.SMILES,
        Modality.SMILES,
    )

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


class AminoAcidToSmilesPairTransform(Transform):
    """
    Transforms a peptide sequence string into a pair of (peptide_sequence, SMILES).
    If the conversion to SMILES fails, the SMILES string will be None.
    """

    input_modality = Modality.AMINO_ACID
    output_modalities = (
        Modality.AMINO_ACID,
        Modality.SMILES,
    )

    def __init__(self, randomize_smiles: bool = False, max_input_length: int | None = None) -> None:
        """
        Parameters
        ----------
        randomize_smiles : bool
            If True, the SMILES string will be randomized (non-canonical).
        max_input_length : int
            The maximum length of the input peptide sequence.
            Sequences longer than this will be truncated prior to conversion.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._randomize_smiles = randomize_smiles
        self._max_input_length = max_input_length

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
        # TODO: Optionally randomize where the peptide is cut
        if self._max_input_length is not None:
            input = input[: self._max_input_length]

        smiles_sequence = convert_aa_to_smiles(input, replace_unknown=False, randomize_smiles=self._randomize_smiles)

        return input, smiles_sequence


class NucleotideToSmilesPairTransform(Transform):
    """
    Transforms a nucleotide sequence string into a pair of (nucleotide_sequence, SMILES).
    If the conversion to SMILES fails, the SMILES string will be None.
    """

    input_modality = Modality.NUCLEOTIDE
    output_modalities = (
        Modality.NUCLEOTIDE,
        Modality.SMILES,
    )

    def __init__(
        self, randomize_smiles: bool = False, randomize_cap: bool = False, max_input_length: int | None = None
    ) -> None:
        """
        Parameters
        ----------
        randomize_smiles : bool
            If True, the SMILES string will be randomized (non-canonical).
        randomize_cap : bool
            If True, the cap of the SMILES string will be randomized.
            Randomization means that phosphate caps might be added to the 5' or 3' end.
        max_input_length : int
            The maximum length of the input nucleotide sequence.
            Sequences longer than this will be truncated prior to conversion.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._randomize_smiles = randomize_smiles
        self._randomize_cap = randomize_cap
        self._max_input_length = max_input_length

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

        # TODO: Optionally randomize where the nucleotide is cut
        if self._max_input_length is not None:
            input = input[: self._max_input_length]

        smiles_sequence = convert_nt_to_smiles(input, cap=cap, randomize_smiles=self._randomize_smiles)

        return input, smiles_sequence


class NucleotideToAminoAcidPairTransform(Transform):
    """
    Transforms a nucleotide sequence string into a pair of (nucleotide_sequence, protein_sequence).
    If the conversion to protein fails, the protein string will be None.
    By default, translation starts from the beginning of the sequence (frame 0).
    """

    input_modality = Modality.NUCLEOTIDE
    output_modalities = (
        Modality.NUCLEOTIDE,
        Modality.AMINO_ACID,
    )

    def __init__(
        self, reading_frame: int = 0, max_input_length: int | None = None, codon_table_path: str | None = None
    ) -> None:
        """
        Parameters
        ----------
        reading_frame : int
            The reading frame to use for translation (0, 1, or 2).
            Default is 0 (start from the beginning).
        max_input_length : int | None
            The maximum length of the input nucleotide sequence.
            Sequences longer than this will be truncated prior to conversion.
        codon_table_path : str | None
            The path to the codon table file to use for translation mapping.
            If None, uses the default CODON_TABLE_PATH.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        if reading_frame not in [0, 1, 2]:
            raise ValueError("reading_frame must be 0, 1, or 2")

        self._reading_frame = reading_frame
        self._max_input_length = max_input_length

        # Set default codon table path if None
        if codon_table_path is None:
            codon_table_path = CODON_TABLE_PATH

        # Load codon mappings
        self._residue_to_codon = json_load(codon_table_path)
        self._codon_to_residue = invert_residue_to_codon_mapping(self._residue_to_codon)

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
        Converts a nucleotide sequence to a protein sequence.

        Parameters
        ----------
        input : str
            The nucleotide sequence string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        tuple[str, str | None]
            A tuple containing the original nucleotide sequence and the converted protein sequence (or None if conversion failed).
        """
        # Canonicalize to upper
        input = input.upper()

        # Truncate if needed
        if self._max_input_length is not None:
            input = input[: self._max_input_length]

        # Apply reading frame offset
        if self._reading_frame > 0:
            input = input[self._reading_frame :]

        try:
            protein_sequence = convert_nt_to_aa(input, self._codon_to_residue)

            return input, protein_sequence

        except (KeyError, ValueError) as e:
            logger.warning(f"Conversion to protein failed for input: {input} with error: {e}")

            return input, None


class AminoAcidToNucleotidePairTransform(Transform):
    """
    Transforms a protein sequence string into a pair of (protein_sequence, nucleotide_sequence).
    If the conversion to nucleotide fails, the nucleotide string will be None.

    Note: This transformation is inherently ambiguous due to codon degeneracy.
    Multiple codons can code for the same amino acid, so the reverse translation
    uses probabilistic sampling based on codon usage frequencies.

    Default vendor codon table is from https://www.genscript.com/tools/codon-frequency-table.
    """

    input_modality = Modality.AMINO_ACID
    output_modalities = (
        Modality.AMINO_ACID,
        Modality.NUCLEOTIDE,
    )

    def __init__(
        self,
        max_input_length: int | None = None,
        vendor_table_path: str | None = None,
        add_stop_codon: bool = True,
        skip_unknown: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        max_input_length : int | None
            The maximum length of the input protein sequence.
            Sequences longer than this will be truncated prior to conversion.
        vendor_table_path : str | None
            The path to the vendor codon table file with usage frequencies.
            If None, uses the default CODON_TABLE_PATH_VENDOR.
        add_stop_codon : bool
            Whether to add a stop codon at the end of the nucleotide sequence.
        skip_unknown : bool
            If True, skip unknown amino acids instead of raising an error.
            If False (default), raise ValueError for unknown amino acids.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._max_input_length = max_input_length
        self._add_stop_codon = add_stop_codon
        self._skip_unknown = skip_unknown

        # Set default vendor table path if None
        if vendor_table_path is None:
            vendor_table_path = CODON_TABLE_PATH_VENDOR

        # Load vendor table for probabilistic sampling
        self._vendor_codon_table = json_load(vendor_table_path)

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
        Converts a protein sequence to a nucleotide sequence using probabilistic codon sampling.

        Parameters
        ----------
        input : str
            The protein sequence string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        tuple[str, str | None]
            A tuple containing the original protein sequence and the converted nucleotide sequence (or None if conversion failed).
        """
        # Canonicalize to upper
        input = input.upper()

        # Truncate if needed
        if self._max_input_length is not None:
            input = input[: self._max_input_length]

        try:
            # Use probabilistic sampling with vendor codon usage frequencies
            nucleotide_sequence = convert_aa_to_nt_probabilistic(
                input, self._vendor_codon_table, add_stop_codon=self._add_stop_codon, skip_unknown=self._skip_unknown
            )

            return input, nucleotide_sequence

        except (KeyError, ValueError) as e:
            logger.warning(f"Conversion to nucleotide failed for input: {input} with error: {e}")

            return input, None


class AminoAcidToNucleotideAndSmilesTransform(Transform):
    """
    Transforms a peptide sequence string into a triplet of (peptide_sequence, nucleotide_sequence, SMILES).
    If any conversion fails, the corresponding output will be None.
    Note: The nucleotide conversion is inherently ambiguous due to codon degeneracy.
    Multiple codons can code for the same amino acid, so the reverse translation
    uses probabilistic sampling based on codon usage frequencies.
    """

    input_modality = Modality.AMINO_ACID
    output_modalities = (
        Modality.AMINO_ACID,
        Modality.NUCLEOTIDE,
        Modality.SMILES,
    )

    def __init__(
        self,
        max_input_length: int | None = None,
        codon_vendor_table_path: str | None = None,
        add_stop_codon: bool = True,
        randomize_smiles: bool = False,
        skip_unknown: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        max_input_length : int | None
            The maximum length of the input peptide sequence.
            Sequences longer than this will be truncated prior to conversion.
        codon_vendor_table_path : str | None
            The path to the vendor codon table file with usage frequencies.
            If None, uses the default CODON_TABLE_PATH_VENDOR.
        add_stop_codon : bool
            Whether to add a stop codon at the end of the nucleotide sequence.
        randomize_smiles : bool
            If True, the SMILES string will be randomized (non-canonical).
        skip_unknown : bool
            If True, skip unknown amino acids instead of raising an error.
            If False (default), raise ValueError for unknown amino acids.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._max_input_length = max_input_length
        self._add_stop_codon = add_stop_codon
        self._randomize_smiles = randomize_smiles
        self._skip_unknown = skip_unknown

        # Set default vendor table path if None
        if codon_vendor_table_path is None:
            codon_vendor_table_path = CODON_TABLE_PATH_VENDOR

        # Load vendor table for probabilistic sampling
        self._vendor_codon_table = json_load(codon_vendor_table_path)

    def _check_inputs(self, inputs: list[Any]) -> None:
        if not inputs:
            raise ValueError(f"{self.__class__.__name__} expects one string input, got none.")
        if len(inputs) > 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise TypeError(f"{self.__class__.__name__} expects a string input, but got type {type(inputs[0])}.")

    def _transform(self, input: str, parameters: dict[str, Any]) -> tuple[str, str | None, str | None]:
        """
        Converts a peptide sequence to both nucleotide and SMILES representations.
        Parameters
        ----------
        input : str
            The peptide sequence string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.
        Returns
        -------
        tuple[str, str | None, str | None]
            A tuple containing:
            - The original peptide sequence
            - The converted nucleotide sequence (or None if conversion failed)
            - The converted SMILES string (or None if conversion failed)
        """
        # Canonicalize to upper
        input = input.upper()

        # Truncate if needed
        if self._max_input_length is not None:
            input = input[: self._max_input_length]

        nucleotide_sequence = None
        smiles_sequence = None

        try:
            # Convert to nucleotide
            nucleotide_sequence = convert_aa_to_nt_probabilistic(
                input, self._vendor_codon_table, add_stop_codon=self._add_stop_codon, skip_unknown=self._skip_unknown
            )
        except (KeyError, ValueError) as e:
            nucleotide_sequence = None
            logger.warning(f"Conversion to nucleotide failed for input: {input} with error: {e}")

        try:
            # Convert to SMILES
            smiles_sequence = convert_aa_to_smiles(
                input, replace_unknown=False, randomize_smiles=self._randomize_smiles
            )
        except (KeyError, ValueError) as e:
            smiles_sequence = None
            logger.warning(f"Conversion to SMILES failed for input: {input} with error: {e}")

        return input, nucleotide_sequence, smiles_sequence


class NucleotideToAminoAcidAndSmilesTransform(Transform):
    """
    Transforms a nucleotide sequence string into a triplet of (nucleotide_sequence, amino_acid_sequence, SMILES).
    If any conversion fails, the corresponding output will be None.
    The nucleotide sequence is expected to start with ATG (start codon).
    If the sequence doesn't start with ATG, the amino acid conversion will fail gracefully.
    """

    input_modality = Modality.NUCLEOTIDE
    output_modalities = (
        Modality.NUCLEOTIDE,
        Modality.AMINO_ACID,
        Modality.SMILES,
    )

    def __init__(
        self,
        max_input_length: int | None = None,
        codon_table_path: str | None = None,
        randomize_smiles: bool = False,
        randomize_cap: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        max_input_length : int | None
            The maximum length of the input nucleotide sequence.
            Sequences longer than this will be truncated prior to conversion.
        codon_table_path : str | None
            The path to the codon table file to use for translation mapping.
            If None, uses the default CODON_TABLE_PATH.
        randomize_smiles : bool
            If True, the SMILES string will be randomized (non-canonical).
        randomize_cap : bool
            If True, the cap of the SMILES string will be randomized.
            Randomization means that phosphate caps might be added to the 5' or 3' end.
        """
        super().__init__()

        # This transform expects a single string input
        self._transformed_types = (str,)

        self._max_input_length = max_input_length
        self._randomize_smiles = randomize_smiles
        self._randomize_cap = randomize_cap

        # Set default codon table path if None
        if codon_table_path is None:
            codon_table_path = CODON_TABLE_PATH

        # Load codon mappings
        self._residue_to_codon = json_load(codon_table_path)
        self._codon_to_residue = invert_residue_to_codon_mapping(self._residue_to_codon)

    def _check_inputs(self, inputs: list[Any]) -> None:
        if not inputs:
            raise ValueError(f"{self.__class__.__name__} expects one string input, got none.")
        if len(inputs) > 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a single string input, but got {len(inputs)} transformable inputs."
            )
        if not isinstance(inputs[0], str):
            raise TypeError(f"{self.__class__.__name__} expects a string input, but got type {type(inputs[0])}.")

    def _transform(self, input: str, parameters: dict[str, Any]) -> tuple[str, str | None, str | None]:
        """
        Converts a nucleotide sequence to both amino acid and SMILES representations.

        Parameters
        ----------
        input : str
            The nucleotide sequence string to convert.
        parameters : dict[str, Any]
             Not used in this transform but part of the interface.

        Returns
        -------
        tuple[str, str | None, str | None]
            A tuple containing:
            - The original nucleotide sequence
            - The converted amino acid sequence (or None if conversion failed)
            - The converted SMILES string (or None if conversion failed)
        """
        # Canonicalize to upper
        input = input.upper()

        # Truncate if needed
        if self._max_input_length is not None:
            input = input[: self._max_input_length]

        amino_acid_sequence = None
        smiles_sequence = None

        original_input = input

        # Check if sequence starts with ATG (start codon)
        if not input.startswith("ATG"):
            logger.warning(f"Nucleotide sequence does not start with ATG start codon: {input[:10]}...")
            amino_acid_sequence = None
        else:
            try:
                # Convert to amino acid
                amino_acid_sequence = convert_nt_to_aa(input, self._codon_to_residue)

            except (KeyError, ValueError) as e:
                amino_acid_sequence = None
                logger.warning(f"Conversion to amino acid failed for input: {input} with error: {e}")

        try:
            # Convert to SMILES
            if self._randomize_cap:
                cap = random.choice(["5'", "3'", "both", None])
            else:
                cap = None

            smiles_sequence = convert_nt_to_smiles(original_input, cap=cap, randomize_smiles=self._randomize_smiles)

        except (KeyError, ValueError) as e:
            smiles_sequence = None
            logger.warning(f"Conversion to SMILES failed for input: {original_input} with error: {e}")

        return original_input, amino_acid_sequence, smiles_sequence
