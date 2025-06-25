"""
Modality detection utilities.

This module provides utilities for detecting the modality of biological and chemical sequences.
"""

import re

from Bio.Seq import Seq
from rdkit import Chem

try:
    from Bio.SeqUtils.IUPACData import ambiguous_dna_letters, protein_letters
except ImportError:
    from Bio.Data.IUPACData import ambiguous_dna_letters, protein_letters

from lobster.constants import Modality


def _validate_dna_sequence(text: str) -> bool:
    """
    Validate DNA sequence using Biopython and IUPAC DNA letters.

    Parameters:
    -----------
    text : str
        The DNA sequence to validate

    Returns:
    --------
    bool
        True if the sequence is valid DNA, False otherwise
    """
    try:
        seq = Seq(text)
        return all(base in ambiguous_dna_letters for base in str(seq))
    except Exception:
        return False


def _validate_protein_sequence(text: str) -> bool:
    """
    Validate protein sequence using Biopython and IUPAC protein letters.

    Parameters:
    -----------
    text : str
        The protein sequence to validate

    Returns:
    --------
    bool
        True if the sequence is valid protein, False otherwise
    """
    try:
        seq = Seq(text)
        return all(residue in protein_letters for residue in str(seq))
    except Exception:
        return False


def _validate_smiles(text: str) -> bool:
    """
    Validate SMILES string using RDKit.

    Parameters:
    -----------
    text : str
        The SMILES string to validate

    Returns:
    --------
    bool
        True if the SMILES is valid, False otherwise
    """
    try:
        # Try to create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(text)
        return mol is not None
    except Exception:
        return False


def _detect_modality(text: str, validate: bool = True) -> Modality:
    """
    Detect the modality of a sequence based on its content.

    Parameters:
    -----------
    text : str
        The text sequence to analyze
    validate : bool, optional
        Whether to validate the sequence using Biopython (for DNA/protein)
        or RDKit (for SMILES). Default is True.

    Returns:
    --------
    Modality
        The detected modality (SMILES, AMINO_ACID, or NUCLEOTIDE)

    Raises:
    -------
    ValueError
        If the sequence is empty, too short (< 3 characters), if no modality can be determined,
        or if validation fails for the detected modality
    """
    text = text.strip().upper()

    # Check for empty or very short sequences
    if len(text) < 3:
        raise ValueError(f"Sequence too short (length {len(text)}). Minimum length required is 3 characters.")

    # DNA patterns: contains only A, T, G, C (check this FIRST since it's more specific)
    dna_pattern = re.compile(r"^[ATGC]+$")
    if dna_pattern.match(text):
        if validate and not _validate_dna_sequence(text):
            raise ValueError(f"Sequence appears to be DNA but failed validation: {text}")
        return Modality.NUCLEOTIDE

    # Amino acid patterns: contains standard amino acid codes (but not just DNA bases)
    aa_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    if aa_pattern.match(text):
        if validate and not _validate_protein_sequence(text):
            raise ValueError(f"Sequence appears to be protein but failed validation: {text}")
        return Modality.AMINO_ACID

    # SMILES patterns: contains molecular symbols and structures
    # First check for complex SMILES with special characters
    smiles_pattern = re.compile(r"[CNOSPFIHBrCl()\[\]=#@+\-\.\\\/]")
    if smiles_pattern.search(text) and any(c in text for c in "()[]=#@"):
        if validate and not _validate_smiles(text):
            raise ValueError(f"Sequence appears to be SMILES but failed validation: {text}")
        return Modality.SMILES

    # Also check for simple SMILES that only contain C, H, O, N, S, P, F, I, Br, Cl
    # but don't have special characters - these are still valid SMILES
    simple_smiles_pattern = re.compile(r"^[CHONSPFIBrCl0123456789]+$")
    if simple_smiles_pattern.match(text) and len(text) >= 3:
        if validate and not _validate_smiles(text):
            raise ValueError(f"Sequence appears to be SMILES but failed validation: {text}")
        return Modality.SMILES

    # If no modality can be determined, raise an error
    raise ValueError(f"Unable to determine modality for sequence: {text}")
