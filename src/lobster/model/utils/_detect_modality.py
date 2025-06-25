"""
Modality detection utilities.

This module provides utilities for detecting the modality of biological and chemical sequences.
"""

import re

from lobster.constants import Modality


def _detect_modality(text: str) -> Modality:
    """
    Detect the modality of a sequence based on its content.

    Parameters:
    -----------
    text : str
        The text sequence to analyze

    Returns:
    --------
    Modality
        The detected modality (SMILES, AMINO_ACID, or NUCLEOTIDE)

    Raises:
    -------
    ValueError
        If the sequence is empty, too short (< 3 characters), or if no modality can be determined
    """
    text = text.strip().upper()

    # Check for empty or very short sequences
    if len(text) < 3:
        raise ValueError(f"Sequence too short (length {len(text)}). Minimum length required is 3 characters.")

    # DNA patterns: contains only A, T, G, C (check this FIRST since it's more specific)
    dna_pattern = re.compile(r"^[ATGC]+$")
    if dna_pattern.match(text):
        return Modality.NUCLEOTIDE

    # Amino acid patterns: contains standard amino acid codes (but not just DNA bases)
    aa_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    if aa_pattern.match(text):
        return Modality.AMINO_ACID

    # SMILES patterns: contains molecular symbols and structures
    smiles_pattern = re.compile(r"[CNOSPFIHBrCl()\[\]=#@+\-\.\\\/]")
    if smiles_pattern.search(text) and any(c in text for c in "()[]=#@"):
        return Modality.SMILES

    # If no modality can be determined, raise an error
    raise ValueError(f"Unable to determine modality for sequence: {text}")
