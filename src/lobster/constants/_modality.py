from enum import StrEnum
from typing import Literal


class Modality(StrEnum):
    SMILES = "SMILES"
    AMINO_ACID = "amino_acid"
    NUCLEOTIDE = "nucleotide"
    COORDINATES_3D = "3d_coordinates"


ModalityType = Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]


def to_modality(modality: str | Modality) -> Modality:
    """Convert a string or Modality enum to a Modality enum."""

    if isinstance(modality, Modality):
        return modality

    if isinstance(modality, str):
        normalized = modality.lower().strip()

        if normalized in ("smiles", "smi", "sm"):
            return Modality.SMILES
        elif normalized in ("amino_acid", "amino acid", "protein", "prot", "aa", "peptide", "polypeptide(D)"):
            return Modality.AMINO_ACID
        elif normalized in ("nucleotide", "nucleotides", "dna", "rna", "nt"):
            return Modality.NUCLEOTIDE

        raise ValueError(f"Invalid modality: '{modality}'. Valid options are: {[m.value for m in Modality]}")
