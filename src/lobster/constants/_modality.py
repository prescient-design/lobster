from enum import StrEnum
from typing import Literal


class Modality(StrEnum):
    SMILES = "SMILES"
    AMINO_ACID = "amino_acid"
    NUCLEOTIDE = "nucleotide"
    COORDINATES_3D = "3d_coordinates"


ModalityType = Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]


def to_modality(modality: str | Modality) -> Modality:
    return Modality(modality) if isinstance(modality, str) else modality
