from enum import Enum
from typing import Literal


class Modality(Enum):
    SMILES = "SMILES"
    AMINO_ACID = "amino_acid"
    NUCLEOTIDE = "nucleotide"
    COORDINATES_3D = "3d_coordinates"


ModalityType = Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]
