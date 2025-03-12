from enum import Enum


class Modality(Enum):
    SMILES = "SMILES"
    AMINO_ACID = "amino_acid"
    NUCLEOTIDE = "nucleotide"
    COORDINATES_3D = "3d_coordinates"
