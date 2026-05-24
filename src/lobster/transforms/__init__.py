# imports below should not be moved to the top of this file,
# otherwise this would create circular imports
# from ._atom3d_ppi_transforms import (
#     Atom3DPPIToSequence,
#     Atom3DPPIToSequenceAndContactMap,
#     PairedSequenceToTokens,
# )
from ._auto_tokenizer_transform import AutoTokenizerTransform
from ._binarize import BinarizeTransform
from ._biopython_features import ProteinToBioPythonFeaturesTransform
from ._equivalence_transforms import (
    AminoAcidToNucleotideAndSmilesTransform,
    AminoAcidToNucleotidePairTransform,
    AminoAcidToSmilesPairTransform,
    NucleotideToAminoAcidAndSmilesTransform,
    NucleotideToAminoAcidPairTransform,
    NucleotideToSmilesPairTransform,
    SmilesToSmilesPairTransform,
)
from ._lambda import Lambda
from ._modality_aware_transform import ComposedModalityAwareTransform, ModalityAwareTransform
from ._rdkit_descriptors import SmilesToRDKitDescriptorsTransform
from ._structure_featurizer import StructureFeaturizer
from ._tokenizer_transform import TokenizerTransform
from ._transform import Transform

# Note: _ligand_chemistry and _ligand_inference have lazy imports to avoid
# circular dependencies with lobster.model.latent_generator.utils.residue_constants.
# Import them directly from the submodules when needed:
#   from lobster.transforms._ligand_chemistry import smiles_to_graph
#   from lobster.transforms._ligand_inference import smiles_to_ligand_input

__all__ = [
    "AutoTokenizerTransform",
    "BinarizeTransform",
    "ProteinToBioPythonFeaturesTransform",
    "NucleotideToAminoAcidPairTransform",
    "NucleotideToAminoAcidAndSmilesTransform",
    "NucleotideToSmilesPairTransform",
    "AminoAcidToSmilesPairTransform",
    "AminoAcidToNucleotidePairTransform",
    "SmilesToSmilesPairTransform",
    "AminoAcidToNucleotideAndSmilesTransform",
    "Lambda",
    "StructureFeaturizer",
    "TokenizerTransform",
    "Transform",
    "ModalityAwareTransform",
    "ComposedModalityAwareTransform",
    "SmilesToRDKitDescriptorsTransform",
]


def __getattr__(name):
    """Lazy import for ligand chemistry functions to avoid circular imports."""
    ligand_chemistry_exports = {
        "smiles_to_graph",
        "graph_to_smiles",
        "atom_types_to_indices",
        "indices_to_atom_types",
        "mol_to_bond_matrix",
        "sdf_to_bond_matrix",
    }
    ligand_inference_exports = {
        "smiles_to_ligand_input",
        "sdf_to_ligand_input",
        "reconstruct_smiles",
        "reconstruct_smiles_from_tokens",
    }

    if name in ligand_chemistry_exports:
        from . import _ligand_chemistry

        return getattr(_ligand_chemistry, name)
    elif name in ligand_inference_exports:
        from . import _ligand_inference

        return getattr(_ligand_inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
