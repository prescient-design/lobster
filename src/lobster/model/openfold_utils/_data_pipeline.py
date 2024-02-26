from typing import Mapping
import numpy as np
from transformers.models.esm.openfold_utils import residue_constants
from . import OFProtein


FeatureDict = Mapping[str, np.ndarray]


def _aatype_to_str_sequence(aatype):
    return "".join([residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))])


def make_sequence_features(sequence: str, description: str, num_res: int) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


def make_protein_features(
    protein_object: OFProtein,
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.0]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.0 if _is_distillation else 0.0).astype(np.float32)

    return pdb_feats


def make_pdb_features(
    protein_object: OFProtein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.0,
) -> FeatureDict:
    pdb_feats = make_protein_features(protein_object, description, _is_distillation=True)

    if is_distillation:
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats
