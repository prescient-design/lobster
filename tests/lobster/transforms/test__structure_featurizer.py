### Not used
# import os

# import pytest
# import torch
# from Bio.PDB import PDBParser
# from lobster.transforms import StructureFeaturizer
# from torch import Size
# from transformers.models.esm.openfold_utils import residue_constants

# atom37_n_atoms = 37
# atom14_n_atoms = 14
# n_euclidean_dims = 3

# torch.backends.cuda.matmul.allow_tf32 = True


# def _extract_sequence_from_pdb(pdb_path, pdb_id=""):
#     pdb_parser = PDBParser()
#     structure = pdb_parser.get_structure(pdb_id, pdb_path)
#     # assume single chain for the purposes of this test
#     chain = list(structure.get_chains())[0]
#     return "".join(residue_constants.restype_3to1[residue.resname] for residue in chain)


# @pytest.fixture
# def example_pdb_path(scope="session"):
#     return os.path.join(os.path.dirname(__file__), "../../../test_data/2ah5A02.pdb")


# @pytest.fixture
# def example_aa_sequence(scope="session"):
#     path = os.path.join(os.path.dirname(__file__), "../../../test_data/2ah5A02.pdb")
#     return _extract_sequence_from_pdb(path)


# @pytest.fixture
# def structure_featurizer(scope="session"):
#     return StructureFeaturizer()


# class TestStructureFeaturizer:
#     def test_pdb_str_to_structure_features(self, example_aa_sequence, example_pdb_path, structure_featurizer):
#         seq_len = len(example_aa_sequence)
#         with open(example_pdb_path, "r") as f:
#             pdb_str = f.read()
#         features = structure_featurizer(pdb_str, seq_len, "test")

#         expected_feature_keys = [
#             "aatype",
#             "between_segment_residues",
#             "domain_name",
#             "residue_index",
#             "seq_length",
#             "sequence",
#             "all_atom_positions",
#             "all_atom_mask",
#             "resolution",
#             "is_distillation",
#             "mask",
#             "rigidgroups_gt_frames",
#             "rigidgroups_gt_exists",
#             "rigidgroups_group_exists",
#             "rigidgroups_group_is_ambiguous",
#             "rigidgroups_alt_gt_frames",
#             "backbone_rigid_tensor",
#             "backbone_rigid_mask",
#             "atom14_atom_exists",
#             "residx_atom14_to_atom37",
#             "residx_atom37_to_atom14",
#             "atom37_atom_exists",
#             "atom14_gt_exists",
#             "atom14_gt_positions",
#             "atom14_alt_gt_positions",
#             "atom14_alt_gt_exists",
#             "atom14_atom_is_ambiguous",
#         ]

#         assert set(expected_feature_keys) == set(features.keys())

#         assert features["aatype"].shape == Size([seq_len])
#         assert features["mask"].shape == Size([seq_len])
#         assert features["all_atom_positions"].shape == Size([seq_len, atom37_n_atoms, 3])
#         assert features["all_atom_mask"].shape == Size([seq_len, atom37_n_atoms])
#         assert features["rigidgroups_gt_frames"].shape == Size([seq_len, 8, 4, 4])
#         assert features["rigidgroups_gt_exists"].shape == Size([seq_len, 8])
#         assert features["rigidgroups_group_exists"].shape == Size([seq_len, 8])
#         assert features["rigidgroups_group_is_ambiguous"].shape == Size([seq_len, 8])
#         assert features["rigidgroups_alt_gt_frames"].shape == Size([seq_len, 8, 4, 4])
#         assert features["backbone_rigid_tensor"].shape == Size([seq_len, 4, 4])
#         assert features["backbone_rigid_mask"].shape == Size([seq_len])
#         assert features["atom14_atom_exists"].shape == Size([seq_len, atom14_n_atoms])
#         assert features["residx_atom14_to_atom37"].shape == Size([seq_len, atom14_n_atoms])
#         assert features["residx_atom37_to_atom14"].shape == Size([seq_len, atom37_n_atoms])
#         assert features["atom37_atom_exists"].shape == Size([seq_len, atom37_n_atoms])
