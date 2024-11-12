import os

import pytest
import torch
from Bio.PDB import PDBParser
from lobster.extern.openfold_utils import backbone_loss
from lobster.model import LobsterPLMFold
from lobster.transforms import StructureFeaturizer
from torch import Size
from transformers.models.esm.openfold_utils import residue_constants

atom37_n_atoms = 37

torch.backends.cuda.matmul.allow_tf32 = True


def _extract_sequence_from_pdb(pdb_path, pdb_id=""):
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_path)
    # assume single chain for the purposes of this test
    chain = list(structure.get_chains())[0]
    return "".join(residue_constants.restype_3to1[residue.resname] for residue in chain)


@pytest.fixture
def example_pdb_path(scope="session"):
    return os.path.join(os.path.dirname(__file__), "../../../../test_data/2ah5A02.pdb")


@pytest.fixture
def example_aa_sequence(scope="session"):
    path = os.path.join(os.path.dirname(__file__), "../../../../test_data/2ah5A02.pdb")
    return _extract_sequence_from_pdb(path)


@pytest.fixture
def structure_featurizer(scope="session"):
    return StructureFeaturizer()


@pytest.fixture
def device(scope="session"):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model_output(example_aa_sequence, device):
    # initialize model
    # m = LobsterPLMFold(model_name="esmfold_v1")  # pre-trained
    m = LobsterPLMFold(model_name="esmfold_v1")  # randomly initialized
    m.to(device)
    m.model.trunk.set_chunk_size(64)
    # m.model.esm = m.model.esm.half()

    tokenized_input = m.tokenizer([example_aa_sequence], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        output = m.model(tokenized_input["input_ids"].to(device))
    return output


@pytest.mark.skip(reason="Not currently used")
class TestBackBoneLoss:
    def test_backbone_loss(self, example_aa_sequence, example_pdb_path, structure_featurizer, device):
        seq_len = len(example_aa_sequence)
        with open(example_pdb_path, "r") as f:
            pdb_str = f.read()
        features = structure_featurizer(pdb_str, seq_len, "test")

        output = _get_model_output(example_aa_sequence, device)
        assert features["backbone_rigid_tensor"].shape == Size([seq_len, 4, 4])
        assert features["backbone_rigid_mask"].shape == Size([seq_len])
        assert output["frames"].shape == Size([8, 1, seq_len, 7])
        backbone_fape_loss = backbone_loss(
            backbone_rigid_tensor=features["backbone_rigid_tensor"].to(device),
            backbone_rigid_mask=features["backbone_rigid_mask"].to(device),
            traj=output["frames"],
        )
        tolerance_threshold_for_random_model = 10000.0
        assert backbone_fape_loss < tolerance_threshold_for_random_model
