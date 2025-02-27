import os
from io import StringIO

import pytest
import torch
from Bio.PDB import PDBParser, Superimposer
from torch import Size, Tensor

from lobster.data import PDBDataModule
from lobster.extern.openfold_utils import backbone_loss
from lobster.model import LobsterPLMFold
from lobster.transforms import StructureFeaturizer

torch.backends.cuda.matmul.allow_tf32 = True


@pytest.fixture
def max_length():
    return 32


@pytest.fixture
def example_fv():
    fv_heavy = "VKLLEQSGAEVKKPGASVKVSCKASGYSFTSYGLHWVRQAPGQRLEWMGWISAGTGNTKYSQKFRGRVTFTRDTSATTAYMGLSSLRPEDTAVYYCARDPYGGGKSEFDYWGQGTLVTVSS"
    fv_light = (
        "ELVMTQSPSSLSASVGDRVNIACRASQGISSALAWYQQKPGKAPRLLIYDASNLESGVPSRFSGSGSGTDFTLTISSLQPEDFAIYYCQQFNSYPLTFGGGTKVEIKRTV"
    )
    return (fv_heavy, fv_light)


@pytest.fixture
def example_fv_pdb(scope="session"):
    return os.path.join(os.path.dirname(__file__), "../../../test_data/fv.pdb")


@pytest.fixture
def example_pdb_dir(scope="session"):
    return os.path.join(os.path.dirname(__file__), "../../../test_data/pdbs")


@pytest.fixture
def structure_featurizer(scope="session"):
    return StructureFeaturizer()


@pytest.fixture
def model(max_length, scope="session"):
    if os.getenv("CI"):
        pytest.skip("large download")
    return LobsterPLMFold(model_name="esmfold_v1", max_length=max_length)


def _get_batch(example_pdb_dir, max_length):
    dm = PDBDataModule(root=example_pdb_dir, batch_size=2, lengths=(0.0, 0.0, 1.0), max_length=max_length)
    dm.setup("predict")
    dataloader = dm.predict_dataloader()
    return next(iter(dataloader))


class TestLobsterPLMFold:
    def test_dataloader_tokenizer(self, model):
        inputs = ["ACDAC", "DEAPLND"]
        max_length = max([len(x) for x in inputs])
        tokenized_input = model.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        assert set(tokenized_input.keys()) == {"input_ids", "attention_mask"}
        assert tokenized_input["input_ids"].device == model.device
        assert isinstance(tokenized_input["input_ids"], Tensor)
        assert tokenized_input["input_ids"].size() == Size((len(inputs), max_length))
        assert tokenized_input["attention_mask"][0, :].sum() == len(inputs[0])
        assert tokenized_input["attention_mask"][1, :].sum() == len(inputs[1])

    @pytest.mark.skip(reason="fwd pass too slow")
    def test_predict_fv(self, model, example_fv):
        pdb_string = model.predict_fv(example_fv[0], example_fv[1])

        # NOTE from zadorozk: ruff checks were failing because ground_truth_file was not defined
        # TODO FIXME
        ground_truth_file = None

        # Parse the input PDB string
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure("pdb_string_structure", StringIO(pdb_string))

        # Parse the ground truth PDB file
        structure2 = parser.get_structure("ground_truth_structure", ground_truth_file)

        # Extract CA atoms for alignment
        atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
        atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]

        # Ensure same number of atoms for RMSD computation
        if len(atoms1) != len(atoms2):
            raise ValueError("Structures do not have the same number of CA atoms.")

        # Superimpose and compute RMSD
        super_imposer = Superimposer()
        super_imposer.set_atoms(atoms2, atoms1)

        # Apply the transformation to structure1
        super_imposer.apply(structure1.get_atoms())

        # Check that RMSD is within tolerance
        rmsd_tolerance = 2.0
        assert super_imposer.rms < rmsd_tolerance

    @pytest.mark.skip(reason="fwd pass too slow")
    def test_forward_pass_and_fape(self, model, example_pdb_dir, max_length):
        batch = _get_batch(example_pdb_dir, max_length)
        with torch.inference_mode():
            outputs = model.forward_pass(batch)

        expected_output_keys = set("sm", "final_atom_mask", "final_affine_tensor", "final_atom_positions")
        assert set(outputs.keys()) == expected_output_keys

        expected_structure_keys = set(
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
            "s_s",
            "s_z",
            "distogram_logits",
            "lm_logits",
            "aatype",
            "atom14_atom_exists",
            "residx_atom14_to_atom37",
            "residx_atom37_to_atom14",
            "atom37_atom_exists",
            "residue_index",
            "lddt_head",
            "plddt",
            "ptm_logits",
            "ptm",
            "aligned_confidence_probs",
            "predicted_aligned_error",
            "max_predicted_aligned_error",
        )
        assert set(outputs["sm"].keys()) == expected_structure_keys
        with torch.inference_mode():
            loss = backbone_loss(
                backbone_rigid_tensor=batch["backbone_rigid_tensor"],
                backbone_rigid_mask=batch["backbone_rigid_mask"],
                traj=outputs["sm"]["frames"],
            )
        fape_tolerance_for_untrained_model = 1000.0
        assert loss < fape_tolerance_for_untrained_model
