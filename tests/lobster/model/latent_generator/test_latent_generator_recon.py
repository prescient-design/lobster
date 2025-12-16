import os
import importlib.util
from pathlib import Path

import pytest
import torch

from lobster.model.latent_generator.cmdline import decode, encode, load_model, methods
from lobster.model.latent_generator.io import load_ligand, load_pdb, writepdb, writepdb_ligand_complex
from lobster.model.latent_generator.tokenizer import L2Loss, LigandL2Loss

# Check for optional dependencies
if importlib.util.find_spec("cpdb"):
    HAS_CPDB = True
else:
    HAS_CPDB = False

if importlib.util.find_spec("rotary_embedding_torch"):
    HAS_ROTARY_EMBEDDING = True
else:
    HAS_ROTARY_EMBEDDING = False

# Test data paths
TEST_DATA_DIR = (
    Path(__file__).parents[4] / "src" / "lobster" / "model" / "latent_generator" / "example" / "example_pdbs"
)
PROTEIN_PDB = TEST_DATA_DIR / "7kdr_protein.pdb"
PROTEIN_LIGAND_PDB = TEST_DATA_DIR / "4erk_protein.pdb"
PROTEIN_LIGAND_SDF = TEST_DATA_DIR / "4erk_ligand.sdf"


loss_fn = L2Loss()
ligand_loss_fn = LigandL2Loss()


@pytest.fixture(scope="module")
def protein_data():
    """Load protein data for testing."""
    return load_pdb(str(PROTEIN_PDB))


@pytest.fixture(scope="module")
def protein_ligand_data():
    """Load protein-ligand data for testing."""
    pdb_data = load_pdb(str(PROTEIN_LIGAND_PDB))
    ligand_data = load_ligand(str(PROTEIN_LIGAND_SDF))

    # Add ligand information to pdb_data
    pdb_data["ligand_coords"] = ligand_data["atom_coords"]
    pdb_data["ligand_mask"] = ligand_data["mask"]
    pdb_data["ligand_residue_index"] = ligand_data["atom_indices"]
    pdb_data["ligand_atom_names"] = ligand_data["atom_names"]
    pdb_data["ligand_indices"] = ligand_data["atom_indices"]

    return pdb_data


@pytest.fixture(scope="module")
def ligand_data():
    """Load ligand-only data for testing."""
    ligand_data = load_ligand(str(PROTEIN_LIGAND_SDF))
    pdb_data = {"protein_coords": None, "protein_mask": None, "protein_seq": None}
    pdb_data["ligand_coords"] = ligand_data["atom_coords"]
    pdb_data["ligand_mask"] = ligand_data["mask"]
    pdb_data["ligand_residue_index"] = ligand_data["atom_indices"]
    pdb_data["ligand_atom_names"] = ligand_data["atom_names"]
    pdb_data["ligand_indices"] = ligand_data["atom_indices"]
    return pdb_data


class TestProteinOnlyModels:
    """Test protein-only model configurations."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "LG 20A seq 3di c6d Aux",
            "LG 20A",
            "LG 10A",
            "LG full attention",
        ],
    )
    @pytest.mark.skipif(
        not HAS_CPDB,
        reason="cpdb is not installed. Please install it with `uv sync --extra struct-gpu` or `uv sync --extra struct-cpu`",
    )
    def test_protein_encoding_decoding(self, model_name, protein_data):
        """Test encoding and decoding of protein structures."""
        # Load model
        load_model(
            methods[model_name].model_config.checkpoint,
            methods[model_name].model_config.config_path,
            methods[model_name].model_config.config_name,
            overrides=methods[model_name].model_config.overrides,
        )

        # Encode protein
        tokens, embeddings = encode(protein_data, return_embeddings=True)

        # Check token shapes
        assert isinstance(tokens, torch.Tensor)
        assert len(tokens.shape) == 3  # (batch, length, n_tokens)

        # Check embedding shapes
        assert isinstance(embeddings, torch.Tensor)
        assert len(embeddings.shape) == 3  # (batch, length, embedding_dim)

        # Decode back to structure
        decoded_outputs = decode(tokens, x_emb=embeddings)
        decoded_outputs = decoded_outputs[0]

        assert isinstance(decoded_outputs, torch.Tensor)
        assert len(decoded_outputs.shape) == 4  # (batch, length, n_atoms, 3)

        # Save decoded structure
        seq = torch.zeros(decoded_outputs.shape[1], dtype=torch.long)[None]
        output_path = f"test_decoded_{model_name.replace(' ', '_')}.pdb"
        writepdb(output_path, decoded_outputs[0], seq[0])

        # calculate the RMSD between the original and decoded structures
        mask = torch.ones(
            decoded_outputs.shape[0], decoded_outputs.shape[1], dtype=torch.bool, device=decoded_outputs.device
        )
        protein_data["coords_res"] = protein_data["coords_res"].to(decoded_outputs.device)
        loss = loss_fn(protein_data, decoded_outputs, mask)
        assert loss.item() < 100, f"Loss for {model_name} is too high: {loss.item()}"

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


class TestProteinLigandModels:
    """Test protein-ligand model configurations."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "LG Ligand 20A seq 3di Aux",
        ],
    )
    @pytest.mark.skipif(
        not HAS_CPDB,
        reason="cpdb is not installed. Please install it with `uv sync --extra struct-gpu` or `uv sync --extra struct-cpu`",
    )
    def test_protein_ligand_encoding_decoding(self, model_name, protein_ligand_data):
        """Test encoding and decoding of protein-ligand structures."""
        # Load model
        load_model(
            methods[model_name].model_config.checkpoint,
            methods[model_name].model_config.config_path,
            methods[model_name].model_config.config_name,
            overrides=methods[model_name].model_config.overrides,
        )

        # Encode protein-ligand
        tokens, embeddings = encode(protein_ligand_data, return_embeddings=True)

        # Check token shapes
        assert isinstance(tokens["protein_tokens"], torch.Tensor)
        assert isinstance(tokens["ligand_tokens"], torch.Tensor)
        assert isinstance(embeddings, torch.Tensor)
        assert len(tokens["protein_tokens"].shape) == 3  # (batch, length, n_tokens)
        assert len(tokens["ligand_tokens"].shape) == 3  # (batch, length, n_tokens)
        assert len(embeddings.shape) == 3  # (batch, length, embedding_dim)

        # Decode back to structure
        decoded_outputs = decode(tokens, x_emb=embeddings)
        decoded_outputs = decoded_outputs[0]
        seq = torch.zeros(decoded_outputs["protein_coords"].shape[1], dtype=torch.long)[None]
        output_path = f"test_decoded_{model_name.replace(' ', '_')}.pdb"
        writepdb_ligand_complex(
            output_path,
            ligand_atoms=decoded_outputs["ligand_coords"][0],
            ligand_atom_names=None,
            ligand_chain="L",
            ligand_resname="LIG",
            protein_atoms=decoded_outputs["protein_coords"][0],
            protein_seq=seq[0],
        )

        # put entries in protein_ligand_data["ligand_coords"] and protein_ligand_data["protein_coords"] to the same device as decoded_outputs
        protein_ligand_data = {
            k: v.to(decoded_outputs["protein_coords"].device)
            for k, v in protein_ligand_data.items()
            if isinstance(v, torch.Tensor)
        }

        # calculate the RMSD between the original and decoded structures
        loss = ligand_loss_fn(protein_ligand_data, decoded_outputs, protein_ligand_data)
        assert loss.item() < 100, f"Loss for {model_name} is too high: {loss.item()}"

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


class TestLigandOnlyModels:
    """Test ligand-only model configurations."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "LG Ligand 20A",
        ],
    )
    @pytest.mark.skipif(
        not HAS_ROTARY_EMBEDDING,
        reason="rotary_embedding_torch is not installed. Please install it to use rotary embeddings.",
    )
    def test_ligand_encoding_decoding(self, model_name, ligand_data):
        """Test encoding and decoding of ligand structures."""
        # Load model
        load_model(
            methods[model_name].model_config.checkpoint,
            methods[model_name].model_config.config_path,
            methods[model_name].model_config.config_name,
            overrides=methods[model_name].model_config.overrides,
        )

        # Encode ligand
        tokens, embeddings = encode(ligand_data, return_embeddings=True)

        # Check token shapes
        assert isinstance(tokens["ligand_tokens"], torch.Tensor)
        assert isinstance(embeddings, torch.Tensor)
        assert len(tokens["ligand_tokens"].shape) == 3  # (batch, length, n_tokens)
        assert len(embeddings.shape) == 3  # (batch, length, embedding_dim)

        # Check embedding shapes
        assert isinstance(embeddings, torch.Tensor)
        assert len(embeddings.shape) == 3  # (batch, length, embedding_dim)

        # Decode back to structure
        decoded_outputs = decode(tokens, x_emb=embeddings)
        decoded_outputs = decoded_outputs[0]

        assert isinstance(decoded_outputs["ligand_coords"], torch.Tensor)
        assert len(decoded_outputs["ligand_coords"].shape) == 3  # (batch, n_atoms, 3)

        # Save decoded structure
        output_path = f"test_decoded_{model_name.replace(' ', '_')}.pdb"

        # Save the reconstructed ligand
        writepdb_ligand_complex(
            output_path,
            ligand_atoms=decoded_outputs["ligand_coords"][0],
            ligand_atom_names=None,  # Optional: provide atom names if available
            ligand_chain="L",
            ligand_resname="LIG",
        )

        # Calculate the RMSD between the original and decoded structures
        ligand_data = {
            k: v.to(decoded_outputs["ligand_coords"].device)
            for k, v in ligand_data.items()
            if isinstance(v, torch.Tensor)
        }

        loss = ligand_loss_fn(ligand_data, decoded_outputs, ligand_data)
        assert loss.item() < 100, f"Loss for {model_name} is too high: {loss.item()}"

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
