"""Tests for ligand inference transforms."""

import torch

from lobster.transforms._ligand_inference import (
    reconstruct_smiles,
    reconstruct_smiles_from_tokens,
    smiles_to_ligand_input,
)
from lobster.transforms._ligand_chemistry import (
    atom_types_to_indices,
    smiles_to_graph,
)
from lobster.model.latent_generator.utils.residue_constants import (
    ELEMENT_VOCAB_EXTENDED_TO_IDX,
)


class TestSmilesToLigandInput:
    """Tests for smiles_to_ligand_input function."""

    def test_single_smiles(self):
        """Test single SMILES string conversion."""
        result = smiles_to_ligand_input("CCO")

        assert "ligand_atom_input_ids" in result
        assert "ligand_mask" in result
        assert "bond_matrix" in result
        assert "atom_types" in result
        assert "smiles" in result

        # Check shapes
        assert result["ligand_atom_input_ids"].shape == (1, 3)
        assert result["ligand_mask"].shape == (1, 3)
        assert result["bond_matrix"].shape == (1, 3, 3)

        # Check values
        assert result["smiles"] == ["CCO"]
        assert result["atom_types"] == [["C", "C", "O"]]

        # Check mask - all atoms should be valid
        assert result["ligand_mask"][0].sum() == 3.0

    def test_batch_smiles(self):
        """Test batch SMILES conversion."""
        smiles_list = ["CCO", "CC(=O)O"]  # Ethanol and Acetic acid
        result = smiles_to_ligand_input(smiles_list)

        # Batch size should be 2
        assert result["ligand_atom_input_ids"].shape[0] == 2

        # Max atoms should be 4 (acetic acid)
        assert result["ligand_atom_input_ids"].shape[1] == 4

        # First molecule should be padded
        assert result["ligand_mask"][0].sum() == 3.0  # CCO has 3 atoms
        assert result["ligand_mask"][1].sum() == 4.0  # CC(=O)O has 4 atoms

    def test_max_atoms_padding(self):
        """Test explicit max_atoms padding."""
        result = smiles_to_ligand_input("CCO", max_atoms=10)

        assert result["ligand_atom_input_ids"].shape == (1, 10)
        assert result["ligand_mask"].shape == (1, 10)
        assert result["bond_matrix"].shape == (1, 10, 10)

        # Only first 3 atoms should be valid
        assert result["ligand_mask"][0][:3].sum() == 3.0
        assert result["ligand_mask"][0][3:].sum() == 0.0

    def test_device_placement(self):
        """Test device placement of tensors."""
        result = smiles_to_ligand_input("CCO", device="cpu")

        assert result["ligand_atom_input_ids"].device.type == "cpu"
        assert result["ligand_mask"].device.type == "cpu"
        assert result["bond_matrix"].device.type == "cpu"

    def test_bond_matrix_values(self):
        """Test that bond matrix has correct values."""
        result = smiles_to_ligand_input("CCO")
        bond_matrix = result["bond_matrix"][0]

        # CCO: C-C single bond, C-O single bond
        # Diagonal should be 0
        assert bond_matrix[0, 0] == 0
        assert bond_matrix[1, 1] == 0
        assert bond_matrix[2, 2] == 0

        # Should have single bonds (value 1)
        assert bond_matrix[0, 1] > 0  # C-C bond
        assert bond_matrix[1, 2] > 0  # C-O bond

        # Should be symmetric
        assert bond_matrix[0, 1] == bond_matrix[1, 0]
        assert bond_matrix[1, 2] == bond_matrix[2, 1]


class TestReconstructSmiles:
    """Tests for reconstruct_smiles function."""

    def test_reconstruction_from_logits(self):
        """Test SMILES reconstruction from logits."""
        # Create fake logits for CCO (ethanol)
        # 3 atoms: C, C, O
        batch_size = 1
        n_atoms = 3
        atom_vocab_size = len(ELEMENT_VOCAB_EXTENDED_TO_IDX)
        num_bond_types = 6

        # Create atom logits with high values at correct positions
        atom_logits = torch.zeros(batch_size, n_atoms, atom_vocab_size)
        c_idx = ELEMENT_VOCAB_EXTENDED_TO_IDX["C"]
        o_idx = ELEMENT_VOCAB_EXTENDED_TO_IDX["O"]

        atom_logits[0, 0, c_idx] = 10.0  # First atom is C
        atom_logits[0, 1, c_idx] = 10.0  # Second atom is C
        atom_logits[0, 2, o_idx] = 10.0  # Third atom is O

        # Create bond logits for CCO
        bond_logits = torch.zeros(batch_size, n_atoms, n_atoms, num_bond_types)
        # Single bonds (type 1)
        bond_logits[0, 0, 1, 1] = 10.0  # C-C
        bond_logits[0, 1, 0, 1] = 10.0  # C-C (symmetric)
        bond_logits[0, 1, 2, 1] = 10.0  # C-O
        bond_logits[0, 2, 1, 1] = 10.0  # C-O (symmetric)

        mask = torch.ones(batch_size, n_atoms)

        smiles_list = reconstruct_smiles(atom_logits, bond_logits, mask)

        assert len(smiles_list) == 1
        # Should reconstruct to a valid SMILES (may not be exactly "CCO" due to canonicalization)
        assert smiles_list[0] != ""

    def test_reconstruction_with_empty_batch(self):
        """Test reconstruction with masked-out batch element."""
        batch_size = 2
        n_atoms = 3
        atom_vocab_size = len(ELEMENT_VOCAB_EXTENDED_TO_IDX)
        num_bond_types = 6

        atom_logits = torch.zeros(batch_size, n_atoms, atom_vocab_size)
        bond_logits = torch.zeros(batch_size, n_atoms, n_atoms, num_bond_types)

        # Create empty mask for first batch element
        mask = torch.zeros(batch_size, n_atoms)
        mask[1, :2] = 1.0  # Only second batch element has valid atoms

        smiles_list = reconstruct_smiles(atom_logits, bond_logits, mask)

        assert len(smiles_list) == 2
        assert smiles_list[0] == ""  # First should be empty


class TestReconstructSmilesFromTokens:
    """Tests for reconstruct_smiles_from_tokens function."""

    def test_roundtrip_reconstruction(self):
        """Test roundtrip: SMILES -> tokens -> SMILES."""
        original_smiles = "CCO"

        # Convert SMILES to graph
        atom_types, bond_matrix = smiles_to_graph(original_smiles)

        # Convert atom types to indices
        atom_tokens = atom_types_to_indices(atom_types).unsqueeze(0)  # Add batch dim
        bond_matrix_batched = bond_matrix.unsqueeze(0)  # Add batch dim
        mask = torch.ones(1, len(atom_types))

        # Reconstruct
        smiles_list = reconstruct_smiles_from_tokens(atom_tokens, bond_matrix_batched, mask)

        assert len(smiles_list) == 1
        # The reconstructed SMILES should represent the same molecule
        # (may be canonically different representation)
        assert smiles_list[0] != ""

    def test_roundtrip_complex_molecule(self):
        """Test roundtrip with more complex molecule."""
        original_smiles = "c1ccccc1"  # Benzene

        atom_types, bond_matrix = smiles_to_graph(original_smiles)
        atom_tokens = atom_types_to_indices(atom_types).unsqueeze(0)
        bond_matrix_batched = bond_matrix.unsqueeze(0)
        mask = torch.ones(1, len(atom_types))

        smiles_list = reconstruct_smiles_from_tokens(atom_tokens, bond_matrix_batched, mask)

        assert len(smiles_list) == 1
        assert smiles_list[0] != ""

    def test_with_padding(self):
        """Test reconstruction with padded tokens."""
        original_smiles = "CCO"

        atom_types, bond_matrix = smiles_to_graph(original_smiles)
        n_atoms = len(atom_types)
        max_atoms = 10

        # Create padded tokens
        atom_tokens = torch.full((1, max_atoms), ELEMENT_VOCAB_EXTENDED_TO_IDX["PAD"], dtype=torch.long)
        atom_tokens[0, :n_atoms] = atom_types_to_indices(atom_types)

        # Create padded bond matrix
        bond_matrix_padded = torch.zeros(1, max_atoms, max_atoms, dtype=torch.long)
        bond_matrix_padded[0, :n_atoms, :n_atoms] = bond_matrix

        # Create mask
        mask = torch.zeros(1, max_atoms)
        mask[0, :n_atoms] = 1.0

        smiles_list = reconstruct_smiles_from_tokens(atom_tokens, bond_matrix_padded, mask)

        assert len(smiles_list) == 1
        assert smiles_list[0] != ""


class TestIntegration:
    """Integration tests for the inference pipeline."""

    def test_full_pipeline(self):
        """Test the full inference pipeline: input -> model-like processing -> output."""
        # 1. Convert SMILES to input
        inputs = smiles_to_ligand_input("CCO")

        # 2. Simulate model processing (identity for now)
        # In real use, this would go through the model
        atom_tokens = inputs["ligand_atom_input_ids"]
        bond_matrix = inputs["bond_matrix"]
        mask = inputs["ligand_mask"]

        # 3. Reconstruct SMILES
        smiles_list = reconstruct_smiles_from_tokens(atom_tokens, bond_matrix, mask)

        assert len(smiles_list) == 1
        assert smiles_list[0] != ""

    def test_batch_pipeline(self):
        """Test full pipeline with batch of molecules."""
        smiles_batch = ["CCO", "CC(=O)O", "c1ccccc1"]

        # 1. Convert to inputs
        inputs = smiles_to_ligand_input(smiles_batch)

        # 2. Verify batch dimensions
        batch_size = len(smiles_batch)
        assert inputs["ligand_atom_input_ids"].shape[0] == batch_size

        # 3. Reconstruct
        smiles_list = reconstruct_smiles_from_tokens(
            inputs["ligand_atom_input_ids"],
            inputs["bond_matrix"],
            inputs["ligand_mask"],
        )

        assert len(smiles_list) == batch_size
        # All should reconstruct successfully
        for smi in smiles_list:
            assert smi != ""
