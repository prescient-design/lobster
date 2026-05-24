"""Tests for ligand chemistry utilities.

TDD tests for:
- smiles_to_graph: SMILES string -> (atom_types, bond_matrix)
- graph_to_smiles: (atom_types, bond_matrix) -> SMILES string
- atom_types_to_indices: element strings -> vocab indices
- indices_to_atom_types: vocab indices -> element strings
"""

import pytest
import torch
from rdkit import Chem


class TestAtomTypeConversion:
    """Tests for atom type <-> index conversion."""

    def test_atom_types_to_indices_basic(self):
        """Test converting basic atom types to indices."""
        from lobster.transforms._ligand_chemistry import atom_types_to_indices

        atom_types = ["C", "N", "O", "S"]
        indices = atom_types_to_indices(atom_types)

        assert isinstance(indices, torch.Tensor)
        assert indices.shape == (4,)
        # Check indices match ELEMENT_VOCAB_EXTENDED
        assert indices[0].item() == 3  # C
        assert indices[1].item() == 4  # N
        assert indices[2].item() == 5  # O
        assert indices[3].item() == 6  # S

    def test_atom_types_to_indices_with_halogens(self):
        """Test converting halogens to indices."""
        from lobster.transforms._ligand_chemistry import atom_types_to_indices

        atom_types = ["C", "F", "Cl", "Br", "I"]
        indices = atom_types_to_indices(atom_types)

        assert indices[0].item() == 3  # C
        assert indices[1].item() == 8  # F
        assert indices[2].item() == 9  # Cl
        assert indices[3].item() == 10  # Br
        assert indices[4].item() == 11  # I

    def test_atom_types_to_indices_unknown(self):
        """Test that unknown elements map to UNK."""
        from lobster.transforms._ligand_chemistry import atom_types_to_indices

        atom_types = ["C", "Xe", "Rn"]  # Xe and Rn not in vocab
        indices = atom_types_to_indices(atom_types)

        assert indices[0].item() == 3  # C
        assert indices[1].item() == 2  # UNK
        assert indices[2].item() == 2  # UNK

    def test_indices_to_atom_types_basic(self):
        """Test converting indices back to atom types."""
        from lobster.transforms._ligand_chemistry import indices_to_atom_types

        indices = torch.tensor([3, 4, 5, 6])  # C, N, O, S
        atom_types = indices_to_atom_types(indices)

        assert atom_types == ["C", "N", "O", "S"]

    def test_indices_to_atom_types_special(self):
        """Test that special tokens are handled."""
        from lobster.transforms._ligand_chemistry import indices_to_atom_types

        indices = torch.tensor([0, 1, 2, 3])  # PAD, MASK, UNK, C
        atom_types = indices_to_atom_types(indices)

        assert atom_types == ["PAD", "MASK", "UNK", "C"]

    def test_roundtrip_atom_types(self):
        """Test roundtrip: atom_types -> indices -> atom_types."""
        from lobster.transforms._ligand_chemistry import (
            atom_types_to_indices,
            indices_to_atom_types,
        )

        original = ["C", "N", "O", "F", "Cl", "Br", "S", "P"]
        indices = atom_types_to_indices(original)
        recovered = indices_to_atom_types(indices)

        assert recovered == original


class TestSmilesToGraph:
    """Tests for SMILES -> graph conversion."""

    def test_smiles_to_graph_ethanol(self):
        """Test parsing ethanol (CCO)."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        atom_types, bond_matrix = smiles_to_graph("CCO")

        # Ethanol has 3 heavy atoms: C, C, O
        assert len(atom_types) == 3
        assert "C" in atom_types
        assert "O" in atom_types

        # Bond matrix should be 3x3
        assert bond_matrix.shape == (3, 3)

        # Should be symmetric
        assert torch.allclose(bond_matrix, bond_matrix.T)

        # Should have 2 bonds (C-C and C-O)
        assert (bond_matrix > 0).sum().item() == 4  # 2 bonds * 2 (symmetric)

    def test_smiles_to_graph_acetic_acid(self):
        """Test parsing acetic acid (CC(=O)O)."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        atom_types, bond_matrix = smiles_to_graph("CC(=O)O")

        # 4 heavy atoms: C, C, O, O
        assert len(atom_types) == 4

        # Bond matrix should be 4x4
        assert bond_matrix.shape == (4, 4)

        # Check for double bond (value 2)
        assert (bond_matrix == 2).any(), "Should have a double bond"

    def test_smiles_to_graph_benzene(self):
        """Test parsing benzene (c1ccccc1)."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        atom_types, bond_matrix = smiles_to_graph("c1ccccc1")

        # 6 carbons
        assert len(atom_types) == 6
        assert all(a == "C" for a in atom_types)

        # Bond matrix should be 6x6
        assert bond_matrix.shape == (6, 6)

        # Aromatic bonds should be value 4
        assert (bond_matrix == 4).any(), "Should have aromatic bonds"

    def test_smiles_to_graph_aspirin(self):
        """Test parsing aspirin."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        atom_types, bond_matrix = smiles_to_graph(smiles)

        # Aspirin has 13 heavy atoms
        assert len(atom_types) == 13
        assert bond_matrix.shape == (13, 13)

    def test_smiles_to_graph_invalid(self):
        """Test that invalid SMILES raises error."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        with pytest.raises(ValueError):
            smiles_to_graph("invalid_smiles_xyz")


class TestGraphToSmiles:
    """Tests for graph -> SMILES conversion."""

    def test_graph_to_smiles_simple(self):
        """Test converting a simple graph back to SMILES."""
        from lobster.transforms._ligand_chemistry import graph_to_smiles

        # Ethanol: C-C-O
        atom_types = ["C", "C", "O"]
        bond_matrix = torch.tensor(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=torch.long,
        )

        smiles = graph_to_smiles(atom_types, bond_matrix)

        # Should be a valid SMILES
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_graph_to_smiles_with_double_bond(self):
        """Test converting graph with double bond."""
        from lobster.transforms._ligand_chemistry import graph_to_smiles

        # Formaldehyde: C=O
        atom_types = ["C", "O"]
        bond_matrix = torch.tensor(
            [
                [0, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )

        smiles = graph_to_smiles(atom_types, bond_matrix)

        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumAtoms() == 2

    def test_graph_to_smiles_with_aromatic(self):
        """Test converting graph with aromatic bonds."""
        from lobster.transforms._ligand_chemistry import graph_to_smiles

        # Benzene ring
        atom_types = ["C"] * 6
        # Aromatic ring: each C bonded to 2 neighbors with aromatic bonds
        bond_matrix = torch.zeros(6, 6, dtype=torch.long)
        for i in range(6):
            bond_matrix[i, (i + 1) % 6] = 4  # aromatic
            bond_matrix[(i + 1) % 6, i] = 4

        smiles = graph_to_smiles(atom_types, bond_matrix)

        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumAtoms() == 6


class TestRoundtrip:
    """Tests for SMILES -> graph -> SMILES roundtrip."""

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CC(C)C",  # isobutane
            "C1CCCCC1",  # cyclohexane
            "CC(=O)Nc1ccc(O)cc1",  # paracetamol
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        ],
    )
    def test_roundtrip_preserves_structure(self, smiles):
        """Test that roundtrip preserves molecular structure."""
        from lobster.transforms._ligand_chemistry import (
            graph_to_smiles,
            smiles_to_graph,
        )

        # Convert to graph
        atom_types, bond_matrix = smiles_to_graph(smiles)

        # Convert back to SMILES
        recovered_smiles = graph_to_smiles(atom_types, bond_matrix)

        # Verify structure is equivalent (canonical SMILES comparison)
        mol_original = Chem.MolFromSmiles(smiles)
        mol_recovered = Chem.MolFromSmiles(recovered_smiles)

        assert mol_recovered is not None, f"Failed to parse recovered SMILES: {recovered_smiles}"

        # Compare canonical SMILES
        canon_original = Chem.MolToSmiles(mol_original)
        canon_recovered = Chem.MolToSmiles(mol_recovered)

        assert canon_original == canon_recovered, (
            f"Roundtrip failed:\n"
            f"  Original: {smiles} -> {canon_original}\n"
            f"  Recovered: {recovered_smiles} -> {canon_recovered}"
        )

    def test_roundtrip_preserves_atom_count(self):
        """Test that roundtrip preserves atom count."""
        from lobster.transforms._ligand_chemistry import (
            graph_to_smiles,
            smiles_to_graph,
        )

        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        atom_types, bond_matrix = smiles_to_graph(smiles)
        recovered_smiles = graph_to_smiles(atom_types, bond_matrix)

        mol_original = Chem.MolFromSmiles(smiles)
        mol_recovered = Chem.MolFromSmiles(recovered_smiles)

        assert mol_original.GetNumAtoms() == mol_recovered.GetNumAtoms()

    def test_roundtrip_with_halogens(self):
        """Test roundtrip with halogenated compounds."""
        from lobster.transforms._ligand_chemistry import (
            graph_to_smiles,
            smiles_to_graph,
        )

        smiles = "Fc1ccc(Cl)cc1Br"  # trihalogenated benzene
        atom_types, bond_matrix = smiles_to_graph(smiles)

        assert "F" in atom_types
        assert "Cl" in atom_types
        assert "Br" in atom_types

        recovered_smiles = graph_to_smiles(atom_types, bond_matrix)
        mol = Chem.MolFromSmiles(recovered_smiles)
        assert mol is not None


class TestBondMatrixFormat:
    """Tests for bond matrix format and values."""

    def test_bond_types_values(self):
        """Test that bond matrix uses correct values."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        # Test single bonds
        _, bond_matrix = smiles_to_graph("CC")
        assert (bond_matrix == 1).any(), "Should have single bonds (value 1)"

        # Test double bonds
        _, bond_matrix = smiles_to_graph("C=C")
        assert (bond_matrix == 2).any(), "Should have double bonds (value 2)"

        # Test triple bonds
        _, bond_matrix = smiles_to_graph("C#C")
        assert (bond_matrix == 3).any(), "Should have triple bonds (value 3)"

        # Test aromatic bonds
        _, bond_matrix = smiles_to_graph("c1ccccc1")
        assert (bond_matrix == 4).any(), "Should have aromatic bonds (value 4)"

    def test_bond_matrix_symmetric(self):
        """Test that bond matrix is always symmetric."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        for smiles in ["CCO", "c1ccccc1", "CC(=O)O", "C1CC1"]:
            _, bond_matrix = smiles_to_graph(smiles)
            assert torch.allclose(bond_matrix, bond_matrix.T), f"Bond matrix not symmetric for {smiles}"

    def test_bond_matrix_diagonal_zero(self):
        """Test that bond matrix diagonal is always zero."""
        from lobster.transforms._ligand_chemistry import smiles_to_graph

        for smiles in ["CCO", "c1ccccc1", "CC(=O)O"]:
            _, bond_matrix = smiles_to_graph(smiles)
            assert (bond_matrix.diag() == 0).all(), f"Diagonal not zero for {smiles}"

