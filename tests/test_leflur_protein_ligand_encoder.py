"""Tests for Gen-UME protein-ligand encoder.

TDD tests for:
- LeFlurProteinLigandEncoderModule: Unified encoder for protein + ligand
- Forward pass with protein-only, ligand-only, and combined inputs
- Bond matrix embedding integration
- Bond matrix prediction integration
"""

import pytest
import torch


class TestLeFlurProteinLigandEncoderModule:
    """Tests for the unified protein-ligand encoder."""

    @pytest.fixture
    def encoder_kwargs(self):
        """Minimal encoder kwargs for testing."""
        return {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
        }

    def test_encoder_creation(self, encoder_kwargs):
        """Test that encoder can be instantiated."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        encoder = LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            **encoder_kwargs,
        )

        assert encoder is not None
        assert hasattr(encoder, "sequence_embedding")
        assert hasattr(encoder, "structure_embedding")
        assert hasattr(encoder, "ligand_atom_embedding")
        assert hasattr(encoder, "ligand_structure_embedding")
        assert hasattr(encoder, "bond_embedding")
        assert hasattr(encoder, "bond_prediction_head")

    def test_forward_protein_only(self, encoder_kwargs):
        """Test forward pass with protein-only input."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        encoder = LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            **encoder_kwargs,
        )

        batch_size, seq_len = 2, 50

        # Protein inputs
        sequence_input_ids = torch.randint(0, 33, (batch_size, seq_len))
        structure_input_ids = torch.randint(0, 4375, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        conditioning_tensor = torch.zeros(batch_size, seq_len, 1)

        # No ligand inputs
        output = encoder(
            sequence_input_ids=sequence_input_ids,
            structure_input_ids=structure_input_ids,
            attention_mask=attention_mask,
            conditioning_tensor=conditioning_tensor,
            # No ligand inputs
        )

        assert "sequence_logits" in output
        assert "structure_logits" in output
        assert output["sequence_logits"].shape == (batch_size, seq_len, 33)
        assert output["structure_logits"].shape == (batch_size, seq_len, 4375)

    def test_forward_protein_ligand(self, encoder_kwargs):
        """Test forward pass with protein + ligand input."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        encoder = LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            **encoder_kwargs,
        )

        batch_size, seq_len, num_atoms = 2, 50, 20

        # Protein inputs
        sequence_input_ids = torch.randint(0, 33, (batch_size, seq_len))
        structure_input_ids = torch.randint(0, 4375, (batch_size, seq_len))
        protein_mask = torch.ones(batch_size, seq_len)
        conditioning_tensor = torch.zeros(batch_size, seq_len, 1)

        # Ligand inputs
        ligand_atom_input_ids = torch.randint(0, 25, (batch_size, num_atoms))
        ligand_structure_input_ids = torch.randint(0, 4375, (batch_size, num_atoms))
        ligand_mask = torch.ones(batch_size, num_atoms)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))

        output = encoder(
            sequence_input_ids=sequence_input_ids,
            structure_input_ids=structure_input_ids,
            attention_mask=protein_mask,
            conditioning_tensor=conditioning_tensor,
            ligand_atom_input_ids=ligand_atom_input_ids,
            ligand_structure_input_ids=ligand_structure_input_ids,
            ligand_mask=ligand_mask,
            bond_matrix=bond_matrix,
        )

        # Protein outputs
        assert "sequence_logits" in output
        assert "structure_logits" in output
        assert output["sequence_logits"].shape == (batch_size, seq_len, 33)
        assert output["structure_logits"].shape == (batch_size, seq_len, 4375)

        # Ligand outputs
        assert "ligand_atom_logits" in output
        assert "ligand_structure_logits" in output
        assert "bond_logits" in output
        assert output["ligand_atom_logits"].shape == (batch_size, num_atoms, 25)
        assert output["ligand_structure_logits"].shape == (batch_size, num_atoms, 4375)
        assert output["bond_logits"].shape == (batch_size, num_atoms, num_atoms, 6)

    def test_forward_ligand_only(self, encoder_kwargs):
        """Test forward pass with ligand-only input (GEOM dataset)."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        encoder = LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            **encoder_kwargs,
        )

        batch_size, num_atoms = 2, 30

        # No protein inputs (empty tensors)
        sequence_input_ids = torch.empty(batch_size, 0, dtype=torch.long)
        structure_input_ids = torch.empty(batch_size, 0, dtype=torch.long)
        protein_mask = torch.empty(batch_size, 0)

        # Ligand inputs only
        ligand_atom_input_ids = torch.randint(0, 25, (batch_size, num_atoms))
        ligand_structure_input_ids = torch.randint(0, 4375, (batch_size, num_atoms))
        ligand_mask = torch.ones(batch_size, num_atoms)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))

        output = encoder(
            sequence_input_ids=sequence_input_ids,
            structure_input_ids=structure_input_ids,
            attention_mask=protein_mask,
            ligand_atom_input_ids=ligand_atom_input_ids,
            ligand_structure_input_ids=ligand_structure_input_ids,
            ligand_mask=ligand_mask,
            bond_matrix=bond_matrix,
        )

        # Ligand outputs should be present
        assert "ligand_atom_logits" in output
        assert "ligand_structure_logits" in output
        assert output["ligand_atom_logits"].shape == (batch_size, num_atoms, 25)

    def test_gradient_flow(self, encoder_kwargs):
        """Test that gradients flow through all components."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        encoder = LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            **encoder_kwargs,
        )

        batch_size, seq_len, num_atoms = 2, 20, 10

        # Inputs
        sequence_input_ids = torch.randint(0, 33, (batch_size, seq_len))
        structure_input_ids = torch.randint(0, 4375, (batch_size, seq_len))
        protein_mask = torch.ones(batch_size, seq_len)
        conditioning_tensor = torch.zeros(batch_size, seq_len, 1)

        ligand_atom_input_ids = torch.randint(0, 25, (batch_size, num_atoms))
        ligand_structure_input_ids = torch.randint(0, 4375, (batch_size, num_atoms))
        ligand_mask = torch.ones(batch_size, num_atoms)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))

        output = encoder(
            sequence_input_ids=sequence_input_ids,
            structure_input_ids=structure_input_ids,
            attention_mask=protein_mask,
            conditioning_tensor=conditioning_tensor,
            ligand_atom_input_ids=ligand_atom_input_ids,
            ligand_structure_input_ids=ligand_structure_input_ids,
            ligand_mask=ligand_mask,
            bond_matrix=bond_matrix,
        )

        # Compute loss from all outputs
        loss = (
            output["sequence_logits"].sum()
            + output["structure_logits"].sum()
            + output["ligand_atom_logits"].sum()
            + output["ligand_structure_logits"].sum()
            + output["bond_logits"].sum()
        )
        loss.backward()

        # Check gradients flow to key components
        assert encoder.sequence_embedding.weight.grad is not None
        assert encoder.ligand_atom_embedding.weight.grad is not None
        assert encoder.bond_embedding.bond_type_embedding.weight.grad is not None
        for param in encoder.bond_prediction_head.parameters():
            assert param.grad is not None

    def test_modality_embedding(self, encoder_kwargs):
        """Test that modality embedding distinguishes protein vs ligand."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        encoder = LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            **encoder_kwargs,
        )

        # Check modality embedding exists
        assert hasattr(encoder, "modality_embedding")
        # Should have at least 2 modalities: protein and ligand
        assert encoder.modality_embedding.num_embeddings >= 2


class TestMixedBatchHandling:
    """Tests for handling mixed batches (protein-only + protein-ligand)."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for mixed batch tests."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        return LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )

    def test_validity_masks(self, encoder):
        """Test that validity masks correctly handle mixed batches."""
        batch_size, seq_len, num_atoms = 4, 30, 15

        # Create mixed batch
        sequence_input_ids = torch.randint(0, 33, (batch_size, seq_len))
        structure_input_ids = torch.randint(0, 4375, (batch_size, seq_len))
        protein_mask = torch.ones(batch_size, seq_len)
        conditioning_tensor = torch.zeros(batch_size, seq_len, 1)

        # Ligand inputs (only some samples have ligands)
        ligand_atom_input_ids = torch.randint(0, 25, (batch_size, num_atoms))
        ligand_structure_input_ids = torch.randint(0, 4375, (batch_size, num_atoms))
        ligand_mask = torch.ones(batch_size, num_atoms)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))

        # Validity masks
        protein_valid_mask = torch.tensor([True, True, True, False])  # Last one is ligand-only
        ligand_valid_mask = torch.tensor([False, True, True, True])  # First one is protein-only

        output = encoder(
            sequence_input_ids=sequence_input_ids,
            structure_input_ids=structure_input_ids,
            attention_mask=protein_mask,
            conditioning_tensor=conditioning_tensor,
            ligand_atom_input_ids=ligand_atom_input_ids,
            ligand_structure_input_ids=ligand_structure_input_ids,
            ligand_mask=ligand_mask,
            bond_matrix=bond_matrix,
            protein_valid_mask=protein_valid_mask,
            ligand_valid_mask=ligand_valid_mask,
        )

        # All outputs should still have correct shapes
        assert output["sequence_logits"].shape == (batch_size, seq_len, 33)
        assert output["ligand_atom_logits"].shape == (batch_size, num_atoms, 25)


class TestBackwardCompatibility:
    """Tests for backward compatibility with protein-only Gen-UME."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for backward compatibility tests."""
        from lobster.model.leflur._leflur_protein_ligand_encoder import (
            LeFlurProteinLigandEncoderModule,
        )

        return LeFlurProteinLigandEncoderModule(
            sequence_token_vocab_size=33,
            structure_token_vocab_size=4375,
            ligand_atom_vocab_size=25,
            ligand_structure_vocab_size=4375,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )

    def test_protein_only_no_ligand_outputs(self, encoder):
        """Test that protein-only forward doesn't produce ligand outputs when not requested."""
        batch_size, seq_len = 2, 50

        output = encoder(
            sequence_input_ids=torch.randint(0, 33, (batch_size, seq_len)),
            structure_input_ids=torch.randint(0, 4375, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            conditioning_tensor=torch.zeros(batch_size, seq_len, 1),
            # No ligand inputs
        )

        # Protein outputs present
        assert "sequence_logits" in output
        assert "structure_logits" in output

        # Ligand outputs should be None or not present when no ligand input
        ligand_atom_logits = output.get("ligand_atom_logits")
        if ligand_atom_logits is not None:
            # If present, should be empty or zeros
            assert ligand_atom_logits.shape[1] == 0 or ligand_atom_logits.sum() == 0

    def test_output_shapes_match_original(self, encoder):
        """Test that protein output shapes match original Gen-UME."""
        batch_size, seq_len = 2, 100

        output = encoder(
            sequence_input_ids=torch.randint(0, 33, (batch_size, seq_len)),
            structure_input_ids=torch.randint(0, 4375, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            conditioning_tensor=torch.zeros(batch_size, seq_len, 1),
        )

        # Same shapes as original UMESequenceStructureEncoderModule
        assert output["sequence_logits"].shape == (batch_size, seq_len, 33)
        assert output["structure_logits"].shape == (batch_size, seq_len, 4375)
        assert "last_hidden_state" in output
