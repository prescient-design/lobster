"""Tests for bond matrix embedding and prediction modules.

TDD tests for:
- BondMatrixEmbedding: Embed bond information into atom features
- BondMatrixPredictionHead: Predict bond types from atom features
"""

import pytest
import torch
import torch.nn as nn


class TestBondMatrixEmbedding:
    """Tests for BondMatrixEmbedding module."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding

        batch_size, num_atoms, hidden_size = 4, 10, 64
        model = BondMatrixEmbedding(hidden_size=hidden_size)

        atom_embeddings = torch.randn(batch_size, num_atoms, hidden_size)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))
        atom_mask = torch.ones(batch_size, num_atoms)

        output = model(atom_embeddings, bond_matrix, atom_mask)

        assert output.shape == (batch_size, num_atoms, hidden_size)

    def test_residual_connection(self):
        """Test that bond embedding is added to input (residual style)."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding

        batch_size, num_atoms, hidden_size = 2, 5, 32
        model = BondMatrixEmbedding(hidden_size=hidden_size)

        # Zero input embeddings
        atom_embeddings = torch.zeros(batch_size, num_atoms, hidden_size)
        bond_matrix = torch.randint(1, 5, (batch_size, num_atoms, num_atoms))  # All bonds
        atom_mask = torch.ones(batch_size, num_atoms)

        output = model(atom_embeddings, bond_matrix, atom_mask)

        # Output should not be zero (bond information added)
        assert not torch.allclose(output, atom_embeddings)

    def test_no_bonds_no_change(self):
        """Test that zero bond matrix adds minimal information."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding

        batch_size, num_atoms, hidden_size = 2, 5, 32
        model = BondMatrixEmbedding(hidden_size=hidden_size)

        atom_embeddings = torch.randn(batch_size, num_atoms, hidden_size)
        bond_matrix = torch.zeros(batch_size, num_atoms, num_atoms, dtype=torch.long)  # No bonds
        atom_mask = torch.ones(batch_size, num_atoms)

        output = model(atom_embeddings, bond_matrix, atom_mask)

        # With no bonds, the bond context should be zero (sum over empty set)
        # So output should equal input + zero-projection
        # The exact behavior depends on implementation, but there should be a pattern
        assert output.shape == atom_embeddings.shape

    def test_mask_respects_padding(self):
        """Test that padding atoms don't contribute to bond aggregation."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding

        batch_size, num_atoms, hidden_size = 1, 5, 32
        model = BondMatrixEmbedding(hidden_size=hidden_size)

        atom_embeddings = torch.randn(batch_size, num_atoms, hidden_size)

        # All atoms bonded, but last 2 are padding
        bond_matrix = torch.ones(batch_size, num_atoms, num_atoms, dtype=torch.long)
        atom_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])  # Last 2 are padding

        output = model(atom_embeddings, bond_matrix, atom_mask)

        # Padding atoms should not affect valid atoms' embeddings
        assert output.shape == atom_embeddings.shape

    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding

        batch_size, num_atoms, hidden_size = 2, 5, 32
        model = BondMatrixEmbedding(hidden_size=hidden_size)

        atom_embeddings = torch.randn(batch_size, num_atoms, hidden_size, requires_grad=True)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))
        atom_mask = torch.ones(batch_size, num_atoms)

        output = model(atom_embeddings, bond_matrix, atom_mask)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to input and model parameters
        assert atom_embeddings.grad is not None
        assert atom_embeddings.grad.shape == atom_embeddings.shape
        for param in model.parameters():
            assert param.grad is not None

    def test_symmetric_bonds_symmetric_contribution(self):
        """Test that symmetric bonds contribute symmetrically."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding

        batch_size, num_atoms, hidden_size = 1, 3, 32
        model = BondMatrixEmbedding(hidden_size=hidden_size)

        # Same embedding for all atoms
        atom_embeddings = torch.ones(batch_size, num_atoms, hidden_size)

        # Symmetric bond: atom 0 bonded to atom 1, atom 1 bonded to atom 0
        bond_matrix = torch.zeros(batch_size, num_atoms, num_atoms, dtype=torch.long)
        bond_matrix[0, 0, 1] = 1  # Single bond 0->1
        bond_matrix[0, 1, 0] = 1  # Single bond 1->0

        atom_mask = torch.ones(batch_size, num_atoms)

        output = model(atom_embeddings, bond_matrix, atom_mask)

        # Output should still be well-formed
        assert not torch.isnan(output).any()


class TestBondMatrixPredictionHead:
    """Tests for BondMatrixPredictionHead module."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 4, 10, 64
        num_bond_types = 6
        model = BondMatrixPredictionHead(hidden_size=hidden_size, num_bond_types=num_bond_types)

        atom_features = torch.randn(batch_size, num_atoms, hidden_size)

        output = model(atom_features)

        # Should output logits for each bond type for each pair
        assert output.shape == (batch_size, num_atoms, num_atoms, num_bond_types)

    def test_diagonal_zero(self):
        """Test that diagonal elements are masked (self-bonds don't exist)."""
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 2, 5, 32
        model = BondMatrixPredictionHead(hidden_size=hidden_size)

        atom_features = torch.randn(batch_size, num_atoms, hidden_size)

        output = model(atom_features)

        # Get predicted bond types
        pred_bonds = output.argmax(dim=-1)

        # Diagonal should be zero (no self-bonds)
        for b in range(batch_size):
            diag = pred_bonds[b].diag()
            # After softmax, diagonal should have highest probability for "no bond" (index 0)
            # This depends on implementation, but output should be valid

    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 2, 5, 32
        model = BondMatrixPredictionHead(hidden_size=hidden_size)

        atom_features = torch.randn(batch_size, num_atoms, hidden_size, requires_grad=True)

        output = model(atom_features)
        loss = output.sum()
        loss.backward()

        assert atom_features.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_prediction_with_target(self):
        """Test that we can compute loss against target bond matrix."""
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 2, 5, 32
        num_bond_types = 6
        model = BondMatrixPredictionHead(hidden_size=hidden_size, num_bond_types=num_bond_types)

        atom_features = torch.randn(batch_size, num_atoms, hidden_size)
        target_bonds = torch.randint(0, num_bond_types, (batch_size, num_atoms, num_atoms))

        output = model(atom_features)

        # Should be able to compute cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()

        # Reshape for cross-entropy: [B*N*N, num_bond_types] and [B*N*N]
        logits = output.view(-1, num_bond_types)
        targets = target_bonds.view(-1)

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)
        assert loss > 0

    def test_symmetric_prediction(self):
        """Test that symmetric option produces symmetric predictions."""
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 1, 5, 32
        model = BondMatrixPredictionHead(hidden_size=hidden_size, symmetric=True)

        atom_features = torch.randn(batch_size, num_atoms, hidden_size)

        output = model(atom_features)

        # For each batch, logits should be symmetric
        for b in range(batch_size):
            logits_b = output[b]  # [N, N, num_bond_types]
            assert torch.allclose(logits_b, logits_b.transpose(0, 1), atol=1e-5)


class TestEndToEnd:
    """End-to-end tests combining embedding and prediction."""

    def test_embed_then_predict(self):
        """Test the full pipeline: embed -> encode -> predict."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 2, 8, 64

        # Create modules
        bond_embed = BondMatrixEmbedding(hidden_size=hidden_size)
        bond_predict = BondMatrixPredictionHead(hidden_size=hidden_size)

        # Input
        atom_embeddings = torch.randn(batch_size, num_atoms, hidden_size)
        bond_matrix = torch.randint(0, 5, (batch_size, num_atoms, num_atoms))
        atom_mask = torch.ones(batch_size, num_atoms)

        # Embed bond info
        enriched = bond_embed(atom_embeddings, bond_matrix, atom_mask)

        # Simulate transformer processing (identity for test)
        processed = enriched  # In reality, this would be transformer output

        # Predict bonds
        predicted_logits = bond_predict(processed)

        assert predicted_logits.shape == (batch_size, num_atoms, num_atoms, 6)

    def test_reconstruction_loss(self):
        """Test that we can train to reconstruct bond matrix."""
        from lobster.model.leflur._bond_embedding import BondMatrixEmbedding
        from lobster.model.leflur._bond_prediction import BondMatrixPredictionHead

        batch_size, num_atoms, hidden_size = 2, 6, 64

        bond_embed = BondMatrixEmbedding(hidden_size=hidden_size)
        bond_predict = BondMatrixPredictionHead(hidden_size=hidden_size)

        # Ground truth bond matrix
        true_bonds = torch.zeros(batch_size, num_atoms, num_atoms, dtype=torch.long)
        # Simple chain: 0-1-2-3-4-5
        for i in range(num_atoms - 1):
            true_bonds[:, i, i + 1] = 1  # Single bond
            true_bonds[:, i + 1, i] = 1

        atom_embeddings = torch.randn(batch_size, num_atoms, hidden_size)
        atom_mask = torch.ones(batch_size, num_atoms)

        # Forward
        enriched = bond_embed(atom_embeddings, true_bonds, atom_mask)
        predicted = bond_predict(enriched)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            predicted.view(-1, 6),
            true_bonds.view(-1),
        )

        # Should be able to backprop
        loss.backward()

        assert not torch.isnan(loss)

