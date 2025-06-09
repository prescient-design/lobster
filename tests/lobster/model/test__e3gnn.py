"""Tests for E3GNN molecular structure encoder."""

import pytest
import torch
import torch.nn.functional as F

from lobster.model import E3GNN


class TestE3GNN:
    """Test suite for E3GNN model."""

    @pytest.fixture
    def sample_molecules(self):
        """Create sample molecular data for testing."""
        # Water molecule (H2O): O at origin, H atoms around it
        water_atoms = torch.tensor([8, 1, 1])  # O, H, H (atomic numbers)
        water_coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # O at origin
                [0.96, 0.0, 0.0],  # H atom 1
                [-0.24, 0.93, 0.0],  # H atom 2
            ]
        )

        # Methane molecule (CH4): C at origin, H atoms in tetrahedral geometry
        methane_atoms = torch.tensor([6, 1, 1, 1, 1])  # C, H, H, H, H
        methane_coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # C at origin
                [1.09, 0.0, 0.0],  # H atom 1
                [-0.36, 1.03, 0.0],  # H atom 2
                [-0.36, -0.51, 0.89],  # H atom 3
                [-0.36, -0.51, -0.89],  # H atom 4
            ]
        )

        return [(water_atoms, water_coords), (methane_atoms, methane_coords)]

    @pytest.fixture
    def batch_data(self, sample_molecules):
        """Create batched and padded molecular data."""
        max_atoms = 10
        batch_atoms = []
        batch_coords = []

        for atoms, coords in sample_molecules:
            n_atoms = len(atoms)

            # Pad with zeros (atomic number 0 = no atom)
            padded_atoms = torch.zeros(max_atoms, dtype=torch.long)
            padded_coords = torch.zeros(max_atoms, 3, dtype=torch.float)

            padded_atoms[:n_atoms] = atoms
            padded_coords[:n_atoms] = coords

            batch_atoms.append(padded_atoms)
            batch_coords.append(padded_coords)

        return torch.stack(batch_atoms), torch.stack(batch_coords)

    def test_model_initialization(self):
        """Test E3GNN model initialization with various configurations."""
        # Test default initialization
        model = E3GNN()
        assert model.input_dim == 119
        assert model.hidden_dim == 128
        assert model.output_dim == 128
        assert model.n_layers == 5

        # Test custom initialization
        model = E3GNN(input_dim=50, hidden_dim=64, output_dim=256, n_layers=3, dropout=0.2, use_atomic_embedding=True)
        assert model.input_dim == 64  # When using embedding, input_dim becomes hidden_dim
        assert model.hidden_dim == 64
        assert model.output_dim == 256
        assert model.n_layers == 3
        assert model.dropout == 0.2

    def test_forward_pass_atomic_embedding(self, batch_data):
        """Test forward pass with atomic embedding enabled."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=True, dropout=0.1)

        model.eval()
        with torch.no_grad():
            embeddings = model(batch_atoms, batch_coords)

        # Check output shape
        assert embeddings.shape == (2, 128)  # batch_size=2, output_dim=128

        # Check embeddings are finite
        assert torch.all(torch.isfinite(embeddings))

        # Check embeddings are not all zeros
        assert not torch.allclose(embeddings, torch.zeros_like(embeddings))

    def test_forward_pass_feature_input(self, batch_data):
        """Test forward pass with feature vectors instead of atomic numbers."""
        batch_atoms, batch_coords = batch_data

        # Convert atomic numbers to one-hot features (simplified)
        max_atomic_num = 20
        batch_size, max_atoms = batch_atoms.shape

        # Create one-hot encoded features
        batch_features = torch.zeros(batch_size, max_atoms, max_atomic_num)
        for i in range(batch_size):
            for j in range(max_atoms):
                if batch_atoms[i, j] > 0:
                    atomic_num = min(batch_atoms[i, j].item(), max_atomic_num - 1)
                    batch_features[i, j, atomic_num] = 1.0

        model = E3GNN(
            input_dim=max_atomic_num, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=False, dropout=0.1
        )

        model.eval()
        with torch.no_grad():
            embeddings = model(batch_features, batch_coords)

        # Check output shape
        assert embeddings.shape == (2, 128)
        assert torch.all(torch.isfinite(embeddings))

    def test_invariance_to_translation(self, batch_data):
        """Test that embeddings are invariant to molecular translation."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=True)

        # Original coordinates
        model.eval()
        with torch.no_grad():
            embeddings_original = model(batch_atoms, batch_coords)

        # Translated coordinates
        translation = torch.tensor([5.0, -3.0, 2.0])
        batch_coords_translated = batch_coords + translation.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embeddings_translated = model(batch_atoms, batch_coords_translated)

        # Embeddings should be (approximately) the same
        assert torch.allclose(embeddings_original, embeddings_translated, atol=1e-6)

    def test_invariance_to_rotation(self, batch_data):
        """Test that embeddings are invariant to molecular rotation."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=True)

        # Original coordinates
        model.eval()
        with torch.no_grad():
            embeddings_original = model(batch_atoms, batch_coords)

        # Create a rotation matrix (rotation around z-axis by 45 degrees)
        angle = torch.pi / 4
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        # Apply rotation
        batch_coords_rotated = torch.matmul(batch_coords, rotation_matrix.T)

        with torch.no_grad():
            embeddings_rotated = model(batch_atoms, batch_coords_rotated)

        # Embeddings should be (approximately) the same
        assert torch.allclose(embeddings_original, embeddings_rotated, atol=1e-5)

    def test_different_molecules_different_embeddings(self, batch_data):
        """Test that different molecules produce different embeddings."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=3, use_atomic_embedding=True)

        model.eval()
        with torch.no_grad():
            embeddings = model(batch_atoms, batch_coords)

        # Water and methane should have different embeddings
        water_embedding = embeddings[0]
        methane_embedding = embeddings[1]

        # Check that embeddings are not identical
        assert not torch.allclose(water_embedding, methane_embedding, atol=1e-4)

        # Check cosine similarity is reasonable but not 1.0
        similarity = F.cosine_similarity(water_embedding.unsqueeze(0), methane_embedding.unsqueeze(0))
        assert 0.0 <= similarity.item() <= 0.99

    def test_batch_consistency(self, batch_data):
        """Test that processing molecules individually vs in batch gives consistent results."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=True)

        model.eval()
        with torch.no_grad():
            # Batch processing
            batch_embeddings = model(batch_atoms, batch_coords)

            # Individual processing
            individual_embeddings = []
            for i in range(batch_atoms.shape[0]):
                atoms_single = batch_atoms[i : i + 1]
                coords_single = batch_coords[i : i + 1]
                embedding = model(atoms_single, coords_single)
                individual_embeddings.append(embedding[0])

            individual_embeddings = torch.stack(individual_embeddings)

        # Should be identical
        assert torch.allclose(batch_embeddings, individual_embeddings, atol=1e-6)

    def test_gradient_flow(self, batch_data):
        """Test that gradients flow properly through the model."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=32, output_dim=64, n_layers=2, use_atomic_embedding=True)

        # Ensure model requires gradients
        model.train()

        # Forward pass
        embeddings = model(batch_atoms, batch_coords)

        # Simple loss (sum of embeddings)
        loss = embeddings.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are not zero for most parameters
        params_with_grad = 0
        total_params = 0

        for param in model.parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                    params_with_grad += 1

        # At least 80% of parameters should have non-zero gradients
        assert params_with_grad / total_params >= 0.8

    def test_atomic_number_limit(self):
        """Test that atomic numbers are properly validated."""
        model = E3GNN(
            input_dim=50,  # Set a lower limit
            hidden_dim=64,
            use_atomic_embedding=True,
        )

        # Create data with atomic number exceeding the limit
        invalid_atoms = torch.tensor([[60, 1, 1]])  # Atomic number 60 > 50
        coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        model.eval()
        with pytest.raises(AssertionError, match="Atomic number too large for embedding"):
            with torch.no_grad():
                model(invalid_atoms, coords)

    def test_empty_molecule_handling(self):
        """Test handling of empty molecules (all atoms are 0)."""
        model = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, use_atomic_embedding=True)

        # Empty molecule (all atomic numbers are 0)
        empty_atoms = torch.zeros(1, 10, dtype=torch.long)
        empty_coords = torch.zeros(1, 10, 3)

        model.eval()
        with torch.no_grad():
            embeddings = model(empty_atoms, empty_coords)

        # Should produce finite embeddings (likely zeros due to masking)
        assert torch.all(torch.isfinite(embeddings))
        assert embeddings.shape == (1, 128)

    def test_model_reproducibility(self, batch_data):
        """Test that model produces reproducible results with fixed seed."""
        batch_atoms, batch_coords = batch_data

        # Set seed for reproducibility
        torch.manual_seed(42)
        model1 = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=True)

        torch.manual_seed(42)
        model2 = E3GNN(input_dim=119, hidden_dim=64, output_dim=128, n_layers=2, use_atomic_embedding=True)

        # Both models should have identical parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

        # Both models should produce identical outputs
        model1.eval()
        model2.eval()
        with torch.no_grad():
            embeddings1 = model1(batch_atoms, batch_coords)
            embeddings2 = model2(batch_atoms, batch_coords)

        assert torch.allclose(embeddings1, embeddings2)

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 5])
    def test_different_layer_counts(self, batch_data, n_layers):
        """Test model with different numbers of layers."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(input_dim=119, hidden_dim=32, output_dim=64, n_layers=n_layers, use_atomic_embedding=True)

        model.eval()
        with torch.no_grad():
            embeddings = model(batch_atoms, batch_coords)

        assert embeddings.shape == (2, 64)
        assert torch.all(torch.isfinite(embeddings))

    @pytest.mark.parametrize("activation", ["SiLU", "GELU"])
    def test_different_activations(self, batch_data, activation):
        """Test model with different activation functions."""
        batch_atoms, batch_coords = batch_data

        model = E3GNN(
            input_dim=119, hidden_dim=32, output_dim=64, n_layers=2, act_fn=activation, use_atomic_embedding=True
        )

        model.eval()
        with torch.no_grad():
            embeddings = model(batch_atoms, batch_coords)

        assert embeddings.shape == (2, 64)
        assert torch.all(torch.isfinite(embeddings))

    def test_invalid_activation_function(self):
        """Test that invalid activation functions raise appropriate errors."""
        with pytest.raises(ValueError, match="Unsupported activation function"):
            E3GNN(act_fn="InvalidActivation")
