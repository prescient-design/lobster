import os
import tempfile
from unittest.mock import MagicMock

import lightning as L
import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule

from lobster.callbacks import PerturbationAnalysisCallback
from lobster.constants import Modality


class DummyModel(L.LightningModule):
    """Dummy model for testing perturbation analysis callback."""

    def __init__(self, embedding_dim: int = 128, seed: int = 42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seed = seed

    def embed_sequences(
        self, sequences: list[str], modality: str = "amino_acid", aggregate: bool = True
    ) -> torch.Tensor:
        """Dummy embed_sequences method that returns deterministic embeddings."""
        batch_size = len(sequences)

        # Create deterministic embeddings based on sequence content
        embeddings = []
        for sequence in sequences:
            # Create a deterministic embedding based on sequence hash
            hash_val = hash(sequence) % 10000
            torch.manual_seed(hash_val + self.seed)  # Add model seed for consistency
            embedding = torch.randn(self.embedding_dim)
            embeddings.append(embedding)

        return torch.stack(embeddings)


class TestPerturbationAnalysisCallback:
    @pytest.fixture
    def test_sequences(self):
        """Create test sequences for analysis."""
        return [
            "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS",
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        ]

    @pytest.fixture
    def test_nucleotide_sequences(self):
        """Create test nucleotide sequences for analysis."""
        return [
            "ATCGATCG",
            "GCTAGCTA",
        ]

    @pytest.fixture
    def test_smiles_sequences(self):
        """Create test SMILES sequences for analysis."""
        return [
            "CCO",
            "CC(C)O",
        ]

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return DummyModel(embedding_dim=64, seed=42)

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_get_default_mutation_tokens_amino_acid(self, temp_output_dir, test_sequences):
        """Test get_default_mutation_tokens method for amino acid modality."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir, sequences=test_sequences, modality=Modality.AMINO_ACID
        )
        tokens = callback.get_default_mutation_tokens(Modality.AMINO_ACID)
        expected_tokens = list("RKHDESTNQAVILMFYWGP")
        assert tokens == expected_tokens

    def test_get_default_mutation_tokens_nucleotide(self, temp_output_dir, test_nucleotide_sequences):
        """Test get_default_mutation_tokens method for nucleotide modality."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir, sequences=test_nucleotide_sequences, modality=Modality.NUCLEOTIDE
        )
        tokens = callback.get_default_mutation_tokens(Modality.NUCLEOTIDE)
        expected_tokens = list("ATCG")
        assert tokens == expected_tokens

    def test_get_default_mutation_tokens_smiles_raises_error(self, temp_output_dir, test_smiles_sequences):
        """Test get_default_mutation_tokens method raises error for SMILES modality."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_smiles_sequences,
            modality=Modality.SMILES,
            mutation_tokens=list("CHNOSPFIBrCl()[]=#@+-.1234567890"),
        )
        with pytest.raises(ValueError, match="Modality 'SMILES' does not have default mutation tokens"):
            callback.get_default_mutation_tokens(Modality.SMILES)

    def test_get_default_mutation_tokens_3d_coordinates_raises_error(self, temp_output_dir, test_sequences):
        """Test get_default_mutation_tokens method raises error for 3D_COORDINATES modality."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_sequences,
            modality=Modality.COORDINATES_3D,
            mutation_tokens=list("ABC"),
        )
        with pytest.raises(ValueError, match="Modality '3d_coordinates' does not have default mutation tokens"):
            callback.get_default_mutation_tokens(Modality.COORDINATES_3D)

    def test_init(self, temp_output_dir, test_sequences):
        """Test the initialization of the callback."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_sequences,
            num_shuffles=5,
            mutation_tokens=list("RKHDESTNQAVILMFYWGP"),
            random_state=42,
            modality=Modality.AMINO_ACID,
        )

        assert str(callback.output_dir) == temp_output_dir
        assert callback.sequences == test_sequences
        assert callback.num_shuffles == 5
        assert callback.mutation_tokens == list("RKHDESTNQAVILMFYWGP")
        assert callback.random_state == 42
        assert callback.modality == Modality.AMINO_ACID
        assert os.path.exists(temp_output_dir)

    def test_init_with_default_mutation_tokens_amino_acid(self, temp_output_dir, test_sequences):
        """Test initialization with default mutation tokens for amino acid modality."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir, sequences=test_sequences, num_shuffles=3, modality=Modality.AMINO_ACID
        )

        # Check that default amino acid tokens are set
        expected_tokens = list("RKHDESTNQAVILMFYWGP")
        assert callback.mutation_tokens == expected_tokens

    def test_init_with_default_mutation_tokens_nucleotide(self, temp_output_dir, test_nucleotide_sequences):
        """Test initialization with default mutation tokens for nucleotide modality."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_nucleotide_sequences,
            num_shuffles=3,
            modality=Modality.NUCLEOTIDE,
        )

        # Check that default nucleotide tokens are set
        expected_tokens = list("ATCG")
        assert callback.mutation_tokens == expected_tokens

    def test_init_without_mutation_tokens_smiles_raises_error(self, temp_output_dir, test_smiles_sequences):
        """Test initialization without mutation_tokens for SMILES modality raises error."""
        with pytest.raises(ValueError, match="Modality 'SMILES' requires explicit mutation_tokens"):
            PerturbationAnalysisCallback(
                output_dir=temp_output_dir, sequences=test_smiles_sequences, num_shuffles=3, modality=Modality.SMILES
            )

    def test_init_without_mutation_tokens_3d_coordinates_raises_error(self, temp_output_dir, test_sequences):
        """Test initialization without mutation_tokens for 3D_COORDINATES modality raises error."""
        with pytest.raises(ValueError, match="Modality '3d_coordinates' requires explicit mutation_tokens"):
            PerturbationAnalysisCallback(
                output_dir=temp_output_dir, sequences=test_sequences, num_shuffles=3, modality=Modality.COORDINATES_3D
            )

    def test_init_with_explicit_mutation_tokens_smiles(self, temp_output_dir, test_smiles_sequences):
        """Test initialization with explicit mutation tokens for SMILES modality."""
        mutation_tokens = list("CHNOSPFIBrCl()[]=#@+-.1234567890")
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_smiles_sequences,
            num_shuffles=3,
            modality=Modality.SMILES,
            mutation_tokens=mutation_tokens,
        )

        assert callback.mutation_tokens == mutation_tokens

    def test_embed_sequences(self, temp_output_dir, test_sequences, dummy_model):
        """Test the _embed_sequences method."""
        callback = PerturbationAnalysisCallback(output_dir=temp_output_dir, sequences=test_sequences)

        embeddings = callback._embed_sequences(dummy_model, test_sequences)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (len(test_sequences), 64)  # batch_size, embedding_dim
        assert embeddings.dtype == torch.float32

    def test_embed_sequences_with_model_model(self, temp_output_dir, test_sequences):
        """Test _embed_sequences when model has model.embed_sequences."""
        # Create a model with nested embed_sequences method
        outer_model = MagicMock(spec=LightningModule)
        inner_model = DummyModel(embedding_dim=32)
        outer_model.model = inner_model

        callback = PerturbationAnalysisCallback(output_dir=temp_output_dir, sequences=test_sequences)

        embeddings = callback._embed_sequences(outer_model, test_sequences)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (len(test_sequences), 32)

    def test_embed_sequences_not_implemented(self, temp_output_dir, test_sequences):
        """Test _embed_sequences raises error when method not implemented."""
        model = MagicMock(spec=LightningModule)
        # Don't add embed_sequences method

        callback = PerturbationAnalysisCallback(output_dir=temp_output_dir, sequences=test_sequences)

        with pytest.raises(NotImplementedError, match="Model must implement embed_sequences"):
            callback._embed_sequences(model, test_sequences)

    def test_compute_mutation_distances(self, temp_output_dir, test_sequences, dummy_model):
        """Test the _compute_mutation_distances method."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir, sequences=test_sequences, mutation_tokens=list("RKHD")
        )

        sequence = test_sequences[0]
        distances = callback._compute_mutation_distances(dummy_model, sequence)

        assert isinstance(distances, np.ndarray)
        assert distances.shape == (len(sequence), 4)  # seq_len, num_mutation_tokens
        assert all(0 <= d <= 2 for d in distances.flatten())  # Cosine distance is between 0 and 2

    def test_create_perturbation_heatmap(self, temp_output_dir, test_sequences):
        """Test the _create_perturbation_heatmap method."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir, sequences=test_sequences, mutation_tokens=list("RKHD")
        )

        sequence = test_sequences[0]
        mutation_distances = np.random.rand(len(sequence), 4)
        output_file = os.path.join(temp_output_dir, "test_heatmap.png")

        callback._create_perturbation_heatmap(mutation_distances, sequence, output_file)

        assert os.path.exists(output_file)

    def test_run_analysis_without_heatmap(self, temp_output_dir, test_sequences, dummy_model):
        """Test _run_analysis without saving heatmap."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_sequences,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            random_state=42,
        )

        # Count files before
        files_before = len(os.listdir(temp_output_dir))

        metrics = callback._run_analysis(dummy_model, test_sequences, save_heatmap=False)

        # Count files after
        files_after = len(os.listdir(temp_output_dir))

        assert files_before == files_after  # No new files should be created
        assert isinstance(metrics, dict)

    def test_run_analysis_empty_sequences(self, temp_output_dir, dummy_model):
        """Test _run_analysis with empty sequences."""
        callback = PerturbationAnalysisCallback(output_dir=temp_output_dir, sequences=[], num_shuffles=3)

        metrics = callback._run_analysis(dummy_model, [])

        assert metrics == {}

    def test_evaluate(self, temp_output_dir, test_sequences, dummy_model):
        """Test the evaluate method."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_sequences,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            random_state=42,
        )

        metrics = callback.evaluate(module=dummy_model, save_heatmap=True)

        assert isinstance(metrics, dict)
        assert "avg_shuffling_distance" in metrics
        assert "avg_mutation_distance" in metrics
        assert "distance_ratio" in metrics

    def test_evaluate_with_custom_output_dir(self, temp_output_dir, test_sequences, dummy_model):
        """Test evaluate method with custom output directory."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_sequences,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            random_state=42,
        )

        custom_output_dir = os.path.join(temp_output_dir, "custom")
        os.makedirs(custom_output_dir, exist_ok=True)

        metrics = callback.evaluate(module=dummy_model, save_heatmap=True, output_dir=custom_output_dir)

        assert isinstance(metrics, dict)

        # Check that heatmap was saved to custom directory (give it a moment to write)
        import time

        time.sleep(0.1)  # Small delay to ensure file is written
        heatmap_files = [f for f in os.listdir(custom_output_dir) if f.endswith(".png")]
        assert len(heatmap_files) > 0

    def test_skip_logic(self, temp_output_dir, test_sequences):
        """Test the _skip method logic."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir, sequences=test_sequences, run_every_n_epochs=2
        )

        trainer_mock = MagicMock()
        trainer_mock.global_rank = 0

        # Test not skipping when run_every_n_epochs is None
        callback.run_every_n_epochs = None
        assert not callback._skip(trainer_mock)

        # Test skipping when not in main process
        callback.run_every_n_epochs = 2
        trainer_mock.global_rank = 1
        assert callback._skip(trainer_mock)

        # Test skipping when not at the right epoch
        trainer_mock.global_rank = 0
        trainer_mock.current_epoch = 1
        assert callback._skip(trainer_mock)

        # Test not skipping when at the right epoch
        trainer_mock.current_epoch = 2
        assert not callback._skip(trainer_mock)

    def test_on_validation_epoch_end(self, temp_output_dir, test_sequences, dummy_model):
        """Test the on_validation_epoch_end method."""
        callback = PerturbationAnalysisCallback(
            output_dir=temp_output_dir,
            sequences=test_sequences,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            run_every_n_epochs=1,
            random_state=42,
        )

        trainer_mock = MagicMock()
        trainer_mock.global_rank = 0
        trainer_mock.current_epoch = 0
        trainer_mock.logger.experiment.add_scalar = MagicMock()

        callback.on_validation_epoch_end(trainer_mock, dummy_model)

        # Check that metrics were logged
        assert trainer_mock.logger.experiment.add_scalar.call_count == 3

        # Check that heatmap was created (give it a moment to write)
        import time

        time.sleep(0.1)  # Small delay to ensure file is written
        heatmap_files = [f for f in os.listdir(temp_output_dir) if f.endswith(".png")]
        assert len(heatmap_files) > 0
