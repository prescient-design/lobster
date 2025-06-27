import os
import tempfile
from unittest.mock import MagicMock

import lightning as L
import pytest
import torch

from lobster.callbacks import PerturbationScoreCallback
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

        # Create deterministic embeddings based on sequence content
        embeddings = []
        for sequence in sequences:
            # Create a deterministic embedding based on sequence hash
            hash_val = hash(sequence) % 10000
            torch.manual_seed(hash_val + self.seed)  # Add model seed for consistency
            embedding = torch.randn(self.embedding_dim)
            embeddings.append(embedding)

        return torch.stack(embeddings)


class TestPerturbationScoreCallback:
    @pytest.fixture
    def test_sequence(self):
        """Create test sequence for analysis."""
        return "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS"

    @pytest.fixture
    def test_nucleotide_sequence(self):
        """Create test nucleotide sequence for analysis."""
        return "ATCGATCG"

    @pytest.fixture
    def test_smiles_sequence(self):
        """Create test SMILES sequence for analysis."""
        return "CCO"

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return DummyModel(embedding_dim=64, seed=42)

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_init(self, temp_output_dir, test_sequence):
        """Test the initialization of the callback."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            num_shuffles=5,
            mutation_tokens=list("RKHDESTNQAVILMFYWGP"),
            random_state=42,
            modality=Modality.AMINO_ACID,
            save_heatmap=True,
        )

        assert str(callback.output_dir) == temp_output_dir
        assert callback.sequence == test_sequence
        assert callback.num_shuffles == 5
        assert callback.mutation_tokens == list("RKHDESTNQAVILMFYWGP")
        assert callback.random_state == 42
        assert callback.modality == Modality.AMINO_ACID
        assert callback.save_heatmap is True
        assert os.path.exists(temp_output_dir)

    def test_init_without_output_dir_and_save_heatmap_false(self, test_sequence):
        """Test initialization without output_dir when save_heatmap is False."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=None,
            save_heatmap=False,
        )

        assert callback.output_dir is None
        assert callback.save_heatmap is False

    def test_init_without_output_dir_and_save_heatmap_true_raises_error(self, test_sequence):
        """Test initialization without output_dir when save_heatmap is True raises error."""
        with pytest.raises(ValueError, match="output_dir must be provided when save_heatmap is True"):
            PerturbationScoreCallback(
                sequence=test_sequence,
                output_dir=None,
                save_heatmap=True,
            )

    def test_init_with_default_mutation_tokens_amino_acid(self, temp_output_dir, test_sequence):
        """Test initialization with default mutation tokens for amino acid modality."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            num_shuffles=3,
            modality=Modality.AMINO_ACID,
        )

        # The callback doesn't set default tokens directly - it passes None to PerturbationScore
        # which then handles the defaults. So mutation_tokens should be None.
        assert callback.mutation_tokens is None

    def test_init_with_default_mutation_tokens_nucleotide(self, temp_output_dir, test_nucleotide_sequence):
        """Test initialization with default mutation tokens for nucleotide modality."""
        callback = PerturbationScoreCallback(
            sequence=test_nucleotide_sequence,
            output_dir=temp_output_dir,
            num_shuffles=3,
            modality=Modality.NUCLEOTIDE,
        )

        # The callback doesn't set default tokens directly - it passes None to PerturbationScore
        # which then handles the defaults. So mutation_tokens should be None.
        assert callback.mutation_tokens is None

    def test_init_with_explicit_mutation_tokens_smiles(self, temp_output_dir, test_smiles_sequence):
        """Test initialization with explicit mutation tokens for SMILES modality."""
        mutation_tokens = list("CHNOSPFIBrCl()[]=#@+-.1234567890")
        callback = PerturbationScoreCallback(
            sequence=test_smiles_sequence,
            output_dir=temp_output_dir,
            num_shuffles=3,
            modality=Modality.SMILES,
            mutation_tokens=mutation_tokens,
        )

        assert callback.mutation_tokens == mutation_tokens

    def test_create_embedding_function(self, temp_output_dir, test_sequence, dummy_model):
        """Test the _create_embedding_function method."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
        )

        embedding_function = callback._create_embedding_function(dummy_model)

        # Test that the embedding function works
        embeddings = embedding_function([test_sequence], Modality.AMINO_ACID)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, 64)  # batch_size, embedding_dim
        assert embeddings.dtype == torch.float32

    def test_compute_scores(self, temp_output_dir, test_sequence, dummy_model):
        """Test the _compute_scores method."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            mutation_tokens=list("RKHD"),
            num_shuffles=3,
            save_heatmap=False,
        )

        metrics = callback._compute_scores(dummy_model)

        assert isinstance(metrics, dict)
        assert "avg_shuffling_embedding_distance" in metrics
        assert "avg_mutation_embedding_distance" in metrics
        assert "shuffling_mutation_ratio" in metrics

    def test_compute_scores_with_heatmap(self, temp_output_dir, test_sequence, dummy_model):
        """Test the _compute_scores method with heatmap generation."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            mutation_tokens=list("RKHD"),
            num_shuffles=3,
            save_heatmap=True,
        )

        # Count files before
        files_before = len(os.listdir(temp_output_dir))

        metrics = callback._compute_scores(dummy_model, step=100)

        # Count files after
        files_after = len(os.listdir(temp_output_dir))

        assert files_before < files_after  # New heatmap file should be created
        assert isinstance(metrics, dict)

    def test_evaluate(self, temp_output_dir, test_sequence, dummy_model):
        """Test the evaluate method."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            random_state=42,
            save_heatmap=True,
        )

        metrics = callback.evaluate(module=dummy_model)

        assert isinstance(metrics, dict)
        assert "avg_shuffling_embedding_distance" in metrics
        assert "avg_mutation_embedding_distance" in metrics
        assert "shuffling_mutation_ratio" in metrics

    def test_evaluate_with_custom_output_dir(self, temp_output_dir, test_sequence, dummy_model):
        """Test evaluate method with custom output directory."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            random_state=42,
            save_heatmap=True,
        )

        custom_output_dir = os.path.join(temp_output_dir, "custom")
        os.makedirs(custom_output_dir, exist_ok=True)

        metrics = callback.evaluate(module=dummy_model, output_dir=custom_output_dir)

        assert isinstance(metrics, dict)

        # Check that heatmap was saved to custom directory (give it a moment to write)
        import time

        time.sleep(0.1)  # Small delay to ensure file is written
        heatmap_files = [f for f in os.listdir(custom_output_dir) if f.endswith(".png")]
        assert len(heatmap_files) > 0

    def test_skip_logic(self, temp_output_dir, test_sequence):
        """Test the _skip method logic."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            run_every_n_epochs=2,
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

    def test_on_validation_epoch_end(self, temp_output_dir, test_sequence, dummy_model):
        """Test the on_validation_epoch_end method."""
        callback = PerturbationScoreCallback(
            sequence=test_sequence,
            output_dir=temp_output_dir,
            num_shuffles=3,
            mutation_tokens=list("RKHD"),
            run_every_n_epochs=1,
            random_state=42,
            save_heatmap=True,
        )

        trainer_mock = MagicMock()
        trainer_mock.global_rank = 0
        trainer_mock.current_epoch = 0
        trainer_mock.global_step = 100
        trainer_mock.logger.experiment.log = MagicMock()

        callback.on_validation_epoch_end(trainer_mock, dummy_model)

        # Check that metrics were logged
        assert trainer_mock.logger.experiment.log.call_count == 3

        # Check that heatmap was created (give it a moment to write)
        import time

        time.sleep(0.1)  # Small delay to ensure file is written
        heatmap_files = [f for f in os.listdir(temp_output_dir) if f.endswith(".png")]
        assert len(heatmap_files) > 0
