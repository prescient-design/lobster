from unittest.mock import Mock, patch

import pytest
import torch

from lobster.callbacks import RandomNeighborScoreCallback


class MockDataset:
    def __init__(self, sequences, size=10):
        self.sequences = sequences
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield self.sequences[i % len(self.sequences)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.sequences[idx % len(self.sequences)]


class MockModel(torch.nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim

    def embed_sequences(self, sequences, modality, aggregate=True):
        batch_size = len(sequences)
        return torch.randn(batch_size, self.embedding_dim)

    def eval(self):
        pass


@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.current_epoch = 0
    trainer.global_rank = 0
    trainer.logger = Mock()
    trainer.logger.experiment = Mock()
    trainer.global_step = 100
    return trainer


@pytest.fixture
def mock_model():
    return MockModel()


def test_callback_initialization():
    # Mock biological sequences for initialization
    bio_sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    mock_bio_dataset = MockDataset(bio_sequences, size=50)

    with patch.object(RandomNeighborScoreCallback.SUPPORTED_DATASETS["AMPLIFY"], "__new__") as mock_dataset_class:
        mock_dataset_class.return_value = mock_bio_dataset

        callback = RandomNeighborScoreCallback(
            dataset_name="AMPLIFY", k=100, biological_dataset_limit=50, num_random_sequences=50, seed=42
        )

        assert callback.dataset_name == "AMPLIFY"
        assert callback.k == 100
        assert callback.biological_dataset_limit == 50
        assert callback.num_random_sequences == 50
        assert callback.seed == 42

        # Verify dataloaders were created during initialization
        assert callback.biological_dataloader is not None
        assert callback.random_dataloader is not None


def test_callback_with_mocked_dataset(mock_trainer, mock_model):
    # Mock biological sequences
    bio_sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    mock_bio_dataset = MockDataset(bio_sequences, size=20)

    with patch.object(RandomNeighborScoreCallback.SUPPORTED_DATASETS["AMPLIFY"], "__new__") as mock_dataset_class:
        mock_dataset_class.return_value = mock_bio_dataset

        callback = RandomNeighborScoreCallback(
            dataset_name="AMPLIFY",
            k=2,
            biological_dataset_limit=20,
            num_random_sequences=20,
            batch_size=4,
            seed=42,
        )

        # Test evaluation
        metrics = callback.evaluate(mock_model, mock_trainer)

        # Check that we get a valid RNS score
        assert "random_neighbor_score" in metrics
        assert isinstance(metrics["random_neighbor_score"], float)
        assert 0.0 <= metrics["random_neighbor_score"] <= 1.0

        # Verify trainer logging was called
        mock_trainer.logger.experiment.log.assert_called_once()


def test_callback_skip_logic(mock_model):
    # Mock biological sequences for initialization
    bio_sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    mock_bio_dataset = MockDataset(bio_sequences, size=20)

    with patch.object(RandomNeighborScoreCallback.SUPPORTED_DATASETS["AMPLIFY"], "__new__") as mock_dataset_class:
        mock_dataset_class.return_value = mock_bio_dataset

        callback = RandomNeighborScoreCallback(
            dataset_name="AMPLIFY",
            run_every_n_epochs=2,
            k=10,
            biological_dataset_limit=20,
            num_random_sequences=20,
        )

        # Test skipping on non-zero rank
        trainer = Mock()
        trainer.global_rank = 1
        trainer.current_epoch = 0
        assert callback._skip(trainer) is True

        # Test skipping based on epoch frequency
        trainer.global_rank = 0
        trainer.current_epoch = 1  # Should skip since run_every_n_epochs=2
        assert callback._skip(trainer) is True

        trainer.current_epoch = 2  # Should not skip
        assert callback._skip(trainer) is False


def test_modality_mapping():
    # Mock biological sequences for initialization
    bio_sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    mock_bio_dataset = MockDataset(bio_sequences, size=10)

    with patch.object(RandomNeighborScoreCallback.SUPPORTED_DATASETS["AMPLIFY"], "__new__") as mock_dataset_class:
        mock_dataset_class.return_value = mock_bio_dataset

        callback = RandomNeighborScoreCallback(dataset_name="AMPLIFY", k=10)
        assert callback._get_modality_for_dataset() == "amino_acid"

        callback.dataset_name = "Calm"
        assert callback._get_modality_for_dataset() == "nucleotide"

        callback.dataset_name = "ZINC"
        assert callback._get_modality_for_dataset() == "SMILES"


def test_unsupported_dataset():
    with pytest.raises(ValueError, match="Dataset 'InvalidDataset' not supported"):
        RandomNeighborScoreCallback(dataset_name="InvalidDataset", k=10)
