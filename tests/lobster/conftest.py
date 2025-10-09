import logging

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


@pytest.fixture(autouse=True)
def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)


class DummyLightningModule(L.LightningModule):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim

    def embed(self, batch):
        return torch.randn(batch["input_ids"].shape[0], self.embedding_dim)

    def embed_sequences(self, sequences, modality="amino_acid", aggregate=True):
        if aggregate:
            return torch.randn(len(sequences), self.embedding_dim)
        else:
            return torch.randn(len(sequences), self.embedding_dim, self.seq_len)


class EmbeddingLightningModule(L.LightningModule):
    """Deterministic model for testing that returns consistent embeddings."""

    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seed = seed

    def embed_sequences(self, sequences, modality=None, aggregate=True):
        """Return random embeddings (deterministic due to seeding)."""
        # Simply return random embeddings - determinism comes from L.seed_everything
        batch_size = len(sequences)
        return torch.randn(batch_size, self.embedding_dim)


class DummyDataset(Dataset):
    def __init__(self, num_samples=32, seq_len=128):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.groups = ["group1", "group2"] * (num_samples // 2)

        if num_samples % 2 == 1:
            self.groups.append("group1")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, 100, (self.seq_len,))
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dataset": self.groups[idx],
        }


class DummyLightningDataModule(L.LightningDataModule):
    """Dummy Lightning data module for evaluation."""

    def __init__(self, num_samples=32, seq_len=128):
        super().__init__()
        self.dataset = DummyDataset(num_samples, seq_len)
        self.num_workers = 0

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, num_workers=self.num_workers)


class MockRegressionDataset(Dataset):
    """Mock dataset for regression testing."""

    def __init__(self):
        self.sequences = [
            "MKLLVLLFGASLLLPSAAQTGAAPVQVLNGLKKLGNLKLSQKFPQYFETLLNDQLTGYGQWKMVDVYQRRSLAQKFPQYFETLLNDQLTGYGQWKMVDVYQRRS",
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA",
            "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
            "AGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGT",
        ]
        self.targets = [1.5, 2.3, 0.8, 3.1, 1.9, 2.7, 1.2, 3.5, 0.9, 2.1]

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)


class MockBinaryClassificationDataset(Dataset):
    """Mock dataset for binary classification testing."""

    def __init__(self):
        self.sequences = [
            "MKLLVLLFGASLLLPSAAQTGAAPVQVLNGLKKLGNLKLSQKFPQYFETLLNDQLTGYGQWKMVDVYQRRSLAQKFPQYFETLLNDQLTGYGQWKMVDVYQRRS",
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        ]
        self.targets = [0, 1, 0, 1]

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.long)

    def __len__(self):
        return len(self.sequences)


class MockMultilabelDataset(Dataset):
    """Mock dataset for multilabel classification testing."""

    def __init__(self):
        self.sequences = [
            "MKLLVLLFGASLLLPSAAQTGAAPVQVLNGLKKLGNLKLSQKFPQYFETLLNDQLTGYGQWKMVDVYQRRSLAQKFPQYFETLLNDQLTGYGQWKMVDVYQRRS",
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ]
        # 5 classes for multilabel
        self.targets = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 1]]
        self.num_label_columns = 5

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.long)

    def __len__(self):
        return len(self.sequences)


class MockCalmDataset(Dataset):
    """Mock CALM dataset that mimics CalmPropertyDataset interface."""

    def __init__(self, task="meltome", species=None):
        self.task = task
        self.species = species

        # Base sequences for all tasks
        self.sequences = [
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            "AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT",
        ]

        if task == "meltome":  # regression
            self.targets = [45.2, 67.8, 52.1]
        elif task == "localization":  # multilabel, 10 classes
            self.targets = [
                [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            ]
            self.num_label_columns = 10
        elif task == "function_bp":  # multilabel, 5 classes
            self.targets = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 1]]
            self.num_label_columns = 5

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.task in ["meltome", "solubility", "protein_abundance", "transcript_abundance"]:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:  # multilabel tasks
            target = torch.tensor(self.targets[idx], dtype=torch.long)
        return seq, target

    def __len__(self):
        return len(self.sequences)


class MockMoleculeACEDataset(Dataset):
    """Mock MoleculeACE dataset."""

    def __init__(self, task="CHEMBL204_Ki", train=True, include_protein_sequences=False):
        self.task = task
        self.train = train
        self.include_protein_sequences = include_protein_sequences

        if train:
            self.smiles = ["CCO", "CCC", "CCCC"]
            self.targets = [2.5, 3.1, 1.8]
            if include_protein_sequences:
                self.protein_seqs = ["MKLLVL", "MKLVLL", "MKLLVV"]
        else:
            self.smiles = ["CCCCC", "CC"]
            self.targets = [2.2, 3.5]
            if include_protein_sequences:
                self.protein_seqs = ["MKLVVL", "MKLLLL"]

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(-1)

        if self.include_protein_sequences:
            protein_seq = self.protein_seqs[idx]
            return (smiles, protein_seq), target
        else:
            return smiles, target

    def __len__(self):
        return len(self.smiles)


class MockPEERDataset(Dataset):
    """Mock PEER dataset."""

    def __init__(self, task, split):
        self.task = task
        self.split = split

        # Simple protein sequences for testing
        if split == "train":
            self.sequences = ["MKLLVLLFGA", "ATCGATCGAT", "AAAAAAAAAA"]
            self.targets = [2.5, 3.1, 1.8]
        else:  # test
            self.sequences = ["CCCCCCCCCC", "GGGGGGGGGG"]
            self.targets = [2.2, 3.5]

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(-1)
        return seq, target

    def __len__(self):
        return len(self.sequences)


@pytest.fixture
def dummy_model():
    return DummyLightningModule()


@pytest.fixture
def dummy_datamodule():
    return DummyLightningDataModule()


@pytest.fixture
def deterministic_model():
    # Set seed for reproducibility before creating model
    L.seed_everything(1)
    return EmbeddingLightningModule(embedding_dim=64, seed=1)


@pytest.fixture
def mock_regression_dataset():
    return MockRegressionDataset()


@pytest.fixture
def mock_binary_classification_dataset():
    return MockBinaryClassificationDataset()


@pytest.fixture
def mock_multilabel_dataset():
    return MockMultilabelDataset()


@pytest.fixture
def mock_calm_dataset():
    return MockCalmDataset


@pytest.fixture
def mock_moleculeace_dataset():
    return MockMoleculeACEDataset


@pytest.fixture
def mock_peer_dataset():
    return MockPEERDataset
