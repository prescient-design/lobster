"""Tests for perturbation score metrics."""

import pytest
import torch

from lobster.constants import Modality
from lobster.metrics._perturbation_score import PerturbationScore, get_default_mutation_tokens


class MockModel:
    """Mock model for testing."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def embed_sequences(self, sequences, modality, aggregate=True):
        """Mock embed_sequences method that returns deterministic embeddings."""
        batch_size = len(sequences)
        # Use a deterministic seed based on the first sequence for testing
        if sequences:
            seed = hash(sequences[0]) % 1000
            torch.manual_seed(seed)
        return torch.randn(batch_size, self.embedding_dim)


def test_get_default_mutation_tokens():
    """Test getting default mutation tokens for different modalities."""
    # Test amino acid modality
    aa_tokens = get_default_mutation_tokens(Modality.AMINO_ACID)
    assert isinstance(aa_tokens, list)
    assert len(aa_tokens) > 0
    assert all(isinstance(token, str) for token in aa_tokens)

    # Test nucleotide modality
    nt_tokens = get_default_mutation_tokens(Modality.NUCLEOTIDE)
    assert isinstance(nt_tokens, list)
    assert len(nt_tokens) == 4  # ATCG
    assert set(nt_tokens) == {"A", "T", "C", "G"}

    # Test SMILES modality
    smiles_tokens = get_default_mutation_tokens(Modality.SMILES)
    assert isinstance(smiles_tokens, list)
    assert len(smiles_tokens) > 0
    assert all(isinstance(token, str) for token in smiles_tokens)

    # Test unsupported modality
    with pytest.raises(ValueError):
        get_default_mutation_tokens("UNSUPPORTED_MODALITY")


def test_perturbation_score_basic():
    """Test basic PerturbationScore functionality."""
    sequence = "ABC"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=2,
        random_state=42,
        save_heatmap=False,
    )

    # Run the analysis
    metric.update()
    metrics = metric.compute()

    assert isinstance(metrics, dict)
    assert "avg_shuffling_embedding_distance" in metrics
    assert "avg_mutation_embedding_distance" in metrics
    assert "shuffling_mutation_ratio" in metrics

    assert isinstance(metrics["avg_shuffling_embedding_distance"], float)
    assert isinstance(metrics["avg_mutation_embedding_distance"], float)
    assert isinstance(metrics["shuffling_mutation_ratio"], float)

    assert metrics["avg_shuffling_embedding_distance"] >= 0
    assert metrics["avg_mutation_embedding_distance"] >= 0
    assert metrics["shuffling_mutation_ratio"] >= 0


def test_perturbation_score_with_heatmap(tmp_path):
    """Test PerturbationScore with heatmap generation."""
    sequence = "ABC"
    output_file = tmp_path / "test_heatmap.png"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=2,
        random_state=42,
        save_heatmap=True,
        output_file=output_file,
    )

    metric.update()
    metrics = metric.compute()

    assert isinstance(metrics, dict)
    assert output_file.exists()


def test_perturbation_score_custom_tokens():
    """Test PerturbationScore with custom mutation tokens."""
    sequence = "ABC"
    custom_tokens = ["X", "Y", "Z"]

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality="CUSTOM",
        num_shuffles=2,
        mutation_tokens=custom_tokens,
        random_state=42,
        save_heatmap=False,
    )

    metric.update()
    metrics = metric.compute()

    assert isinstance(metrics, dict)
    assert all(
        key in metrics
        for key in ["avg_shuffling_embedding_distance", "avg_mutation_embedding_distance", "shuffling_mutation_ratio"]
    )


def test_perturbation_score_empty_sequence():
    """Test PerturbationScore with empty sequence."""
    sequence = ""

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    with pytest.raises(ValueError):
        PerturbationScore(
            sequence=sequence,
            embedding_function=embedding_function,
            modality=Modality.AMINO_ACID,
            num_shuffles=2,
            random_state=42,
            save_heatmap=False,
        )


def test_perturbation_score_missing_tokens():
    """Test PerturbationScore with unsupported modality and no tokens."""
    sequence = "ABC"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    with pytest.raises(ValueError):
        PerturbationScore(
            sequence=sequence,
            embedding_function=embedding_function,
            modality="UNSUPPORTED",
            num_shuffles=2,
            random_state=42,
            save_heatmap=False,
        )


def test_perturbation_score_missing_output_file():
    """Test PerturbationScore with save_heatmap=True but no output_file."""
    sequence = "ABC"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=2,
        random_state=42,
        save_heatmap=True,
        output_file=None,
    )

    with pytest.raises(ValueError, match="output_file must be provided when save_heatmap is True"):
        metric.update()


def test_perturbation_score_compute_without_update():
    """Test PerturbationScore compute without calling update first."""
    sequence = "ABC"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=2,
        random_state=42,
        save_heatmap=False,
    )

    with pytest.raises(ValueError, match="No perturbation analysis results available"):
        metric.compute()


def test_perturbation_score_deterministic():
    """Test that PerturbationScore produces deterministic results with same seed."""
    sequence = "ABCDEF"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    # Run with same seed
    metric1 = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=5,
        random_state=42,
        save_heatmap=False,
    )
    metric1.update()
    metrics1 = metric1.compute()

    metric2 = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=5,
        random_state=42,
        save_heatmap=False,
    )
    metric2.update()
    metrics2 = metric2.compute()

    # Results should be identical with same seed
    assert metrics1["avg_shuffling_embedding_distance"] == metrics2["avg_shuffling_embedding_distance"]
    assert metrics1["avg_mutation_embedding_distance"] == metrics2["avg_mutation_embedding_distance"]
    assert metrics1["shuffling_mutation_ratio"] == metrics2["shuffling_mutation_ratio"]


def test_perturbation_score_different_seeds():
    """Test that PerturbationScore produces different results with different seeds."""
    sequence = "ABCDEF"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    # Run with different seeds
    metric1 = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=5,
        random_state=42,
        save_heatmap=False,
    )
    metric1.update()
    metrics1 = metric1.compute()

    metric2 = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=5,
        random_state=123,
        save_heatmap=False,
    )
    metric2.update()
    metrics2 = metric2.compute()

    # Results should be different with different seeds
    assert metrics1["avg_shuffling_embedding_distance"] != metrics2["avg_shuffling_embedding_distance"]


def test_perturbation_score_nucleotide_modality():
    """Test PerturbationScore with nucleotide modality."""
    sequence = "ATCGATCG"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.NUCLEOTIDE,
        num_shuffles=3,
        random_state=42,
        save_heatmap=False,
    )

    metric.update()
    metrics = metric.compute()

    assert isinstance(metrics, dict)
    assert all(
        key in metrics
        for key in ["avg_shuffling_embedding_distance", "avg_mutation_embedding_distance", "shuffling_mutation_ratio"]
    )


def test_perturbation_score_smiles_modality():
    """Test PerturbationScore with SMILES modality."""
    sequence = "CCO"

    def embedding_function(sequences, modality):
        batch_size = len(sequences)
        return torch.randn(batch_size, 128)

    metric = PerturbationScore(
        sequence=sequence,
        embedding_function=embedding_function,
        modality=Modality.SMILES,
        num_shuffles=3,
        mutation_tokens=list("CHNOSPFIBrCl()[]=#@+-.1234567890"),
        random_state=42,
        save_heatmap=False,
    )

    metric.update()
    metrics = metric.compute()

    assert isinstance(metrics, dict)
    assert all(
        key in metrics
        for key in ["avg_shuffling_embedding_distance", "avg_mutation_embedding_distance", "shuffling_mutation_ratio"]
    )


if __name__ == "__main__":
    pytest.main([__file__])
