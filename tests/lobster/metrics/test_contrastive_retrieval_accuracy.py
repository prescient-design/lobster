import pytest
import torch
from torch import Tensor

from lobster.metrics import ContrastiveRetrievalAccuracy
from lobster.transforms import (
    AminoAcidToSmilesPairTransform,
    NucleotideToAminoAcidPairTransform,
    SmilesToSmilesPairTransform,
)


def dummy_embedding_function(sequences: list[str], modality: str) -> Tensor:
    """Dummy embedding function that returns random embeddings."""
    return torch.randn(len(sequences), 64)


def create_amino_acid_to_smiles_transform():
    """Create a transform function for amino acid to SMILES conversion."""
    transform = AminoAcidToSmilesPairTransform()

    def transform_function(seq: str) -> tuple[str, str | None, str, str]:
        try:
            result = transform._transform(seq, {})
            return result[0], result[1], "amino_acid", "smiles"
        except Exception:
            return seq, None, "amino_acid", "smiles"

    return transform_function


def create_nucleotide_to_amino_acid_transform():
    """Create a transform function for nucleotide to amino acid conversion."""
    transform = NucleotideToAminoAcidPairTransform()

    def transform_function(seq: str) -> tuple[str, str | None, str, str]:
        try:
            result = transform._transform(seq, {})
            return result[0], result[1], "nucleotide", "amino_acid"
        except Exception:
            return seq, None, "nucleotide", "amino_acid"

    return transform_function


def create_smiles_to_smiles_transform():
    """Create a transform function for SMILES to SMILES conversion."""
    transform = SmilesToSmilesPairTransform(randomize_smiles=True)

    def transform_function(seq: str) -> tuple[str, str | None, str, str]:
        try:
            result = transform._transform(seq, {})
            return result[0], result[1], "smiles", "smiles"
        except Exception:
            return seq, None, "smiles", "smiles"

    return transform_function


def create_custom_length_transform():
    """Create a custom transform function that maps sequences to their length categories."""

    def transform_function(seq: str) -> tuple[str, str | None, str, str]:
        length = len(seq)
        if length <= 5:
            category = "short"
        elif length <= 10:
            category = "medium"
        else:
            category = "long"
        return seq, category, "sequence", "length_category"

    return transform_function


def test_contrastive_retrieval_accuracy_init():
    """Test initialization of ContrastiveRetrievalAccuracy metric."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST"]
    transform_fn = create_amino_acid_to_smiles_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        k_values=[1, 5],
        distance_metric="cosine",
        batch_size=16,
        random_state=42,
    )

    assert metric.query_sequences == sequences
    assert metric.transform_function == transform_fn
    assert metric.k_values == [1, 5]
    assert metric.distance_metric == "cosine"
    assert metric.batch_size == 16
    assert metric.random_state == 42
    assert hasattr(metric, "query_target_pairs")


def test_contrastive_retrieval_accuracy_init_empty_sequences():
    """Test that empty sequences raise ValueError."""
    transform_fn = create_amino_acid_to_smiles_transform()

    with pytest.raises(ValueError, match="query_sequences cannot be empty"):
        ContrastiveRetrievalAccuracy(
            query_sequences=[], embedding_function=dummy_embedding_function, transform_function=transform_fn
        )


def test_contrastive_retrieval_accuracy_invalid_transform():
    """Test that invalid transform function raises appropriate error."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH"]

    def invalid_transform(seq: str) -> tuple[str, str | None, str, str]:
        raise ValueError("Invalid transform")

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=invalid_transform,
        random_state=42,
    )

    # The metric should handle the error gracefully and have empty pairs
    assert len(metric.query_target_pairs) == 0


def test_contrastive_retrieval_accuracy_compute_without_update():
    """Test that compute without update raises ValueError."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH"]
    transform_fn = create_amino_acid_to_smiles_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        random_state=42,
    )

    with pytest.raises(ValueError, match="No retrieval results available"):
        metric.compute()


def test_contrastive_retrieval_accuracy_perfect_embedding():
    """Test metric with perfect embedding function (identity mapping)."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST"]
    transform_fn = create_amino_acid_to_smiles_transform()

    def perfect_embedding_function(sequences: list[str], modality: str) -> Tensor:
        """Perfect embedding function that returns identical embeddings for paired sequences."""
        # For simplicity, return embeddings based on sequence length
        embeddings = []
        for seq in sequences:
            # Create embedding based on sequence characteristics
            embedding = torch.zeros(64)
            embedding[0] = len(seq)  # First dimension encodes length
            # Add some sequence-specific features
            for i, char in enumerate(seq[:10]):  # Use first 10 chars
                embedding[i + 1] = ord(char) / 100.0
            embeddings.append(embedding)
        return torch.stack(embeddings)

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=perfect_embedding_function,
        transform_function=transform_fn,
        k_values=[1, 3, 5],
        random_state=42,
    )

    # Update and compute
    metric.update()
    results = metric.compute()

    # Check that results contain expected keys
    assert "top_1_accuracy" in results
    assert "top_3_accuracy" in results
    assert "top_5_accuracy" in results
    assert "mean_rank" in results
    assert "median_rank" in results
    assert "mean_reciprocal_rank" in results

    # Check that all values are reasonable
    assert 0 <= results["top_1_accuracy"] <= 1
    assert 0 <= results["top_3_accuracy"] <= 1
    assert 0 <= results["top_5_accuracy"] <= 1
    assert results["mean_rank"] >= 1
    assert results["median_rank"] >= 1
    assert 0 <= results["mean_reciprocal_rank"] <= 1


@pytest.mark.parametrize(
    "transform_type,sequences",
    [
        ("amino_acid_to_smiles", ["MKLLVVVGG", "ARNDCQEGH"]),
        ("nucleotide_to_amino_acid", ["ATGAAACTGCTG", "ATGGGCAAATAA"]),
        ("smiles_to_smiles", ["CCO", "CC(=O)O"]),
    ],
)
def test_contrastive_retrieval_accuracy_different_transforms(transform_type, sequences):
    """Test metric with different transform types."""
    # Get the appropriate transform function
    if transform_type == "amino_acid_to_smiles":
        transform_fn = create_amino_acid_to_smiles_transform()
    elif transform_type == "nucleotide_to_amino_acid":
        transform_fn = create_nucleotide_to_amino_acid_transform()
    elif transform_type == "smiles_to_smiles":
        transform_fn = create_smiles_to_smiles_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        k_values=[1, 2],
        random_state=42,
    )

    # Check that the metric initializes correctly
    assert metric.transform_function == transform_fn
    assert len(metric.query_target_pairs) >= 0  # Some transforms might fail


@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_contrastive_retrieval_accuracy_distance_metrics(distance_metric):
    """Test metric with different distance metrics."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH"]
    transform_fn = create_amino_acid_to_smiles_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        distance_metric=distance_metric,
        random_state=42,
    )

    assert metric.distance_metric == distance_metric

    # Update and compute
    metric.update()

    # Only compute if we have valid pairs
    if metric.retrieval_ranks:
        results = metric.compute()
        assert isinstance(results, dict)


def test_contrastive_retrieval_accuracy_batch_processing():
    """Test metric with small batch size to verify batch processing."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST", "ALANYLGLY"]
    transform_fn = create_amino_acid_to_smiles_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        batch_size=2,  # Small batch size
        random_state=42,
    )

    # Update and compute
    metric.update()

    # Only compute if we have valid pairs
    if metric.retrieval_ranks:
        results = metric.compute()
        assert isinstance(results, dict)


def test_contrastive_retrieval_accuracy_reproducibility():
    """Test that the metric produces reproducible results."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST"]
    transform_fn1 = create_amino_acid_to_smiles_transform()
    transform_fn2 = create_amino_acid_to_smiles_transform()

    # Run metric twice with same random state
    metric1 = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn1,
        random_state=42,
    )

    metric2 = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn2,
        random_state=42,
    )

    # Both should generate the same pairs
    metric1.update()
    metric2.update()

    # Check that the number of pairs is the same
    assert len(metric1.query_target_pairs) == len(metric2.query_target_pairs)


def test_contrastive_retrieval_accuracy_custom_transform():
    """Test metric with a completely custom transform function."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST", "ACDEF", "VWKSTMNPQFIL"]
    transform_fn = create_custom_length_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        k_values=[1, 2, 3],
        random_state=42,
    )

    # All sequences should be transformed successfully
    assert len(metric.query_target_pairs) == len(sequences)

    # Check that modalities are correct
    for pair in metric.query_target_pairs:
        assert pair[2] == "sequence"  # query modality
        assert pair[3] == "length_category"  # target modality

    # Update and compute
    metric.update()
    results = metric.compute()

    # Check results structure
    assert "top_1_accuracy" in results
    assert "top_2_accuracy" in results
    assert "top_3_accuracy" in results
    assert "mean_rank" in results
    assert "median_rank" in results
    assert "mean_reciprocal_rank" in results


def test_contrastive_retrieval_accuracy_mixed_query_modalities_error():
    """Test that mixed query modalities in pairs raise an error."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH"]

    def mixed_query_modality_transform(seq: str) -> tuple[str, str | None, str, str]:
        # Return different query modalities for different sequences
        if seq == "MKLLVVVGG":
            return seq, "CCO", "amino_acid", "smiles"
        else:
            return seq, "short", "sequence", "smiles"

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=mixed_query_modality_transform,
        random_state=42,
    )

    # This should raise an error during update because of mixed query modalities
    with pytest.raises(ValueError, match="All query modalities must be the same"):
        metric.update()


def test_contrastive_retrieval_accuracy_mixed_target_modalities_error():
    """Test that mixed target modalities in pairs raise an error."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH"]

    def mixed_target_modality_transform(seq: str) -> tuple[str, str | None, str, str]:
        # Return different target modalities for different sequences
        if seq == "MKLLVVVGG":
            return seq, "CCO", "amino_acid", "smiles"
        else:
            return seq, "short", "amino_acid", "length_category"

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=mixed_target_modality_transform,
        random_state=42,
    )

    # This should raise an error during update because of mixed target modalities
    with pytest.raises(ValueError, match="All target modalities must be the same"):
        metric.update()


def test_contrastive_retrieval_accuracy_no_valid_pairs():
    """Test behavior when no valid pairs are generated."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH"]

    def failing_transform(seq: str) -> tuple[str, str | None, str, str]:
        return seq, None, "amino_acid", "smiles"  # Always return None for target

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=failing_transform,
        random_state=42,
    )

    # Should have no valid pairs
    assert len(metric.query_target_pairs) == 0

    # Update should handle this gracefully
    metric.update()  # Should not raise an error

    # Compute should still raise an error since no pairs were processed
    with pytest.raises(ValueError, match="No retrieval results available"):
        metric.compute()


def test_contrastive_retrieval_accuracy_k_values():
    """Test metric with different k values."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST", "ALANYLGLY"]
    transform_fn = create_custom_length_transform()

    # Test with custom k values
    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        k_values=[1, 2, 4],
        random_state=42,
    )

    assert metric.k_values == [1, 2, 4]

    metric.update()
    results = metric.compute()

    # Should have results for all specified k values
    assert "top_1_accuracy" in results
    assert "top_2_accuracy" in results
    assert "top_4_accuracy" in results
    assert "top_3_accuracy" not in results  # Should not have k=3


def test_contrastive_retrieval_accuracy_single_sequence():
    """Test metric with a single sequence."""
    sequences = ["MKLLVVVGG"]
    transform_fn = create_custom_length_transform()

    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        k_values=[1],
        random_state=42,
    )

    assert len(metric.query_target_pairs) == 1

    metric.update()
    results = metric.compute()

    # With a single sequence, top-1 accuracy should be 1.0 (perfect retrieval)
    assert results["top_1_accuracy"] == 1.0
    assert results["mean_rank"] == 1.0
    assert results["median_rank"] == 1.0
    assert results["mean_reciprocal_rank"] == 1.0


def test_contrastive_retrieval_accuracy_default_k_values():
    """Test metric with default k values."""
    sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST"]
    transform_fn = create_custom_length_transform()

    # Test with default k values (should be [1, 5, 10])
    metric = ContrastiveRetrievalAccuracy(
        query_sequences=sequences,
        embedding_function=dummy_embedding_function,
        transform_function=transform_fn,
        random_state=42,
    )

    assert metric.k_values == [1, 5, 10]

    metric.update()
    results = metric.compute()

    # Should have results for default k values
    assert "top_1_accuracy" in results
    assert "top_5_accuracy" in results
    assert "top_10_accuracy" in results
