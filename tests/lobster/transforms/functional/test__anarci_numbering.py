"""Tests for ANARCI numbering functionality."""

import pytest

# Check if anarci is available
pytest.importorskip("anarci", reason="anarci not installed")

from lobster.transforms.functional.anarci_numbering import (
    anarci_numbering,
    get_aligned_kabat_sequences,
)


class TestAnarciNumbering:
    """Test suite for anarci_numbering function."""

    def test_anarci_numbering_basic(self):
        """Test basic ANARCI numbering with a known antibody sequence."""
        # Heavy chain variable region sequence (example)
        sequences = ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"]

        result = anarci_numbering(sequences)

        assert isinstance(result, list)
        assert len(result) == 1
        # The result should not be None for a valid antibody sequence
        assert result[0] is not None
        # The result should be a string
        assert isinstance(result[0], str)

    def test_anarci_numbering_multiple_sequences(self):
        """Test ANARCI numbering with multiple sequences."""
        sequences = [
            "QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY",
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
        ]

        result = anarci_numbering(sequences)

        assert isinstance(result, list)
        assert len(result) == 2
        # Both should be valid antibody sequences
        assert result[0] is not None
        assert result[1] is not None

    def test_anarci_numbering_with_invalid_sequence(self):
        """Test ANARCI numbering with an invalid sequence."""
        sequences = ["INVALIDSEQ"]

        result = anarci_numbering(sequences)

        assert isinstance(result, list)
        assert len(result) == 1
        # Invalid sequence should return None
        assert result[0] is None

    def test_anarci_numbering_with_metadata(self):
        """Test ANARCI numbering with metadata return."""
        sequences = ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"]

        result, metadata = anarci_numbering(sequences, return_metadata=True)

        assert isinstance(result, list)
        assert isinstance(metadata, list)
        assert len(result) == 1
        assert len(metadata) == 1
        # For valid sequence, metadata should not be None
        assert metadata[0] is not None
        # Metadata should contain scheme_indexes
        assert "scheme_indexes" in metadata[0]

    def test_anarci_numbering_kabat_scheme(self):
        """Test ANARCI numbering with Kabat scheme."""
        sequences = ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"]

        result = anarci_numbering(sequences, scheme="kabat")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not None
        assert isinstance(result[0], str)

    def test_anarci_numbering_chothia_scheme(self):
        """Test ANARCI numbering with Chothia scheme."""
        sequences = ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"]

        result = anarci_numbering(sequences, scheme="chothia")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not None
        assert isinstance(result[0], str)

    def test_anarci_numbering_imgt_scheme(self):
        """Test ANARCI numbering with IMGT scheme."""
        sequences = ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"]

        result = anarci_numbering(sequences, scheme="imgt")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not None
        assert isinstance(result[0], str)

    def test_anarci_numbering_with_none_in_list(self):
        """Test ANARCI numbering with None values in input."""
        sequences = [
            "QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY",
            None,
        ]

        result = anarci_numbering(sequences)

        assert isinstance(result, list)
        assert len(result) == 2
        # First should be valid
        assert result[0] is not None
        # Second should be None (was None in input)
        assert result[1] is None

    def test_anarci_numbering_invalid_scheme(self):
        """Test ANARCI numbering with invalid scheme."""
        sequences = ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"]

        with pytest.raises(ValueError, match="Unknown `scheme`"):
            anarci_numbering(sequences, scheme="invalid_scheme")

    def test_anarci_numbering_string_input_error(self):
        """Test that passing a single string raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of strings"):
            anarci_numbering("QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY")


class TestGetAlignedKabatSequences:
    """Test suite for get_aligned_kabat_sequences function."""

    def test_get_aligned_kabat_sequences_basic(self):
        """Test basic aligned Kabat sequence generation."""
        sequences = [
            "QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY",
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
        ]

        (
            aligned_sequences,
            kabat_positions,
            v_genes,
            j_genes,
        ) = get_aligned_kabat_sequences(sequences)

        assert isinstance(aligned_sequences, list)
        assert isinstance(kabat_positions, list)
        assert isinstance(v_genes, list)
        assert isinstance(j_genes, list)

        # All lists should have the same number of elements as input sequences
        assert len(aligned_sequences) == 2
        assert len(v_genes) == 2
        assert len(j_genes) == 2

        # All aligned sequences should have the same length
        if aligned_sequences[0] and aligned_sequences[1]:
            assert len(aligned_sequences[0]) == len(aligned_sequences[1])
            # Length should match the number of Kabat positions
            assert len(aligned_sequences[0]) == len(kabat_positions)

    def test_get_aligned_kabat_sequences_with_gaps(self):
        """Test aligned Kabat sequences with sequences containing gaps."""
        sequences = [
            "QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY",
        ]

        (
            aligned_sequences,
            kabat_positions,
            v_genes,
            j_genes,
        ) = get_aligned_kabat_sequences(sequences)

        assert isinstance(aligned_sequences, list)
        assert len(aligned_sequences) == 1
        # Should handle sequences with or without gaps
        assert aligned_sequences[0] is not None

    def test_get_aligned_kabat_sequences_with_required_indexes(self):
        """Test aligned Kabat sequences with required indexes."""
        sequences = [
            "QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY",
        ]

        required_indexes = {"1A", "2A"}

        (
            aligned_sequences,
            kabat_positions,
            v_genes,
            j_genes,
        ) = get_aligned_kabat_sequences(sequences, required_kabat_indexes=required_indexes)

        assert isinstance(aligned_sequences, list)
        assert isinstance(kabat_positions, list)

        # Required indexes should be present in kabat_positions
        for idx in required_indexes:
            assert idx in kabat_positions

    def test_get_aligned_kabat_sequences_empty_input(self):
        """Test aligned Kabat sequences with empty input."""
        sequences = []

        (
            aligned_sequences,
            kabat_positions,
            v_genes,
            j_genes,
        ) = get_aligned_kabat_sequences(sequences)

        assert isinstance(aligned_sequences, list)
        assert isinstance(kabat_positions, list)
        assert isinstance(v_genes, list)
        assert isinstance(j_genes, list)
        assert len(aligned_sequences) == 0
        assert len(v_genes) == 0
        assert len(j_genes) == 0
