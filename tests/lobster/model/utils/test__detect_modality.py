"""
Tests for the _detect_modality function.
"""

import pytest

from lobster.constants import Modality
from lobster.model.utils._detect_modality import _detect_modality


class TestDetectModality:
    """Test cases for the _detect_modality function."""

    def test_valid_dna_sequences(self):
        """Test detection of valid DNA sequences."""
        test_cases = [
            "ATGC",
            "GATACA",
            "ATGCATGCATGC",
            "GCTAGCTAGCTA",
        ]

        for sequence in test_cases:
            modality = _detect_modality(sequence)
            assert modality == Modality.NUCLEOTIDE

    def test_valid_protein_sequences(self):
        """Test detection of valid protein sequences."""
        test_cases = [
            "MKTVRQERLKSIVRILERSKEPVSGAQL",
            "ACDEFGHIKLMNPQRSTVWY",
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        ]

        for sequence in test_cases:
            modality = _detect_modality(sequence)
            assert modality == Modality.AMINO_ACID

    def test_valid_smiles_sequences(self):
        """Test detection of valid SMILES sequences."""
        test_cases = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        ]

        for sequence in test_cases:
            modality = _detect_modality(sequence)
            assert modality == Modality.SMILES

    def test_case_insensitive(self):
        """Test that the function handles case-insensitive input."""
        # DNA
        assert _detect_modality("atgc") == Modality.NUCLEOTIDE
        assert _detect_modality("AtGc") == Modality.NUCLEOTIDE

        # Protein
        assert _detect_modality("mkTvrQERLKSIVRILERSKEPVSGAQL") == Modality.AMINO_ACID

        # SMILES (should be case-sensitive for chemical symbols)
        assert _detect_modality("CCO") == Modality.SMILES

    def test_whitespace_handling(self):
        """Test that the function handles whitespace correctly."""
        # DNA with whitespace
        assert _detect_modality("  ATGC  ") == Modality.NUCLEOTIDE
        assert _detect_modality("\tGATACA\n") == Modality.NUCLEOTIDE

        # Protein with whitespace
        assert _detect_modality("  MKTVRQERLKSIVRILERSKEPVSGAQL  ") == Modality.AMINO_ACID

    def test_short_sequences(self):
        """Test that very short sequences are rejected."""
        short_sequences = ["", "A", "AT", "AB"]

        for sequence in short_sequences:
            with pytest.raises(ValueError, match="Sequence too short"):
                _detect_modality(sequence)

    def test_ambiguous_sequences(self):
        """Test sequences that don't clearly match any modality."""
        ambiguous_sequences = [
            "XYZ123",
            "ATGCXYZ",
            "MKTVR123",
            # "CC(=O)XYZ",  # This actually matches SMILES pattern but fails validation
        ]

        for sequence in ambiguous_sequences:
            with pytest.raises(ValueError, match="Unable to determine modality"):
                _detect_modality(sequence)

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        # This should work even if the sequence might not be perfectly valid
        # but matches the pattern
        modality = _detect_modality("ATGC", validate=False)
        assert modality == Modality.NUCLEOTIDE

    def test_dna_validation_failure(self):
        """Test DNA sequences that fail Biopython validation."""
        # These should still be detected as DNA by pattern but might fail validation
        # depending on Biopython's strictness
        try:
            modality = _detect_modality("ATGC")
            assert modality == Modality.NUCLEOTIDE
        except ValueError as e:
            # If validation fails, it should be a validation error
            assert "failed validation" in str(e)

    def test_protein_validation_failure(self):
        """Test protein sequences that fail Biopython validation."""
        # These should still be detected as protein by pattern but might fail validation
        try:
            modality = _detect_modality("ACDEFGHIKLMNPQRSTVWY")
            assert modality == Modality.AMINO_ACID
        except ValueError as e:
            # If validation fails, it should be a validation error
            assert "failed validation" in str(e)

    def test_smiles_validation_failure(self):
        """Test SMILES sequences that fail RDKit validation."""
        # Invalid SMILES that match the pattern
        invalid_smiles = [
            "C(C(C(C(C",  # Unclosed parentheses
            "CC(=O)XYZ",  # Invalid element in otherwise valid SMILES
            "C[Invalid]C",  # Invalid element
            "C=====C======C",  # Invalid bond structure (too many equals signs)
        ]

        for smiles in invalid_smiles:
            with pytest.raises(ValueError, match="failed validation"):
                _detect_modality(smiles)

    def test_smiles_validation_success(self):
        """Test valid SMILES sequences that pass RDKit validation."""
        valid_smiles = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        ]

        for smiles in valid_smiles:
            modality = _detect_modality(smiles)
            assert modality == Modality.SMILES

    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        # Very long sequences
        long_dna = "ATGC" * 1000
        assert _detect_modality(long_dna) == Modality.NUCLEOTIDE

        long_protein = "ACDEFGHIKLMNPQRSTVWY" * 100
        assert _detect_modality(long_protein) == Modality.AMINO_ACID

        # Sequences with mixed case that should still be valid
        mixed_case_dna = "AtGcAtGc"
        assert _detect_modality(mixed_case_dna) == Modality.NUCLEOTIDE

    def test_priority_order(self):
        """Test that DNA is checked before protein (since DNA bases are subset of protein)."""
        # "ATGC" could be interpreted as protein (contains only valid amino acids)
        # but should be detected as DNA due to priority order
        assert _detect_modality("ATGC") == Modality.NUCLEOTIDE

        # But "ACDEFGHIKLMNPQRSTVWY" should be protein (contains amino acids not in DNA)
        assert _detect_modality("ACDEFGHIKLMNPQRSTVWY") == Modality.AMINO_ACID
