import pytest

from lobster.model.utils import _detect_modality


class TestModalityDetection:
    """Test the modality detection function."""

    def test_smiles_sequences(self):
        """Test detection of SMILES sequences."""
        test_cases = [
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "SMILES"),  # Ibuprofen
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "SMILES"),  # Aspirin
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "SMILES"),  # Caffeine
        ]

        for sequence, expected in test_cases:
            detected = _detect_modality(sequence)
            assert detected.value == expected, f"Expected {expected}, got {detected.value} for {sequence}"

    def test_amino_acid_sequences(self):
        """Test detection of amino acid sequences."""
        test_cases = [
            ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "amino_acid"),
            ("ACDEFGHIKLMNPQRSTVWY", "amino_acid"),
            (
                "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWAAIGLNKVVVKAAWG",
                "amino_acid",
            ),
        ]

        for sequence, expected in test_cases:
            detected = _detect_modality(sequence)
            assert detected.value == expected, f"Expected {expected}, got {detected.value} for {sequence}"

    def test_dna_sequences(self):
        """Test detection of DNA sequences."""
        test_cases = [
            ("ATGCATGCATGCATGCATGC", "nucleotide"),
            ("GCTAGCTAGCTAGCTAGCTA", "nucleotide"),
            ("ATGCGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA", "nucleotide"),
        ]

        for sequence, expected in test_cases:
            detected = _detect_modality(sequence)
            assert detected.value == expected, f"Expected {expected}, got {detected.value} for {sequence}"

    def test_edge_cases(self):
        """Test edge cases and invalid sequences."""
        # Test cases that should raise ValueError
        error_cases = [
            ("", r"Sequence too short \(length 0\)\. Minimum length required is 3 characters\."),
            ("A", r"Sequence too short \(length 1\)\. Minimum length required is 3 characters\."),
            ("AT", r"Sequence too short \(length 2\)\. Minimum length required is 3 characters\."),
            ("invalid sequence", r"Unable to determine modality for sequence: INVALID SEQUENCE"),
            ("12345", r"Sequence appears to be SMILES but failed validation: 12345"),
        ]

        for sequence, expected_error in error_cases:
            with pytest.raises(ValueError, match=expected_error):
                _detect_modality(sequence)

    def test_simple_value_error(self):
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError):
            _detect_modality("")

    def test_debug_error_messages(self):
        """Debug test to see actual error messages."""
        test_cases = ["", "A", "AT", "invalid sequence", "12345"]

        for sequence in test_cases:
            try:
                _detect_modality(sequence)
            except ValueError as e:
                print(f"Sequence '{sequence}' -> Error: {e}")
            else:
                print(f"Sequence '{sequence}' -> No error raised")
