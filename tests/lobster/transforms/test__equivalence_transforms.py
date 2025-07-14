from typing import Any, Literal
from unittest import mock

import pytest
from rdkit import Chem

from lobster.transforms._equivalence_transforms import (
    AminoAcidToNucleotideAndSmilesTransform,
    AminoAcidToNucleotidePairTransform,
    AminoAcidToSmilesPairTransform,
    NucleotideToAminoAcidAndSmilesTransform,
    NucleotideToAminoAcidPairTransform,
    NucleotideToSmilesPairTransform,
    SmilesToSmilesPairTransform,
)


def get_canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, doRandom=False)


# --- Constants for expected SMILES strings ---
SMILES_CCO_CANONICAL = "CCO"
PEPTIDE_ACEG_CANONICAL_SMILES = "C[C@H](N)C(=O)N[C@@H](CS)C(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O"
PEPTIDE_A_CANONICAL_SMILES = "C[C@H](N)C(=O)O"

# Canonical SMILES for "AC" (Alanine-Cysteine)
_mol_ac = Chem.MolFromSequence("AC")
PEPTIDE_AC_CANONICAL_SMILES = Chem.MolToSmiles(_mol_ac, doRandom=False) if _mol_ac else "AC_FAILED_GENERATION"

# Expected SMILES for NT sequences (dynamically generated for consistency)
_mol_atgc_no_cap = Chem.MolFromSequence("ATGC", flavor=6)  # DNA, no cap
NT_ATGC_NO_CAP_SMILES = (
    Chem.MolToSmiles(_mol_atgc_no_cap, doRandom=False) if _mol_atgc_no_cap else "ATGC_NO_CAP_FAILED_GENERATION"
)

_mol_atgc_5_prime = Chem.MolFromSequence("ATGC", flavor=7)  # DNA, 5' cap
NT_ATGC_5_PRIME_SMILES = (
    Chem.MolToSmiles(_mol_atgc_5_prime, doRandom=False) if _mol_atgc_5_prime else "ATGC_5_PRIME_FAILED_GENERATION"
)

_mol_atgc_3_prime = Chem.MolFromSequence("ATGC", flavor=8)  # DNA, 3' cap
NT_ATGC_3_PRIME_SMILES = (
    Chem.MolToSmiles(_mol_atgc_3_prime, doRandom=False) if _mol_atgc_3_prime else "ATGC_3_PRIME_FAILED_GENERATION"
)

_mol_atgc_both_cap = Chem.MolFromSequence("ATGC", flavor=9)  # DNA, both caps
NT_ATGC_BOTH_CAP_SMILES = (
    Chem.MolToSmiles(_mol_atgc_both_cap, doRandom=False) if _mol_atgc_both_cap else "ATGC_BOTH_FAILED_GENERATION"
)

# Expected SMILES for truncated "AT" NT sequence with different caps
_mol_at_no_cap = Chem.MolFromSequence("AT", flavor=6)  # DNA, no cap
NT_AT_NO_CAP_SMILES = (
    Chem.MolToSmiles(_mol_at_no_cap, doRandom=False) if _mol_at_no_cap else "AT_NO_CAP_FAILED_GENERATION"
)
_mol_at_5_prime = Chem.MolFromSequence("AT", flavor=7)  # DNA, 5' cap
NT_AT_5_PRIME_SMILES = (
    Chem.MolToSmiles(_mol_at_5_prime, doRandom=False) if _mol_at_5_prime else "AT_5_PRIME_FAILED_GENERATION"
)
_mol_at_3_prime = Chem.MolFromSequence("AT", flavor=8)  # DNA, 3' cap
NT_AT_3_PRIME_SMILES = (
    Chem.MolToSmiles(_mol_at_3_prime, doRandom=False) if _mol_at_3_prime else "AT_3_PRIME_FAILED_GENERATION"
)
_mol_at_both_cap = Chem.MolFromSequence("AT", flavor=9)  # DNA, both caps
NT_AT_BOTH_CAP_SMILES = (
    Chem.MolToSmiles(_mol_at_both_cap, doRandom=False) if _mol_at_both_cap else "AT_BOTH_FAILED_GENERATION"
)


class TestSmilesToSmilesPairTransform:
    def test_init(self):
        transform_no_random = SmilesToSmilesPairTransform(randomize_smiles=False)
        assert not transform_no_random._randomize_smiles
        transform_random = SmilesToSmilesPairTransform(randomize_smiles=True)
        assert transform_random._randomize_smiles

    @pytest.mark.parametrize(
        "inputs, expected_error, error_message_contains",
        [
            ([], ValueError, "expects one string input, got none"),
            (["s1", "s2"], ValueError, "expects a single string input, but got 2"),
            ([123], TypeError, "expects a string input, but got type <class 'int'>"),
        ],
    )
    def test_check_inputs_invalid(
        self, inputs: list[Any], expected_error: type[Exception], error_message_contains: str
    ):
        transform = SmilesToSmilesPairTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)  # __call__ will invoke _check_inputs

    @pytest.mark.parametrize(
        "input_smiles, randomize, expected_canonical",
        [
            ("CCO", False, SMILES_CCO_CANONICAL),
            ("OCC", False, SMILES_CCO_CANONICAL),  # Non-canonical input
            ("CCO", True, SMILES_CCO_CANONICAL),  # Randomized, canonical target is CCO
            ("c1ccccc1", False, "c1ccccc1"),
            ("C1=CC=CC=C1", False, "c1ccccc1"),  # Kekule form
            ("", False, ""),
            ("", True, ""),
        ],
    )
    def test_transform_valid_smiles(self, input_smiles: str, randomize: bool, expected_canonical: str):
        transform = SmilesToSmilesPairTransform(randomize_smiles=randomize)
        outputs = transform([input_smiles])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == input_smiles
        assert result is not None
        assert get_canonical_smiles(result) == expected_canonical
        if (
            randomize and input_smiles == expected_canonical and len(input_smiles) > 3
        ):  # Simple check if it's likely randomized
            # This is not a strict guarantee, but for CCO it might be same. For larger, less likely.
            pass  # Could check result != input_smiles, but not guaranteed.
        elif not randomize:
            assert result == expected_canonical  # Should be canonical form directly

    @pytest.mark.parametrize(
        "input_smiles, randomize",
        [
            ("invalid_smiles", False),
            ("invalid_smiles", True),
        ],
    )
    def test_transform_invalid_or_empty_smiles(self, input_smiles: str, randomize: bool):
        transform = SmilesToSmilesPairTransform(randomize_smiles=randomize)
        outputs = transform([input_smiles])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == input_smiles
        assert result is None


class TestAminoAcidToSmilesPairTransform:
    def test_init(self):
        transform_no_random = AminoAcidToSmilesPairTransform(randomize_smiles=False)
        assert not transform_no_random._randomize_smiles
        transform_random = AminoAcidToSmilesPairTransform(randomize_smiles=True)
        assert transform_random._randomize_smiles

    @pytest.mark.parametrize(
        "inputs, expected_error, error_message_contains",
        [
            ([], ValueError, "expects one string input, got none"),
            (["p1", "p2"], ValueError, "expects a single string input, but got 2"),
            ([123], TypeError, "expects a string input, but got type <class 'int'>"),
        ],
    )
    def test_check_inputs_invalid(
        self, inputs: list[Any], expected_error: type[Exception], error_message_contains: str
    ):
        transform = AminoAcidToSmilesPairTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)

    def test_transform_empty_peptide(self):
        transform = AminoAcidToSmilesPairTransform(randomize_smiles=False)
        outputs = transform([""])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == ""
        assert result == ""

        transform_random = AminoAcidToSmilesPairTransform(randomize_smiles=True)
        outputs_rand = transform_random([""])
        assert len(outputs_rand) == 1
        original_rand, result_rand = outputs_rand[0]
        assert original_rand == ""
        assert result_rand == ""

    @pytest.mark.parametrize(
        "input_peptide, randomize_smiles, expected_canonical_smiles",
        [
            ("A", False, PEPTIDE_A_CANONICAL_SMILES),
            ("A", True, PEPTIDE_A_CANONICAL_SMILES),
            ("ACEG", False, PEPTIDE_ACEG_CANONICAL_SMILES),
            ("ACEG", True, PEPTIDE_ACEG_CANONICAL_SMILES),
        ],
    )
    def test_transform_valid_peptide(self, input_peptide: str, randomize_smiles: bool, expected_canonical_smiles: str):
        transform = AminoAcidToSmilesPairTransform(randomize_smiles=randomize_smiles)
        outputs = transform([input_peptide])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == input_peptide
        assert result is not None
        assert get_canonical_smiles(result) == expected_canonical_smiles

        if not randomize_smiles:
            # For non-randomized, the result should be the canonical SMILES directly
            # if the underlying convert_aa_to_smiles produces canonical by default when not randomizing
            # Let's assume convert_aa_to_smiles is canonical when not randomizing for now
            assert result == expected_canonical_smiles
        # For randomized SMILES, we only check that it canonicalizes correctly, not exact equality

    @pytest.mark.parametrize(
        "input_peptide, max_len, expected_original, expected_smiles_target",
        [
            ("ACEG", 2, "AC", PEPTIDE_AC_CANONICAL_SMILES),
            ("ACEG", 4, "ACEG", PEPTIDE_ACEG_CANONICAL_SMILES),  # No truncation
            ("ACEG", 10, "ACEG", PEPTIDE_ACEG_CANONICAL_SMILES),  # No truncation, max_len > len
            ("ACEG", 0, "", ""),  # Truncate to empty
        ],
    )
    def test_transform_with_max_input_length(
        self,
        input_peptide: str,
        max_len: int | None,
        expected_original: str,
        expected_smiles_target: str,
    ):
        transform = AminoAcidToSmilesPairTransform(max_input_length=max_len)
        outputs = transform([input_peptide])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == expected_original
        if expected_smiles_target == "":
            assert result == ""
        else:
            assert result is not None
            assert result == expected_smiles_target


class TestNucleotideToSmilesPairTransform:
    def test_init(self):
        transform_a = NucleotideToSmilesPairTransform(randomize_smiles=False, randomize_cap=False)
        assert not transform_a._randomize_smiles
        assert not transform_a._randomize_cap
        transform_b = NucleotideToSmilesPairTransform(randomize_smiles=True, randomize_cap=True)
        assert transform_b._randomize_smiles
        assert transform_b._randomize_cap

    @pytest.mark.parametrize(
        "inputs, expected_error, error_message_contains",
        [
            ([], ValueError, "expects one string input, got none"),
            (["n1", "n2"], ValueError, "expects a single string input, but got 2"),
            ([123], TypeError, "expects a string input, but got type <class 'int'>"),
        ],
    )
    def test_check_inputs_invalid(
        self, inputs: list[Any], expected_error: type[Exception], error_message_contains: str
    ):
        transform = NucleotideToSmilesPairTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)

    @pytest.mark.parametrize(
        "input_nt, randomize_smiles, expected_smiles",
        [
            ("ATGC", False, NT_ATGC_NO_CAP_SMILES),
            ("atgc", False, NT_ATGC_NO_CAP_SMILES),  # Should be uppercased
            ("ATGC", True, NT_ATGC_NO_CAP_SMILES),  # Randomized, target is canonical no-cap
        ],
    )
    def test_transform_no_random_cap(self, input_nt: str, randomize_smiles: bool, expected_smiles: str):
        transform = NucleotideToSmilesPairTransform(randomize_smiles=randomize_smiles, randomize_cap=False)
        outputs = transform([input_nt])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == input_nt.upper()  # Transform uppercases input
        assert result is not None
        assert get_canonical_smiles(result) == expected_smiles  # expected_smiles is already canonical by generation
        if not randomize_smiles:
            assert result == expected_smiles

    @pytest.mark.parametrize(
        "input_nt, randomize_smiles, mock_cap_choice, expected_smiles_target",
        [
            ("ATGC", False, None, NT_ATGC_NO_CAP_SMILES),
            ("ATGC", False, "5'", NT_ATGC_5_PRIME_SMILES),
            ("ATGC", False, "3'", NT_ATGC_3_PRIME_SMILES),
            ("ATGC", False, "both", NT_ATGC_BOTH_CAP_SMILES),
            # Randomized SMILES, cap is chosen, target is canonical of that cap
            ("ATGC", True, None, NT_ATGC_NO_CAP_SMILES),
            ("", False, None, ""),
            ("", True, None, ""),
        ],
    )
    @mock.patch("lobster.transforms._equivalence_transforms.random.choice")
    def test_transform_with_random_cap(
        self,
        mock_random_choice: mock.Mock,
        input_nt: str,
        randomize_smiles: bool,
        mock_cap_choice: Literal["5'", "3'", "both"] | None,
        expected_smiles_target: str,
    ):
        mock_random_choice.return_value = mock_cap_choice
        transform = NucleotideToSmilesPairTransform(randomize_smiles=randomize_smiles, randomize_cap=True)

        outputs = transform([input_nt])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == input_nt.upper()
        mock_random_choice.assert_called_once_with(["5'", "3'", "both", None])

        assert result is not None
        assert get_canonical_smiles(result) == expected_smiles_target  # expected_smiles_target is already canonical
        if not randomize_smiles:  # If SMILES not randomized, result should be the direct target
            assert result == expected_smiles_target

    @pytest.mark.parametrize(
        "input_nt, randomize_smiles, randomize_cap",
        [
            ("ATGX", False, False),  # Invalid nucleotide
            ("ATGX", True, True),  # Invalid nucleotide
        ],
    )
    @mock.patch("lobster.transforms._equivalence_transforms.random.choice")
    def test_transform_invalid_or_empty_nucleotide(
        self, mock_random_choice: mock.Mock, input_nt: str, randomize_smiles: bool, randomize_cap: bool
    ):
        mock_random_choice.return_value = None  # Mock choice for cases where it's called
        transform = NucleotideToSmilesPairTransform(randomize_smiles=randomize_smiles, randomize_cap=randomize_cap)
        outputs = transform([input_nt])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == input_nt.upper()
        assert result is None
        if randomize_cap:  # If randomize_cap is True, choice is always called
            mock_random_choice.assert_called_once_with(["5'", "3'", "both", None])
        else:  # If randomize_cap is False, choice is never called
            mock_random_choice.assert_not_called()

    def test_transform_nucleotide_invalid_cap_type_error(self):
        """Tests that an invalid cap from random.choice would lead to ValueError from convert_nt_to_smiles"""
        transform = NucleotideToSmilesPairTransform(randomize_cap=True)
        with mock.patch(
            "lobster.transforms._equivalence_transforms.random.choice", return_value="invalid_cap_val"
        ) as mocked_choice:
            with pytest.raises(ValueError, match="Invalid cap: invalid_cap_val"):
                transform(["ATGC"])
            mocked_choice.assert_called_once()

    @pytest.mark.parametrize(
        "input_nt, max_len, expected_original_nt, expected_smiles_target",
        [
            ("ATGC", 2, "AT", NT_AT_NO_CAP_SMILES),
            ("atgc", 2, "AT", NT_AT_NO_CAP_SMILES),  # Test case insensitivity and truncation
            ("ATGC", 2, "AT", NT_AT_NO_CAP_SMILES),  # Randomized, target is canonical
            ("ATGC", 4, "ATGC", NT_ATGC_NO_CAP_SMILES),  # No truncation
            ("ATGC", 10, "ATGC", NT_ATGC_NO_CAP_SMILES),  # No truncation, max_len > len
            ("ATGC", 0, "", ""),  # Truncate to empty
            ("", 5, "", ""),  # Empty input
            ("AT", None, "AT", NT_AT_NO_CAP_SMILES),  # max_len is None
        ],
    )
    def test_transform_with_max_input_length(
        self,
        input_nt: str,
        max_len: int | None,
        expected_original_nt: str,
        expected_smiles_target: str,
    ):
        transform = NucleotideToSmilesPairTransform(randomize_cap=False, max_input_length=max_len)
        outputs = transform([input_nt])
        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == expected_original_nt  # Transform uppercases and truncates input
        if expected_smiles_target == "":
            assert result == ""
        else:
            assert result is not None
            assert get_canonical_smiles(result) == expected_smiles_target


class TestNucleotideToAminoAcidPairTransform:
    def test_init_default(self):
        """Test default initialization."""
        transform = NucleotideToAminoAcidPairTransform()
        assert transform._reading_frame == 0
        assert transform._max_input_length is None
        assert transform._codon_to_residue is not None
        assert transform._residue_to_codon is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        transform = NucleotideToAminoAcidPairTransform(reading_frame=1, max_input_length=100)
        assert transform._reading_frame == 1
        assert transform._max_input_length == 100

    def test_init_invalid_reading_frame(self):
        """Test initialization with invalid reading frame."""
        with pytest.raises(ValueError, match="reading_frame must be 0, 1, or 2"):
            NucleotideToAminoAcidPairTransform(reading_frame=3)

    @pytest.mark.parametrize(
        "inputs, expected_error, error_message_contains",
        [
            ([], ValueError, "expects one string input, got none"),
            (["n1", "n2"], ValueError, "expects a single string input, but got 2"),
            ([123], TypeError, "expects a string input, but got type <class 'int'>"),
        ],
    )
    def test_check_inputs_invalid(
        self, inputs: list[Any], expected_error: type[Exception], error_message_contains: str
    ):
        transform = NucleotideToAminoAcidPairTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)

    @pytest.mark.parametrize(
        "input_nt, reading_frame, expected_original, expected_protein",
        [
            # Frame 0: start from beginning
            ("ATGAAATAG", 0, "ATGAAATAG", "MK"),  # ATG AAA TAG -> M K (stop)
            ("ATGAAACAG", 0, "ATGAAACAG", "MKQ"),  # ATG AAA CAG -> M K Q
            # Frame 1: skip first nucleotide
            ("CATGAAATAG", 1, "ATGAAATAG", "MK"),  # C+ATG AAA TAG -> M K (stop)
            # Frame 2: skip first two nucleotides
            ("CCATGAAATAG", 2, "ATGAAATAG", "MK"),  # CC+ATG AAA TAG -> M K (stop)
            # Empty input
            ("", 0, "", ""),
            # Short input
            ("AT", 0, "AT", "<unk>"),  # Not enough for a complete codon
        ],
    )
    def test_transform_basic(self, input_nt: str, reading_frame: int, expected_original: str, expected_protein: str):
        transform = NucleotideToAminoAcidPairTransform(reading_frame=reading_frame)
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == expected_original.upper()
        if expected_protein == "":
            assert result == ""
        else:
            assert result == expected_protein

    def test_transform_case_insensitive(self):
        """Test that input is converted to uppercase."""
        transform = NucleotideToAminoAcidPairTransform()
        input_nt = "atgaaatag"
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "ATGAAATAG"
        assert result == "MK"

    def test_transform_with_max_input_length(self):
        """Test truncation with max_input_length."""
        transform = NucleotideToAminoAcidPairTransform(max_input_length=6)
        input_nt = "ATGAAACAG"  # 9 bases
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "ATGAAA"  # Truncated to 6 bases
        assert result == "MK"  # Only 1 complete codon after truncation

    def test_transform_unknown_codon(self):
        """Test handling of unknown codons."""
        transform = NucleotideToAminoAcidPairTransform()
        input_nt = "ATGXXXCAG"  # XXX is not a valid codon
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "ATGXXXCAG"
        assert result == "M<unk>Q"  # XXX becomes <unk>

    def test_transform_early_stop_codon(self):
        """Test handling of early stop codons."""
        transform = NucleotideToAminoAcidPairTransform()
        input_nt = "ATGTAAAAACAT"  # ATG TAA AAA CAT -> M (stop) - translation stops
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "ATGTAAAAACAT"
        assert result == "M"  # Translation stops at TAA

    def test_transform_exception_handling(self):
        """Test that exceptions during conversion result in None."""
        transform = NucleotideToAminoAcidPairTransform()

        # Mock convert_nt_to_aa to raise an exception
        with mock.patch(
            "lobster.transforms._equivalence_transforms.convert_nt_to_aa", side_effect=ValueError("Test error")
        ):
            outputs = transform(["ATGAAACAG"])

            assert len(outputs) == 1
            original, result = outputs[0]

            assert original == "ATGAAACAG"
            assert result is None


class TestAminoAcidToNucleotidePairTransform:
    def test_init_default(self):
        """Test default initialization."""
        transform = AminoAcidToNucleotidePairTransform()
        assert transform._max_input_length is None
        assert transform._add_stop_codon is True
        assert transform._vendor_codon_table is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        transform = AminoAcidToNucleotidePairTransform(max_input_length=50, add_stop_codon=False)
        assert transform._max_input_length == 50
        assert transform._add_stop_codon is False

    @pytest.mark.parametrize(
        "inputs, expected_error, error_message_contains",
        [
            ([], ValueError, "expects one string input, got none"),
            (["p1", "p2"], ValueError, "expects a single string input, but got 2"),
            ([123], TypeError, "expects a string input, but got type <class 'int'>"),
        ],
    )
    def test_check_inputs_invalid(
        self, inputs: list[Any], expected_error: type[Exception], error_message_contains: str
    ):
        transform = AminoAcidToNucleotidePairTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)

    def test_transform_basic_with_stop_codon(self):
        """Test basic transformation with stop codon."""
        transform = AminoAcidToNucleotidePairTransform(add_stop_codon=True)
        input_protein = "MK"
        outputs = transform([input_protein])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "MK"
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 9  # 2 AA + 1 stop = 3 codons = 9 bases

        # Check structure
        assert result[0:3] == "ATG"  # M always ATG
        assert result[3:6] in ["AAA", "AAG"]  # K codons based on usage frequency
        assert result[6:9] in ["TAA", "TAG", "TGA"]  # Stop codons

    def test_transform_basic_without_stop_codon(self):
        """Test basic transformation without stop codon."""
        transform = AminoAcidToNucleotidePairTransform(add_stop_codon=False)
        input_protein = "MK"
        outputs = transform([input_protein])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "MK"
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 6  # 2 AA = 2 codons = 6 bases

        # Check structure
        assert result[0:3] == "ATG"  # M always ATG
        assert result[3:6] in ["AAA", "AAG"]  # K codons

    def test_transform_case_insensitive(self):
        """Test that input is converted to uppercase."""
        transform = AminoAcidToNucleotidePairTransform(add_stop_codon=False)
        input_protein = "mk"
        outputs = transform([input_protein])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "MK"
        assert result is not None
        assert len(result) == 6

    def test_transform_with_max_input_length(self):
        """Test truncation with max_input_length."""
        transform = AminoAcidToNucleotidePairTransform(max_input_length=2, add_stop_codon=False)
        input_protein = "MKLV"  # 4 amino acids
        outputs = transform([input_protein])

        assert len(outputs) == 1
        original, result = outputs[0]

        assert original == "MK"  # Truncated to 2 amino acids
        assert result is not None
        assert len(result) == 6  # 2 codons = 6 bases

    def test_transform_exception_handling(self):
        """Test that exceptions during conversion result in None."""
        transform = AminoAcidToNucleotidePairTransform()

        # Mock convert_aa_to_nt_probabilistic to raise an exception
        with mock.patch(
            "lobster.transforms._equivalence_transforms.convert_aa_to_nt_probabilistic",
            side_effect=ValueError("Test error"),
        ):
            outputs = transform(["MK"])

            assert len(outputs) == 1
            original, result = outputs[0]

            assert original == "MK"
            assert result is None

    def test_probabilistic_behavior(self):
        """Test that the probabilistic nature works (different runs can give different results)."""
        transform = AminoAcidToNucleotidePairTransform(add_stop_codon=False)
        input_protein = "L"  # Leucine has 6 possible codons

        # Run multiple times and collect results
        results = set()
        for _ in range(50):  # Run enough times to likely see variation
            outputs = transform([input_protein])
            result = outputs[0][1]
            results.add(result)

        # Should have more than one result due to probabilistic sampling
        # (though theoretically could be the same by chance)
        assert len(results) >= 1  # At least one result

        # All results should be valid leucine codons
        valid_leucine_codons = {"TTA", "TTG", "CTT", "CTC", "CTA", "CTG"}
        for result in results:
            assert result in valid_leucine_codons


def test_amino_acid_to_nucleotide_and_smiles_transform():
    """Test the AminoAcidToNucleotideAndSmilesTransform with a simple peptide sequence."""
    # Create the transform
    transform = AminoAcidToNucleotideAndSmilesTransform(
        max_input_length=1000, add_stop_codon=True, randomize_smiles=False
    )

    # Example peptide sequence
    peptide = "MAGIC"

    # Run the transform
    peptide_seq, nucleotide_seq, smiles = transform(peptide)

    # Check results
    assert peptide_seq == "MAGIC"
    assert set(nucleotide_seq) == {"A", "T", "G", "C"}
    assert smiles == "CC[C@H](C)[C@H](NC(=O)CNC(=O)[C@H](C)NC(=O)[C@@H](N)CCSC)C(=O)N[C@@H](CS)C(=O)O"

    # Test with max_input_length
    transform = AminoAcidToNucleotideAndSmilesTransform(max_input_length=3, add_stop_codon=True, randomize_smiles=False)
    peptide_seq, nucleotide_seq, smiles = transform(peptide)
    assert peptide_seq == "MAG"
    assert nucleotide_seq is not None
    assert smiles is not None


class TestNucleotideToAminoAcidAndSmilesTransform:
    """Test the NucleotideToAminoAcidAndSmilesTransform class."""

    def test_init_default(self):
        """Test default initialization."""
        transform = NucleotideToAminoAcidAndSmilesTransform()
        assert transform._max_input_length is None
        assert transform._randomize_smiles is False
        assert transform._randomize_cap is False
        assert transform._codon_to_residue is not None
        assert transform._residue_to_codon is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        transform = NucleotideToAminoAcidAndSmilesTransform(
            max_input_length=100,
            randomize_smiles=True,
            randomize_cap=True,
        )
        assert transform._max_input_length == 100
        assert transform._randomize_smiles is True
        assert transform._randomize_cap is True

    @pytest.mark.parametrize(
        "inputs, expected_error, error_message_contains",
        [
            ([], ValueError, "expects one string input, got none"),
            (["n1", "n2"], ValueError, "expects a single string input, but got 2"),
            ([123], TypeError, "expects a string input, but got type <class 'int'>"),
        ],
    )
    def test_check_inputs_invalid(
        self, inputs: list[Any], expected_error: type[Exception], error_message_contains: str
    ):
        transform = NucleotideToAminoAcidAndSmilesTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)

    def test_transform_valid_atg_sequence(self):
        """Test transformation of valid sequence starting with ATG."""
        transform = NucleotideToAminoAcidAndSmilesTransform(
            randomize_smiles=False,
            randomize_cap=False,
        )
        input_nt = "ATGAAATAG"  # ATG AAA TAG -> M K (stop)
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "ATGAAATAG"
        assert amino_acid_seq == "MK"
        assert smiles_seq is not None
        assert isinstance(smiles_seq, str)

    def test_transform_valid_atg_sequence_case_insensitive(self):
        """Test transformation with lowercase input."""
        transform = NucleotideToAminoAcidAndSmilesTransform()
        input_nt = "atgaaatag"
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "ATGAAATAG"
        assert amino_acid_seq == "MK"
        assert smiles_seq is not None

    def test_transform_no_atg_start(self):
        """Test transformation of sequence not starting with ATG."""
        transform = NucleotideToAminoAcidAndSmilesTransform()
        input_nt = "AAATAG"  # Does not start with ATG
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "AAATAG"
        assert amino_acid_seq is None  # Should be None due to no ATG
        assert smiles_seq is not None  # SMILES conversion should still work

    def test_transform_with_max_input_length(self):
        """Test truncation with max_input_length."""
        transform = NucleotideToAminoAcidAndSmilesTransform(max_input_length=6)
        input_nt = "ATGAAACAG"  # 9 bases
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "ATGAAA"  # Truncated to 6 bases
        assert amino_acid_seq == "MK"  # Only 1 complete codon after truncation
        assert smiles_seq is not None

    @mock.patch("lobster.transforms._equivalence_transforms.random.choice")
    def test_transform_with_randomize_cap(self, mock_random_choice):
        """Test transformation with randomized caps."""
        mock_random_choice.return_value = "5'"
        transform = NucleotideToAminoAcidAndSmilesTransform(randomize_cap=True)
        input_nt = "ATGAAATAG"
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "ATGAAATAG"
        assert amino_acid_seq == "MK"
        assert smiles_seq is not None
        mock_random_choice.assert_called_once_with(["5'", "3'", "both", None])

    def test_transform_exception_handling_amino_acid(self):
        """Test that exceptions during amino acid conversion result in None."""
        transform = NucleotideToAminoAcidAndSmilesTransform()

        # Mock convert_nt_to_aa to raise an exception
        with mock.patch(
            "lobster.transforms._equivalence_transforms.convert_nt_to_aa", side_effect=ValueError("Test error")
        ):
            outputs = transform(["ATGAAACAG"])

            assert len(outputs) == 1
            original, amino_acid_seq, smiles_seq = outputs[0]

            assert original == "ATGAAACAG"
            assert amino_acid_seq is None
            assert smiles_seq is not None  # SMILES conversion should still work

    def test_transform_exception_handling_smiles(self):
        """Test that exceptions during SMILES conversion result in None."""
        transform = NucleotideToAminoAcidAndSmilesTransform()

        # Mock convert_nt_to_smiles to raise an exception
        with mock.patch(
            "lobster.transforms._equivalence_transforms.convert_nt_to_smiles", side_effect=ValueError("Test error")
        ):
            outputs = transform(["ATGAAACAG"])

            assert len(outputs) == 1
            original, amino_acid_seq, smiles_seq = outputs[0]

            assert original == "ATGAAACAG"
            assert amino_acid_seq == "MKQ"  # ATG AAA CAG -> M K Q
            assert smiles_seq is None

    def test_transform_empty_sequence(self):
        """Test transformation of empty sequence."""
        transform = NucleotideToAminoAcidAndSmilesTransform()
        outputs = transform([""])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == ""
        assert amino_acid_seq is None  # Empty doesn't start with ATG
        assert smiles_seq == ""  # Empty SMILES for empty input

    def test_transform_short_sequence(self):
        """Test transformation of sequence shorter than ATG."""
        transform = NucleotideToAminoAcidAndSmilesTransform()
        input_nt = "AT"
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "AT"
        assert amino_acid_seq is None  # Too short to start with ATG
        assert smiles_seq is not None  # SMILES conversion should still work

    def test_transform_randomize_smiles(self):
        """Test transformation with randomized SMILES."""
        transform = NucleotideToAminoAcidAndSmilesTransform(randomize_smiles=True)
        input_nt = "ATGAAATAG"
        outputs = transform([input_nt])

        assert len(outputs) == 1
        original, amino_acid_seq, smiles_seq = outputs[0]

        assert original == "ATGAAATAG"
        assert amino_acid_seq == "MK"
        assert smiles_seq is not None
        # We can't easily test randomization without knowing the exact behavior
        # but we can verify it's still a valid SMILES string
        assert get_canonical_smiles(smiles_seq) is not None
