from typing import Any, Literal
from unittest import mock

import pytest
from lobster.transforms._equivalence_transforms import (
    NucleotideToSmilesPairTransform,
    PeptideToSmilesPairTransform,
    SmilesToSmilesPairTransform,
)
from rdkit import Chem


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


class TestPeptideToSmilesPairTransform:
    def test_init(self):
        transform_no_random = PeptideToSmilesPairTransform(randomize_smiles=False)
        assert not transform_no_random._randomize_smiles
        transform_random = PeptideToSmilesPairTransform(randomize_smiles=True)
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
        transform = PeptideToSmilesPairTransform()
        with pytest.raises(expected_error, match=error_message_contains):
            transform(inputs)

    def test_transform_empty_peptide(self):
        transform = PeptideToSmilesPairTransform(randomize_smiles=False)
        outputs = transform([""])
        assert len(outputs) == 1
        original, result = outputs[0]
        assert original == ""
        assert result == ""

        transform_random = PeptideToSmilesPairTransform(randomize_smiles=True)
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
        transform = PeptideToSmilesPairTransform(randomize_smiles=randomize_smiles)
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
        # No easy check for randomization other than it not being None and canonicalizing correctly
        # unless we know specific non-canonical forms are likely for specific inputs.

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
        transform = PeptideToSmilesPairTransform(max_input_length=max_len)
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
