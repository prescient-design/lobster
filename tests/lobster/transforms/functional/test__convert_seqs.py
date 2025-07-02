import importlib.resources
import re

import pytest
from rdkit import Chem

from lobster.transforms.functional import (
    convert_aa_to_nt,
    convert_aa_to_nt_probabilistic,
    convert_aa_to_selfies,
    convert_aa_to_smiles,
    convert_nt_to_aa,
    convert_nt_to_selfies_via_aa,
    convert_nt_to_smiles,
    convert_selfies_to_nt_via_aa,
    convert_selfies_to_smiles,
    convert_smiles_to_selfies,
    convert_smiles_to_smiles,
    invert_residue_to_codon_mapping,
    json_load,
    replace_target_symbol,
    replace_unknown_symbols,
    uniform_sample,
)


# Helper to get canonical SMILES using RDKit for test comparison
def get_canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, doRandom=False)
    return None


@pytest.fixture(scope="class")
def residue_to_codon_map() -> dict[str, list[str]]:
    path = importlib.resources.files("lobster") / "assets" / "codon_tables" / "codon_table.json"
    residue_to_codon = json_load(path)
    return residue_to_codon


@pytest.fixture(scope="class")
def codon_to_residue_map(residue_to_codon_map: dict[str, list[str]]) -> dict[str, str]:
    codon_to_residue = invert_residue_to_codon_mapping(residue_to_codon_map)
    return codon_to_residue


def split_by_two_characters(s: str, char1: str, char2: str) -> list[str]:
    pattern = f"[{re.escape(char1)}{re.escape(char2)}]+"
    return [item for item in re.split(pattern, s) if item]


class TestConvertSeqs:
    def test_convert_aa_to_nt(
        self,
        residue_to_codon_map: dict[str, list[str]],
        codon_to_residue_map: dict[str, str],
    ):
        aa_seq = "EVQLVESGGGLVQPGGSLRLS"
        nt_seq = convert_aa_to_nt(aa_seq, residue_to_codon_map, uniform_sample)
        assert isinstance(nt_seq, str), f"Failed for aa seq {aa_seq}, nt seq should be a str"
        assert len(nt_seq) == 3 * (len(aa_seq) + 1), (
            f"Failed for AA seq {aa_seq}, nt seq does not have the expected length"
        )
        assert "STOP" not in nt_seq, f"Failed for AA seq {aa_seq}, nt seq shouldn't STOP character"
        aa_seq_2 = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == aa_seq_2, f"Failed for AA seq {aa_seq}, nt seq is not str"

        # unknown character
        aa_seq = "EVQLVESGXGGLVQPGGSLRLS"
        nt_seq = convert_aa_to_nt(aa_seq, residue_to_codon_map, uniform_sample)
        assert isinstance(nt_seq, str), f"Failed for AA seq {aa_seq}, nt seq should be a str"
        assert "<unk>" in nt_seq, f"Failed for AA seq {aa_seq}, nt seq should have <unk> token"
        assert len(nt_seq) == 71

    def test_convert_nt_to_aa(self, codon_to_residue_map: dict[str, str]):
        # seq len divisible by 3 (no stop codon)
        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert isinstance(aa_seq, str), f"Failed for nt seq {nt_seq}, AA seq should be a str"
        assert len(nt_seq) == 3 * (len(aa_seq)), f"Failed for nt seq {nt_seq}, AA seq does not have the expected length"
        assert aa_seq == "EVQLVESGGGLV", f"Failed for nt seq {nt_seq}, wrong aa_seq"

        # seq len not divisible by 3
        nt_seq = "GAGGTGCAACTAGTCGAGT"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert "<unk>" in aa_seq

        # unknown character (e.g U) in seq of 3n bases
        nt_seq = "GAGGTGCAACTAGUCGAGTCCGGAGGGGGGCTTGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert isinstance(aa_seq, str), f"Failed for nt seq {nt_seq}, AA seq should be a str"
        assert "<unk>" in aa_seq, f"Failed for nt seq {nt_seq}, aa_seq shuodl have <unk> token"
        assert aa_seq == "EVQL<unk>ESGGGLV", f"Failed for nt seq {nt_seq}, wrong AA seq"

        # stop codon at the end
        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTGTATGA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert isinstance(aa_seq, str), f"Failed for nt seq {nt_seq}, AA seq should be a str"
        assert len(nt_seq) == 3 * (len(aa_seq) + 1), (
            f"Failed for nt_seq {nt_seq}, AA seq does not have the expected length"
        )
        assert aa_seq == "EVQLVESGGGLV", f"Failed for nt seq {nt_seq}, wrong AA seq"

        # early stop codons (TAA, TAG, TGA)
        nt_seq = "GAGGTGTAACAACTAGTCGAGTCCGGAGGGGGGCTTGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == "EV"

        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTTAGGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == "EVQLVESGGGL"

        nt_seq = "TGAGAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTGATTGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == "", f"Failed for nt seq {nt_seq}, AA seq should be None due to early STOP codon"

    @pytest.mark.parametrize(
        "aa_seq, allowed_aa, replace_unknown, randomize_smiles, expected_smiles_pattern, raises_error",
        [
            (
                "EVQLV",
                {"E", "V", "Q", "L"},
                True,
                False,
                "CC(C)C[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)CCC(=O)O)C(C)C)C(=O)N[C@H](C(=O)O)C(C)C",
                None,
            ),
            (
                "EVQLV",
                {"E", "V", "Q", "L"},
                True,
                True,
                "CC(C)C[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)CCC(=O)O)C(C)C)C(=O)N[C@H](C(=O)O)C(C)C",
                None,
            ),
            ("AXP", {"A", "P"}, True, False, "C[C@H](N)C(=O)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O", None),
            ("AXP", {"A", "P"}, False, False, None, None),
            ("A", {"A"}, True, False, "C[C@H](N)C(=O)O", None),
            ("", {"A"}, True, False, "", None),
            ("EVQLV", None, True, False, None, ValueError),
            (
                "evqlv",
                {"E", "V", "Q", "L"},
                True,
                False,
                "CC(C)C[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)CCC(=O)O)C(C)C)C(=O)N[C@H](C(=O)O)C(C)C",
                None,
            ),
            ("Z", {"A"}, True, False, "C[C@H](N)C(=O)O", None),
            ("Z", {"A"}, False, False, None, None),
            ("ACEG", None, False, False, "C[C@H](N)C(=O)N[C@@H](CS)C(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O", None),
            ("ACEG", None, False, True, "C[C@H](N)C(=O)N[C@@H](CS)C(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O", None),
            ("AXG", None, False, False, None, None),
            ("<unk>CEG", None, False, False, None, None),
            ("", None, False, False, "", None),
        ],
    )
    def test_convert_aa_to_smiles_parameterized(
        self,
        aa_seq: str,
        allowed_aa: set[str] | None,
        replace_unknown: bool,
        randomize_smiles: bool,
        expected_smiles_pattern: str | None,
        raises_error: type[BaseException] | None,
    ):
        if raises_error:
            with pytest.raises(raises_error):
                convert_aa_to_smiles(
                    aa_seq, allowed_aa=allowed_aa, replace_unknown=replace_unknown, randomize_smiles=randomize_smiles
                )
        else:
            result_smiles = convert_aa_to_smiles(
                aa_seq, allowed_aa=allowed_aa, replace_unknown=replace_unknown, randomize_smiles=randomize_smiles
            )
            if randomize_smiles:
                assert result_smiles is not None, f"Randomized SMILES should not be None for valid input {aa_seq}"
                mol_from_random = Chem.MolFromSmiles(result_smiles)
                assert mol_from_random is not None, (
                    f"Randomized SMILES '{result_smiles}' is not valid for input {aa_seq}"
                )
                assert expected_smiles_pattern is not None, (
                    "Canonical SMILES pattern must be provided for randomized tests"
                )
                assert get_canonical_smiles(result_smiles) == expected_smiles_pattern
            elif expected_smiles_pattern is None:
                assert result_smiles is None
            else:
                assert result_smiles == expected_smiles_pattern
                if result_smiles:
                    assert result_smiles == get_canonical_smiles(result_smiles), (
                        f"SMILES {result_smiles} should be canonical"
                    )

    @pytest.mark.parametrize(
        "nt_seq, cap, randomize_smiles, expected_output, raises_error",
        [
            (
                "ATGC",
                None,
                False,
                "Cc1cn([C@H]2C[C@H](OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(=O)[nH]c(N)nc54)C[C@@H]3OP(=O)(O)OC[C@H]3O[C@@H](n4ccc(N)nc4=O)C[C@@H]3O)[C@@H](COP(=O)(O)O[C@H]3C[C@H](n4cnc5c(N)ncnc54)O[C@@H]3CO)O2)c(=O)[nH]c1=O",
                None,
            ),
            (
                "AUGC",
                None,
                False,
                "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4cnc5c(=O)[nH]c(N)nc54)O[C@@H]3COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4ccc(=O)[nH]c4=O)O[C@@H]3COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4cnc5c(N)ncnc54)O[C@@H]3CO)[C@@H](O)[C@H]2O)c(=O)n1",
                None,
            ),
            (
                "atgc",
                None,
                False,
                "Cc1cn([C@H]2C[C@H](OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(=O)[nH]c(N)nc54)C[C@@H]3OP(=O)(O)OC[C@H]3O[C@@H](n4ccc(N)nc4=O)C[C@@H]3O)[C@@H](COP(=O)(O)O[C@H]3C[C@H](n4cnc5c(N)ncnc54)O[C@@H]3CO)O2)c(=O)[nH]c1=O",
                None,
            ),
            (
                "augc",
                None,
                False,
                "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4cnc5c(=O)[nH]c(N)nc54)O[C@@H]3COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4ccc(=O)[nH]c4=O)O[C@@H]3COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4cnc5c(N)ncnc54)O[C@@H]3CO)[C@@H](O)[C@H]2O)c(=O)n1",
                None,
            ),
            (
                "ATGC",
                "5'",
                False,
                "Cc1cn([C@H]2C[C@H](OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(=O)[nH]c(N)nc54)C[C@@H]3OP(=O)(O)OC[C@H]3O[C@@H](n4ccc(N)nc4=O)C[C@@H]3O)[C@@H](COP(=O)(O)O[C@H]3C[C@H](n4cnc5c(N)ncnc54)O[C@@H]3COP(=O)(O)O)O2)c(=O)[nH]c1=O",
                None,
            ),
            (
                "AUGC",
                "3'",
                False,
                "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4cnc5c(=O)[nH]c(N)nc54)O[C@@H]3COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4ccc(=O)[nH]c4=O)O[C@@H]3COP(=O)(O)O[C@H]3[C@@H](O)[C@H](n4cnc5c(N)ncnc54)O[C@@H]3CO)[C@@H](OP(=O)(O)O)[C@H]2O)c(=O)n1",
                None,
            ),
            (
                "ATGC",
                "both",
                False,
                "Cc1cn([C@H]2C[C@H](OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(=O)[nH]c(N)nc54)C[C@@H]3OP(=O)(O)OC[C@H]3O[C@@H](n4ccc(N)nc4=O)C[C@@H]3OP(=O)(O)O)[C@@H](COP(=O)(O)O[C@H]3C[C@H](n4cnc5c(N)ncnc54)O[C@@H]3COP(=O)(O)O)O2)c(=O)[nH]c1=O",
                None,
            ),
            (
                "ATGC",
                None,
                True,
                "Cc1cn([C@H]2C[C@H](OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(=O)[nH]c(N)nc54)C[C@@H]3OP(=O)(O)OC[C@H]3O[C@@H](n4ccc(N)nc4=O)C[C@@H]3O)[C@@H](COP(=O)(O)O[C@H]3C[C@H](n4cnc5c(N)ncnc54)O[C@@H]3CO)O2)c(=O)[nH]c1=O",
                None,
            ),
            ("", None, False, "", None),
            ("ATGX", None, False, None, None),  # Invalid char in sequence
            ("AUGC", "invalid_cap", False, None, ValueError),
        ],
    )
    def test_convert_nt_to_smiles_parameterized(
        self,
        nt_seq: str,
        cap: str | None,
        randomize_smiles: bool,
        expected_output: str | None,
        raises_error: type[BaseException] | None,
    ):
        if raises_error:
            with pytest.raises(raises_error):
                convert_nt_to_smiles(nt_seq, cap=cap, randomize_smiles=randomize_smiles)  # type: ignore[arg-type]
        else:
            result_smiles = convert_nt_to_smiles(nt_seq, cap=cap, randomize_smiles=randomize_smiles)  # type: ignore[arg-type]

            if randomize_smiles:
                assert result_smiles is not None, (
                    f"Randomized SMILES should not be None for valid input {nt_seq} with cap {cap}"
                )
                assert Chem.MolFromSmiles(result_smiles) is not None, (
                    f"Randomized SMILES '{result_smiles}' is not valid"
                )
                # expected_output is the canonical SMILES of the non-randomized version (with the same cap)
                assert expected_output is not None, "Canonical SMILES target must be provided for randomized tests"
                assert get_canonical_smiles(result_smiles) == expected_output
            elif expected_output is None:
                assert result_smiles is None
            else:  # Non-randomized, direct match
                assert result_smiles == expected_output
                if result_smiles:  # If not None, it should be canonical already
                    assert result_smiles == get_canonical_smiles(result_smiles), (
                        f"SMILES {result_smiles} should be canonical"
                    )

    @pytest.mark.parametrize(
        "input_smiles, randomize_smiles, expected_smiles_pattern",
        [
            ("CCO", False, "CCO"),
            ("OCC", False, "CCO"),
            ("c1ccccc1", False, "c1ccccc1"),
            ("C1=CC=CC=C1", False, "c1ccccc1"),
            ("CCO", True, "CCO"),
            ("invalid_smiles_string", False, None),
            ("CC(C)C[C@H](N)C(=O)O", False, "CC(C)C[C@H](N)C(=O)O"),
            ("CC(C)C[C@H](N)C(=O)O", True, "CC(C)C[C@H](N)C(=O)O"),
            ("", False, ""),
        ],
    )
    def test_convert_smiles_to_smiles_parameterized(
        self, input_smiles: str, randomize_smiles: bool, expected_smiles_pattern: str | None
    ):
        result_smiles = convert_smiles_to_smiles(input_smiles, randomize_smiles=randomize_smiles)

        if expected_smiles_pattern is None:
            assert result_smiles is None
        elif randomize_smiles:
            assert result_smiles is not None, f"Randomized SMILES should not be None for valid input {input_smiles}"
            mol_from_random = Chem.MolFromSmiles(result_smiles)
            assert mol_from_random is not None, f"Randomized SMILES '{result_smiles}' is not valid"
            assert get_canonical_smiles(result_smiles) == expected_smiles_pattern
        else:
            assert result_smiles == expected_smiles_pattern
            if result_smiles:
                assert result_smiles == get_canonical_smiles(result_smiles), "SMILES should be canonical"

    def test_replace_target_symbol(self):
        target_symbol = "<unk>"
        replacement_symbol = "A"

        seq1 = "EVQLV"
        assert seq1 == replace_target_symbol(seq1, target_symbol, replacement_symbol)

        seq1 = "EVALV"
        seq2 = "EV<unk>LV"
        assert seq1 == replace_target_symbol(seq2, target_symbol, replacement_symbol)

        seq1 = "EVLLA"
        seq2 = "EVLL<unk>"
        assert seq1 == replace_target_symbol(seq2, target_symbol, replacement_symbol)

        seq1 = "AVLLV"
        seq2 = "<unk>VLLV"
        assert seq1 == replace_target_symbol(seq2, target_symbol, replacement_symbol)

        seq1 = "AVLAV"
        seq2 = "<unk>VL<unk>V"
        assert seq1 == replace_target_symbol(seq2, target_symbol, replacement_symbol)

    def test_replace_unknown_symbols(self):
        allowed_set = {
            "T",
            "G",
            "C",
            "Y",
            "E",
            "I",
            "R",
            "W",
            "H",
            "L",
            "Q",
            "S",
            "F",
            "N",
            "V",
            "P",
            "D",
            "A",
            "K",
            "M",
            "STOP",
        }
        replacement_symbol = "A"

        seq1 = "EVQLV"
        assert seq1 == replace_unknown_symbols(seq1, allowed_set, replacement_symbol)

        seq1 = "EVQxV"
        seq2 = "EVQAV"
        assert seq2 == replace_unknown_symbols(seq1, allowed_set, replacement_symbol)

    def test_convert_aa_to_selfies(self):
        allowed_aa = {
            "T",
            "G",
            "C",
            "Y",
            "E",
            "I",
            "R",
            "W",
            "H",
            "L",
            "Q",
            "S",
            "F",
            "N",
            "V",
            "P",
            "D",
            "A",
            "K",
            "M",
            "STOP",
        }
        seq1 = "EVQLV"
        seq2 = convert_aa_to_selfies(seq1, allowed_aa)
        assert seq2 is not None, "convert_aa_to_selfies should return a string for valid inputs"
        symbols = split_by_two_characters(seq2, "[", "]")

        assert isinstance(seq2, str)
        assert len(symbols) == 72
        exp_seq2 = (
            "[C][C][Branch1][C][C][C][C@H1][Branch2][Ring2][=N][N][C][=Branch1][C][=O][C@H1]"
            "[Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1][C][=O][C@@H1][Branch1][P][N]"
            "[C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O][C][Branch1][C][C][C]"
            "[C][=Branch1][C][=O][N][C@H1][Branch1][=Branch1][C][=Branch1][C][=O][O][C][Branch1][C][C][C]"
        )
        assert seq2 == exp_seq2

        seq3 = "EvqLV"
        assert convert_aa_to_selfies(seq3, allowed_aa) == seq2

    def test_convert_nt_to_selfies(self, codon_to_residue_map: dict[str, str]):
        allowed_aa = {
            "T",
            "G",
            "C",
            "Y",
            "E",
            "I",
            "R",
            "W",
            "H",
            "L",
            "Q",
            "S",
            "F",
            "N",
            "V",
            "P",
            "D",
            "A",
            "K",
            "M",
            "STOP",
        }
        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTGTA"
        sf_seq = convert_nt_to_selfies_via_aa(nt_seq, codon_to_residue_map, allowed_aa)
        assert sf_seq is not None, "convert_nt_to_selfies_via_aa should return a string for valid inputs"
        exp_seq = (
            "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N]"
            "[C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1]"
            "[C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1]"
            "[C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O]"
            "[C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@H1][Branch1][=Branch1][C]"
            "[=Branch1][C][=O][O][C][Branch1][C][C][C]"
        )

        symbols = split_by_two_characters(sf_seq, "[", "]")
        assert isinstance(sf_seq, str)
        assert len(symbols) == 143
        assert sf_seq == exp_seq

        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTGT"
        sf_seq = convert_nt_to_selfies_via_aa(nt_seq, codon_to_residue_map, allowed_aa)
        assert sf_seq is not None
        assert (
            sf_seq
            == "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1][C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1][C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1][C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1][C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O][C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1][#Branch1][C][C][Branch1][C][N][=O][C][=Branch1][C][=O][N][C@@H1][Branch1][=Branch1][C][C][C][C][N][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][O]"
        )

        nt_seq = "GAGGUGCAA"
        sf_seq = convert_nt_to_selfies_via_aa(nt_seq, codon_to_residue_map, allowed_aa)
        assert sf_seq is not None
        exp_seq2 = (
            "[C][C@H1][Branch2][Ring1][#Branch2][N][C][=Branch1][C][=O][C@H1][Branch1][C][C][N][C]"
            "[=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O][C][=Branch1][C][=O][N][C@@H1]"
            "[Branch1][#Branch1][C][C][Branch1][C][N][=O][C][=Branch1][C][=O][N][C@@H1][Branch1][=Branch1][C][C]"
            "[C][C][N][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1]"
            "[Branch2][C][C][C][Branch1][C][N][=O][C][=Branch1][C][=O][O]"
        )
        assert sf_seq == exp_seq2

    def _test_convert_selfies_to_aa(self):
        pass

    def _test_convert_selfies_to_nt(
        self,
        residue_to_codon_map,
    ):
        sf_seq = (
            "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N]"
            "[C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1]"
            "[C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1]"
            "[C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O]"
            "[C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@H1][Branch1][=Branch1][C]"
            "[=Branch1][C][=O][O][C][Branch1][C][C][C]"
        )
        nt_seq = convert_selfies_to_nt_via_aa(sf_seq, residue_to_codon_map, uniform_sample)

        assert isinstance(nt_seq, str)

    def test_convert_selfies_to_smiles(self):
        sf_seq = (
            "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N]"
            "[C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1]"
            "[C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1]"
            "[C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O]"
            "[C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@H1][Branch1][=Branch1][C]"
            "[=Branch1][C][=O][O][C][Branch1][C][C][C]"
        )
        smiles_seq = convert_selfies_to_smiles(sf_seq)

        assert isinstance(smiles_seq, str)
        assert smiles_seq == (
            "CC(C)C[C@H1](NC(=O)CNC(=O)CNC(=O)CNC(=O)[C@H1](CO)NC(=O)[C@H1](CCC(=O)O)NC(=O)"
            "[C@@H1](NC(=O)[C@H1](CC(C)C)NC(=O)[C@H1](CCC(N)=O)NC(=O)[C@@H1](NC(=O)[C@@H1](N)CCC(=O)O)C(C)C)C"
            "(C)C)C(=O)N[C@H1](C(=O)O)C(C)C"
        )

    def test_convert_smiles_to_selfies(self):
        smiles_seq = (
            "CC(C)C[C@H1](NC(=O)CNC(=O)CNC(=O)CNC(=O)[C@H1](CO)NC(=O)[C@H1](CCC(=O)O)NC(=O)"
            "[C@@H1](NC(=O)[C@H1](CC(C)C)NC(=O)[C@H1](CCC(N)=O)NC(=O)[C@@H1](NC(=O)[C@@H1](N)CCC(=O)O)C(C)C)C"
            "(C)C)C(=O)N[C@H1](C(=O)O)C(C)C"
        )

        exp_sf_seq = (
            "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N]"
            "[C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1]"
            "[C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1]"
            "[C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1]"
            "[C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O]"
            "[C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@H1][Branch1][=Branch1][C]"
            "[=Branch1][C][=O][O][C][Branch1][C][C][C]"
        )

        sf_seq = convert_smiles_to_selfies(smiles_seq)

        assert isinstance(sf_seq, str)
        assert sf_seq == exp_sf_seq

    @pytest.mark.parametrize(
        "aa_seq, vendor_codon_table, add_stop_codon, expected_length_multiple",
        [
            ("M", {"M": {"ATG": 1.0}, "*": {"TAA": 0.5, "TAG": 0.3, "TGA": 0.2}}, True, 2),  # M + stop = 6 bases
            (
                "MA",
                {"M": {"ATG": 1.0}, "A": {"GCT": 0.3, "GCC": 0.4, "GCA": 0.2, "GCG": 0.1}, "*": {"TAA": 1.0}},
                True,
                3,
            ),  # M + A + stop = 9 bases
            ("M", {"M": {"ATG": 1.0}}, False, 1),  # M only = 3 bases
            ("", {}, True, 0),  # Empty sequence
            ("", {}, False, 0),  # Empty sequence
        ],
    )
    def test_convert_aa_to_nt_probabilistic_parameterized(
        self,
        aa_seq: str,
        vendor_codon_table: dict[str, dict[str, float]],
        add_stop_codon: bool,
        expected_length_multiple: int,
    ):
        """Test probabilistic amino acid to nucleotide conversion."""
        result = convert_aa_to_nt_probabilistic(aa_seq, vendor_codon_table, add_stop_codon)

        assert isinstance(result, str)
        assert len(result) == expected_length_multiple * 3

        # Check if result contains only valid nucleotides
        valid_chars = set("ATGC")
        assert all(c in valid_chars for c in result)

    def test_convert_aa_to_nt_probabilistic_real_table(self):
        """Test probabilistic conversion with real vendor codon table data."""
        aa_seq = "MKL"  # Start with known amino acids

        # Use a subset of the real vendor table for testing
        vendor_table = {
            "M": {"ATG": 1.0},
            "K": {"AAA": 0.43, "AAG": 0.57},
            "L": {"TTA": 0.08, "TTG": 0.14, "CTT": 0.15, "CTC": 0.18, "CTA": 0.09, "CTG": 0.36},
            "*": {"TAA": 0.27, "TAG": 0.23, "TGA": 0.50},
        }

        result = convert_aa_to_nt_probabilistic(aa_seq, vendor_table, add_stop_codon=True)

        assert isinstance(result, str)
        assert len(result) == 12  # 3 AA + 1 stop = 4 codons = 12 bases

        # Check structure: should be 4 codons
        assert result[0:3] == "ATG"  # M always ATG
        assert result[3:6] in ["AAA", "AAG"]  # K codons
        assert result[6:9] in ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]  # L codons
        assert result[9:12] in ["TAA", "TAG", "TGA"]  # Stop codons

    def test_convert_aa_to_nt_probabilistic_unknown_residue(self):
        """Test probabilistic conversion with unknown amino acids raises ValueError."""
        aa_seq = "MXK"  # X is unknown
        vendor_table = {"M": {"ATG": 1.0}, "K": {"AAA": 0.5, "AAG": 0.5}, "*": {"TAA": 1.0}}

        with pytest.raises(ValueError, match="Unknown amino acid residue 'X' not found in vendor codon table"):
            convert_aa_to_nt_probabilistic(aa_seq, vendor_table, add_stop_codon=True)

    def test_convert_aa_to_nt_probabilistic_case_insensitive(self):
        """Test that function handles lowercase input."""
        aa_seq = "mkl"  # lowercase
        vendor_table = {"M": {"ATG": 1.0}, "K": {"AAA": 1.0}, "L": {"TTG": 1.0}, "*": {"TAA": 1.0}}

        result = convert_aa_to_nt_probabilistic(aa_seq, vendor_table, add_stop_codon=True)

        assert isinstance(result, str)
        assert result == "ATGAAATTGTAA"

    def test_convert_aa_to_nt_probabilistic_no_stop_codon(self):
        """Test conversion without adding stop codon."""
        aa_seq = "MK"
        vendor_table = {"M": {"ATG": 1.0}, "K": {"AAA": 1.0}}

        result = convert_aa_to_nt_probabilistic(aa_seq, vendor_table, add_stop_codon=False)

        assert isinstance(result, str)
        assert result == "ATGAAA"  # No stop codon
