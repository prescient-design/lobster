import importlib.resources
import re
from typing import Dict, List

import pytest
from lobster.transforms import (
    convert_aa_to_nt,
    convert_aa_to_selfies,
    convert_nt_to_aa,
    convert_nt_to_selfies,
    convert_selfies_to_nt,
    convert_selfies_to_smiles,
    convert_smiles_to_selfies,
    invert_residue_to_codon_mapping,
    json_load,
    replace_target_symbol,
    replace_unknown_symbols,
    uniform_sample,
)
from rdkit import Chem


@pytest.fixture(scope="class")
def residue_to_codon_map() -> Dict[str, List[str]]:
    path = importlib.resources.files("lobster") / "assets" / "codon_tables" / "codon_table.json"
    residue_to_codon = json_load(path)
    return residue_to_codon


@pytest.fixture(scope="class")
def codon_to_residue_map(residue_to_codon_map: Dict[str, List[str]]) -> Dict[str, str]:
    codon_to_residue = invert_residue_to_codon_mapping(residue_to_codon_map)
    return codon_to_residue


def split_by_two_characters(s: str, char1: str, char2: str) -> List[str]:
    pattern = f"[{re.escape(char1)}{re.escape(char2)}]+"
    return [item for item in re.split(pattern, s) if item]


class TestConvertSeqs:
    def test_convert_aa_to_nt(
        self,
        residue_to_codon_map: Dict[str, List[str]],
        codon_to_residue_map: Dict[str, str],
    ):
        aa_seq = "EVQLVESGGGLVQPGGSLRLS"
        nt_seq = convert_aa_to_nt(aa_seq, residue_to_codon_map, uniform_sample)
        assert isinstance(nt_seq, str), f"Failed for aa seq {aa_seq}, nt seq should be a str"
        assert len(nt_seq) == 3 * (
            len(aa_seq) + 1
        ), f"Failed for AA seq {aa_seq}, nt seq does not have the expected length"
        assert "STOP" not in nt_seq, f"Failed for AA seq {aa_seq}, nt seq shouldn't STOP character"
        aa_seq_2 = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == aa_seq_2, f"Failed for AA seq {aa_seq}, nt seq is not str"

        # unknown character
        aa_seq = "EVQLVESGXGGLVQPGGSLRLS"
        nt_seq = convert_aa_to_nt(aa_seq, residue_to_codon_map, uniform_sample)
        # TODO: change, now unk token -> do assert
        assert isinstance(nt_seq, str), f"Failed for AA seq {aa_seq}, nt seq should be a str"
        assert "<unk>" in nt_seq, f"Failed for AA seq {aa_seq}, nt seq should have <unk> token"
        assert len(nt_seq) == 71

    def test_convert_nt_to_aa(self, codon_to_residue_map: Dict[str, str]):
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
        # assert aa_seq is None, f"Failed for nt seq {nt_seq}, AA seq should be None"

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
        assert len(nt_seq) == 3 * (
            len(aa_seq) + 1
        ), f"Failed for nt_seq {nt_seq}, AA seq does not have the expected length"
        assert aa_seq == "EVQLVESGGGLV", f"Failed for nt seq {nt_seq}, wrong AA seq"

        # early stop codons (TAA, TAG, TGA)
        nt_seq = "GAGGTGTAACAACTAGTCGAGTCCGGAGGGGGGCTTGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == "EV"
        # assert aa_seq is None, f"Failed for nt seq {nt_seq}, AA seq should be None due to early STOP codon"

        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTTAGGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == "EVQLVESGGGL"
        # assert aa_seq is None, f"Failed for nt seq {nt_seq}, AA seq should be None due to early STOP codon"

        nt_seq = "TGAGAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTGATTGTA"
        aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue_map)
        assert aa_seq == "", f"Failed for nt seq {nt_seq}, AA seq should be None due to early STOP codon"

    def test_convert_aa_to_smiles(self):
        aa_seq = "EVQLV"
        mol = Chem.MolFromSequence(aa_seq)
        smi_seq = Chem.MolToSmiles(mol)
        assert isinstance(smi_seq, str)
        # print(smi_seq)
        assert len(smi_seq) == 100, f"Failed for AA seq {aa_seq}, smiles seq does not have the expected length"
        assert smi_seq == (
            "CC(C)C[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)CCC(=O)O)C(C)C)C(=O)N[C@H](C(=O)O)C(C)C"
        )

    def test_convert_smiles_to_aa(self):
        """
        smiles_seq = ("CC(C)C[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)CCC(=O)O)C(C)C)"
        "C(=O)N[C@H](C(=O)O)C(C)C")
        aa_seq = convert_smiles_to_aa(smiles_seq)
        assert isinstance(aa_seq, str)
        assert len(aa_seq) == 5
        assert aa_seq == "EVQLV"
        """
        pass

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
        # all residues in codon table
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
        # print(seq2)
        symbols = split_by_two_characters(seq2, "[", "]")
        # print(len(symbols))

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

    def test_convert_nt_to_selfies(self, codon_to_residue_map: Dict[str, str]):
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
        sf_seq = convert_nt_to_selfies(nt_seq, codon_to_residue_map, allowed_aa)
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

        # seq length not divisible by 3
        nt_seq = "GAGGTGCAACTAGTCGAGTCCGGAGGGGGGCTTGT"
        sf_seq = convert_nt_to_selfies(nt_seq, codon_to_residue_map, allowed_aa)
        print(sf_seq)
        assert (
            sf_seq
            == "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1][C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1][C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1][C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1][C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O][C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1][#Branch1][C][C][Branch1][C][N][=O][C][=Branch1][C][=O][N][C@@H1][Branch1][=Branch1][C][C][C][C][N][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][O]"
        )

        # seq length divisible by 3 but one unknown token (U) -> mapped to Ala
        nt_seq = "GAGGUGCAA"
        sf_seq = convert_nt_to_selfies(nt_seq, codon_to_residue_map, allowed_aa)
        exp_seq2 = (
            "[C][C@H1][Branch2][Ring1][#Branch2][N][C][=Branch1][C][=O][C@H1][Branch1][C][C][N][C]"
            "[=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O][C][=Branch1][C][=O][N][C@@H1]"
            "[Branch1][#Branch1][C][C][Branch1][C][N][=O][C][=Branch1][C][=O][N][C@@H1][Branch1][=Branch1][C][C]"
            "[C][C][N][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@@H1][Branch1]"
            "[Branch2][C][C][C][Branch1][C][N][=O][C][=Branch1][C][=O][O]"
        )
        assert sf_seq == exp_seq2

        # TODO: add special cases

    def _test_convert_selfies_to_aa(self):
        # sf_seq = (
        #     "[C][C][Branch1][C][C][C][C@H1][Branch2][=Branch2][Ring2][N][C][=Branch1][C][=O][C][N]"
        #     "[C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C][N][C][=Branch1][C][=O][C@H1][Branch1][Ring1]"
        #     "[C][O][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][=Branch1][C][=O][O][N][C][=Branch1]"
        #     "[C][=O][C@@H1][Branch2][Branch1][O][N][C][=Branch1][C][=O][C@H1][Branch1][#Branch1][C][C][Branch1]"
        #     "[C][C][C][N][C][=Branch1][C][=O][C@H1][Branch1][Branch2][C][C][C][Branch1][C][N][=O][N][C][=Branch1]"
        #     "[C][=O][C@@H1][Branch1][P][N][C][=Branch1][C][=O][C@@H1][Branch1][C][N][C][C][C][=Branch1][C][=O][O]"
        #     "[C][Branch1][C][C][C][C][Branch1][C][C][C][C][=Branch1][C][=O][N][C@H1][Branch1][=Branch1][C]"
        #     "[=Branch1][C][=O][O][C][Branch1][C][C][C]"
        # )
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
        # print(len(sf_seq))
        nt_seq = convert_selfies_to_nt(sf_seq, residue_to_codon_map, uniform_sample)

        assert isinstance(nt_seq, str)
        # print(nt_seq)
        # TODO: problem: conversion back from selfies not reversible
        # -> commented lines below do not work yet
        # assert len(nt_seq) == 36
        # assert set(nt_seq) <= allowed_nt

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
