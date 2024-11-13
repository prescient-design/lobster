import importlib
from typing import Any, Dict, Optional, Tuple

import Bio.PDB.Polypeptide as Poly
import torch
from pandas import DataFrame

from lobster.tokenization._pmlm_tokenizer import PmlmTokenizer

from ._transform import Transform


def _protein_letters_3to1_wrapper(three_letter: str) -> str:
    """
    Simple wrapper to use in pandas.Series apply fn
    """
    assert len(three_letter) == 3
    return Poly.protein_letters_3to1[three_letter]


def _get_contact_map(seqA: str, seqB: str, ixs: list) -> torch.Tensor:
    """
    Creates a tensor of size len(seqA) x len(seqB) using the contact residues
    from ixs
    """
    num_rows, num_cols = len(seqA), len(seqB)
    matrix = torch.zeros((num_rows, num_cols))  # Initialize matrix with zeros

    for r, c in ixs:
        if 0 <= r < num_rows and 0 <= c < num_cols:  # NOTE - There are Atom3D indices out-of-bounds, unclear why
            matrix[r - 1, c - 1] = 1  # Set the corresponding element to 1

    return matrix


class Atom3DPPIToSequenceAndContactMap(Transform):
    """
    Transforms an Atom3DPPIDataset output (feature, target) to its constituent protein sequences
    and a 2D contact map (sequenceA, sequenceB, contact_map). Use as a joint_transform arg
    when instantiating a Dataset
    """

    def __init__(
        self,
    ):
        super().__init__()

    def transform(
        self,
        feat: DataFrame,
        target: DataFrame,
    ) -> Tuple[Tuple[Optional[str], Optional[str]], Optional[torch.Tensor]]:
        """
        Return list of tuples of (id, sequence) for different chains of monomers in a given dataframe
        Adapted from atom3d.protein.sequence
        """
        # Keep only alpha carbons (CA) of standard residues
        feat = feat[feat["name"] == "CA"].drop_duplicates()
        if feat.empty:  # If no more alpha carbons
            return (None, None), None

        feat = feat[feat["resname"].apply(lambda x: Poly.is_aa(x, standard=True))]
        feat["resname"] = feat["resname"].apply(_protein_letters_3to1_wrapper)
        chain_sequences = []
        chain_residue_ixs = []
        for _, chain in feat.groupby(["ensemble", "subunit", "structure", "model", "chain"]):
            # Get sequence and chain info
            seq = "".join(chain["resname"])
            chain_sequences.append(seq)
            # Get indices of aa residues (instead of HOH groups)
            contact_ixs = set(chain["residue"])
            chain_residue_ixs.append(contact_ixs)

        # Get contact map
        if not len(chain_sequences) == 2:
            return (None, None), None  # If not a protein-protein complex
        a_ixs, b_ixs = tuple(chain_residue_ixs)
        is_residue_mask = target["residue0"].isin(a_ixs) & target["residue1"].isin(b_ixs)
        target = target[is_residue_mask]
        ixs = list(zip(target["residue0"], target["residue1"]))
        contact_map = _get_contact_map(*chain_sequences, ixs)

        return tuple(chain_sequences), contact_map

    def validate(self, flat_inputs: list[Any]) -> None:
        pass


class PairedSequenceToTokens(Transform):
    """
    Transforms a (sequence, sequence) pair to tokens
    """

    def __init__(self, vocab_file: str = None, tokenizer_dir="pmlm_tokenizer"):
        super().__init__()

        # TODO - make this a hydra param to allow different vocab files?
        if vocab_file is None:
            vocab_file = (importlib.resources.files("lobster") / "assets" / "pmlm_vocab.txt").as_posix()

        self._tokenizer_dir = tokenizer_dir
        path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
        self._tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)

    def transform(
        self,
        inpt: Optional[Tuple[str, str]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return a tuple of tokens representing sequenceA and sequenceB
        """
        if inpt is None or not isinstance(inpt, tuple):
            return inpt

        sequence1, sequence2 = inpt
        if sequence1 is None or sequence2 is None:
            return (None, None)

        tokens1 = self._tokenizer(text=sequence1)
        tokens2 = self._tokenizer(text=sequence2)
        if len(tokens1) == 1:
            tokens1 = tokens1[0].flatten()
        if len(tokens2) == 1:
            tokens2 = tokens2[0].flatten()

        # Return input ids only, no attns
        if not isinstance(tokens1, torch.Tensor) or not isinstance(tokens1, list):
            tokens1 = tokens1["input_ids"]
            tokens2 = tokens2["input_ids"]

        tokens1, tokens2 = torch.tensor(tokens1, dtype=torch.int), torch.tensor(tokens2, dtype=torch.int)

        return tokens1, tokens2


class Atom3DPPIToSequence(Transform):
    """
    Transforms an Atom3DPPIDataset feature to its constituent protein sequences
    """

    def __init__(
        self,
    ):
        super().__init__()

    def transform(
        self,
        inpt: DataFrame,
    ) -> Dict:
        """
        Return a tuple of sequences for different chains of monomers in a given dataframe
        Adapted from atom3d.protein.sequence
        """
        feat = inpt
        # Keep only alpha carbons (CA) of standard residues
        feat = feat[feat["name"] == "CA"].drop_duplicates()
        if feat.empty:  # If no more alpha carbons, no protein sequence available
            return {"sequences": None, "features": None}
        feat = feat[feat["resname"].apply(lambda x: Poly.is_aa(x, standard=True))]
        feat["resname"] = feat["resname"].apply(_protein_letters_3to1_wrapper)
        chain_sequences = []
        for _, chain in feat.groupby(["ensemble", "subunit", "structure", "model", "chain"]):
            seq = "".join(chain["resname"])
            chain_sequences.append(seq)

        return {"sequences": chain_sequences, "features": feat}  # needed for contact map transform
