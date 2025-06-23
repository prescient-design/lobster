import random
from collections.abc import Callable
from importlib.util import find_spec
from typing import Literal

_SELFIES_AVAILABLE = False
_RDKIT_AVAILABLE = False

if find_spec("selfies"):
    import selfies as sf

    _SELFIES_AVAILABLE = True

if find_spec("rdkit"):
    from rdkit import Chem

    _RDKIT_AVAILABLE = True


def convert_nt_to_aa(
    nt_seq: str,
    codon_to_residue: dict[str, str],
) -> str:
    if not nt_seq.isupper():
        nt_seq = nt_seq.upper()
    aa_seq = ""

    for i in range(0, len(nt_seq), 3):
        try:
            codon = nt_seq[i : i + 3]
            residue = codon_to_residue[codon]
        except KeyError:
            # unknown triplet of characters
            residue = "<unk>"

        if residue == "STOP":
            return aa_seq

        aa_seq += residue
    return aa_seq


def convert_aa_to_nt(
    aa_seq: str,
    residue_to_codon: dict[str, list[str]],
    sample_fn: Callable,
) -> str:
    if not aa_seq.isupper():
        aa_seq = aa_seq.upper()

    nt_seq = ""
    for residue in aa_seq:
        try:
            codons = residue_to_codon[residue]
            codon = sample_fn(codons)
        except KeyError:
            codon = "<unk>"
        nt_seq += codon

    stop_codons = residue_to_codon["STOP"]
    stop_codon = sample_fn(stop_codons)
    nt_seq += stop_codon
    return nt_seq


def convert_aa_to_nt_probabilistic(
    aa_seq: str,
    vendor_codon_table: dict[str, dict[str, float]],
    add_stop_codon: bool = True,
    skip_unknown: bool = False,
) -> str:
    """
    Convert amino acid sequence to nucleotide sequence using probabilistic codon sampling.

    Parameters
    ----------
    aa_seq : str
        The amino acid sequence to convert
    vendor_codon_table : dict[str, dict[str, float]]
        A dictionary mapping amino acids to dictionaries mapping codons to their usage frequencies.
    add_stop_codon : bool
        Whether to add a stop codon at the end
    skip_unknown : bool
        If True, skip unknown amino acids instead of raising an error.
        If False (default), raise ValueError for unknown amino acids.

    Returns
    -------
    str
        The nucleotide sequence
    """
    if not aa_seq.isupper():
        aa_seq = aa_seq.upper()

    nt_seq = ""
    for residue in aa_seq:
        if residue not in vendor_codon_table:
            if skip_unknown:
                continue
            raise ValueError(f"Unknown amino acid residue '{residue}' not found in vendor codon table")

        try:
            available_codons = list(vendor_codon_table[residue].keys())
            probabilities = list(vendor_codon_table[residue].values())
            codon = random.choices(available_codons, weights=probabilities, k=1)[0]

        except (KeyError, IndexError) as e:
            if skip_unknown:
                continue
            raise Exception(f"Error processing residue '{residue}': {e}") from e

        nt_seq += codon

    if add_stop_codon and aa_seq:  # Only add stop codon if sequence is not empty
        nt_seq += "TAA"

    return nt_seq


def convert_aa_to_smiles(
    aa_seq: str, allowed_aa: set | None = None, replace_unknown: bool = True, randomize_smiles: bool = False
) -> str | None:
    """Convert an amino acid sequence to a SMILES string.

    Parameters
    ----------
    aa_seq : str
        The amino acid sequence to convert.
    allowed_aa : set | None
        The set of allowed amino acids.
    replace_unknown : bool
        Whether to replace unknown amino acids with Ala.
    randomize_smiles : bool
        Whether to randomize the SMILES string.

    Returns
    -------
    str | None
        The SMILES string, or None if conversion fails.
    """
    assert _RDKIT_AVAILABLE, "RDKit not available. This dependency is part of the mgm extra"

    if not aa_seq.isupper():
        aa_seq = aa_seq.upper()

    # substitute unknown tokens with Ala
    if replace_unknown:
        if allowed_aa is None:
            raise ValueError("allowed_aa must be provided if replace_unknown is True")
        aa_seq = replace_target_symbol(aa_seq, target_symbol="<unk>", replacement_symbol="A")
        aa_seq = replace_unknown_symbols(aa_seq, allowed_set=allowed_aa, replacement="A")

    try:
        mol = Chem.MolFromSequence(aa_seq)
    except SystemError:  # likely TypeError in RDKit
        return None

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, doRandom=randomize_smiles)


def convert_smiles_to_aa(
    smiles_seq: str,
) -> str | None:
    assert _RDKIT_AVAILABLE, "rdkit not available. This dependency is part of the mgm extra"

    try:
        mol = Chem.MolFromSmiles(smiles_seq)
        aa_seq = Chem.MolToSequence(mol)
        return aa_seq
    except TypeError:
        # TODO: check whether tokenized seq can be None
        return None


def convert_smiles_to_selfies(
    smiles_seq: str,
) -> str | None:
    assert _SELFIES_AVAILABLE, "selfies not available. This dependency is part of the mgm extra"
    try:
        sf_seq = sf.encoder(smiles_seq)
        return sf_seq
    except sf.EncoderError:
        return None


def convert_selfies_to_smiles(
    selfies_seq: str,
) -> str | None:
    # TODO: add conversion of unknown selfies tokens to Ala selfies
    assert _SELFIES_AVAILABLE, "selfies not available. This dependency is part of the mgm extra"
    try:
        smiles_seq = sf.decoder(selfies_seq)
        return smiles_seq  # type: ignore[no-any-return]
    except sf.DecoderError:
        return None


def convert_aa_to_selfies(aa_seq: str, allowed_aa: set) -> str | None:
    if not aa_seq.isupper():
        aa_seq = aa_seq.upper()

    smiles_seq = convert_aa_to_smiles(aa_seq, allowed_aa)
    if smiles_seq is None:
        return None
    sf_seq = convert_smiles_to_selfies(smiles_seq)
    return sf_seq


def convert_selfies_to_aa(
    sf_seq: str,
) -> str | None:
    # TODO: problem: SELFIES TO SMILES conversion is not reversible!
    smiles_seq = convert_selfies_to_smiles(sf_seq)
    if smiles_seq is None:
        return None
    aa_seq = convert_smiles_to_aa(smiles_seq)
    return aa_seq


def convert_nt_to_selfies_via_aa(nt_seq: str, codon_to_residue: dict[str, str], allowed_aa: set) -> str | None:
    aa_seq = convert_nt_to_aa(nt_seq, codon_to_residue)
    sf_seq = convert_aa_to_selfies(aa_seq, allowed_aa)
    return sf_seq


def convert_selfies_to_nt_via_aa(
    sf_seq: str,
    residue_to_codon: dict[str, list[str]],
    sample_fn: Callable,
) -> str | None:
    aa_seq = convert_selfies_to_aa(sf_seq)
    if aa_seq is None:
        return None
    nt_seq = convert_aa_to_nt(aa_seq, residue_to_codon, sample_fn)
    return nt_seq


def convert_nt_to_smiles(
    nt_seq: str, cap: Literal["5'", "3'", "both"] | None = None, randomize_smiles: bool = False
) -> str | None:
    """Convert a nucleic acid sequence to a SMILES string.

    DNA/RNA is recognized automatically by the presence of "U" in the sequence.
    Defaults to DNA if "U" is not present (even if there is no "T").

    Parameters
    ----------
    nt_seq : str
        The nucleic acid sequence to convert.
    cap : Literal["5'", "3'", "both"] | None
        The type of phosphate cap to use.
            - "5'": 5' cap
            - "3'": 3' cap
            - "both": both caps
    randomize_smiles : bool
        Whether to randomize the SMILES string (non-canonical).

    Returns
    -------
    str | None
        The SMILES string, or None if conversion fails.
    """
    assert _RDKIT_AVAILABLE, "RDKit not available. This dependency is part of the mgm extra"

    is_rna = "U" in nt_seq or "u" in nt_seq
    match is_rna, cap:
        case True, None:
            flavor = 2
        case True, "5'":
            flavor = 3
        case True, "3'":
            flavor = 4
        case True, "both":
            flavor = 5
        case False, None:
            flavor = 6
        case False, "5'":
            flavor = 7
        case False, "3'":
            flavor = 8
        case False, "both":
            flavor = 9
        case _:
            raise ValueError(f"Invalid cap: {cap} for {'RNA' if is_rna else 'DNA'} sequence {nt_seq}")

    try:
        mol = Chem.MolFromSequence(nt_seq, flavor=flavor)
    except SystemError:  # likely TypeError in RDKit
        return None

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, doRandom=randomize_smiles)


def convert_smiles_to_smiles(smiles_seq: str, randomize_smiles: bool = False) -> str | None:
    """Convert a SMILES string to a canonical or randomized SMILES string.

    Parameters
    ----------
    smiles_seq : str
        The SMILES string to convert.
    randomize_smiles : bool
        Whether to randomize the SMILES string (non-canonical). If False,
        the canonical SMILES string is returned.

    Returns
    -------
    str | None
        The canonical or randomized SMILES string, or None if conversion fails.
    """
    assert _RDKIT_AVAILABLE, "RDKit not available. This dependency is part of the mgm extra"

    try:
        mol = Chem.MolFromSmiles(smiles_seq)
    except SystemError:  # likely TypeError in RDKit
        return None

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, doRandom=randomize_smiles)


def replace_target_symbol(seq: str, target_symbol: str, replacement_symbol: str) -> str:
    if target_symbol not in seq:
        return seq

    parts = seq.split(target_symbol)
    new_seq = replacement_symbol.join(parts)
    return new_seq


def replace_unknown_symbols(seq: str, allowed_set: set, replacement: str) -> str:
    if set(seq) == allowed_set:
        return seq

    new_seq = ""
    for c in seq:
        if c in allowed_set:
            new_seq += c
        else:
            new_seq += replacement
    return new_seq
