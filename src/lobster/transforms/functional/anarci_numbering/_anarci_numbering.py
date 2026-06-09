import inspect
import logging
import pathlib
import shutil
import sys
from functools import cache


from ._fix_aho_sequences import fix_aho_sequences

_ANARCI_DEFAULT_KWARGS = {
    "assign_germline": False,
    "allowed_species": {"rabbit", "rat", "mouse", "human"},
}

KABAT_NUMBERS_HC = (
    [str(i) for i in range(1, 36)]
    + ["35a", "35b"]
    + [str(i) for i in range(36, 53)]
    + ["52a", "52b", "52c"]
    + [str(i) for i in range(53, 83)]
    + ["82a", "82b", "82c"]
    + [str(i) for i in range(83, 101)]
    + ["100" + c for c in "abcdefghi"]
    + [str(i) for i in range(101, 114)]
)
# Kappa won't have 106a; lambda won't have 108
# AMW: We are omitting Kabat number 108 (K) and 107 (L) for the same reason that we omit AHo number 149 for the light chain.
KABAT_NUMBERS_LC_K = (
    [str(i) for i in range(1, 28)]
    + ["27" + c for c in "abcdef"]
    + [str(i) for i in range(28, 96)]
    + ["95" + c for c in "abcdef"]
    + [str(i) for i in range(96, 108)]
)
KABAT_NUMBERS_LC_L = (
    [str(i) for i in range(1, 28)]
    + ["27" + c for c in "abcdef"]
    + [str(i) for i in range(28, 96)]
    + ["95" + c for c in "abcdef"]
    + [str(i) for i in range(96, 107)]
    + ["106a"]
)


@cache
def hmmscan_parent_path():
    if hmmscan := shutil.which("hmmscan"):
        return str(pathlib.Path(hmmscan).parent)

    hmmscan = pathlib.Path(sys.prefix) / "bin" / "hmmscan"
    if hmmscan.exists():
        return str(hmmscan.parent)

    raise RuntimeError("unable to find hmmscan executable")


def repair_kabat(numbering: list[tuple[tuple[str, str], str]], metadata: dict | None) -> str:
    if not metadata:
        logging.warning(
            "Lack of metadata will make it possible to make mistakes in HC vs LC Kabat numbering. Checking length"
        )
        metadata = {}

    use_hc_numbering = False
    use_lambda_numbering = False
    chain_type = metadata.get("chain_type")

    # neither lc does not have a residue 109.
    # lambda lc is less common, and it's not hard to detect anyway
    use_hc_numbering = chain_type == "H" or any(it[0][0] == 109 for it in numbering)
    use_lambda_numbering = chain_type == "L" or any(
        it[0][0] == 106 and it[0][1].lower().strip() == "a" for it in numbering
    )

    numbering_str = ""
    number_list = (
        KABAT_NUMBERS_HC if use_hc_numbering else KABAT_NUMBERS_LC_L if use_lambda_numbering else KABAT_NUMBERS_LC_K
    )

    def get_index_key(it):
        return (str(it[0][0]) + it[0][1].lower()).strip()

    numbering_dict = {get_index_key(it): it[1] for it in numbering}

    for num in number_list:
        if num in numbering_dict:
            numbering_str += numbering_dict[num]
        else:
            numbering_str += "-"

    return numbering_str


def anarci_numbering(
    sequences: list[str],
    *,
    scheme: str | None = None,
    allow_fix: bool = True,
    return_metadata: bool = False,
    **kwargs,
) -> list[str | None] | tuple[list[str | None], list[dict | None]]:
    """
    Return numbered sequences using ANARCI based on the selected `scheme`.
    For sequences where numbering fails, return None.

    Example usage:

        ```
        from prescient.transforms.functional import anarci_numbering

        numbered_sequences = anarci_numbering(["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"])
        ```

        ```
        from prescient.utilities import query_cpu

        numbered_sequences, metadata = anarci_numbering(
            ["QVQLQQSGAELARPGASVKMSCKASGYTFTNYGMNWVRQAPGKGLEWVSAITWNSGHIDY"],
            ncpu=query_cpu(),
            assign_germline=True,
            return_metadata=True,
            )
        ```

    Parameters
    ----------
    sequences : List[str]
        A list of sequences to be numbered.
    scheme : Optional[str], optional
        The numbering scheme to use.
        One of {"a", "aho", "c", "chothia", "i", "imgt", "k", "kabat", "martin", "wolfguy"}.
        If not provided, defaults to "aho".
    allow_fix: bool
        Allow fixing anarci output by adding or removing extra elements like dashes or
        the sequence itself (depending on the kwargs).
        Currently implemented only for AHo numbering. Default True.
    return_metadata : bool, optional
        If True, returns a tuple containing a list of numbered sequences and optional metadata.
        Metadata contains ANARCI annotations such as species, germline (if `assign_germline` is True),
        bitscores etc. If False, returns a list of numbered sequences (default)
    **kwargs
        Any further keyword arguments will be passed to `anarci.run_anarci` and
        additional arguments related to ANARCI AHo numbering fixes passed to `fix_aho_sequences`
        from `prescient.transforms.functional._anarci_numbering._fix_aho_sequences`. These include:

        desired_length: Optional[int], default None
            If provided, the computed AHo sequences will be padded with dashes, or have dashes
            removed, to match the desired sequence length. Adding or removing dashes is always
            done at the end of the sequence. Default is None
        fix_truncation: bool, default False
            If `True`, attempt to fix errors where a computed AHo sequence is missing residues
            from the end on the input sequence by appending these residues to the AHo
            sequence. Do not use with Fab regions!!
        allow_sequence_alteration: bool, default True
            If `False`, return None if the computed AHo sequence differs from
            the input sequence. If `True`, allow the computed sequence to differ from the
            input (for example, when computing AHo alignment using input sequences which
            include some or all of the constant domain).
            By default True to allow for variable region extraction from Fab regions.

    Returns
    -------
    Union[List[Optional[str]], Tuple[List[Optional[str]], List[Optional[dict]]] ]
        If `return_metadata` is False, returns a list of numbered sequences.
        If `return_metadata` is True, returns a tuple containing a list of numbered sequences
        and optional metadata dictionary for each sequence.
        A list of numbered sequences. If a sequence could not be properly numbered, its entry is None.
    """
    try:
        import anarci
    except ImportError as err:
        raise Exception(
            """
            anarci_numbering requires anarci to be installed.

            anarci can be installed from conda:

                conda install -c bioconda anarci
            """
        ) from err

    if isinstance(sequences, str):
        raise TypeError("argument sequences must be a sequence of strings not a single string")

    # Validate the scheme
    scheme = scheme if scheme is not None else "aho"
    if scheme not in {
        "a",
        "aho",
        "c",
        "chothia",
        "i",
        "imgt",
        "k",
        "kabat",
        "martin",
        "wolfguy",
    }:
        raise ValueError(
            f"Unknown `scheme` {scheme}.\n"
            "Must be one of: "
            "`a`, `aho`, `c`, `chothia`, `i`, `imgt`, `k`, `kabat`, `martin`, `wolfguy`"
        )

    # ANARCI expects tuples of (ID, sequence) as inputs and valid sequences (no None objects)
    sequences_input = [(i, seq) for i, seq in list(enumerate(sequences)) if isinstance(seq, str)]

    # ANARCI keyword arguments are missing when inspecting with `inspect` tool, list manually
    anarci_kwargs = _ANARCI_DEFAULT_KWARGS | {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "database",
            "hmmerpath",
            "output",
            "ncpu",
            "outfile",
            "csv",
            "assign_germline",
            "allowed_species",
            "bit_score_threshold",
            "allow",
        }
    }

    # only automatically append hmmerpath if anarci supports it
    # this is to support the pyhmmer version of anarci
    if "hmmerpath" in inspect.signature(anarci.anarci).parameters and "hmmerpath" not in anarci_kwargs:
        anarci_kwargs["hmmerpath"] = hmmscan_parent_path()

    # Run ANARCI numbering
    try:
        _, numbered, metadata_, _ = anarci.run_anarci(sequences_input, scheme=scheme, **anarci_kwargs)

    except Exception as e:
        logging.warning(
            f"""ANARCI batch numbering failed. Running ANARCI for each sequence separately which might take a long time.
            exception: {e}
            """
        )
        numbered = []
        metadata_ = []

        for i, seq in sequences_input:
            try:
                _, numbered_entry, metadata_entry, _ = anarci.run_anarci([(i, seq)], scheme=scheme, **anarci_kwargs)
                numbered.extend(numbered_entry)
                metadata_.extend(metadata_entry)

            except Exception:
                numbered.extend([None])
                metadata_.extend([None])

    # Extract numbered sequences from the output and validate the output
    numbered_sequences: list[str | None] = [None] * len(sequences)
    metadata: list[dict | None] = [None] * len(sequences)

    for (i, _), numbered_entry, metadata_entry in zip(sequences_input, numbered, metadata_):
        if numbered_entry is None or len(numbered_entry) > 1:
            continue

        numbering, _, _ = numbered_entry[0]

        numbered_sequences[i] = "".join([aa for (_, _), aa in numbering])
        scheme_indexes = [f"{pos}{ins}".strip() for (pos, ins), _ in numbering]
        metadata[i] = metadata_entry[0] if metadata_entry is not None else None
        metadata[i]["scheme_indexes"] = scheme_indexes

    # If the scheme is AHo, fix sequences
    if scheme in {"a", "aho"} and allow_fix:
        numbered_sequences = fix_aho_sequences(
            input_sequences=sequences,
            anarci_outputs=numbered_sequences,
            **kwargs,
        )

    if return_metadata:
        return numbered_sequences, metadata

    else:
        return numbered_sequences


def _sort_alphanumerically(s: list[str]):
    # a list like ["1B", "1A", "2A", "2B"] should be sorted as ["1A", "1B", "2A", "2B"]
    num_part = "".join(filter(str.isdigit, s))
    alpha_part = "".join(filter(str.isalpha, s))
    return (
        int(num_part),
        alpha_part,
    )  # sort by integer first, then alphabetically


def get_aligned_kabat_sequences(
    sequences: list[str], required_kabat_indexes: set[str] | None = None
) -> tuple[list[str], list[str], list[str | None], list[str | None]]:
    """
    Get aligned kabat sequences with insertions.
    This function uses ANARCI to number the sequences and then aligns them
    based on the presence of other insertions in the sequences.

    Parameters
    ----------
    sequences : list[str]
        A list of sequences to be numbered. These can be AHo or Kabat sequences.
    required_kabat_indexes :  Optional[set[str]]
        If provided, these indexes will be included even if they are not present
        in the anarci-numbered sequences. E.g. if required_kabat_indexes = {"1A", "2A"},
        and the sequences have indexes {"1", "2"} (EV), gaps will be inserted so that the final output
        positions will be {"1", "1A", "2", "2A"} (E-V-).

    Returns
    -------
    aligned_sequences : list[str]
        A list of the now-aligned sequences in Kabat numbering.
    kabat_positions : list[str]
        One list of kabat positions for the aligned sequences.
        These positions are guaranteed to be the same for all sequences,
        so just one list of positions is returned.
    v_genes : list[Optional[str]]
        A list of the V-gene germlines for each sequence.
    j_genes : list[Optional[str]]
        A list of the J-gene germlines for each sequence.
    """
    cleaned_sequences = [seq.replace("-", "") for seq in sequences]
    numbered_results, metadata = anarci_numbering(
        cleaned_sequences,
        scheme="kabat",
        return_metadata=True,
        assign_germline=True,
    )
    # --------------------
    # Find all kabat positions
    # --------------------
    # get the set of positions from all sequences
    all_positions = set()
    for metadata_entry in metadata:
        if metadata_entry is None:  # happens if invalid input (e.g. a None sequence)
            continue

        kabat_indexes = metadata_entry.get("scheme_indexes", [])
        if not kabat_indexes:
            continue
        all_positions.update(kabat_indexes)

    if required_kabat_indexes:
        all_positions.update(required_kabat_indexes)

    sorted_positions = sorted(all_positions, key=_sort_alphanumerically)

    # --------------------
    # Recreate sequences with insertions
    # --------------------
    # insert "-" if a sequence is missing an index present in the other ones
    aligned_sequences = []
    v_genes = []
    j_genes = []

    for i, sequence in enumerate(numbered_results):
        metadata_for_sequence = metadata[i]

        if not sequence or not metadata_for_sequence:
            aligned_sequences.append("")
            v_genes.append(None)
            j_genes.append(None)
            continue

        kabat_indexes_for_sequence = metadata_for_sequence.get("scheme_indexes", [])

        residue_dict = {kabat_index: residue for kabat_index, residue in zip(kabat_indexes_for_sequence, sequence)}

        aligned_seq = [residue_dict.get(pos, "-") for pos in sorted_positions]

        aligned_sequences.append("".join(aligned_seq))

        germlines = metadata_for_sequence.get("germlines", {})

        v_gene_info = germlines.get("v_gene", [[None, None]])[0]
        v_genes.append(v_gene_info[1] if v_gene_info else None)

        j_gene_info = germlines.get("j_gene", [[None, None]])[0]
        j_genes.append(j_gene_info[1] if j_gene_info else None)

    return aligned_sequences, sorted_positions, v_genes, j_genes
