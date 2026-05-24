import logging
from typing import Optional


def fix_aho_sequences(
    input_sequences: list[Optional[str]],
    anarci_outputs: list[Optional[str]],
    *,
    desired_length: Optional[int] = None,
    remove_terminal_arginine: bool = True,
    fix_truncation: bool = False,
    allow_sequence_alteration: bool = True,
    **kwargs,
) -> list[Optional[str]]:
    """
    Get the AHo-aligned versions of a list of sequences, as a list of strings containing
    dashes. The AHo alignment is unreliable, so if the computed AHo alignment is not of
    the desired length, dashes are added or removed at the end of the sequence to
    compensate. Any sequences for which alignment fails and cannot be repaired will return
    a None in the output list.

    Parameters
    ----------
    input_sequences: list[Optional[str]]
        A list of input amino acid sequences, as strings (e.g., "VQLVETGGRL...").
    anarci_outputs: list[Optional[str]]
        A list of numbered sequences as strings (e.g., "--VQL--VETGG---RL...").
    desired_length: Optional[int]
        If provided, the computed AHo sequences will be padded with dashes, or have dashes
        removed, to match the desired sequence length. Adding or removing dashes is always
        done at the end of the sequence.
    remove_terminal_arginine: bool
        If `True`, remove the terminal arginine from the AHo sequence. Terminal R is often
        present in light chains when using ANARCI to extract the variable region from a
        Fab sequence. Removing R leads to better manufacturability for certain
        expression vectors. Default True.
    fix_truncation: bool
        If `True`, attempt to fix errors where a computed AHo sequence is missing residues
        from the end on the input sequence by appending these residues to the AHo
        sequence.
    allow_sequence_alteration: bool
        If `False`, return None if the computed AHo sequence differs from
        the input sequence. If `True`, allow the computed sequence to differ from the
        input (for example, when computing AHo alignment using input sequences which
        include some or all of the constant domain).
        By default True to allow for variable region extraction from Fab regions.
    **kwargs: dict
        kwargs (unused)

    Returns
    -------
    A list of AHo-aligned sequences, as strings (e.g., "-VQLVET-GGRL..."). If numbering
    fails for any sequence in the input list, the corresponding entry in the output list
    will be None.
    """

    if fix_truncation:
        logging.critical(
            "Please make sure your input sequences are only variable regions (not Fab)! Using `fix_truncation` with Fab will result in an unexpected behavior. "
        )
    results = [
        _fix_aho_sequence(
            input_sequence,
            anarci_output,
            desired_length=desired_length,
            fix_truncation=fix_truncation,
            allow_sequence_alteration=allow_sequence_alteration,
            remove_terminal_arginine=remove_terminal_arginine,
        )
        for input_sequence, anarci_output in zip(input_sequences, anarci_outputs)
    ]

    return results


def _fix_aho_sequence(
    input_sequence: str,
    anarci_output: Optional[str] = None,
    *,
    desired_length: Optional[int] = None,
    remove_terminal_arginine: bool = True,
    fix_truncation: bool = False,
    allow_sequence_alteration: bool = True,
) -> Optional[str]:
    """
    Process a single AHo-aligned sequence, possibly returning None, as described in
    `get_aho_sequence`.
    """
    if anarci_output is None:
        return None

    aho_sequence = anarci_output
    aho_sequence_without_dashes = aho_sequence.replace("-", "")

    if aho_sequence_without_dashes != input_sequence:
        if fix_truncation and input_sequence.startswith(aho_sequence_without_dashes):
            aho_sequence = _fix_truncated_aho_sequence(aho_sequence, input_sequence)

        elif not allow_sequence_alteration:
            return None

    if desired_length is not None:
        aho_sequence = _fix_sequence_length(aho_sequence, desired_length)

    if remove_terminal_arginine:
        aho_sequence = _remove_terminal_arginine(aho_sequence)

    return aho_sequence


def _remove_terminal_arginine(aho_sequence: str) -> str:
    """
    Remove the terminal arginine from an AHo sequence, if present.
    """
    return aho_sequence[:-1] if aho_sequence.endswith("R") else aho_sequence


def _fix_truncated_aho_sequence(aho_sequence: str, original_sequence: str) -> str:
    """
    Attempt to fix a truncated AHo sequence by appending missing residues. Assumes that
    `aho_sequence.replace("-", "")` is a prefix of `original_sequence`.`.
    """
    num_missing_residues = len(original_sequence) - len(aho_sequence.replace("-", ""))
    missing_residues = original_sequence[-num_missing_residues:]

    return aho_sequence + missing_residues


def _fix_sequence_length(aho_sequence: str, desired_length: int) -> Optional[str]:
    """
    Attempt to fix an AHo sequence by either removing or appending dashes until it is of
    the desired length. Dashes are always added or removed at the end of the sequence
    (i.e., the rightmost N dashes are removed, or the sequence is padded with N dashes at
    the right). Returns the modified sequence as a string.
    """
    aho_sequence_length = len(aho_sequence)

    if aho_sequence_length == desired_length:
        return aho_sequence

    elif aho_sequence_length < desired_length:
        num_missing_residues = desired_length - aho_sequence_length
        return aho_sequence + ("-" * num_missing_residues)

    else:
        num_extra_residues = aho_sequence_length - desired_length
        aho_sequence_string = _remove_last_n_dashes(aho_sequence, num_extra_residues)

        if len(aho_sequence_string) == desired_length:
            return aho_sequence_string

        else:
            # We removed all the dashes, but the sequence is still too long.
            return None


def _remove_last_n_dashes(string: str, num_dashes_to_remove: int) -> str:
    reversed_string = string[::-1]
    reversed_string_with_first_n_dashes_removed = reversed_string.replace("-", "", num_dashes_to_remove)
    return reversed_string_with_first_n_dashes_removed[::-1]
