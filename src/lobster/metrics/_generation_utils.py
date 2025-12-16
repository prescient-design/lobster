import torch
from tmtools import tm_align
import logging
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import csv
from lobster.model.latent_generator.utils import kabsch_torch_batched
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x


# Set up logging
logging.basicConfig(level=logging.INFO)


def _is_sequence_pattern(mask_spec: str) -> bool:
    """Check if mask_spec is an amino acid pattern like 'GGGG'.

    A valid pattern is:
    - String of 2+ uppercase letters
    - All characters are the same amino acid
    - Not a numeric range (no digits)
    - Not a random spec (doesn't contain '|')
    - Not an index/range spec (no commas or hyphens with numbers)

    Args:
        mask_spec: Mask specification string

    Returns:
        True if this is an amino acid pattern, False otherwise

    Examples:
        >>> _is_sequence_pattern("GGGG")
        True
        >>> _is_sequence_pattern("AAA")
        True
        >>> _is_sequence_pattern("GG")
        True
        >>> _is_sequence_pattern("G")
        False  # Too short
        >>> _is_sequence_pattern("GGGA")
        False  # Mixed amino acids
        >>> _is_sequence_pattern("10-20")
        False  # Numeric range
        >>> _is_sequence_pattern("random | 3 | 10-30")
        False  # Random spec
    """
    if not mask_spec or not isinstance(mask_spec, str):
        return False

    mask_spec = mask_spec.strip()

    # Check minimum length
    if len(mask_spec) < 2:
        return False

    # Check if it contains special characters that indicate other spec types
    if "|" in mask_spec:  # Random spec
        return False

    if any(char.isdigit() for char in mask_spec):  # Contains numbers
        return False

    if "," in mask_spec:  # Index list
        return False

    if "-" in mask_spec:  # Range spec
        return False

    # Check if all characters are uppercase letters
    if not mask_spec.isalpha() or not mask_spec.isupper():
        return False

    # Check if all characters are the same amino acid
    if len(set(mask_spec)) != 1:
        return False

    # Check if it's a valid amino acid
    if mask_spec[0] not in restype_order_with_x:
        return False

    return True


def _find_consecutive_amino_acid_runs(
    pattern: str, sequence: torch.Tensor, device: torch.device
) -> list[tuple[int, int]]:
    """Find all consecutive runs of amino acid that meet or exceed pattern length.

    Args:
        pattern: Amino acid pattern (e.g., "GGGG")
        sequence: ORIGINAL (pre-tokenized) sequence tensor of shape (L,)
                  with raw amino acid indices (0-19 for standard AAs, 20 for X)
        device: torch device

    Returns:
        List of (start, end) tuples (inclusive) for qualifying runs.
        Empty list if no runs found.

    Example:
        pattern = "GGGG"
        sequence = [L, F, G, G, G, G, G, G, G, G, G, G, G, G, A, R, D]
                   indices: [11, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 17, 3]
        G_index = restype_order_with_x['G'] = 6
        Returns: [(2, 13)]  # Entire run of 12 G's
    """
    # Get the target amino acid and its index
    target_aa = pattern[0]  # All characters are the same (validated by _is_sequence_pattern)
    min_length = len(pattern)

    if target_aa not in restype_order_with_x:
        logger.warning(f"Invalid amino acid in pattern: {target_aa}")
        return []

    target_idx = restype_order_with_x[target_aa]

    # Convert sequence to numpy for easier processing
    seq_array = sequence.cpu().numpy() if sequence.is_cuda else sequence.numpy()

    # Find all runs of the target amino acid
    runs = []
    current_run_start = None
    current_run_length = 0

    for i, aa_idx in enumerate(seq_array):
        if aa_idx == target_idx:
            # Continue or start a run
            if current_run_start is None:
                current_run_start = i
                current_run_length = 1
            else:
                current_run_length += 1
        else:
            # End of a run (or not in a run)
            if current_run_start is not None and current_run_length >= min_length:
                # Record this run (inclusive end)
                runs.append((current_run_start, current_run_start + current_run_length - 1))
            # Reset
            current_run_start = None
            current_run_length = 0

    # Check if we ended with a qualifying run
    if current_run_start is not None and current_run_length >= min_length:
        runs.append((current_run_start, current_run_start + current_run_length - 1))

    return runs


def _create_sequence_pattern_masks(
    pattern: str,
    sequences: list[torch.Tensor],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Create per-sample inpainting masks based on sequence patterns.

    Args:
        pattern: Amino acid pattern (e.g., "GGGG")
        sequences: List of ORIGINAL (pre-tokenized) sequence tensors,
                   one per sample (each shape: (L,) with raw AA indices 0-19)
        max_length: Maximum sequence length for padding
        device: torch device

    Returns:
        Binary mask tensor of shape (B, max_length) where 1=inpaint, 0=keep

    Note:
        - Each sample gets its own mask based on its specific sequence
        - Works with original_sequences (pre-tokenization) not input_sequences
        - Mask positions correspond to positions in the original structure
    """
    B = len(sequences)
    mask = torch.zeros((B, max_length), dtype=torch.long, device=device)

    total_masked_positions = 0
    samples_with_matches = 0

    for i, seq in enumerate(sequences):
        # Find runs in this sequence
        runs = _find_consecutive_amino_acid_runs(pattern, seq, device)

        if runs:
            samples_with_matches += 1
            sample_masked = 0

            for start, end in runs:
                # Mark these positions as masked (inclusive end)
                length = end - start + 1
                mask[i, start : end + 1] = 1
                sample_masked += length

                logger.info(
                    f"Sample {i}: Found {pattern[0]} run of length {length} at positions {start}-{end} "
                    f"({length} positions masked)"
                )

            total_masked_positions += sample_masked
        else:
            logger.info(f"Sample {i}: No runs of {pattern[0]} with length >= {len(pattern)} found")

    # Summary logging
    if samples_with_matches > 0:
        logger.info(
            f"Pattern '{pattern}' found in {samples_with_matches}/{B} samples "
            f"({samples_with_matches / B * 100:.1f}% of batch)"
        )
        logger.info(f"Total sequence inpainting: {total_masked_positions} positions across {B} samples")
    else:
        logger.warning(f"Pattern '{pattern}' not found in any samples")

    return mask


def generate_random_mask_regions(
    num_regions: int,
    span_min: int,
    span_max: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate non-overlapping random masked regions.

    Args:
        num_regions: Number of random regions to generate
        span_min: Minimum span length (inclusive)
        span_max: Maximum span length (inclusive)
        length: Total sequence length
        device: Device to create tensor on

    Returns:
        Binary mask tensor of shape (1, length) where 1=mask, 0=keep
    """
    # Validate and adjust span range
    if span_min > span_max:
        logger.warning(f"Invalid span range {span_min}-{span_max}, swapping to {span_max}-{span_min}")
        span_min, span_max = span_max, span_min

    # Clamp to sequence length
    if span_max > length:
        logger.warning(f"Max span {span_max} exceeds sequence length {length}, clamping to {length}")
        span_max = length

    if span_min > length:
        logger.warning(f"Min span {span_min} exceeds sequence length {length}, clamping to {length}")
        span_min = length

    # Theoretical maximum regions that could fit
    theoretical_max = length // span_min
    if num_regions > theoretical_max:
        logger.warning(
            f"Requested {num_regions} regions may not fit (theoretical max ~{theoretical_max}), "
            f"will attempt to fit as many as possible"
        )

    # Initialize mask (all zeros = keep all)
    mask = torch.zeros((1, length), dtype=torch.long, device=device)

    # Track available positions (0-indexed)
    available_positions = set(range(length))

    # Track generated regions for logging
    generated_regions = []

    # Generate regions
    for region_idx in range(num_regions):
        # Randomly select span length for this region (torch.randint is exclusive on upper bound)
        span_length = torch.randint(span_min, span_max + 1, (1,), device=device).item()

        # Find all valid starting positions
        valid_starts = []
        for start_pos in range(length - span_length + 1):
            # Check if all positions in [start_pos, start_pos + span_length) are available
            region_positions = set(range(start_pos, start_pos + span_length))
            if region_positions.issubset(available_positions):
                valid_starts.append(start_pos)

        # Check if we can place this region
        if not valid_starts:
            logger.warning(
                f"Could only generate {region_idx} out of {num_regions} regions due to sequence fragmentation"
            )
            break

        # Randomly select a valid starting position using torch
        random_idx = torch.randint(0, len(valid_starts), (1,), device=device).item()
        start_pos = valid_starts[random_idx]
        end_pos = start_pos + span_length

        # Mark positions as masked
        for pos in range(start_pos, end_pos):
            mask[0, pos] = 1
            available_positions.remove(pos)

        # Track for logging (store as inclusive range)
        generated_regions.append((start_pos, end_pos - 1))

    # Log results
    num_masked = mask.sum().item()
    regions_str = ", ".join([f"[{start}-{end}]" for start, end in generated_regions])
    logger.info(
        f"Generated {len(generated_regions)} random regions: {regions_str} ({num_masked}/{length} positions masked)"
    )

    return mask


def _parse_random_mask_spec(mask_spec: str, length: int, device: torch.device) -> torch.Tensor:
    """Parse and execute random mask specification.

    Args:
        mask_spec: String in format "random | num_regions | span_range"
        length: Total sequence length
        device: Device to create tensor on

    Returns:
        Binary mask tensor of shape (1, length) where 1=mask, 0=keep
    """
    parts = [p.strip() for p in mask_spec.split("|")]

    if len(parts) != 3:
        raise ValueError(f"Random mask spec must have 3 parts: 'random | num_regions | span_range', got: {mask_spec}")

    keyword, num_regions_str, span_range = parts

    # Validation
    if keyword.lower() != "random":
        raise ValueError(f"Expected 'random' keyword, got: {keyword}")

    num_regions = int(num_regions_str)

    # Parse span range
    if "-" not in span_range:
        raise ValueError(f"Span range must be 'min-max', got: {span_range}")

    span_min_str, span_max_str = span_range.split("-", 1)
    span_min = int(span_min_str.strip())
    span_max = int(span_max_str.strip())

    # Call the generation function
    return generate_random_mask_regions(num_regions, span_min, span_max, length, device)


def parse_mask_indices(mask_spec: str | list | None, length: int, device: torch.device) -> torch.Tensor:
    """Parse mask indices specification and return a binary mask tensor.

    Args:
        mask_spec: Specification of indices to mask. Can be:
            - None or empty string "": return all-zero tensor (no masking)
            - String with ranges: "10-20,30-35" (inclusive ranges)
            - String with comma-separated indices: "10,15,20,25"
            - List of integers: [10, 15, 20, 25]
            - List of tuples for ranges: [(10, 20), (30, 35)]
            - Random regions: "random | N | min-max" (e.g., "random | 3 | 10-30")
        length: Total length of the sequence/structure
        device: Device to create the tensor on

    Returns:
        Binary mask tensor of shape (1, length) where 1=mask/generate, 0=keep fixed.
        Returns all-zero tensor if mask_spec is None or empty string (no masking).
    """
    # Initialize mask with all zeros (keep all positions)
    mask = torch.zeros((1, length), dtype=torch.long, device=device)

    if mask_spec is None or mask_spec == "":
        return mask

    # Check for random specification
    if isinstance(mask_spec, str) and "|" in mask_spec:
        if mask_spec.strip().lower().startswith("random"):
            return _parse_random_mask_spec(mask_spec, length, device)

    indices_to_mask = set()

    if isinstance(mask_spec, str):
        # Parse string specification
        parts = mask_spec.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Range specification (e.g., "10-20")
                start_str, end_str = part.split("-")
                start = int(start_str.strip())
                end = int(end_str.strip())
                indices_to_mask.update(range(start, end + 1))  # Inclusive
            else:
                # Single index
                indices_to_mask.add(int(part))

    elif isinstance(mask_spec, list):
        for item in mask_spec:
            if isinstance(item, tuple):
                # Range as tuple (start, end)
                start, end = item
                indices_to_mask.update(range(start, end + 1))  # Inclusive
            elif isinstance(item, int):
                # Single index
                indices_to_mask.add(item)
            else:
                raise ValueError(f"Invalid mask specification item: {item}")

    else:
        raise ValueError(f"Invalid mask specification type: {type(mask_spec)}")

    # Convert to mask tensor
    for idx in indices_to_mask:
        if 0 <= idx < length:
            mask[0, idx] = 1
        else:
            logger.warning(f"Mask index {idx} out of bounds [0, {length}), ignoring")

    num_masked = mask.sum().item()
    logger.info(f"Parsed mask specification: {num_masked}/{length} positions to generate")

    return mask


def add_linker_to_sequence(sequence: str, residue_index_offset: int = 512, chain_linker: str = "G" * 25):
    """Add a linker to a sequence.
    Args:
        sequence: The sequence to encode
        residue_index_offset: The offset for the residue indices
        chain_linker: The linker to use for the chain breaks
    Returns:
        sequence: The sequence with linker
        residx: The residue indices accounting for the linker
        linker_mask: The mask for the linker
    """

    chains = sequence.split(":")
    seq = chain_linker.join(chains)

    residx = torch.arange(len(seq))

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += i * residue_index_offset
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(residx, dtype=torch.float32)
    offset = 0
    for i, chain in enumerate(chains):
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    return seq, residx, linker_mask


def filter_residues_by_chains(
    seq_i: torch.Tensor,
    coords: torch.Tensor,
    chains_i: torch.Tensor,
    target_chain_ids: list[int],
    mask_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Filter residues to only include those from specified chains.

    This function extracts residues belonging to specific chain IDs from multi-chain
    protein complexes. Useful for chain-specific ESMFold predictions.

    Args:
        seq_i: Sequence tensor (L,) with amino acid indices (0-19 for standard AAs, 20 for X)
        coords: Coordinate tensor (L, 3, 3) or (L, num_atoms, 3) with atomic coordinates
        chains_i: Chain ID tensor (L,) with chain identifier for each residue
        target_chain_ids: List of chain IDs to include (e.g., [200, 400])
        mask_tensor: Optional mask tensor (L,) to filter as well

    Returns:
        filtered_seq: Sequence tensor with only target chain residues
        filtered_coords: Coordinate tensor with only target chain residues
        filtered_chains: Chain ID tensor with only target chain residues
        filtered_mask: Mask tensor with only target chain residues (None if mask_tensor is None)

    Example:
        >>> # Filter to only chains 200 and 400 from a 3-chain complex
        >>> seq_filtered, coords_filtered, chains_filtered, _ = filter_residues_by_chains(
        ...     seq, coords, chains, [200, 400]
        ... )
        >>> # chains_filtered will only contain residues from chains 200 and 400
    """
    # Create boolean mask for residues in target chains
    chain_mask = torch.zeros_like(chains_i, dtype=torch.bool)
    for chain_id in target_chain_ids:
        chain_mask |= chains_i == chain_id

    # Filter tensors
    filtered_seq = seq_i[chain_mask]
    filtered_coords = coords[chain_mask]
    filtered_chains = chains_i[chain_mask]
    filtered_mask = mask_tensor[chain_mask] if mask_tensor is not None else None

    return filtered_seq, filtered_coords, filtered_chains, filtered_mask


def build_multichain_sequence_string(seq_i: torch.Tensor, chains_i: torch.Tensor, restype_order_inv: dict) -> str:
    """Build sequence string with chain breaks marked by ':'.

    This function converts a sequence tensor and chain IDs into a string representation
    where different chains are separated by colons. This format is used by ESMFold
    to understand chain boundaries in multi-chain complexes.

    Args:
        seq_i: Sequence tensor (L,) with amino acid indices
        chains_i: Chain ID tensor (L,) with chain identifier for each residue
        restype_order_inv: Dictionary mapping AA index to single-letter code
                          (e.g., {0: 'A', 1: 'C', ...})

    Returns:
        sequence_str: Sequence string with ':' separating chains (e.g., "ACGT:DEFG:HIJK")

    Example:
        >>> # Convert multi-chain structure to sequence string
        >>> from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
        >>> seq_str = build_multichain_sequence_string(seq, chains, restype_order_with_x_inv)
        >>> print(seq_str)  # "ACDEFG:HIKLMN:PQRST" (3 chains)
    """
    sequence_str = ""
    prev_chain = None
    for aa_idx, chain_id in zip(seq_i, chains_i):
        if prev_chain is not None and chain_id.item() != prev_chain:
            sequence_str += ":"
        sequence_str += restype_order_inv[aa_idx.item()]
        prev_chain = chain_id.item()
    return sequence_str


def predict_structure_with_esmfold(
    plm_fold,
    seq_i: torch.Tensor,
    chains_i: torch.Tensor,
    orig_coords: torch.Tensor,
    gen_coords: torch.Tensor | None,
    mask_i: torch.Tensor,
    cfg,
    device: torch.device,
    restype_order_inv: dict,
    inpainting_mask_seq_i: torch.Tensor | None = None,
    inpainting_mask_struc_i: torch.Tensor | None = None,
    chain_group: list[int] | None = None,
) -> dict:
    """Predict structure using ESMFold.

    This function consolidates ESMFold prediction logic used in both inverse folding
    and inpainting modes. It handles:
    - Building multi-chain sequences with proper chain breaks
    - Adding linkers between chains for ESMFold
    - Running ESMFold prediction
    - Computing metrics (TM-score, RMSD)
    - Optionally filtering to specific chains only

    Args:
        plm_fold: ESMFold model wrapper with .tokenizer and .model attributes
        seq_i: Sequence tensor (L,) with amino acid indices
        chains_i: Chain ID tensor (L,) with chain identifier for each residue
        orig_coords: Original structure coordinates (L, 3, 3)
        gen_coords: Generated structure coordinates (L, 3, 3). Required for inpainting
                   mode (to compute alignment RMSD). None for inverse folding mode.
        mask_i: Structure mask (L,)
        cfg: Configuration object with .generation.get() method
        device: torch device
        restype_order_inv: Dictionary mapping AA index to single-letter code
        inpainting_mask_seq_i: Optional sequence inpainting mask (L,). Used to compute
                              RMSD only on inpainted regions.
        inpainting_mask_struc_i: Optional structure inpainting mask (L,). Used to compute
                                RMSD only on inpainted regions.
        chain_group: Optional list of chain IDs to predict. If None, use all chains.

    Returns:
        Dictionary containing:
            - 'folded_structure_metrics': Metrics dict with TM-score, pLDDT, etc.
            - 'pred_coords': Predicted coordinates from ESMFold (1, L, 3, 3)
            - 'gen_coords_aligned': Generated coords aligned to prediction (L, 3, 3)
                                   Only present if gen_coords was provided
            - 'rmsd_inpainted': RMSD on inpainted region (float)
                               Only present if gen_coords was provided
            - 'sequence_str': Sequence string used (with chain breaks removed)
            - 'esmfold_outputs': Raw ESMFold model outputs
            - 'chain_group': Chain IDs that were predicted (for tracking)
            - 'num_residues': Number of residues predicted
            - 'num_chains': Number of chains predicted

    Example Usage:
        # Inverse folding (no generated coords):
        result = predict_structure_with_esmfold(
            plm_fold, seq, chains, orig_coords, None, mask, cfg, device, restype_inv
        )
        tm_score = result['folded_structure_metrics']['_tm_score']

        # Inpainting (with generated coords and alignment):
        result = predict_structure_with_esmfold(
            plm_fold, seq, chains, orig_coords, gen_coords, mask, cfg, device,
            restype_inv, inpaint_mask_seq, inpaint_mask_struc
        )
        rmsd = result['rmsd_inpainted']
    """
    # 1. OPTIONAL: Filter by chains if specified
    if chain_group is not None:
        # Save original tensors before filtering (needed for filtering other tensors)
        seq_i_original = seq_i
        orig_coords_original = orig_coords
        chains_i_original = chains_i

        # Filter primary tensors
        seq_i, orig_coords, chains_i, mask_i = filter_residues_by_chains(
            seq_i, orig_coords, chains_i, chain_group, mask_tensor=mask_i
        )

        # Filter gen_coords using ORIGINAL tensors (gen_coords is still unfiltered size)
        if gen_coords is not None:
            _, gen_coords, _, _ = filter_residues_by_chains(seq_i_original, gen_coords, chains_i_original, chain_group)

        # Filter inpainting masks using ORIGINAL tensors (still unfiltered size)
        if inpainting_mask_seq_i is not None:
            _, _, _, inpainting_mask_seq_i = filter_residues_by_chains(
                seq_i_original, orig_coords_original, chains_i_original, chain_group, mask_tensor=inpainting_mask_seq_i
            )
        if inpainting_mask_struc_i is not None:
            _, _, _, inpainting_mask_struc_i = filter_residues_by_chains(
                seq_i_original,
                orig_coords_original,
                chains_i_original,
                chain_group,
                mask_tensor=inpainting_mask_struc_i,
            )

    # 2. Build sequence string with chain breaks
    sequence_str = build_multichain_sequence_string(seq_i, chains_i, restype_order_inv)

    # 3. Add linkers between chains for ESMFold
    sequence_str, position_ids, linker_mask = add_linker_to_sequence(sequence_str)

    # 4. Tokenize the sequence
    tokenized_input = plm_fold.tokenizer.encode_plus(
        sequence_str,
        padding=True,
        truncation=True,
        max_length=cfg.generation.get("max_length", 512),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    # 5. Fold with ESMFold
    with torch.no_grad():
        outputs = plm_fold.model(tokenized_input, position_ids=position_ids.unsqueeze(0).to(device))

    # 6. Remove linkers from outputs
    outputs["positions"] = outputs["positions"][:, :, linker_mask == 1, :, :]
    outputs["plddt"] = outputs["plddt"][:, linker_mask == 1]
    outputs["predicted_aligned_error"] = outputs["predicted_aligned_error"][:, linker_mask == 1]

    # Remove linker from sequence_str
    sequence_list = list(sequence_str)
    sequence_str = "".join([seq_char for seq_char, mask_val in zip(sequence_list, linker_mask) if mask_val == 1])

    # 7. Get folded structure metrics (TM-score, etc.)
    folded_structure_metrics, pred_coords = get_folded_structure_metrics(
        outputs, orig_coords[None], [sequence_str], mask=mask_i[None], device=device
    )

    # 8. Prepare return dictionary with common results
    result = {
        "folded_structure_metrics": folded_structure_metrics,
        "pred_coords": pred_coords,
        "sequence_str": sequence_str,
        "esmfold_outputs": outputs,
        "chain_group": chain_group if chain_group is not None else chains_i.unique().tolist(),
        "num_residues": len(seq_i),
        "num_chains": len(chains_i.unique()),
    }

    # 9. OPTIONAL: Align generated coords to prediction (inpainting mode only)
    if gen_coords is not None:
        gen_coords_aligned, rmsd_inpainted = align_and_compute_rmsd_inpainted(
            gen_coords=gen_coords,
            pred_coords=pred_coords[0],
            inpainting_mask_seq=inpainting_mask_seq_i,
            inpainting_mask_struc=inpainting_mask_struc_i,
            device=device,
        )
        result["gen_coords_aligned"] = gen_coords_aligned
        result["rmsd_inpainted"] = rmsd_inpainted

    return result


def calculate_percent_identity(
    ground_truth_seq: torch.Tensor, generated_seq: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate percent identity between ground truth and generated sequences.

    Parameters
    ----------
    ground_truth_seq : torch.Tensor
        Ground truth sequence tensor of shape (B, L).
    generated_seq : torch.Tensor
        Generated sequence tensor of shape (B, L).
    mask : torch.Tensor, optional
        Optional mask tensor of shape (B, L) to ignore padded positions. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor of percent identities for each sequence in the batch.
    """
    # Ensure both sequences have the same shape
    assert ground_truth_seq.shape == generated_seq.shape, "Sequences must have the same shape"

    # Calculate matches
    matches = (ground_truth_seq == generated_seq).float()

    if mask is not None:
        # Only consider positions where mask is 1
        matches = matches * mask.float()
        valid_positions = mask.sum(dim=1).float()
        # Avoid division by zero
        valid_positions = torch.clamp(valid_positions, min=1.0)
        percent_identity = (matches.sum(dim=1) / valid_positions) * 100.0
    else:
        # Consider all positions
        sequence_length = ground_truth_seq.shape[1]
        percent_identity = (matches.sum(dim=1) / sequence_length) * 100.0

    return percent_identity


def get_folded_structure_metrics(outputs, ref_coords, ref_seq, prefix="", mask=None, device=None):
    """Get the metrics of the folded structure.

    Parameters
    ----------
    outputs : dict
        The outputs of the ESMFold model.
    ref_coords : torch.Tensor
        The reference coordinates of the structure. Shape [B, L, 3, 3].
    ref_seq : list of str
        The reference sequence list of strings.
    prefix : str, optional
        Optional prefix for the returned metric keys. Default is "".
    mask : torch.Tensor, optional
        The mask of the structure. Shape [B, L]. Default is None.
    device : torch.device, optional
        Device for computation. If None, inferred from pred_coords. Default is None.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - plddt: The average pLDDT scores of the batch
        - predicted_aligned_error: The average predicted aligned error of the batch
        - tm_score: The average TM-score of the predicted structure vs the reference structure of the batch
        - rmsd: The average RMSD (from Kabsch alignment) of the predicted structure vs the reference structure of the batch
    torch.Tensor
        The predicted coordinates of the structure. Shape [B, L, 3, 3].
    """
    pred_coords = outputs["positions"][-1][:, :, :3, :]  # [B, L, 3, 3]
    plddt_scores = outputs["plddt"].mean(dim=(-1, -2))  # [B]
    predicted_aligned_error = outputs["predicted_aligned_error"].mean(dim=(-1, -2))  # [B]

    # Infer device if not provided
    if device is None:
        device = pred_coords.device

    tm_score = []
    rmsd = []
    for i in range(pred_coords.shape[0]):
        if mask is not None:
            pred_coords_i = pred_coords[i, mask[i] == 1, :, :]
            ref_coords_i = ref_coords[i, mask[i] == 1, :, :]
            # get correct index for string
            ref_seq_i = ref_seq[i]
            ref_seq_i = "".join([ref_seq_i[j] for j in range(len(ref_seq_i)) if mask[i][j] == 1])

        else:
            pred_coords_i = pred_coords[i, :, :, :]
            ref_coords_i = ref_coords[i, :, :, :]
            ref_seq_i = ref_seq[i]

        # Calculate TM-score using tm_align (still needed for structural similarity)
        tm_out = tm_align(
            pred_coords_i[:, 1, :].cpu().numpy(), ref_coords_i[:, 1, :].detach().cpu().numpy(), ref_seq_i, ref_seq_i
        )
        tm_score.append(tm_out.tm_norm_chain1)

        # Calculate Kabsch RMSD using our implementation (PRIMARY METRIC)
        rmsd_kabsch = align_and_compute_rmsd(
            coords1=pred_coords_i,
            coords2=ref_coords_i,
            mask=None,  # Already masked by this point
            return_aligned=False,
            device=device,
        )
        rmsd.append(rmsd_kabsch)

    tm_score = torch.tensor(tm_score).to(pred_coords.device)
    rmsd = torch.tensor(rmsd).to(pred_coords.device)

    # set masked coords to 0
    if mask is not None:
        pred_coords[mask == 0] = 0

    return {
        f"{prefix}_plddt": plddt_scores.mean(),
        f"{prefix}_predicted_aligned_error": predicted_aligned_error.mean(),
        f"{prefix}_tm_score": tm_score.mean(),
        f"{prefix}_rmsd": rmsd.mean(),
    }, pred_coords


class MetricsPlotter:
    """Helper class to create plots from metrics data."""

    def __init__(self, output_dir: Path, mode: str):
        """Initialize plotter for a specific generation mode.

        Args:
            output_dir: Directory to save plot files
            mode: Generation mode (unconditional, inverse_folding, forward_folding)
        """
        self.output_dir = output_dir
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_box_plots_from_csv(self, csv_path: Path):
        """Create box and whisker plots from CSV metrics data.

        Args:
            csv_path: Path to the CSV file containing metrics
        """

        # Read CSV data
        df = pd.read_csv(csv_path)

        # Define metrics to plot based on mode
        if self.mode == "unconditional":
            metrics = ["plddt", "predicted_aligned_error", "tm_score", "rmsd"]
            length_col = "sequence_length"
        elif self.mode == "inverse_folding":
            metrics = ["percent_identity", "plddt", "predicted_aligned_error", "tm_score", "rmsd"]
            length_col = "sequence_length"
        elif self.mode == "forward_folding":
            metrics = ["tm_score", "rmsd"]
            length_col = "sequence_length"
        elif self.mode == "inpainting":
            metrics = [
                "percent_identity_masked",
                "percent_identity_unmasked",
                "rmsd_inpainted",
                "plddt",
                "predicted_aligned_error",
                "tm_score",
                "rmsd",
            ]
            length_col = "sequence_length"
        else:
            logger.warning(f"Unknown mode for plotting: {self.mode}")
            return

        # Filter out empty values and convert to numeric
        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")

        # Remove rows with NaN values
        df = df.dropna(subset=metrics)

        if df.empty:
            logger.warning("No valid data found for plotting")
            return

        # Create plots for each metric
        for metric in metrics:
            if metric not in df.columns:
                continue

            self._create_single_box_plot(df, metric, length_col)

        # Create combined plot
        self._create_combined_box_plot(df, metrics, length_col)

    def _create_single_box_plot(self, df: pd.DataFrame, metric: str, length_col: str):
        """Create a single box plot for one metric."""
        plt.figure(figsize=(10, 6))

        # Group data by length
        lengths = sorted(df[length_col].unique())
        data_by_length = [df[df[length_col] == length][metric].dropna() for length in lengths]

        # Create box plot without outliers
        box_plot = plt.boxplot(data_by_length, labels=lengths, patch_artist=True, showfliers=False)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(lengths)))
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title(
            f"{metric.replace('_', ' ').title()} by Sequence Length\n({self.mode.replace('_', ' ').title()} Generation)"
        )
        plt.xlabel("Sequence Length")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)

        # Add horizontal line at 2.0 for RMSD plots
        if "rmsd" in metric.lower():
            plt.axhline(y=2.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="RMSD = 2.0 Å")
            plt.legend()

        # Rotate x-axis labels if there are many lengths
        if len(lengths) > 5:
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{self.mode}_{metric}_boxplot_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved box plot: {plot_path}")

    def _create_combined_box_plot(self, df: pd.DataFrame, metrics: list, length_col: str):
        """Create a combined subplot with all metrics."""
        lengths = sorted(df[length_col].unique())

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            ax = axes[i]

            # Group data by length
            data_by_length = [df[df[length_col] == length][metric].dropna() for length in lengths]

            # Create box plot without outliers
            box_plot = ax.boxplot(data_by_length, labels=lengths, patch_artist=True, showfliers=False)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(lengths)))
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

            # Add horizontal line at 2.0 for RMSD plots
            if "rmsd" in metric.lower():
                ax.axhline(y=2.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="RMSD = 2.0 Å")
                ax.legend(fontsize=8)

            # Rotate x-axis labels if there are many lengths
            if len(lengths) > 5:
                ax.tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"Metrics by Sequence Length - {self.mode.replace('_', ' ').title()} Generation", fontsize=16)
        plt.tight_layout()

        # Save combined plot
        plot_path = self.output_dir / f"{self.mode}_combined_boxplots_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved combined box plots: {plot_path}")

    def create_correlation_plots_from_csv(self, csv_path: Path):
        """Create correlation scatter plots from CSV data.

        Creates two sets of plots:
        1. Forward folding TM-score vs ESMFold metrics
        2. Inverse folding percent identity vs ESMFold metrics

        Only applicable for unconditional mode with self-reflection enabled.

        Args:
            csv_path: Path to the CSV file containing metrics
        """
        if self.mode != "unconditional":
            logger.info("Correlation plots only available for unconditional mode")
            return

        # Read CSV data
        df = pd.read_csv(csv_path)

        # Check if self-reflection columns exist
        required_cols = [
            "tm_score_unconditional_to_forward",
            "percent_identity_self_reflection",
            "plddt_refined",
            "pae_refined",
            "tm_score_esmfold_refined",
            "rmsd_esmfold_refined",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.info(f"Self-reflection metrics not found ({missing_cols}), skipping correlation plots")
            return

        # Filter out rows with missing data
        df_filtered = df[required_cols].copy()

        # Convert to numeric and drop NaN
        for col in required_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

        df_filtered = df_filtered.dropna()

        if df_filtered.empty or len(df_filtered) < 2:
            logger.warning(f"Insufficient valid data for correlation plots (n={len(df_filtered)})")
            return

        # Create correlation plots
        self._create_forward_tm_correlation_plot(df_filtered)
        self._create_percent_identity_correlation_plot(df_filtered)

    def _create_forward_tm_correlation_plot(self, df: pd.DataFrame):
        """Create scatter plots of forward folding TM-score vs ESMFold metrics.

        Args:
            df: DataFrame with filtered data
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        x_data = df["tm_score_unconditional_to_forward"]

        metrics = [
            ("plddt_refined", "pLDDT (Refined)", "Higher is better"),
            ("pae_refined", "PAE (Refined) [Å]", "Lower is better"),
            ("tm_score_esmfold_refined", "TM-Score (ESMFold vs Refined)", "Higher is better"),
            ("rmsd_esmfold_refined", "RMSD (ESMFold vs Refined) [Å]", "Lower is better"),
        ]

        for i, (col, ylabel, trend) in enumerate(metrics):
            ax = axes[i]
            y_data = df[col]

            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color="steelblue")

            # Calculate correlation
            correlation = x_data.corr(y_data)

            # Add trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Labels
            ax.set_xlabel("Forward Folding TM-Score\n(Unconditional → Forward)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel}\nCorrelation: {correlation:.3f}", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add trend annotation
            ax.text(
                0.05,
                0.95,
                trend,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.suptitle("Forward Folding Quality vs ESMFold Metrics", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{self.mode}_forward_tm_correlation_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved forward TM-score correlation plot: {plot_path}")

    def _create_percent_identity_correlation_plot(self, df: pd.DataFrame):
        """Create scatter plots of inverse folding percent identity vs ESMFold metrics.

        Args:
            df: DataFrame with filtered data
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        x_data = df["percent_identity_self_reflection"]

        metrics = [
            ("plddt_refined", "pLDDT (Refined)", "Higher is better"),
            ("pae_refined", "PAE (Refined) [Å]", "Lower is better"),
            ("tm_score_esmfold_refined", "TM-Score (ESMFold vs Refined)", "Higher is better"),
            ("rmsd_esmfold_refined", "RMSD (ESMFold vs Refined) [Å]", "Lower is better"),
        ]

        for i, (col, ylabel, trend) in enumerate(metrics):
            ax = axes[i]
            y_data = df[col]

            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color="seagreen")

            # Calculate correlation
            correlation = x_data.corr(y_data)

            # Add trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Labels
            ax.set_xlabel("Sequence Recovery (%)\n(Initial → Refined Percent Identity)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel}\nCorrelation: {correlation:.3f}", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add trend annotation
            ax.text(
                0.05,
                0.95,
                trend,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )

        plt.suptitle("Sequence Recovery vs ESMFold Metrics", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / (f"{self.mode}_percent_identity_correlation_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved percent identity correlation plot: {plot_path}")


class MetricsCSVWriter:
    """Helper class to write metrics to CSV files."""

    def __init__(self, output_dir: Path, mode: str):
        """Initialize CSV writer for a specific generation mode.

        Args:
            output_dir: Directory to save CSV files
            mode: Generation mode (unconditional, inverse_folding, forward_folding, inpainting)
        """
        self.output_dir = output_dir
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create CSV file path for metrics
        self.csv_path = output_dir / f"{mode}_metrics_{self.timestamp}.csv"

        # Create CSV file path for sequences
        self.sequences_csv_path = output_dir / f"sequences_{mode}_{self.timestamp}.csv"

        # Initialize CSV files with headers
        self._initialize_csv()
        self._initialize_sequences_csv()

    def _initialize_csv(self):
        """Initialize CSV file with appropriate headers based on mode."""
        headers = ["run_id", "timestamp", "mode"]

        if self.mode == "unconditional":
            headers.extend(
                [
                    "plddt",
                    "predicted_aligned_error",
                    "tm_score",
                    "rmsd",
                    "sequence_length",
                    "num_samples",
                    # Self-reflection refinement metrics (optional, empty if not enabled)
                    "percent_identity_self_reflection",
                    "tm_score_unconditional_to_forward",
                    "rmsd_unconditional_to_forward",
                    "tm_score_forward_to_inverse",
                    "rmsd_forward_to_inverse",
                    # ESMFold baseline metrics
                    "plddt_unconditional",
                    "pae_unconditional",
                    "tm_score_esmfold_unconditional",
                    "rmsd_esmfold_unconditional",
                    # ESMFold refined metrics
                    "plddt_refined",
                    "pae_refined",
                    "tm_score_esmfold_refined",
                    "rmsd_esmfold_refined",
                    # ESMFold improvement metrics
                    "plddt_improvement",
                    "pae_improvement",
                    "tm_score_improvement",
                    "rmsd_improvement",
                    # ESMFold structure comparison metrics
                    "tm_score_unconditional_to_esmfold",
                    "rmsd_unconditional_to_esmfold",
                    "tm_score_forward_to_esmfold",
                    "rmsd_forward_to_esmfold",
                    "tm_score_esmfold_agreement_improvement",
                    "rmsd_esmfold_agreement_improvement",
                ]
            )
        elif self.mode == "inverse_folding":
            headers.extend(
                [
                    "percent_identity",
                    "plddt",
                    "predicted_aligned_error",
                    "tm_score",
                    "rmsd",
                    "sequence_length",
                    "input_file",
                ]
            )
        elif self.mode == "forward_folding":
            headers.extend(["tm_score", "rmsd", "sequence_length", "input_file"])
        elif self.mode == "inpainting":
            headers.extend(
                [
                    "percent_identity_masked",
                    "percent_identity_unmasked",
                    "rmsd_inpainted",
                    "plddt",
                    "predicted_aligned_error",
                    "tm_score",
                    "rmsd",
                    "sequence_length",
                    "num_masked_seq",
                    "num_masked_struc",
                    "input_file",
                ]
            )

        # Write headers to CSV
        with open(self.csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        logger.info(f"Initialized CSV metrics file: {self.csv_path}")

    def _initialize_sequences_csv(self):
        """Initialize sequences CSV file with headers."""
        headers = [
            "run_id",
            "iteration",
            "sample_idx",
            "sequence",
            "original_sequence",
            "inpainted_sequence",
            "original_inpainted_sequence",
            "length",
            "generation_mode",
            "input_structure",
            "num_chains",
            "chain_ids",
            "trial_selected",
            "percent_identity_original",
            "masked_positions",
            "sequence_type",
            "timestamp",
        ]

        # Write headers to sequences CSV
        with open(self.sequences_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        logger.info(f"Initialized sequences CSV file: {self.sequences_csv_path}")

    def write_batch_metrics(self, metrics: dict, run_id: str, **kwargs):
        """Write batch metrics to CSV.

        Args:
            metrics: Dictionary containing metric values
            run_id: Unique identifier for this run
            **kwargs: Additional data to include (input_file, sequence_length, etc.)
        """

        def _to_scalar(value):
            """Convert tensor to scalar value."""
            if value is None or value == "":
                return ""
            if hasattr(value, "item"):
                return value.item()
            elif hasattr(value, "cpu"):
                return value.cpu().item()
            else:
                return float(value)

        row = [run_id, datetime.now().isoformat(), self.mode]

        if self.mode == "unconditional":
            row.extend(
                [
                    _to_scalar(metrics.get("_plddt", "")),
                    _to_scalar(metrics.get("_predicted_aligned_error", "")),
                    _to_scalar(metrics.get("_tm_score", "")),
                    _to_scalar(metrics.get("_rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("num_samples", ""),
                    # Self-reflection refinement metrics (empty if not available)
                    _to_scalar(metrics.get("percent_identity_self_reflection", "")),
                    _to_scalar(metrics.get("tm_score_unconditional_to_forward", "")),
                    _to_scalar(metrics.get("rmsd_unconditional_to_forward", "")),
                    _to_scalar(metrics.get("rmsd_unconditional_to_forward_kabsch", "")),
                    _to_scalar(metrics.get("tm_score_forward_to_inverse", "")),
                    _to_scalar(metrics.get("rmsd_forward_to_inverse", "")),
                    _to_scalar(metrics.get("rmsd_forward_to_inverse_kabsch", "")),
                    # ESMFold baseline metrics
                    _to_scalar(metrics.get("plddt_unconditional", "")),
                    _to_scalar(metrics.get("pae_unconditional", "")),
                    _to_scalar(metrics.get("tm_score_esmfold_unconditional", "")),
                    _to_scalar(metrics.get("rmsd_esmfold_unconditional", "")),
                    # ESMFold refined metrics
                    _to_scalar(metrics.get("plddt_refined", "")),
                    _to_scalar(metrics.get("pae_refined", "")),
                    _to_scalar(metrics.get("tm_score_esmfold_refined", "")),
                    _to_scalar(metrics.get("rmsd_esmfold_refined", "")),
                    # ESMFold improvement metrics
                    _to_scalar(metrics.get("plddt_improvement", "")),
                    _to_scalar(metrics.get("pae_improvement", "")),
                    _to_scalar(metrics.get("tm_score_improvement", "")),
                    _to_scalar(metrics.get("rmsd_improvement", "")),
                ]
            )
        elif self.mode == "inverse_folding":
            row.extend(
                [
                    _to_scalar(kwargs.get("percent_identity", "")),
                    _to_scalar(metrics.get("_plddt", "")),
                    _to_scalar(metrics.get("_predicted_aligned_error", "")),
                    _to_scalar(metrics.get("_tm_score", "")),
                    _to_scalar(metrics.get("_rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("input_file", ""),
                ]
            )
        elif self.mode == "forward_folding":
            row.extend(
                [
                    _to_scalar(metrics.get("tm_score", "")),
                    _to_scalar(metrics.get("rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("input_file", ""),
                ]
            )
        elif self.mode == "inpainting":
            row.extend(
                [
                    _to_scalar(kwargs.get("percent_identity_masked", "")),
                    _to_scalar(kwargs.get("percent_identity_unmasked", "")),
                    _to_scalar(kwargs.get("rmsd_inpainted", "")),
                    _to_scalar(metrics.get("_plddt", "")),
                    _to_scalar(metrics.get("_predicted_aligned_error", "")),
                    _to_scalar(metrics.get("_tm_score", "")),
                    _to_scalar(metrics.get("_rmsd", "")),
                    kwargs.get("sequence_length", ""),
                    kwargs.get("num_masked_seq", ""),
                    kwargs.get("num_masked_struc", ""),
                    kwargs.get("input_file", ""),
                ]
            )

        # Write row to CSV
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def write_aggregate_stats(self, aggregate_stats: dict, length: int = None):
        """Write aggregate statistics to a separate summary CSV.

        Args:
            aggregate_stats: Dictionary containing aggregate statistics
            length: Optional length parameter for per-length aggregation
        """
        if length is not None:
            summary_csv_path = self.output_dir / f"{self.mode}_summary_length_{length}_{self.timestamp}.csv"
        else:
            summary_csv_path = self.output_dir / f"{self.mode}_summary_{self.timestamp}.csv"

        headers = ["metric", "value", "count", "mode", "timestamp"]
        if length is not None:
            headers.insert(-1, "length")

        with open(summary_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for metric_name, (value, count) in aggregate_stats.items():
                row = [metric_name, value, count, self.mode]
                if length is not None:
                    row.append(length)
                row.append(datetime.now().isoformat())
                writer.writerow(row)

        logger.info(f"Saved aggregate statistics to: {summary_csv_path}")

    def write_pass_rates(self, pass_rates: dict, length: int = None, threshold: float = 2.0):
        """Write RMSD pass rate statistics to a separate CSV.

        Args:
            pass_rates: Dictionary mapping metric names to (pass_count, total_count, pass_rate) tuples
            length: Optional length parameter for per-length aggregation
            threshold: RMSD threshold used for pass/fail determination
        """
        if length is not None:
            pass_rate_csv_path = self.output_dir / f"{self.mode}_pass_rates_length_{length}_{self.timestamp}.csv"
        else:
            pass_rate_csv_path = self.output_dir / f"{self.mode}_pass_rates_{self.timestamp}.csv"

        headers = [
            "metric",
            "pass_count",
            "total_count",
            "pass_rate_percent",
            "threshold_angstrom",
            "mode",
            "timestamp",
        ]
        if length is not None:
            headers.insert(-2, "length")

        with open(pass_rate_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for metric_name, (pass_count, total_count, pass_rate) in pass_rates.items():
                row = [metric_name, pass_count, total_count, pass_rate, threshold, self.mode]
                if length is not None:
                    row.append(length)
                row.append(datetime.now().isoformat())
                writer.writerow(row)

        logger.info(f"Saved pass rate statistics to: {pass_rate_csv_path}")

    def write_diversity_metrics(self, diversity_metrics: dict, length: int):
        """Write Foldseek diversity metrics to a separate CSV.

        Args:
            diversity_metrics: Dictionary containing diversity metrics:
                - total_structures: Number of structures passing RMSD threshold
                - num_clusters: Number of Foldseek clusters found
                - diversity_percentage: (num_clusters / total_structures) * 100
                - tmscore_threshold: TM-score threshold used for clustering
                - rmsd_threshold: RMSD threshold used for filtering
            length: Sequence length for this diversity analysis
        """
        diversity_csv_path = self.output_dir / f"{self.mode}_diversity_length_{length}_{self.timestamp}.csv"

        headers = [
            "length",
            "total_structures",
            "num_clusters",
            "diversity_percentage",
            "tmscore_threshold",
            "rmsd_threshold",
            "mode",
            "timestamp",
        ]

        with open(diversity_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            row = [
                length,
                diversity_metrics["total_structures"],
                diversity_metrics["num_clusters"],
                diversity_metrics["diversity_percentage"],
                diversity_metrics["tmscore_threshold"],
                diversity_metrics["rmsd_threshold"],
                self.mode,
                datetime.now().isoformat(),
            ]
            writer.writerow(row)

        logger.info(f"Saved diversity metrics to: {diversity_csv_path}")

    def write_sequences(
        self,
        sequences: list[str],
        run_id: str,
        iteration: int = 0,
        original_sequences: list[str] | None = None,
        inpainted_sequences: list[str] | None = None,
        original_inpainted_sequences: list[str] | None = None,
        input_structure: str | list[str] | None = None,
        trial_number: int | None = None,
        chain_ids: list[list[int]] | None = None,
        sequence_type: str | None = None,
        percent_identities: list[float] | None = None,
        masked_positions: list[list[int]] | None = None,
    ):
        """Write generated sequences to CSV for diversity analysis.

        Args:
            sequences: List of generated amino acid sequence strings (full sequences)
            run_id: Unique identifier for this generation run
            iteration: Iteration number (for unconditional mode)
            original_sequences: List of original sequences from input structures (for comparison modes)
            inpainted_sequences: List of ONLY the generated inpainted region sequences (for inpainting mode)
            original_inpainted_sequences: List of ONLY the original inpainted region sequences (for inpainting mode)
            input_structure: Source structure filename(s)
            trial_number: Selected trial number (if using n_trials > 1)
            chain_ids: Chain IDs for each sequence (list of lists)
            sequence_type: Reserved for future use (e.g., "unconditional" for unconditional mode)
            percent_identities: Percent identity with original (one per sequence)
            masked_positions: List of masked position indices (for inpainting)
        """
        timestamp = datetime.now().isoformat()

        with open(self.sequences_csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            for sample_idx, sequence in enumerate(sequences):
                # Handle input_structure (can be single string or list)
                if isinstance(input_structure, list):
                    input_struct = input_structure[sample_idx] if sample_idx < len(input_structure) else ""
                else:
                    input_struct = input_structure or ""

                # Handle original_sequences
                original_seq = ""
                if original_sequences and sample_idx < len(original_sequences):
                    original_seq = original_sequences[sample_idx]

                # Handle inpainted_sequences (generated)
                inpainted_seq = ""
                if inpainted_sequences and sample_idx < len(inpainted_sequences):
                    inpainted_seq = inpainted_sequences[sample_idx]

                # Handle original_inpainted_sequences
                original_inpainted_seq = ""
                if original_inpainted_sequences and sample_idx < len(original_inpainted_sequences):
                    original_inpainted_seq = original_inpainted_sequences[sample_idx]

                # Handle chain_ids
                chains = ""
                num_chains = 0
                if chain_ids and sample_idx < len(chain_ids):
                    chains = ",".join(map(str, chain_ids[sample_idx]))
                    num_chains = len(set(chain_ids[sample_idx]))

                # Handle percent_identities
                percent_id = ""
                if percent_identities and sample_idx < len(percent_identities):
                    percent_id = f"{percent_identities[sample_idx]:.2f}"

                # Handle masked_positions
                masked_pos = ""
                if masked_positions and sample_idx < len(masked_positions):
                    masked_pos = ",".join(map(str, masked_positions[sample_idx]))

                writer.writerow(
                    [
                        run_id,
                        iteration,
                        sample_idx,
                        sequence,
                        original_seq,
                        inpainted_seq,
                        original_inpainted_seq,
                        len(sequence),
                        self.mode,
                        input_struct,
                        num_chains,
                        chains,
                        trial_number or "",
                        percent_id,
                        masked_pos,
                        sequence_type or "",
                        timestamp,
                    ]
                )


def calculate_aggregate_stats(metric_lists: dict) -> dict:
    """Calculate aggregate statistics from lists of metrics.

    Args:
        metric_lists: Dictionary mapping metric names to lists of values

    Returns:
        Dictionary mapping metric names to (average, count) tuples
    """
    aggregate_stats = {}

    for metric_name, values in metric_lists.items():
        if values:
            # Convert tensors to scalars and filter out invalid values
            valid_values = []
            for v in values:
                # Convert tensor to scalar if needed
                if hasattr(v, "item"):
                    scalar_v = v.item()
                elif hasattr(v, "cpu"):
                    scalar_v = v.cpu().item()
                else:
                    scalar_v = float(v)

                # Filter out invalid values (inf, nan)
                if scalar_v != float("inf") and not (isinstance(scalar_v, float) and scalar_v != scalar_v):
                    valid_values.append(scalar_v)

            if valid_values:
                avg_value = sum(valid_values) / len(valid_values)
                aggregate_stats[metric_name] = (avg_value, len(valid_values))
            else:
                aggregate_stats[metric_name] = (0.0, 0)
        else:
            aggregate_stats[metric_name] = (0.0, 0)

    return aggregate_stats


def align_and_compute_rmsd(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: torch.Tensor | None = None,
    return_aligned: bool = False,
    device: torch.device = None,
) -> tuple[torch.Tensor | None, float] | float:
    """Align two structures using Kabsch algorithm and compute RMSD.

    This function performs optimal superposition (Kabsch alignment) on CA atoms
    and computes RMSD on the aligned structures. Optionally can compute RMSD
    only on a masked region.

    Args:
        coords1: First structure coordinates, shape (N, 3, 3) - [N, CA, C] atoms
        coords2: Second structure (reference), shape (N, 3, 3)
        mask: Optional mask for which positions to use, shape (N,)
              If None, uses all positions. 1 = use, 0 = ignore
        return_aligned: If True, return aligned coords1; if False, return None
        device: torch device (inferred from coords1 if not provided)

    Returns:
        If return_aligned=True: (aligned_coords1, rmsd)
        If return_aligned=False: rmsd only

        - aligned_coords1: Aligned version of coords1 (or None)
        - rmsd: RMSD on CA atoms after alignment (float)

    Note:
        - Alignment is performed on CA atoms (index 1)
        - RMSD is computed on CA atoms after alignment
        - If mask is provided, only masked positions are used for alignment and RMSD
    """
    if device is None:
        device = coords1.device

    N = coords1.shape[0]

    # Create mask if not provided (use all positions)
    if mask is None:
        mask = torch.ones(N, device=device)
    else:
        mask = mask.float()

    # Check if there are positions to align
    num_positions = mask.sum().item()
    if num_positions == 0:
        logger.warning("No positions to align (mask is all zeros)")
        if return_aligned:
            return coords1, 0.0
        else:
            return 0.0

    # Extract CA atoms (index 1)
    coords1_ca = coords1[:, 1, :]  # Shape: (N, 3)
    coords2_ca = coords2[:, 1, :]  # Shape: (N, 3)

    # Add batch dimension for kabsch_torch_batched
    coords1_ca_batch = coords1_ca.unsqueeze(0)  # Shape: (1, N, 3)
    coords2_ca_batch = coords2_ca.unsqueeze(0)  # Shape: (1, N, 3)
    mask_batch = mask.unsqueeze(0)  # Shape: (1, N)

    # Apply Kabsch alignment - get rotation R and translation t
    coords1_ca_aligned_batch, (R, t) = kabsch_torch_batched(
        P=coords1_ca_batch,  # Source: coords1 CA atoms
        Q=coords2_ca_batch,  # Target: coords2 CA atoms (reference)
        mask=mask_batch,  # Positions to use for alignment
        return_transform=True,  # Need R and t to transform all atoms
    )

    coords1_ca_aligned = coords1_ca_aligned_batch.squeeze(0)  # Shape: (N, 3)

    # Calculate RMSD on masked region CA atoms after alignment
    masked_coords1_ca = coords1_ca_aligned[mask.bool()]  # (M, 3)
    masked_coords2_ca = coords2_ca[mask.bool()]  # (M, 3)

    rmsd = torch.sqrt(torch.mean((masked_coords1_ca - masked_coords2_ca) ** 2)).item()

    # If we need to return aligned full structure (all atoms, not just CA)
    if return_aligned:
        # Apply the same transformation (R, t) to ALL atoms (N, CA, C)
        R = R.squeeze(0)  # Shape: (3, 3)
        t = t.squeeze(0)  # Shape: (3,)

        # Compute centroids used by Kabsch
        centroid_coords1_ca = torch.sum(coords1_ca * mask.unsqueeze(-1), dim=0) / mask.sum()
        centroid_coords2_ca = torch.sum(coords2_ca * mask.unsqueeze(-1), dim=0) / mask.sum()

        # Center all atoms around the CA centroid
        coords1_centered = coords1 - centroid_coords1_ca.unsqueeze(0).unsqueeze(0)  # (N, 3, 3)

        # Apply rotation to all atoms
        coords1_rotated = torch.matmul(
            coords1_centered.reshape(N * 3, 3),  # Flatten to (N*3, 3)
            R.transpose(0, 1),  # Rotate
        ).reshape(N, 3, 3)  # Reshape back to (N, 3, 3)

        # Translate to target centroid
        coords1_aligned = coords1_rotated + centroid_coords2_ca.unsqueeze(0).unsqueeze(0)

        return coords1_aligned, rmsd
    else:
        return rmsd


def align_and_compute_rmsd_inpainted(
    gen_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    inpainting_mask_seq: torch.Tensor | None,
    inpainting_mask_struc: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Align generated structure to ESMFold prediction and compute RMSD on inpainted region.

    This is a specialized wrapper around align_and_compute_rmsd for inpainting mode.
    It creates the union of sequence and structure inpainting masks and uses that for alignment.

    Args:
        gen_coords: Generated structure coordinates, shape (N, 3, 3) - [N, CA, C] atoms
        pred_coords: ESMFold prediction coordinates, shape (N, 3, 3) - [N, CA, C] atoms
        inpainting_mask_seq: Sequence inpainting mask, shape (N,) or None
        inpainting_mask_struc: Structure inpainting mask, shape (N,) or None
        device: torch device

    Returns:
        gen_coords_aligned: Aligned generated structure, shape (N, 3, 3)
        rmsd_inpainted: RMSD on inpainted region CA atoms after alignment (float)
    """
    N = gen_coords.shape[0]

    # Create union mask for all inpainted regions
    if inpainting_mask_seq is None:
        inpaint_mask_seq = torch.zeros(N, device=device)
    else:
        inpaint_mask_seq = inpainting_mask_seq

    if inpainting_mask_struc is None:
        inpaint_mask_struc = torch.zeros(N, device=device)
    else:
        inpaint_mask_struc = inpainting_mask_struc

    # Union: any position that was inpainted in sequence OR structure
    inpaint_mask_union = (inpaint_mask_seq.bool() | inpaint_mask_struc.bool()).float()

    # Check if there are any inpainted positions
    num_inpainted = inpaint_mask_union.sum().item()
    if num_inpainted == 0:
        logger.warning("No inpainted positions found, skipping alignment")
        return gen_coords, 0.0

    # Use generalized alignment function
    gen_coords_aligned, rmsd_inpainted = align_and_compute_rmsd(
        coords1=gen_coords,
        coords2=pred_coords,
        mask=inpaint_mask_union,
        return_aligned=True,
        device=device,
    )

    return gen_coords_aligned, rmsd_inpainted
