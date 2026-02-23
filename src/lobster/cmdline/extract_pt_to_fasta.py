#!/usr/bin/env python3
"""
Extract sequences from .pt files and save to FASTA format.
"""

import torch
from pathlib import Path
import glob
from tqdm import tqdm

# Standard amino acid mapping (same as lobster)
RESTYPES = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X"]


def extract_sequence_from_pt(pt_file):
    """Extract amino acid sequence from .pt file."""
    try:
        # Load .pt file
        data = torch.load(pt_file, map_location="cpu")

        # Try to find sequence - check multiple possible keys
        sequence_tensor = None
        if "sequence" in data:
            sequence_tensor = data["sequence"]
        elif "seq" in data:
            sequence_tensor = data["seq"]
        elif "aatype" in data:
            sequence_tensor = data["aatype"]
        else:
            print(f"No sequence found in {pt_file}. Available keys: {list(data.keys())}")
            return None

        # Convert tensor to numpy if needed
        if isinstance(sequence_tensor, torch.Tensor):
            sequence_indices = sequence_tensor.cpu().numpy()
        else:
            sequence_indices = sequence_tensor

        # Flatten if multidimensional
        if sequence_indices.ndim > 1:
            sequence_indices = sequence_indices.flatten()

        # Convert integer codes to amino acids
        sequence = "".join([RESTYPES[int(i)] if 0 <= int(i) < len(RESTYPES) else "X" for i in sequence_indices])

        return sequence

    except Exception as e:
        print(f"Error loading {pt_file}: {e}")
        return None


def extract_all_sequences_to_fasta(input_dir, output_fasta, truncate_at_x=False):
    """Extract all sequences from .pt files to FASTA.

    Args:
        input_dir: Directory containing .pt files
        output_fasta: Output FASTA file path
        truncate_at_x: If True, truncate sequences at first X (unknown residue)
    """

    input_path = Path(input_dir)

    # Find all .pt files
    pt_files = sorted(glob.glob(str(input_path / "*.pt")))

    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return

    print(f"Found {len(pt_files)} .pt files")
    print(f"Output FASTA: {output_fasta}")
    if truncate_at_x:
        print("Mode: Truncate at first X (unknown residue)")
    print()

    sequences_written = 0
    errors = 0
    total_residues = 0
    truncated_count = 0

    with open(output_fasta, "w") as fasta_out:
        for pt_file in tqdm(pt_files, desc="Extracting sequences"):
            # Get structure name from filename (without .pt)
            structure_name = Path(pt_file).stem

            # Remove common suffixes
            for suffix in ["_processed", "_cleaned"]:
                if structure_name.endswith(suffix):
                    structure_name = structure_name[: -len(suffix)]

            # Extract sequence
            sequence = extract_sequence_from_pt(pt_file)

            if sequence:
                # Optionally truncate at first X
                if truncate_at_x and "X" in sequence:
                    first_x = sequence.index("X")
                    if first_x > 0:  # Only truncate if there's something before X
                        sequence = sequence[:first_x]
                        truncated_count += 1

                # Skip empty sequences
                if len(sequence) == 0:
                    errors += 1
                    continue

                # Write to FASTA format
                fasta_out.write(f">{structure_name}\n")

                # Write sequence in 80 character lines (standard FASTA)
                for i in range(0, len(sequence), 80):
                    fasta_out.write(sequence[i : i + 80] + "\n")

                sequences_written += 1
                total_residues += len(sequence)
            else:
                errors += 1

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully extracted: {sequences_written} sequences")
    print(f"✗ Failed to process: {errors} files")
    if truncate_at_x and truncated_count > 0:
        print(f"  Sequences truncated: {truncated_count}")
    print(f"  Total residues: {total_residues:,}")

    if sequences_written > 0:
        print(f"  Average length: {total_residues / sequences_written:.1f} residues")

    print(f"\nOutput saved to: {output_fasta}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract sequences from .pt files to FASTA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all sequences (default)
  python extract_pt_to_fasta.py

  # Specify input and output
  python extract_pt_to_fasta.py --input-dir /path/to/pt/files --output sequences.fasta

  # Truncate at first X (unknown residue)
  python extract_pt_to_fasta.py --truncate-at-x
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data2/lisanzas/multi_flow_data/test_set_filtered_pt",
        help="Directory containing .pt files (default: /data2/lisanzas/multi_flow_data/test_set_filtered_pt)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="test_set_filtered_sequences.fasta",
        help="Output FASTA file path (default: test_set_filtered_sequences.fasta)",
    )

    parser.add_argument("--truncate-at-x", action="store_true", help="Truncate sequences at first X (unknown residue)")

    args = parser.parse_args()

    print("=" * 60)
    print("PT to FASTA Extractor")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output FASTA: {args.output}\n")

    extract_all_sequences_to_fasta(args.input_dir, args.output, args.truncate_at_x)
