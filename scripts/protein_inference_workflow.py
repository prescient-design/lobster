#!/usr/bin/env python3
"""
Automated Protein Inference and Intervention Workflow

This script provides a command-line interface for:
1. Loading protein datasets
2. Running inference with various models
3. Performing concept-based interventions
4. Computing and exporting metrics

Usage:
    # Inference
    python scripts/protein_inference_workflow.py inference \
        --fasta test_data/query.fasta \
        --model pmlm \
        --checkpoint asalam91/lobster_24M \
        --output outputs/inference_results

    # Intervention
    python scripts/protein_inference_workflow.py intervene \
        --sequence "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF" \
        --concept gravy \
        --direction negative \
        --edits 5 \
        --iterations 3 \
        --output outputs/intervention_results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    import Levenshtein
except ImportError:
    print("Warning: python-Levenshtein not installed. Edit distance calculations will be unavailable.")
    Levenshtein = None

from lobster.model import (
    LobsterPMLM,
    LobsterCBMPMLM,
    LobsterPCLM,
)


def load_sequences(source: str, source_type: str = "fasta") -> tuple[list[str], list[str]]:
    """Load sequences from various sources.

    Parameters
    ----------
    source : str
        Path to FASTA file or direct sequence string
    source_type : str
        Type of source: 'fasta' or 'direct'

    Returns
    -------
    tuple[list[str], list[str]]
        Sequences and their IDs
    """
    sequences = []
    seq_ids = []

    if source_type == "fasta":
        if not os.path.exists(source):
            raise FileNotFoundError(f"FASTA file not found: {source}")

        for record in SeqIO.parse(source, "fasta"):
            sequences.append(str(record.seq))
            seq_ids.append(record.id)

        print(f"✓ Loaded {len(sequences)} sequences from {source}")

    elif source_type == "direct":
        seq = source.strip().replace(" ", "").replace("\n", "")
        sequences = [seq]
        seq_ids = ["sequence_1"]
        print(f"✓ Loaded sequence from direct input (length: {len(seq)})")

    return sequences, seq_ids


def compute_bio_metrics(sequences: list[str], seq_ids: list[str]) -> pd.DataFrame:
    """Compute biological metrics for protein sequences.

    Parameters
    ----------
    sequences : list[str]
        Protein sequences
    seq_ids : list[str]
        Sequence identifiers

    Returns
    -------
    pd.DataFrame
        DataFrame with biological metrics
    """
    metrics_data = []

    for seq_id, seq in zip(seq_ids, sequences):
        try:
            analyzer = ProteinAnalysis(seq)
            ss = analyzer.secondary_structure_fraction()

            metrics = {
                "seq_id": seq_id,
                "length": len(seq),
                "molecular_weight": analyzer.molecular_weight(),
                "aromaticity": analyzer.aromaticity(),
                "instability_index": analyzer.instability_index(),
                "isoelectric_point": analyzer.isoelectric_point(),
                "gravy": analyzer.gravy(),
                "helix_fraction": ss[0],
                "turn_fraction": ss[1],
                "sheet_fraction": ss[2],
            }
            metrics_data.append(metrics)
        except Exception as e:
            print(f"Warning: Could not compute metrics for {seq_id}: {e}")

    return pd.DataFrame(metrics_data)


def run_inference(
    sequences: list[str],
    seq_ids: list[str],
    model_type: str,
    checkpoint: str,
    output_dir: str,
    pooling: str = "both",
    device: str = "auto",
) -> dict[str, Any]:
    """Run inference on protein sequences.

    Parameters
    ----------
    sequences : list[str]
        Protein sequences
    seq_ids : list[str]
        Sequence identifiers
    model_type : str
        Model type: 'pmlm', 'cbm', 'pclm'
    checkpoint : str
        Model checkpoint identifier
    output_dir : str
        Output directory
    pooling : str
        Pooling method: 'cls', 'mean', 'max', 'both'
    device : str
        Device to use: 'auto', 'cuda', 'cpu'

    Returns
    -------
    dict[str, Any]
        Inference results
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model: {model_type} ({checkpoint})")
    print(f"Device: {device}")

    # Load model
    if model_type == "pmlm":
        model = LobsterPMLM(checkpoint).to(device)
    elif model_type == "cbm":
        model = LobsterCBMPMLM(checkpoint).to(device)
    elif model_type == "pclm":
        model = LobsterPCLM(checkpoint).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Run inference
    print(f"\nRunning inference on {len(sequences)} sequences...")
    results = {}

    with torch.inference_mode():
        # Get latent representations
        latents = model.sequences_to_latents(sequences)[-1]
        print(f"✓ Generated latent representations: {latents.shape}")

        # Apply pooling
        if pooling in ["cls", "both"]:
            cls_embeddings = latents[:, 0, :]
            results["cls_embeddings"] = cls_embeddings.cpu().numpy()
            print(f"✓ CLS token embeddings: {cls_embeddings.shape}")

        if pooling in ["mean", "both"]:
            mean_embeddings = torch.mean(latents, dim=1)
            results["mean_embeddings"] = mean_embeddings.cpu().numpy()
            print(f"✓ Mean pooled embeddings: {mean_embeddings.shape}")

        if pooling == "max":
            max_embeddings = torch.max(latents, dim=1)[0]
            results["max_embeddings"] = max_embeddings.cpu().numpy()
            print(f"✓ Max pooled embeddings: {max_embeddings.shape}")

        # For CBM models, also get concepts
        if model_type == "cbm":
            concepts = model.sequences_to_concepts(sequences)[-1]
            results["concepts"] = concepts.cpu().numpy()
            print(f"✓ Extracted concepts: {concepts.shape}")

    # Compute biological metrics
    print("\nComputing biological metrics...")
    metrics_df = compute_bio_metrics(sequences, seq_ids)
    results["metrics"] = metrics_df

    # Export results
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting results to: {output_dir}")

    # Save embeddings
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            np.save(f"{output_dir}/{key}.npy", value)
            print(f"✓ Saved {key}.npy")

    # Save metrics
    if not metrics_df.empty:
        metrics_df.to_csv(f"{output_dir}/sequence_metrics.csv", index=False)
        print(f"✓ Saved sequence_metrics.csv")

    # Save sequence IDs
    with open(f"{output_dir}/sequence_ids.txt", "w") as f:
        f.write("\n".join(seq_ids))
    print(f"✓ Saved sequence_ids.txt")

    # Save summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("Inference Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Number of Sequences: {len(sequences)}\n")
        f.write(f"Pooling Method: {pooling}\n")
        f.write(f"Device: {device}\n\n")
        f.write("Results:\n")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                f.write(f"  {key}: {value.shape}\n")
    print(f"✓ Saved summary.txt")

    print("\n✓ Inference complete!")
    return results


def run_intervention(
    sequence: str,
    seq_id: str,
    concept: str,
    direction: str,
    edits: int,
    iterations: int,
    checkpoint: str,
    output_dir: str,
    device: str = "auto",
) -> dict[str, Any]:
    """Run concept-based intervention on a protein sequence.

    Parameters
    ----------
    sequence : str
        Protein sequence
    seq_id : str
        Sequence identifier
    concept : str
        Target concept
    direction : str
        Intervention direction: 'positive' or 'negative'
    edits : int
        Number of edits per iteration
    iterations : int
        Number of iterations
    checkpoint : str
        Model checkpoint identifier
    output_dir : str
        Output directory
    device : str
        Device to use: 'auto', 'cuda', 'cpu'

    Returns
    -------
    dict[str, Any]
        Intervention results
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading Concept Bottleneck Model: {checkpoint}")
    print(f"Device: {device}")

    # Load model
    model = LobsterCBMPMLM(checkpoint).to(device)
    model.eval()

    concept_names = model.list_supported_concept()
    if concept not in concept_names:
        print(f"\n❌ Concept '{concept}' not found!")
        print(f"Available concepts: {', '.join(concept_names)}")
        return {}

    print(f"\nIntervention Configuration:")
    print(f"  Sequence: {seq_id} ({len(sequence)} aa)")
    print(f"  Concept: {concept}")
    print(f"  Direction: {direction}")
    print(f"  Edits per iteration: {edits}")
    print(f"  Total iterations: {iterations}")

    # Get original concepts
    with torch.inference_mode():
        original_concepts = model.sequences_to_concepts([sequence])[-1][0]

    concept_idx = concept_names.index(concept)
    original_value = original_concepts[concept_idx].item()
    print(f"  Original concept value: {original_value:.4f}")

    # Run intervention
    print(f"\nRunning intervention...")
    intervention_results = []
    current_seq = sequence

    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}...")

        [new_seq] = model.intervene_on_sequences(
            [current_seq],
            concept,
            edits=edits,
            intervention_type=direction
        )

        # Compute metrics
        edit_distance = Levenshtein.distance(sequence, new_seq) if Levenshtein else -1

        # Get new concepts
        with torch.inference_mode():
            new_concepts = model.sequences_to_concepts([new_seq])[-1][0]

        new_value = new_concepts[concept_idx].item()
        concept_change = new_value - original_value

        intervention_results.append({
            "iteration": iteration + 1,
            "sequence": new_seq,
            "edit_distance": edit_distance,
            "concept_value": new_value,
            "concept_change": concept_change,
        })

        print(f"  Edit distance from original: {edit_distance}")
        print(f"  Concept value: {new_value:.4f}")
        print(f"  Concept change: {concept_change:+.4f}")

        current_seq = new_seq

    # Compute biological metrics
    print(f"\nComputing biological metrics...")
    original_metrics = compute_bio_metrics([sequence], [f"{seq_id}_original"])
    modified_metrics = compute_bio_metrics(
        [intervention_results[-1]["sequence"]],
        [f"{seq_id}_modified"]
    )

    # Export results
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting results to: {output_dir}")

    # Save sequences
    with open(f"{output_dir}/sequences.txt", "w") as f:
        f.write(f"Original Sequence:\n{sequence}\n\n")
        f.write(f"Modified Sequence:\n{intervention_results[-1]['sequence']}\n")
    print(f"✓ Saved sequences.txt")

    # Save metrics comparison
    comparison_df = pd.concat([original_metrics, modified_metrics], ignore_index=True)
    comparison_df.to_csv(f"{output_dir}/metrics_comparison.csv", index=False)
    print(f"✓ Saved metrics_comparison.csv")

    # Save intervention trajectory
    trajectory_df = pd.DataFrame(intervention_results)
    trajectory_df.to_csv(f"{output_dir}/intervention_trajectory.csv", index=False)
    print(f"✓ Saved intervention_trajectory.csv")

    # Save summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("Intervention Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original Sequence ID: {seq_id}\n")
        f.write(f"Target Concept: {concept}\n")
        f.write(f"Intervention Direction: {direction}\n")
        f.write(f"Edits per Iteration: {edits}\n")
        f.write(f"Total Iterations: {iterations}\n\n")
        f.write("Results:\n")
        f.write(f"  Total Edit Distance: {intervention_results[-1]['edit_distance']}\n")
        f.write(f"  Concept Change: {intervention_results[-1]['concept_change']:+.4f}\n")
        if intervention_results[-1]['edit_distance'] > 0:
            identity = 100 * (1 - intervention_results[-1]['edit_distance'] / len(sequence))
            f.write(f"  Sequence Identity: {identity:.2f}%\n")
    print(f"✓ Saved summary.txt")

    print("\n✓ Intervention complete!")
    return {
        "results": intervention_results,
        "original_metrics": original_metrics,
        "modified_metrics": modified_metrics,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated Protein Inference and Intervention Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on protein sequences")
    inference_parser.add_argument("--fasta", type=str, help="Path to FASTA file")
    inference_parser.add_argument("--sequence", type=str, help="Direct sequence input")
    inference_parser.add_argument(
        "--model",
        type=str,
        choices=["pmlm", "cbm", "pclm"],
        default="pmlm",
        help="Model type"
    )
    inference_parser.add_argument(
        "--checkpoint",
        type=str,
        default="asalam91/lobster_24M",
        help="Model checkpoint"
    )
    inference_parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "mean", "max", "both"],
        default="both",
        help="Pooling method"
    )
    inference_parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference_results",
        help="Output directory"
    )
    inference_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use"
    )

    # Intervention command
    intervene_parser = subparsers.add_parser("intervene", help="Run concept-based intervention")
    intervene_parser.add_argument("--fasta", type=str, help="Path to FASTA file (first sequence)")
    intervene_parser.add_argument("--sequence", type=str, help="Direct sequence input")
    intervene_parser.add_argument(
        "--concept",
        type=str,
        default="gravy",
        help="Target concept"
    )
    intervene_parser.add_argument(
        "--direction",
        type=str,
        choices=["positive", "negative"],
        default="negative",
        help="Intervention direction"
    )
    intervene_parser.add_argument(
        "--edits",
        type=int,
        default=5,
        help="Number of edits per iteration"
    )
    intervene_parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations"
    )
    intervene_parser.add_argument(
        "--checkpoint",
        type=str,
        default="asalam91/cb_lobster_24M",
        help="Model checkpoint"
    )
    intervene_parser.add_argument(
        "--output",
        type=str,
        default="outputs/intervention_results",
        help="Output directory"
    )
    intervene_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use"
    )

    # List concepts command
    list_parser = subparsers.add_parser("list-concepts", help="List available concepts")
    list_parser.add_argument(
        "--checkpoint",
        type=str,
        default="asalam91/cb_lobster_24M",
        help="Model checkpoint"
    )

    args = parser.parse_args()

    if args.command == "inference":
        # Load sequences
        if args.fasta:
            sequences, seq_ids = load_sequences(args.fasta, "fasta")
        elif args.sequence:
            sequences, seq_ids = load_sequences(args.sequence, "direct")
        else:
            print("Error: Either --fasta or --sequence must be provided")
            sys.exit(1)

        # Run inference
        run_inference(
            sequences=sequences,
            seq_ids=seq_ids,
            model_type=args.model,
            checkpoint=args.checkpoint,
            output_dir=args.output,
            pooling=args.pooling,
            device=args.device,
        )

    elif args.command == "intervene":
        # Load sequence
        if args.fasta:
            sequences, seq_ids = load_sequences(args.fasta, "fasta")
            sequence = sequences[0]
            seq_id = seq_ids[0]
        elif args.sequence:
            sequences, seq_ids = load_sequences(args.sequence, "direct")
            sequence = sequences[0]
            seq_id = seq_ids[0]
        else:
            print("Error: Either --fasta or --sequence must be provided")
            sys.exit(1)

        # Run intervention
        run_intervention(
            sequence=sequence,
            seq_id=seq_id,
            concept=args.concept,
            direction=args.direction,
            edits=args.edits,
            iterations=args.iterations,
            checkpoint=args.checkpoint,
            output_dir=args.output,
            device=args.device,
        )

    elif args.command == "list-concepts":
        # Load model
        print(f"Loading model: {args.checkpoint}")
        model = LobsterCBMPMLM(args.checkpoint)

        # List concepts
        concepts = model.list_supported_concept()
        print(f"\nAvailable Concepts ({len(concepts)} total):")
        print("=" * 60)
        for i, concept in enumerate(concepts, 1):
            print(f"{i:3d}. {concept}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
