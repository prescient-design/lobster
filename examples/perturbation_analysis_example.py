#!/usr/bin/env python3
"""
This example shows how to set up the callback to analyze model robustness
through sequence perturbations (shuffling and mutations) for different modalities.

Credits: Josh Southern for the original perturbation analysis notebook.
"""

import lightning as L
import torch
import numpy as np
from lobster.callbacks import PerturbationAnalysisCallback
from lobster.constants import Modality

# Example sequences for different modalities
EXAMPLE_SEQUENCES = {
    Modality.AMINO_ACID: [
        "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS",
    ],
    Modality.SMILES: [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
    ],
    Modality.NUCLEOTIDE: [
        "ATCGATCG",
        "GCTAGCTA",
    ],
}


class DummyModel(L.LightningModule):
    """Dummy model for testing perturbation analysis callback."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def embed_sequences(self, sequences: list[str], modality: str = "amino_acid", aggregate: bool = True) -> torch.Tensor: # noqa: F821
        """Dummy embed_sequences method that returns random embeddings."""
        batch_size = len(sequences)
        
        return torch.randn(batch_size, self.embedding_dim)


def analyze_modality(model, sequences, modality, output_dir, mutation_tokens=None, num_shuffles=5):
    """Helper function to analyze a specific modality."""
    print(f"\n=== Analyzing {modality.upper()} sequences ===")
    
    # Create the perturbation analysis callback
    callback_kwargs = {
        "output_dir": f"{output_dir}/{modality}",
        "sequences": sequences,
        "num_shuffles": num_shuffles,
        "random_state": 42,
        "modality": modality,
    }
    
    # Add mutation_tokens for modalities that require them
    if mutation_tokens is not None:
        callback_kwargs["mutation_tokens"] = mutation_tokens
    
    callback = PerturbationAnalysisCallback(**callback_kwargs)
    
    # Run the evaluation
    print(f"Running perturbation analysis for {modality}...")
    metrics = callback.evaluate(
        module=model,
        save_heatmap=True,
        output_dir=f"{output_dir}/{modality}"  
    )
    
    print(f"Results for {modality}:")
    print(f"  Average shuffling distance: {metrics['avg_shuffling_distance']:.6f}")
    print(f"  Average mutation distance: {metrics['avg_mutation_distance']:.6f}")
    print(f"  Distance ratio (shuffle/mutation): {metrics['distance_ratio']:.6f}")
    
    return metrics


def main():
    """Main function demonstrating the PerturbationAnalysisCallback usage for different modalities."""
    
    # Create a dummy model for testing
    print("Creating dummy model...")
    model = DummyModel(embedding_dim=128)
    
    output_dir = "perturbation_analysis_results"
    
    # Analyze each modality
    all_metrics = {}
    
    # AMINO_ACID and NUCLEOTIDE have default mutation tokens
    for modality in [Modality.AMINO_ACID, Modality.NUCLEOTIDE]:
        try:
            sequences = EXAMPLE_SEQUENCES[modality]
            metrics = analyze_modality(model, sequences, modality, output_dir)
            all_metrics[modality] = metrics
        except Exception as e:
            print(f"Error analyzing {modality}: {e}")
    
    # SMILES requires explicit mutation_tokens
    try:
        smiles_sequences = EXAMPLE_SEQUENCES[Modality.SMILES]
        smiles_tokens = list("CHNOSPFIBrCl()[]=#@+-.1234567890")
        metrics = analyze_modality(
            model, 
            smiles_sequences, 
            Modality.SMILES, 
            output_dir, 
            mutation_tokens=smiles_tokens
        )
        all_metrics[Modality.SMILES] = metrics
    except Exception as e:
        print(f"Error analyzing {Modality.SMILES}: {e}")
    
    # Demonstrate error for 3D_COORDINATES without mutation_tokens
    print(f"\n=== Demonstrating error for {Modality.COORDINATES_3D} ===")
    try:
        # This should raise an error
        analyze_modality(model, ["dummy_sequence"], Modality.COORDINATES_3D, output_dir)
    except ValueError as e:
        print(f"Expected error for {Modality.COORDINATES_3D}: {e}")
    
    # Compare results across modalities
    print(f"\n=== Cross-Modality Comparison ===")
    for modality, metrics in all_metrics.items():
        print(f"{modality.upper()}:")
        print(f"  Shuffling distance: {metrics['avg_shuffling_distance']:.6f}")
        print(f"  Mutation distance: {metrics['avg_mutation_distance']:.6f}")
        print(f"  Distance ratio: {metrics['distance_ratio']:.6f}")
    
    print(f"\nResults saved to: {output_dir}/")
    print("Check the subdirectories for modality-specific heatmap visualizations!")


if __name__ == "__main__":
    main() 