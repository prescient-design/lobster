#!/usr/bin/env python3
"""
Example script demonstrating how to use the perturbation score metrics.

This script shows how to use the PerturbationScore class to analyze model robustness 
through sequence perturbations.
"""

import logging
from pathlib import Path

import lightning as L
import torch
from upath import UPath

from lobster.constants import Modality
from lobster.metrics import PerturbationScore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_amino_acid_perturbation_score():
    """Example using the PerturbationScore class for amino acid sequences."""
    logger.info("=== Amino Acid Perturbation Score Example ===")
    
    # Example protein sequence
    protein_sequence = "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS"
    
    # Mock model (replace with your actual model)
    class MockModel(L.LightningModule):
        def embed_sequences(self, sequences, modality, aggregate=True):
            # Mock embedding - replace with actual model
            return torch.randn(len(sequences), 128)
    
    model = MockModel()
    
    # Create embedding function
    def embedding_function(sequences, modality):
        model.eval()
        with torch.no_grad():
            return model.embed_sequences(sequences, modality=modality, aggregate=True)
    
    # Create PerturbationScore metric
    metric = PerturbationScore(
        sequence=protein_sequence,
        embedding_function=embedding_function,
        modality=Modality.AMINO_ACID,
        num_shuffles=5,  # Reduced for example
        random_state=42,
        save_heatmap=True,
        output_file=UPath("outputs/amino_acid_perturbation_score.png"),
    )
    
    # Run the analysis
    metric.update()
    metrics = metric.compute()
    
    logger.info(f"Amino acid perturbation score results: {metrics}")


def example_nucleotide_perturbation_score():
    """Example using the PerturbationScore class for nucleotide sequences."""
    logger.info("=== Nucleotide Perturbation Score Example ===")
    
    # Example nucleotide sequence
    nucleotide_sequence = "ATCGATCGATCGATCGATCG"
    
    # Mock model (replace with your actual model)
    class MockModel(L.LightningModule):
        def embed_sequences(self, sequences, modality, aggregate=True):
            # Mock embedding - replace with actual model
            return torch.randn(len(sequences), 128)
    
    model = MockModel()
    
    # Create embedding function
    def embedding_function(sequences, modality):
        model.eval()
        with torch.no_grad():
            return model.embed_sequences(sequences, modality=modality, aggregate=True)
    
    # Create PerturbationScore metric
    metric = PerturbationScore(
        sequence=nucleotide_sequence,
        embedding_function=embedding_function,
        modality=Modality.NUCLEOTIDE,
        num_shuffles=5,  # Reduced for example
        random_state=42,
        save_heatmap=True,
        output_file=UPath("outputs/nucleotide_perturbation_score.png"),
    )
    
    # Run the analysis
    metric.update()
    metrics = metric.compute()
    
    logger.info(f"Nucleotide perturbation score results: {metrics}")


def example_smiles_perturbation_score():
    """Example of analyzing SMILES sequences."""
    logger.info("=== SMILES Perturbation Score Example ===")
    
    # Example SMILES sequence
    smiles_sequence = "CCO"  # Ethanol
    
    # Mock model (replace with your actual model)
    class MockModel(L.LightningModule):
        def embed_sequences(self, sequences, modality, aggregate=True):
            # Mock embedding - replace with actual model
            return torch.randn(len(sequences), 128)
    
    model = MockModel()
    
    # Create embedding function
    def embedding_function(sequences, modality):
        model.eval()
        with torch.no_grad():
            return model.embed_sequences(sequences, modality=modality, aggregate=True)
    
    # Create PerturbationScore metric
    metric = PerturbationScore(
        sequence=smiles_sequence,
        embedding_function=embedding_function,
        modality=Modality.SMILES,
        num_shuffles=5,  # Reduced for example
        random_state=42,
        save_heatmap=True,
        output_file=UPath("outputs/smiles_perturbation_score.png"),
    )
    
    # Run the analysis
    metric.update()
    metrics = metric.compute()
    
    logger.info(f"SMILES perturbation score results: {metrics}")


def example_custom_embedding_function():
    """Example using a custom embedding function."""
    logger.info("=== Custom Embedding Function Example ===")
    
    # Custom sequence
    custom_sequence = "ABCDEFGHIJKL"
    
    # Custom mutation tokens
    custom_tokens = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Custom embedding function (could be from any model/library)
    def custom_embedding_function(sequences, modality):
        # This could be a call to any embedding model
        # For example: return some_embedding_model.encode(sequences)
        batch_size = len(sequences)
        # Mock embedding - replace with actual embedding function
        return torch.randn(batch_size, 128)
    
    # Create PerturbationScore metric with custom embedding function
    metric = PerturbationScore(
        sequence=custom_sequence,
        embedding_function=custom_embedding_function,
        modality="CUSTOM",  # Custom modality
        num_shuffles=10,
        mutation_tokens=custom_tokens,
        random_state=123,
        save_heatmap=False,  # Don't save heatmap
    )
    
    # Run the analysis
    metric.update()
    metrics = metric.compute()
    
    logger.info(f"Custom perturbation score results: {metrics}")


def main():
    """Run all perturbation score examples."""
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Run examples
    example_amino_acid_perturbation_score()
    print()
    
    example_nucleotide_perturbation_score()
    print()
    
    example_smiles_perturbation_score()
    print()
    
    example_custom_embedding_function()
    print()
    
    logger.info("All examples completed! Check the 'outputs' directory for heatmap images.")
    logger.info("The PerturbationScore class provides comprehensive analysis of model robustness.")


if __name__ == "__main__":
    main() 