#!/usr/bin/env python3
"""
Script to generate a synthetic dataset of molecular and biological sequences.
Generates 100 SMILES strings, 100 amino acid strings, and 100 DNA strings.
Saves as a HuggingFace dataset with train/val/test split of 0.9/0.05/0.05.

The completions aren't actually used for training with GRPO.
"""

import random
import json
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict


def generate_random_smiles(length_range: tuple[int, int] = (10, 50)) -> str:
    """Generate a random SMILES string with basic molecular structure patterns."""
    # Common SMILES patterns and building blocks
    atoms = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    bonds = ["", "=", "#"]
    rings = ["1", "2", "3", "4", "5", "6"]
    branches = ["(", ")"]
    
    # Start with a simple pattern
    patterns = [
        "CCO",  # ethanol-like
        "c1ccccc1",  # benzene-like
        "CC(=O)O",  # acetic acid-like
        "CC(C)C",  # isobutane-like
        "C1CCCCC1",  # cyclohexane-like
        "CCN",  # ethylamine-like
        "CCOC",  # ether-like
        "CC#N",  # acetonitrile-like
        "CCOCC",  # diethyl ether-like
        "CC(C)(C)C",  # t-butyl-like
    ]
    
    # Choose a base pattern and extend it
    base = random.choice(patterns)
    length = random.randint(*length_range)
    
    # Extend the pattern by adding random elements
    result = base
    while len(result) < length:
        if random.random() < 0.3:  # 30% chance to add a new atom
            result += random.choice(atoms)
        elif random.random() < 0.2:  # 20% chance to add a bond
            result += random.choice(bonds)
        elif random.random() < 0.1:  # 10% chance to add a ring
            result += random.choice(rings)
        elif random.random() < 0.1:  # 10% chance to add a branch
            result += random.choice(branches)
        else:  # 30% chance to add a number (for ring closure)
            result += str(random.randint(1, 9))
    
    return result


def generate_random_amino_acid_sequence(length_range: tuple[int, int] = (20, 100)) -> str:
    """Generate a random amino acid sequence using standard one-letter codes."""
    # Standard amino acid one-letter codes
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    length = random.randint(*length_range)
    sequence = ""
    
    for _ in range(length):
        sequence += random.choice(amino_acids)
    
    return sequence


def generate_random_dna_sequence(length_range: tuple[int, int] = (50, 200)) -> str:
    """Generate a random DNA sequence using A, T, G, C."""
    # DNA nucleotides
    nucleotides = "ATGC"
    
    length = random.randint(*length_range)
    sequence = ""
    
    for _ in range(length):
        sequence += random.choice(nucleotides)
    
    return sequence


def generate_smiles_prompts() -> List[str]:
    """Generate prompts for SMILES string generation."""
    prompts = [
        "Generate a valid SMILES string representing a small organic molecule. \
            The molecule should be at least 10 characters long. You MUST only use the following atoms: \
            C, H, O, N, P, S, F, Cl, Br, I. You MUST produce valid SMILES strings. \
                Generate only one SMILES string and nothing else. You MUST enclose the SMILES string in <smiles> and </smiles> tags.",
    ]
    return [random.choice(prompts) for _ in range(100)]


def generate_amino_acid_prompts() -> List[str]:
    """Generate prompts for amino acid sequence generation."""
    prompts = [
        "Generate a valid amino acid sequence using one-letter codes. \
            The sequence should be at least 10 characters long. You MUST only use the following amino acids: \
            A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y. There must be no other characters in the sequence. \
                Generate only one amino acid sequence and nothing else. You MUST enclose the amino acid sequence in <amino_acid> and </amino_acid> tags.",
    ]
    return [random.choice(prompts) for _ in range(100)]


def generate_dna_prompts() -> List[str]:
    """Generate prompts for DNA sequence generation."""
    prompts = [
        "Generate a valid DNA sequence using A, T, G, C. The sequence should be at least 20 characters long. \
            You MUST only use the following nucleotides: A, T, G, C. There must be no other characters in the sequence. \
                Generate only one DNA sequence and nothing else. You MUST enclose the DNA sequence in <dna> and </dna> tags.",
    ]
    return [random.choice(prompts) for _ in range(100)]


def validate_smiles(smiles: str) -> bool:
    """Basic validation for SMILES strings."""
    if len(smiles) < 5:
        return False
    
    # Check for basic SMILES characters
    valid_chars = set("CNOSPFIHBrCl()[]=#@+-.\\/0123456789")
    return all(c in valid_chars for c in smiles)


def validate_amino_acid(sequence: str) -> bool:
    """Validate amino acid sequence."""
    if len(sequence) < 10:
        return False
    
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return all(c in valid_aa for c in sequence.upper())


def validate_dna(sequence: str) -> bool:
    """Validate DNA sequence."""
    if len(sequence) < 20:
        return False
    
    valid_dna = set("ATGC")
    return all(c in valid_dna for c in sequence.upper())


def main():
    """Main function to generate the dataset."""
    print("Generating synthetic molecular and biological sequences...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate SMILES sequences
    print("Generating SMILES strings...")
    smiles_prompts = generate_smiles_prompts()
    smiles_sequences = []
    
    for i in range(100):
        smiles = generate_random_smiles()
        if validate_smiles(smiles):
            smiles_sequences.append(smiles)
        else:
            # Generate a simple valid SMILES if validation fails
            smiles_sequences.append("CCO")
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/100 SMILES sequences")
    
    # Generate amino acid sequences
    print("Generating amino acid sequences...")
    aa_prompts = generate_amino_acid_prompts()
    aa_sequences = []
    
    for i in range(100):
        aa_seq = generate_random_amino_acid_sequence()
        if validate_amino_acid(aa_seq):
            aa_sequences.append(aa_seq)
        else:
            # Generate a simple valid sequence if validation fails
            aa_sequences.append("ACDEFGHIKLMNPQRSTVWY")
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/100 amino acid sequences")
    
    # Generate DNA sequences
    print("Generating DNA sequences...")
    dna_prompts = generate_dna_prompts()
    dna_sequences = []
    
    for i in range(100):
        dna_seq = generate_random_dna_sequence()
        if validate_dna(dna_seq):
            dna_sequences.append(dna_seq)
        else:
            # Generate a simple valid sequence if validation fails
            dna_sequences.append("ATGCATGCATGCATGCATGC")
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/100 DNA sequences")
    
    print(f"Generated {len(smiles_sequences)} SMILES sequences")
    print(f"Generated {len(aa_sequences)} amino acid sequences")
    print(f"Generated {len(dna_sequences)} DNA sequences")
    
    # Create dataset
    dataset_data = []
    
    # Add SMILES sequences
    for i, seq in enumerate(smiles_sequences):
        dataset_data.append({
            "prompt": smiles_prompts[i],
            "completion": seq
        })
    
    # Add amino acid sequences
    for i, seq in enumerate(aa_sequences):
        dataset_data.append({
            "prompt": aa_prompts[i],
            "completion": seq
        })
    
    # Add DNA sequences
    for i, seq in enumerate(dna_sequences):
        dataset_data.append({
            "prompt": dna_prompts[i],
            "completion": seq
        })
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_data)
    
    # Create train/validation/test splits (90/5/5)
    print("Creating train/validation/test splits...")
    
    # First split: 90% train, 10% temp (for val+test)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]
    
    # Second split: split the 10% into 5% validation and 5% test
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create DatasetDict with all splits
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
        "full": dataset
    })
    
    # Save the entire DatasetDict
    output_path = "synthetic_molecular_dataset"
    dataset_dict.save_to_disk(output_path)
    
    # Save as JSON for easy inspection
    with open("synthetic_molecular_dataset.json", "w") as f:
        json.dump(dataset_data, f, indent=2)
    
    print(f"Dataset saved to {output_path}/")
    print(f"Available splits: train, validation, test, full")
    print(f"Full dataset also saved as synthetic_molecular_dataset.json")
    print(f"Total sequences in dataset: {len(dataset_data)}")
    
    # Print some examples
    print("\nExample sequences:")
    for i, item in enumerate(dataset_data[:6]):
        seq_type = "SMILES" if i < 2 else "Amino Acid" if i < 4 else "DNA"
        print(f"{seq_type}:")
        print(f"  Prompt: {item['prompt']}")
        print(f"  Completion: {item['completion'][:50]}...")
        print()


if __name__ == "__main__":
    main() 