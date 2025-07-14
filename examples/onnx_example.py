#!/usr/bin/env python3
"""Example script demonstrating ONNX export and inference for UME models."""

from lobster.model import UME
from lobster.model._onnx_utils import run_onnx_inference, compare_onnx_pytorch, benchmark_onnx_pytorch
from lobster.constants import Modality


def main():
    """Demonstrate ONNX export and inference with UME models."""
    print("🚀 UME ONNX Example")
    print("=" * 50)
    
    # Initialize model
    print("📦 Initializing UME model...")
    ume = UME(model_name="UME_mini")
    ume.freeze()  # Freeze for inference
    
    # Example sequences
    smiles_sequences = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # Ibuprofen
    ]
    
    protein_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQL",  # Protein sequence 1
        "ACDEFGHIKL",  # Protein sequence 2
    ]
    
    # Export models for different modalities
    print("\n📤 Exporting models to ONNX...")
    
    # Export for SMILES
    ume.export_onnx("ume_smiles.onnx", modality=Modality.SMILES)
    print("✅ Exported UME model for SMILES sequences")
    
    # Export for proteins
    ume.export_onnx("ume_protein.onnx", modality=Modality.AMINO_ACID)
    print("✅ Exported UME model for protein sequences")
    
    # Test ONNX inference
    print("\n🔍 Testing ONNX inference...")
    
    # SMILES inference
    smiles_embeddings = run_onnx_inference("ume_smiles.onnx", smiles_sequences, Modality.SMILES)
    print(f"📊 SMILES embeddings shape: {smiles_embeddings.shape}")
    
    # Protein inference
    protein_embeddings = run_onnx_inference("ume_protein.onnx", protein_sequences, Modality.AMINO_ACID)
    print(f"📊 Protein embeddings shape: {protein_embeddings.shape}")
    
    # Compare ONNX vs PyTorch
    print("\n⚖️  Comparing ONNX vs PyTorch outputs...")
    
    # Test SMILES comparison
    smiles_match = compare_onnx_pytorch("ume_smiles.onnx", ume, smiles_sequences, Modality.SMILES)
    print(f"🔬 SMILES outputs match: {smiles_match}")
    
    # Test protein comparison
    protein_match = compare_onnx_pytorch("ume_protein.onnx", ume, protein_sequences, Modality.AMINO_ACID)
    print(f"🔬 Protein outputs match: {protein_match}")
    
    # Benchmark performance
    print("\n⚡ Benchmarking performance...")
    
    # Benchmark SMILES
    smiles_results = benchmark_onnx_pytorch("ume_smiles.onnx", ume, smiles_sequences, Modality.SMILES, num_runs=5)
    print(f"📈 SMILES - PyTorch: {smiles_results['pytorch_time']:.4f}s, ONNX: {smiles_results['onnx_time']:.4f}s, Speedup: {smiles_results['speedup']:.2f}x")
    
    # Benchmark proteins
    protein_results = benchmark_onnx_pytorch("ume_protein.onnx", ume, protein_sequences, Modality.AMINO_ACID, num_runs=5)
    print(f"📈 Protein - PyTorch: {protein_results['pytorch_time']:.4f}s, ONNX: {protein_results['onnx_time']:.4f}s, Speedup: {protein_results['speedup']:.2f}x")
    
    print("\n🎉 ONNX example completed successfully!")
    print("\n📝 Usage examples:")
    print("  # Export model")
    print("  ume.export_onnx('my_model.onnx', modality=Modality.SMILES)")
    print("  ")
    print("  # Run inference")
    print("  embeddings = run_onnx_inference('my_model.onnx', ['CC(=O)O'], Modality.SMILES)")
    print("  ")
    print("  # Compare outputs")
    print("  match = compare_onnx_pytorch('my_model.onnx', ume, ['CC(=O)O'], Modality.SMILES)")
    print("  ")
    print("  # Benchmark performance")
    print("  results = benchmark_onnx_pytorch('my_model.onnx', ume, ['CC(=O)O'], Modality.SMILES)")


if __name__ == "__main__":
    main() 