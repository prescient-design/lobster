"""
Sanity check for UME HuggingFace integration.

This script compares the HuggingFace integrated UME model with the native UME model
to ensure they produce identical results. It tests:

1. Tokenization consistency across all modalities (amino_acid, SMILES, nucleotide)
2. Model output consistency across all modalities

Both tokenized inputs/attention masks and model embeddings should be identical
(within numerical precision) between the two implementations.
"""

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from lobster.model import UME
from lobster.constants import Modality


def load_models():
    """Load both HF and normal UME models."""
    print("Loading models...")
    
    # Load HF model
    print("  - Loading HuggingFace UME model...")
    hf_config = AutoConfig.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True)
    hf_model = AutoModel.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True, config=hf_config)
    hf_model.eval()
    
    # Load normal UME model
    print("  - Loading normal UME model...")
    ume_model = UME.from_pretrained("ume-mini-base-12M")
    ume_model.eval()
    ume_model.freeze()
    
    print("Both models loaded successfully")
    return hf_model, ume_model


def load_tokenizers():
    """Load tokenizers for all modalities."""
    print("Loading tokenizers...")
    
    modalities = {
        "amino_acid": "tokenizer_amino_acid",
        "SMILES": "tokenizer_smiles", 
        "nucleotide": "tokenizer_nucleotide"
    }
    
    hf_tokenizers = {}
    for modality, subfolder in modalities.items():
        print(f"  - Loading HF tokenizer for {modality}...")
        hf_tokenizers[modality] = AutoTokenizer.from_pretrained(
            "karina-zadorozhny/ume",
            subfolder=subfolder,
            trust_remote_code=True
        )
    
    print("All tokenizers loaded successfully")
    return hf_tokenizers


def get_test_sequences():
    """Get test sequences for each modality."""
    return {
        "amino_acid": [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA",
            "ACDEFGHIKLMNPQRSTVWY",
            "MKLLILLFGLLSSVATANGATPGGKPKAGSPKAGGAAA"
        ],
        "SMILES": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CCO",  # Ethanol
            "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid
        ],
        "nucleotide": [
            "ATGCGTACGTACGTACGTACGT",
            "AAAAAAAAAA",
            "GCGCGCGCGCGCGCGCGC"
        ]
    }


def compare_tokenization(hf_tokenizers, ume_model, test_sequences):
    """Compare tokenization between HF and UME models."""
    print("\n" + "="*60)
    print("TOKENIZATION COMPARISON")
    print("="*60)
    
    all_passed = True
    
    for modality_str, sequences in test_sequences.items():
        print(f"\nTesting {modality_str} tokenization...")
        
        # Get tokenizers
        hf_tokenizer = hf_tokenizers[modality_str]
        modality_enum = Modality(modality_str if modality_str != "smiles" else "SMILES")
        ume_tokenizer = ume_model.get_tokenizer(modality_enum)
        
        for i, sequence in enumerate(sequences):
            print(f"  Sequence {i+1}: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
            
            # Tokenize with HF
            hf_inputs = hf_tokenizer([sequence], return_tensors="pt", padding=True, truncation=True)
            hf_input_ids = hf_inputs["input_ids"]
            hf_attention_mask = hf_inputs["attention_mask"]
            
            # Tokenize with UME 
            ume_inputs = ume_tokenizer([sequence], return_tensors="pt", padding=True, truncation=True)
            ume_input_ids = ume_inputs["input_ids"]
            ume_attention_mask = ume_inputs["attention_mask"]
            
            # Compare input_ids
            input_ids_match = torch.equal(hf_input_ids, ume_input_ids)
            attention_mask_match = torch.equal(hf_attention_mask, ume_attention_mask)
            
            if input_ids_match and attention_mask_match:
                print(f"    PASS - Tokenization matches")
            else:
                print(f"    FAIL - Tokenization mismatch!")
                print(f"       Input IDs match: {input_ids_match}")
                print(f"       Attention mask match: {attention_mask_match}")
                
                if not input_ids_match:
                    print(f"       HF input_ids shape: {hf_input_ids.shape}")
                    print(f"       UME input_ids shape: {ume_input_ids.shape}")
                    print(f"       HF input_ids: {hf_input_ids}")
                    print(f"       UME input_ids: {ume_input_ids}")
                
                if not attention_mask_match:
                    print(f"       HF attention_mask: {hf_attention_mask}")
                    print(f"       UME attention_mask: {ume_attention_mask}")
                
                all_passed = False
    
    if all_passed:
        print(f"\nAll tokenization tests passed!")
    else:
        print(f"\nSome tokenization tests failed!")
    
    return all_passed


def compare_model_outputs(hf_model, ume_model, hf_tokenizers, test_sequences, rtol=1e-4, atol=5e-6):
    """Compare model outputs between HF and UME models."""
    print("\n" + "="*60)
    print("MODEL OUTPUT COMPARISON")
    print("="*60)
    
    all_passed = True
    
    for modality_str, sequences in test_sequences.items():
        print(f"\nTesting {modality_str} model outputs...")
        
        # Get tokenizers
        hf_tokenizer = hf_tokenizers[modality_str]
        modality_enum = Modality(modality_str if modality_str != "smiles" else "SMILES")
        
        for i, sequence in enumerate(sequences):
            print(f"  Sequence {i+1}: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
            
            # Tokenize
            hf_inputs = hf_tokenizer([sequence], return_tensors="pt", padding=True, truncation=True)
            hf_input_ids = hf_inputs["input_ids"]
            hf_attention_mask = hf_inputs["attention_mask"]
            
            # Get HF model output
            with torch.no_grad():
                # HF model expects 3D input: (batch_size, num_views, seq_len)
                hf_output = hf_model(hf_input_ids.unsqueeze(1), hf_attention_mask.unsqueeze(1))
            
            # Get UME model output
            with torch.no_grad():
                ume_output = ume_model.embed_sequences([sequence], modality_enum)
            
            # Compare outputs
            outputs_close = torch.allclose(hf_output, ume_output, rtol=rtol, atol=atol)
            
            if outputs_close:
                print(f"    PASS - Model outputs match (within tolerance)")
                print(f"       Output shape: {hf_output.shape}")
                print(f"       Max absolute difference: {torch.max(torch.abs(hf_output - ume_output)).item():.2e}")
            else:
                print(f"    FAIL - Model outputs don't match!")
                print(f"       HF output shape: {hf_output.shape}")
                print(f"       UME output shape: {ume_output.shape}")
                print(f"       Max absolute difference: {torch.max(torch.abs(hf_output - ume_output)).item():.2e}")
                print(f"       Relative tolerance: {rtol}")
                print(f"       Absolute tolerance: {atol}")
                
                # Show some sample values for debugging
                print(f"       HF output sample: {hf_output.flatten()[:5]}")
                print(f"       UME output sample: {ume_output.flatten()[:5]}")
                
                all_passed = False
    
    if all_passed:
        print(f"\nAll model output tests passed!")
    else:
        print(f"\nSome model output tests failed!")
    
    return all_passed


def run_comprehensive_sanity_check():
    """Run comprehensive sanity check comparing HF and UME models."""
    print("Starting comprehensive UME HuggingFace sanity check...")
    print("="*80)
    
    # Load models
    hf_model, ume_model = load_models()
    
    # Load tokenizers
    hf_tokenizers = load_tokenizers()
    
    # Get test sequences
    test_sequences = get_test_sequences()
    
    # Run tokenization comparison
    tokenization_passed = compare_tokenization(hf_tokenizers, ume_model, test_sequences)
    
    # Run model output comparison
    output_passed = compare_model_outputs(hf_model, ume_model, hf_tokenizers, test_sequences)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if tokenization_passed and output_passed:
        print("ALL TESTS PASSED - HuggingFace and UME models are consistent.")
    else:
        print("SOME TESTS FAILED!")
        print(f"   Tokenization tests: {'PASSED' if tokenization_passed else 'FAILED'}")
        print(f"   Model output tests: {'PASSED' if output_passed else 'FAILED'}")
    
    return tokenization_passed and output_passed


if __name__ == "__main__":
    success = run_comprehensive_sanity_check()
    exit(0 if success else 1)
