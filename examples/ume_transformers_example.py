"""
Example script demonstrating UME integration with HuggingFace Transformers.

This script shows how to:
1. Register UME with Transformers AutoModel classes
2. Use UME with the standard Transformers API
3. Perform masked language modeling with UME
4. Load pretrained UME models as Transformers models
5. Use UME for embedding extraction
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from lobster.model.ume import (
    register_ume_with_transformers,
    UMEConfig,
    UMEForMaskedLM,
    UMETokenizer,
    load_ume_as_transformers_model,
)
from lobster.constants import Modality


def main():
    print("🧬 UME Transformers Integration Example\n")
    
    # Step 1: Register UME with Transformers
    print("1. Registering UME with Transformers...")
    register_ume_with_transformers()
    print("✅ UME successfully registered!\n")
    
    # Step 2: Create a UME model using Transformers API
    print("2. Creating UME model using Transformers API...")
    config = UMEConfig(
        model_name="UME_mini",
        max_length=128,
        hidden_size=768,
        vocab_size=1536,
    )
    model = UMEForMaskedLM(config)
    tokenizer = UMETokenizer(max_length=128)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Step 3: Test basic tokenization and embedding
    print("3. Testing multi-modal tokenization...")
    
    # Example sequences from different modalities
    test_sequences = {
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O"],  # Aspirin
        "nucleotide": ["ATGCGTACGTAGCTACGTACG"],
    }
    
    for modality, sequences in test_sequences.items():
        print(f"  Testing {modality}:")
        
        # Tokenize using UME's multi-modal tokenizer
        inputs = tokenizer(sequences, modality=modality, return_tensors="pt", padding=True)
        print(f"    Input shape: {inputs['input_ids'].shape}")
        
        # Get embeddings
        with torch.no_grad():
            outputs = model.ume(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            embeddings = outputs.pooler_output
            print(f"    Embedding shape: {embeddings.shape}")
            print(f"    Embedding norm: {torch.norm(embeddings, dim=-1).item():.3f}")
        print()
    
    # Step 4: Demonstrate masked language modeling
    print("4. Testing Masked Language Modeling...")
    
    # Create a sequence with a mask token
    protein_sequence = "MKTVRQERLK[MASK]IVRILERSKEPVSGAQL"
    inputs = tokenizer([protein_sequence], modality="amino_acid", return_tensors="pt")
    
    # Get predictions for the masked token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Find the masked token position
        mask_token_id = tokenizer.tokenizer_transform.mask_token_id
        mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) > 0:
            batch_idx, seq_idx = mask_positions[0][0], mask_positions[1][0]
            masked_logits = logits[batch_idx, seq_idx]
            predicted_token_id = masked_logits.argmax().item()
            
            # Convert back to token (simplified)
            predicted_token = f"<token_{predicted_token_id}>"
            
            print(f"    Original: {protein_sequence}")
            print(f"    Predicted token ID: {predicted_token_id}")
            print(f"    Predicted token: {predicted_token}")
        print()
    
    # Step 5: Use with standard Transformers utilities (if registered)
    print("5. Testing AutoModel compatibility...")
    try:
        # This would work if UME was actually registered in the global registry
        # For now, we'll demonstrate the concept
        print("    Creating model via AutoModelForMaskedLM.from_config...")
        auto_model = AutoModelForMaskedLM.from_config(config)
        print(f"    ✅ AutoModel created successfully!")
        print(f"    Model type: {type(auto_model).__name__}")
    except Exception as e:
        print(f"    ⚠️  AutoModel creation failed (expected in this demo): {e}")
    print()
    
    # Step 6: Demonstrate loading from Lightning checkpoint (conceptual)
    print("6. Loading from UME Lightning checkpoint (conceptual)...")
    print("    # To load a pretrained UME as a Transformers model:")
    print("    # model = load_ume_as_transformers_model('path/to/ume/checkpoint.ckpt')")
    print("    # tokenizer = UMETokenizer(max_length=8192)")
    print("    # inputs = tokenizer(['MKTVRQERLK'], modality='amino_acid', return_tensors='pt')")
    print("    # outputs = model(**inputs)")
    print()
    
    # Step 7: Show the standard Transformers workflow
    print("7. Standard Transformers workflow example:")
    print("""
    # Once registered, you can use UME just like any other Transformers model:
    
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    
    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained("path/to/ume/model")
    tokenizer = AutoTokenizer.from_pretrained("path/to/ume/model")
    
    # Tokenize input
    text = "The protein sequence is MKTVRQERLK[MASK]IVRILERSKEPVSGAQL"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get predictions
    outputs = model(**inputs)
    
    # Find masked token prediction
    mask_token_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    print("Predicted token:", predicted_token)
    """)
    
    print("\n🎉 UME Transformers integration example completed!")


if __name__ == "__main__":
    main() 