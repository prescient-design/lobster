#!/usr/bin/env python3
"""
Test script to validate UME HuggingFace deployment.

This script tests that the model files in model/ directory work correctly
with standard HuggingFace transformers library.
"""

import json
import sys
import torch
from pathlib import Path

def test_config_loading():
    """Test that config.json loads correctly."""
    print("🔧 Testing config loading...")
    
    config_path = Path("model/config.json")
    if not config_path.exists():
        print("❌ model/config.json not found!")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ["model_type", "architectures", "hidden_size", "vocab_size"]
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        print(f"✅ Config loaded: {config['model_type']}")
        print(f"   Architecture: {config['architectures'][0]}")
        print(f"   Hidden size: {config['hidden_size']}")
        print(f"   Vocab size: {config['vocab_size']}")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_tokenizer_loading():
    """Test that the tokenizer loads and works."""
    print("\n🔧 Testing tokenizer loading...")
    
    # Check tokenizer file exists
    tokenizer_path = Path("model/tokenization_ume.py")
    if not tokenizer_path.exists():
        print("❌ model/tokenization_ume.py not found!")
        return False
    
    try:
        # Add model directory to path so we can import
        sys.path.insert(0, str(Path("model").absolute()))
        
        from tokenization_ume import UMETokenizer
        
        # Create tokenizer
        tokenizer = UMETokenizer()
        print(f"✅ {tokenizer.__class__.__name__}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        
        # Test basic tokenization
        test_sequence = "MKTVRQERLK"
        tokens = tokenizer.tokenize(test_sequence)
        print(f"   Test tokenization: {test_sequence} -> {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False
    finally:
        # Clean up path
        if str(Path("model").absolute()) in sys.path:
            sys.path.remove(str(Path("model").absolute()))

def test_model_creation():
    """Test model creation from config."""
    print("\n🔧 Testing model creation...")
    
    try:
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoModel
        
        # Load config from local file
        config = AutoConfig.from_pretrained("model")
        print(f"✅ Config loaded from model/")
        
        # Create model for masked LM
        model_mlm = AutoModelForMaskedLM.from_config(config)
        num_params = sum(p.numel() for p in model_mlm.parameters())
        print(f"✅ {model_mlm.__class__.__name__} ({num_params/1e6:.1f}M params)")
        
        # Create base model  
        model_base = AutoModel.from_config(config)
        print(f"✅ {model_base.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_multi_modal_tokenization():
    """Test multi-modal tokenization capabilities."""
    print("\n🔧 Testing multi-modal tokenization...")
    
    try:
        # Import tokenizer  
        sys.path.insert(0, str(Path("model").absolute()))
        from tokenization_ume import UMETokenizer
        
        tokenizer = UMETokenizer()
        
        # Test sequences for different modalities
        test_sequences = {
            "amino_acid": "MKTVRQERLKSIVRILERSKEPVSGAQL",
            "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "nucleotide": "ATGCGTACGTAGCTACGTACG"
        }
        
        for modality, sequence in test_sequences.items():
            # Test explicit modality
            result = tokenizer([sequence], modality=modality, return_tensors="pt", padding=True)
            print(f"   {modality:12s}: {sequence[:20]}... -> {result['input_ids'].shape}")
            
            # Test auto-detection
            result_auto = tokenizer([sequence], modality=None, return_tensors="pt", padding=True)
            print(f"   auto-detected  : {sequence[:20]}... -> {result_auto['input_ids'].shape}")
        
        print("✅ All modalities working")
        return True
        
    except Exception as e:
        print(f"❌ Multi-modal tokenization failed: {e}")
        return False
    finally:
        if str(Path("model").absolute()) in sys.path:
            sys.path.remove(str(Path("model").absolute()))

def test_inference():
    """Test full inference pipeline."""
    print("\n🔧 Testing inference...")
    
    try:
        from transformers import AutoConfig, AutoModel
        
        # Import tokenizer
        sys.path.insert(0, str(Path("model").absolute()))
        from tokenization_ume import UMETokenizer
        
        # Load model and tokenizer
        config = AutoConfig.from_pretrained("model")
        model = AutoModel.from_config(config)
        tokenizer = UMETokenizer()
        
        # Test inference
        test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQL"
        inputs = tokenizer([test_sequence], modality="amino_acid", return_tensors="pt")
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Inference successful")
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        print(f"   Hidden size: {outputs.last_hidden_state.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    finally:
        if str(Path("model").absolute()) in sys.path:
            sys.path.remove(str(Path("model").absolute()))

def test_embedding_extraction():
    """Test embedding extraction for molecular similarity."""
    print("\n🔧 Testing embedding extraction...")
    
    try:
        from transformers import AutoConfig, AutoModel
        import torch.nn.functional as F
        
        # Import tokenizer
        sys.path.insert(0, str(Path("model").absolute()))
        from tokenization_ume import UMETokenizer
        
        # Load model and tokenizer
        config = AutoConfig.from_pretrained("model")
        model = AutoModel.from_config(config)
        tokenizer = UMETokenizer()
        
        # Test with similar proteins
        protein1 = "MKTVRQERLKSIVRILERSKEPVSGAQL"
        protein2 = "MKTVRQERLKSIVRILERSKEPVSGAQL"  # Same protein
        protein3 = "AVKTVRQERLKSIVRILERSKEPVSGAQL"  # Different first AA
        
        def get_embedding(sequence):
            inputs = tokenizer([sequence], modality="amino_acid", return_tensors="pt")
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                return outputs.last_hidden_state[:, 0]  # CLS token
        
        emb1 = get_embedding(protein1)
        emb2 = get_embedding(protein2)  
        emb3 = get_embedding(protein3)
        
        # Calculate similarities
        sim_identical = F.cosine_similarity(emb1, emb2, dim=1).item()
        sim_similar = F.cosine_similarity(emb1, emb3, dim=1).item()
        
        print(f"✅ Embedding extraction successful")
        print(f"   Embedding shape: {emb1.shape}")
        print(f"   Identical proteins similarity: {sim_identical:.4f}")
        print(f"   Similar proteins similarity: {sim_similar:.4f}")
        
        # Sanity check: identical should be 1.0 (or very close)
        if abs(sim_identical - 1.0) > 0.01:
            print(f"⚠️  Warning: Identical proteins similarity not close to 1.0")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding extraction failed: {e}")
        return False
    finally:
        if str(Path("model").absolute()) in sys.path:
            sys.path.remove(str(Path("model").absolute()))

def main():
    """Run all tests."""
    print("🧪 UME HuggingFace Deployment Tests")
    print("=" * 50)
    
    # Check working directory
    if not Path("model").exists():
        print("❌ model/ directory not found!")
        print("   Make sure you're running this from examples/ume-transformers/")
        sys.exit(1)
    
    tests = [
        test_config_loading,
        test_tokenizer_loading,
        test_model_creation,
        test_multi_modal_tokenization,
        test_inference,
        test_embedding_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print("❌ Test failed!")
        except Exception as e:
            print(f"❌ Test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! UME is ready for HuggingFace Hub deployment.")
        print("\n✨ Next steps:")
        print("   1. Run: python upload_to_hub.py")
        print("   2. Run: python create_demo_weights.py") 
        print("   3. Test deployed model from Hub")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main() 