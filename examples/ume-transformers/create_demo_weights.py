#!/usr/bin/env python3
"""
Create and upload demo weights for UME model.

This script creates a randomly initialized UME model based on the config
in model/ directory and uploads it to HuggingFace Hub.
"""

import sys
import tempfile
from pathlib import Path

def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check model directory
    if not Path("model").exists():
        print("❌ model/ directory not found!")
        print("   Make sure you're running this from examples/ume-transformers/")
        return False
        
    # Check required files
    required_files = ["model/config.json", "model/tokenization_ume.py"]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    
    # Check dependencies
    try:
        import transformers
        import torch
        import huggingface_hub
        print(f"✅ Dependencies available")
        print(f"   transformers: {transformers.__version__}")
        print(f"   torch: {torch.__version__}")
        print(f"   huggingface_hub: {huggingface_hub.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_demo_model():
    """Create randomly initialized model from config."""
    print("\n🏗️  Creating demo model...")
    
    try:
        from transformers import AutoConfig, AutoModel
        
        # Load config from model directory
        config = AutoConfig.from_pretrained("model")
        print(f"✅ Config loaded: {config.model_type}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Layers: {config.num_hidden_layers}")
        print(f"   Vocab size: {config.vocab_size}")
        
        # Create model with random weights
        model = AutoModel.from_config(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {num_params/1e6:.1f}M parameters")
        
        return model, config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None, None

def save_demo_model(model, repo_id):
    """Save model to temporary directory and upload."""
    print(f"\n💾 Saving demo model...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save model
            model.save_pretrained(temp_path)
            print(f"✅ Model saved to temporary directory")
            
            # Upload to Hub
            from huggingface_hub import HfApi
            api = HfApi()
            
            print(f"📤 Uploading demo weights to {repo_id}...")
            
            # Upload model files
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    print(f"   Uploading {file_path.name}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
            
            print(f"✅ Demo weights uploaded successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def test_deployed_model(repo_id):
    """Test the deployed model from Hub."""
    print(f"\n🧪 Testing deployed model from Hub...")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Load from Hub
        print(f"   Loading model from {repo_id}...")
        model = AutoModel.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        
        # Test inference
        test_protein = "MKTVRQERLKSIVRILERSKEPVSGAQL"
        inputs = tokenizer([test_protein], modality="amino_acid", return_tensors="pt")
        
        import torch
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Hub model test successful!")
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        print(f"   Embedding norm: {torch.norm(outputs.last_hidden_state[:, 0]).item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hub model test failed: {e}")
        return False

def main():
    """Main demo weights creation process."""
    print("🎲 UME Demo Weights Creation")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Check authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        username = user_info["name"]
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please login with: huggingface-cli login")
        sys.exit(1)
    
    # Get repository info
    repo_name = input(f"\n📝 Repository name (default: ume-base): ").strip() or "ume-base"
    repo_id = f"{username}/{repo_name}"
    
    print(f"\n🎯 Creating demo weights for: {repo_id}")
    print(f"⚠️  Note: This will create randomly initialized weights for testing purposes.")
    print(f"   Replace with actual trained weights when available.")
    
    confirm = input(f"\n🤔 Upload demo weights to Hub? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled.")
        sys.exit(0)
    
    # Create model
    model, config = create_demo_model()
    if model is None:
        print("❌ Failed to create model.")
        sys.exit(1)
    
    # Upload demo weights
    if save_demo_model(model, repo_id):
        print(f"\n📤 Uploading demo model weights to Hub...")
        
        # Test deployed model
        if test_deployed_model(repo_id):
            print(f"\n🎉 Demo weights successfully uploaded and tested!")
            print(f"🔗 Model with weights: https://huggingface.co/{repo_id}")
            print(f"\n🧪 Users can now test with:")
            print(f"```python")
            print(f"from transformers import AutoModel, AutoTokenizer")
            print(f"model = AutoModel.from_pretrained('{repo_id}')")
            print(f"tokenizer = AutoTokenizer.from_pretrained('{repo_id}', trust_remote_code=True)")
            print(f"```")
        else:
            print(f"⚠️  Upload successful but Hub test failed. Check model manually.")
    else:
        print(f"❌ Failed to upload demo weights.")
        sys.exit(1)

if __name__ == "__main__":
    main() 