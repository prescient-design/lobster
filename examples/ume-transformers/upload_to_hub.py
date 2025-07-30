#!/usr/bin/env python3
"""
Upload UME model files to HuggingFace Hub.

This script uploads the prepared model files from the model/ directory
to HuggingFace Hub, making UME available as a standard HF model.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami


def check_prerequisites():
    """Check if all prerequisites are met before uploading."""
    print("🔍 Checking prerequisites...")
    
    # Check if model directory exists
    model_dir = Path("model")
    if not model_dir.exists():
        print("❌ model/ directory not found!")
        print("   Make sure you're running this from examples/ume-transformers/")
        return False
    
    # Check required files
    required_files = [
        "model/config.json",
        "model/tokenization_ume.py", 
        "model/tokenizer_config.json",
        "model/special_tokens_map.json",
        "model/README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check huggingface_hub availability
    try:
        import huggingface_hub
        print(f"✅ huggingface_hub {huggingface_hub.__version__} available")
    except ImportError:
        print("❌ huggingface_hub not installed!")
        print("   Install with: uv add huggingface_hub")
        return False
    
    return True

def check_authentication():
    """Check HuggingFace authentication."""
    print("\n🔐 Checking HuggingFace authentication...")
    
    try:
        user_info = whoami()
        username = user_info["name"]
        print(f"✅ Authenticated as: {username}")
        return username
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please login with: huggingface-cli login")
        return None

def upload_to_hub(username, repo_name="ume-base"):
    """Upload model files to HuggingFace Hub."""
    repo_id = f"{username}/{repo_name}"
    print(f"\n📤 Uploading to: https://huggingface.co/{repo_id}")
    
    # Initialize HF API
    api = HfApi()
    
    try:
        # Create repository
        print("🏗️  Creating repository...")
        create_repo(repo_id=repo_id, exist_ok=True)
        print(f"✅ Repository created/updated: {repo_id}")
        
        # Upload model directory
        print("\n📁 Uploading model files...")
        api.upload_folder(
            folder_path="model",
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"\n🎉 Upload successful!")
        print(f"🔗 Model available at: https://huggingface.co/{repo_id}")
        
        # Test loading
        print(f"\n🧪 To test your deployed model:")
        print(f"```python")
        print(f"from transformers import AutoModel, AutoTokenizer")
        print(f"model = AutoModel.from_pretrained('{repo_id}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{repo_id}', trust_remote_code=True)")
        print(f"```")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main upload process."""
    print("🚀 UME HuggingFace Hub Upload")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Check authentication  
    username = check_authentication()
    if not username:
        print("\n❌ Authentication required. Please login to HuggingFace.")
        sys.exit(1)
    
    # Confirm upload
    repo_name = input(f"\n📝 Repository name (default: ume-base): ").strip() or "ume-base"
    repo_id = f"{username}/{repo_name}"
    
    print(f"\n🎯 Ready to upload to: {repo_id}")
    print(f"📁 Files to upload from model/ directory:")
    
    model_dir = Path("model")
    for file_path in sorted(model_dir.iterdir()):
        if file_path.is_file():
            size = file_path.stat().st_size / 1024  # KB
            print(f"   - {file_path.name} ({size:.1f} KB)")
    
    confirm = input(f"\n🤔 Proceed with upload? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Upload cancelled.")
        sys.exit(0)
    
    # Upload to Hub
    if upload_to_hub(username, repo_name):
        print(f"\n✨ UME successfully deployed to HuggingFace Hub!")
    else:
        print(f"\n❌ Upload failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 