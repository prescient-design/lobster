"""
Create and upload real weights for UME model from checkpoint.

This script loads a UME model from a PyTorch Lightning checkpoint,
converts it to HuggingFace format, and uploads it to HuggingFace Hub.
"""

import logging
import tempfile
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi, whoami
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lobster.model import UME

logger = logging.getLogger(__name__)


def validate_prerequisites() -> bool:
    """Validate that all prerequisites are met.
    
    Returns
    -------
    bool
        True if all prerequisites are met, False otherwise.
    """
    logger.info("Validating prerequisites")
    
    # Check model directory
    if not Path("model").exists():
        logger.error("model/ directory not found. Ensure running from examples/ume-transformers/")
        return False
        
    # Check required files
    required_files = ["model/config.json", "model/tokenization_ume.py"]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        logger.error(f"Missing required files: {missing}")
        return False
    
    return True
    


def load_ume_model(checkpoint_path: str) -> UME:
    """Load UME model from PyTorch Lightning checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (local or S3 URL).
        
    Returns
    -------
    UME
        Loaded UME model.
        
    Raises
    ------
    Exception
        If model loading fails.
    """
    logger.info(f"Loading UME model from checkpoint: {checkpoint_path}")
    
    ume_model = UME.load_from_checkpoint(
        checkpoint_path,
        use_flash_attn=False,
        device="cpu"
    )
    ume_model.eval()
    
    num_params = sum(p.numel() for p in ume_model.parameters())
    logger.info(
        f"UME model loaded successfully - "
        f"model: {ume_model.hparams.model_name}, "
        f"max_length: {ume_model.hparams.max_length}, "
        f"hidden_size: {ume_model.model.config.hidden_size}, "
        f"vocab_size: {ume_model.model.config.vocab_size}, "
        f"parameters: {num_params/1e6:.1f}M"
    )
    
    return ume_model


def convert_to_huggingface_format(ume_model: UME) -> AutoModel:
    """Convert UME model to HuggingFace format.
    
    Parameters
    ----------
    ume_model : UME
        The loaded UME model to convert.
        
    Returns
    -------
    AutoModel
        Converted HuggingFace model.
        
    Raises
    ------
    Exception
        If conversion fails.
    """
    logger.info("Converting UME model to HuggingFace format")
    
    # Load HuggingFace config
    hf_config = AutoConfig.from_pretrained("model")
    hf_model = AutoModel.from_config(hf_config)
    
    # Get UME state dict
    ume_state_dict = ume_model.model.state_dict()
    
    # Map weights from UME to HuggingFace model
    hf_state_dict = {}
    unmapped_params = []
    
    for name, param in hf_model.named_parameters():
        if name in ume_state_dict:
            hf_state_dict[name] = ume_state_dict[name].clone()
        else:
            # Try alternative naming patterns
            possible_names = [
                name,
                name.replace('embeddings.', ''),
                name.replace('encoder.', ''),
                f"bert.{name}",
                f"backbone.{name}",
            ]
            
            mapped = False
            for possible_name in possible_names:
                if possible_name in ume_state_dict:
                    hf_state_dict[name] = ume_state_dict[possible_name].clone()
                    mapped = True
                    break
            
            if not mapped:
                unmapped_params.append(name)
    
    if unmapped_params:
        logger.warning(f"Could not map {len(unmapped_params)} parameters, keeping random initialization")
    
    # Load mapped weights
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys during state dict loading: {len(missing_keys)} parameters")
    if unexpected_keys:
        logger.warning(f"Unexpected keys during state dict loading: {len(unexpected_keys)} parameters")
    
    logger.info("Successfully converted UME model to HuggingFace format")
    return hf_model


def create_huggingface_model_from_checkpoint(checkpoint_path: str) -> tuple[AutoModel, AutoConfig]:
    """Create HuggingFace model from UME checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the UME checkpoint.
        
    Returns
    -------
    tuple[AutoModel, AutoConfig]
        Tuple of (converted model, config).
        
    Raises
    ------
    Exception
        If model creation fails.
    """
    logger.info("Creating HuggingFace model from UME checkpoint")
    
    # Load UME model
    ume_model = load_ume_model(checkpoint_path)
    
    # Convert to HuggingFace format
    hf_model = convert_to_huggingface_format(ume_model)
    
    # Load config
    config = AutoConfig.from_pretrained("model")
    
    return hf_model, config


def upload_model_to_hub(model: AutoModel, repo_id: str) -> bool:
    """Save model and upload to HuggingFace Hub.
    
    Parameters
    ----------
    model : AutoModel
        The model to upload.
    repo_id : str
        Repository ID in format 'username/repo_name'.
        
    Returns
    -------
    bool
        True if upload successful, False otherwise.
    """
    logger.info(f"Uploading model to Hub: {repo_id}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save model to temporary directory
            model.save_pretrained(temp_path)
            logger.info("Model saved to temporary directory")
            
            # Upload to Hub
            api = HfApi()
            
            # Upload all model files
            uploaded_files = []
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    uploaded_files.append(file_path.name)
            
            logger.info(f"Successfully uploaded {len(uploaded_files)} files to {repo_id}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to upload model to Hub: {e}")
        return False


def validate_deployed_model(repo_id: str) -> bool:
    """Validate the deployed model from HuggingFace Hub.
    
    Parameters
    ----------
    repo_id : str
        Repository ID to validate.
        
    Returns
    -------
    bool
        True if validation successful, False otherwise.
    """
    logger.info(f"Validating deployed model: {repo_id}")
    
    try:
        # Load model and tokenizer from Hub
        model = AutoModel.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        
        # Test inference with sample protein sequence
        test_protein = "MKTVRQERLKSIVRILERSKEPVSGAQL"
        inputs = tokenizer([test_protein], modality="amino_acid", return_tensors="pt")
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding_norm = torch.norm(outputs.last_hidden_state[:, 0]).item()
        
        logger.info(
            f"Model validation successful - "
            f"input_shape: {inputs['input_ids'].shape}, "
            f"output_shape: {outputs.last_hidden_state.shape}, "
            f"embedding_norm: {embedding_norm:.3f}"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def create_and_upload_ume_weights(
    checkpoint_path: str,
    repo_id: str,
    validate_after_upload: bool = True
) -> bool:
    """Create and upload UME model weights from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the UME checkpoint (local or S3 URL).
    repo_id : str
        HuggingFace repository ID in format 'username/repo_name'.
    validate_after_upload : bool, default=True
        Whether to validate the model after upload.
        
    Returns
    -------
    bool
        True if process completed successfully, False otherwise.
    """
    logger.info(f"Starting UME weights creation and upload process")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Target repository: {repo_id}")
    
    # Validate prerequisites
    if not validate_prerequisites():
        logger.error("Prerequisites validation failed")
        return False
    
    # Check authentication
    try:
        user_info = whoami()
        username = user_info["name"]
        logger.info(f"Authenticated as: {username}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}. Please login with: huggingface-cli login")
        return False
    
    # Create model from checkpoint
    try:
        model, config = create_huggingface_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to create model from checkpoint: {e}")
        return False
    
    # Upload model to Hub
    if not upload_model_to_hub(model, repo_id):
        logger.error("Failed to upload model to Hub")
        return False
    
    # Validate deployed model if requested
    if validate_after_upload:
        if not validate_deployed_model(repo_id):
            logger.warning("Model upload successful but validation failed")
            return False
    
    logger.info(f"Successfully created and uploaded UME weights to {repo_id}")
    logger.info(f"Model available at: https://huggingface.co/{repo_id}")
    
    return True


def main():
    """Main execution function."""
    # Default configuration
    checkpoint_path = "s3://prescient-lobster/ume/runs/2025-06-29T00-35-21/step-250000.ckpt"
    repo_id = "prescientai/ume-mini-base-12M"  # Default repo ID
    
    # Execute the process
    success = create_and_upload_ume_weights(
        checkpoint_path=checkpoint_path,
        repo_id=repo_id,
        validate_after_upload=True
    )
    
    if success:
        logger.info("Process completed successfully")
    else:
        logger.error("Process failed")


if __name__ == "__main__":
    main() 