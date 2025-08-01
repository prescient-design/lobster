"""
Register UME model and tokenizer with AutoClass API and upload to HuggingFace Hub
"""

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lobster.model.integrations.ume_huggingface.configuration_ume import UMEConfig
from lobster.model.integrations.ume_huggingface.modeling_ume import UMEModel

from lobster.model.integrations.ume_huggingface.tokenization_ume import (
    UMEAminoAcidTokenizer,
    UMESmilesTokenizer,
    UMENucleotideTokenizer,
)
from lobster.constants import HF_UME_REPO_ID, HF_UME_MODEL_DIRPATH
import os


def register_and_save_model(upload_to_hf: bool = False):
    """Register the model and tokenizers with AutoClass and save them"""
    # Register model
    AutoConfig.register("ume", UMEConfig)
    AutoModel.register(UMEConfig, UMEModel)
    config = UMEConfig(model_name="ume-mini-base-12M")
    config.register_for_auto_class()

    model = UMEModel(config)
    model.register_for_auto_class("AutoModel")

    # Register individual tokenizers for each modality
    AutoTokenizer.register(UMEAminoAcidTokenizer, UMEAminoAcidTokenizer)
    AutoTokenizer.register(UMESmilesTokenizer, UMESmilesTokenizer)
    AutoTokenizer.register(UMENucleotideTokenizer, UMENucleotideTokenizer)

    # Create tokenizer instances for each modality
    tokenizers = {
        "amino_acid": UMEAminoAcidTokenizer(),
        "smiles": UMESmilesTokenizer(),
        "nucleotide": UMENucleotideTokenizer(),
    }

    # Register each tokenizer for auto class
    for tokenizer in tokenizers.values():
        tokenizer.register_for_auto_class()

    # Save model and config to main directory
    config.save_pretrained(HF_UME_MODEL_DIRPATH)
    model.save_pretrained(HF_UME_MODEL_DIRPATH)

    # Save each tokenizer to its own modality-specific subdirectory
    for modality, tokenizer in tokenizers.items():
        modality_dir = os.path.join(HF_UME_MODEL_DIRPATH, f"tokenizer_{modality}")
        tokenizer.save_pretrained(modality_dir)

    # Save one tokenizer directly (not ideal but we need this)
    tokenizer.save_pretrained(HF_UME_MODEL_DIRPATH)

    print("Model and all tokenizers registered and saved successfully!")
    print(f"Tokenizers saved to: {', '.join([f'tokenizer_{mod}' for mod in tokenizers.keys()])}")

    if upload_to_hf:
        print("Uploading `model` folder to HuggingFace Hub...")

        api = HfApi()
        api.upload_folder(
            folder_path=HF_UME_MODEL_DIRPATH,
            repo_id=HF_UME_REPO_ID,
        )
        print("Folder uploaded to HuggingFace Hub successfully!")


if __name__ == "__main__":
    register_and_save_model(upload_to_hf=True)
