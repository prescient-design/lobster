"""
Register UME model and tokenizer with AutoClass API and upload to HuggingFace Hub
"""

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel

from lobster.model.integrations.ume_huggingface.configuration_ume import UMEConfig
from lobster.model.integrations.ume_huggingface.modeling_ume import UMEModel

# from lobster.model.integrations.ume_huggingface.tokenization_ume import UMETokenizer
from lobster.constants import HF_UME_REPO_ID, HF_UME_MODEL_FILEPATH


def register_and_save_model(upload_to_hf: bool = False):
    """Register the model and tokenizer with AutoClass and save them"""
    # Register model
    AutoConfig.register("ume", UMEConfig)
    AutoModel.register(UMEConfig, UMEModel)

    # # Register tokenizer
    # AutoTokenizer.register(UMEConfig, UMETokenizer)

    config = UMEConfig(model_name="ume-mini-base-12M")
    config.register_for_auto_class()

    model = UMEModel(config)
    model.register_for_auto_class("AutoModel")

    # # Create and register tokenizer
    # tokenizer = UMETokenizer(max_length=512)
    # tokenizer.register_for_auto_class()

    # Save everything
    config.save_pretrained(HF_UME_MODEL_FILEPATH)
    model.save_pretrained(HF_UME_MODEL_FILEPATH)
    # tokenizer.save_pretrained(HF_UME_MODEL_FILEPATH)

    print("Model and config registered and saved successfully!")

    if upload_to_hf:
        print("Uploading model folder to HuggingFace Hub...")

        api = HfApi()
        api.upload_folder(
            folder_path=HF_UME_MODEL_FILEPATH,
            repo_id=HF_UME_REPO_ID,
        )
        print("Model folder uploaded to HuggingFace Hub successfully!")


if __name__ == "__main__":
    register_and_save_model(upload_to_hf=True)
