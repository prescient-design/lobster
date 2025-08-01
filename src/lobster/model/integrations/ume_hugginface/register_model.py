"""
Register UME model with AutoClass API and upload the model to HuggingFace Hub
"""

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel

from lobster.model.integrations.ume_hugginface.configuration_ume import UMEConfig
from lobster.model.integrations.ume_hugginface.modeling_ume import UMEModel
from lobster.constants import HF_UME_REPO_ID, HF_UME_MODEL_FILEPATH


def register_and_save_model(upload_to_hf: bool = False):
    """Register the model with AutoClass and save it"""
    AutoConfig.register("ume", UMEConfig)
    AutoModel.register(UMEConfig, UMEModel)

    config = UMEConfig(model_name="ume-mini-base-12M")
    config.register_for_auto_class()

    model = UMEModel(config)
    model.register_for_auto_class("AutoModel")

    config.save_pretrained(HF_UME_MODEL_FILEPATH)
    model.save_pretrained(HF_UME_MODEL_FILEPATH)

    print("Model and config registered and saved successfully!")

    if upload_to_hf:
        print("Uploading model to HuggingFace Hub...")

        api = HfApi()
        api.upload_folder(
            folder_path=HF_UME_MODEL_FILEPATH,
            repo_id=HF_UME_REPO_ID,
        )
        print("Model uploaded to HuggingFace Hub successfully!")


if __name__ == "__main__":
    register_and_save_model(upload_to_hf=True)
