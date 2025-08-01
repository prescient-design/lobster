from lobster.model import UME
from lobster.constants import HF_UME_MODEL_DIRPATH, UME_MODEL_VERSIONS


def export_ume_models_to_onnx():
    for model_version in UME_MODEL_VERSIONS:
        model = UME.from_pretrained(model_version)
        model.export_onnx(HF_UME_MODEL_DIRPATH / f"{model_version}.onnx")


if __name__ == "__main__":
    export_ume_models_to_onnx()
