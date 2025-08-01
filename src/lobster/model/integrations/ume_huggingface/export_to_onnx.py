import onnxruntime as ort
import torch

from lobster.constants import HF_UME_MODEL_DIRPATH, UME_MODEL_VERSIONS, Modality
from lobster.model import UME


def export_ume_models_to_onnx():
    for model_version in UME_MODEL_VERSIONS:
        model = UME.from_pretrained(model_version)
        model.export_onnx(HF_UME_MODEL_DIRPATH / f"{model_version}.onnx", modality=Modality.SMILES)

        print(f"Exported {model_version} to ONNX")

        print("Verifying the exported model")
        # Try running ONNX inference with exported model
        input_ids = torch.randint(0, 100, (1, 1, 10))
        attention_mask = torch.ones_like(input_ids)

        onnx_session = ort.InferenceSession(HF_UME_MODEL_DIRPATH / f"{model_version}.onnx")

        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        output = onnx_session.run(None, ort_inputs)
        onnx_output = torch.from_numpy(output[0])
        print(f"Output shape: {onnx_output.shape}")

        print(f"Successfully exported {model_version} to ONNX")


if __name__ == "__main__":
    export_ume_models_to_onnx()
