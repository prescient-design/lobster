import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
from transformers import PreTrainedModel

from .configuration_ume import UMEConfig


class UMEModel(PreTrainedModel):
    config_class = UMEConfig

    def _get_onnx_path(self, model_name: str) -> str:
        """Download ONNX file from HuggingFace Hub"""
        onnx_filename = f"{model_name}.onnx"
        repo_id = "karina-zadorozhny/ume-mini-base-12M-test"

        # HF handles caching automatically
        return hf_hub_download(
            repo_id=repo_id,
            filename=onnx_filename,
        )

    def __init__(self, config: UMEConfig) -> None:
        super().__init__(config)
        onnx_path = self._get_onnx_path(self.config.model_name)
        self.onnx_session = ort.InferenceSession(onnx_path)

        # Add a dummy parameter to ensure the model has PyTorch parameters
        # This helps with torch_dtype detection in transformers
        self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        input_ids = input_ids.detach().cpu().numpy()
        attention_mask = attention_mask.detach().cpu().numpy()

        outputs = self.onnx_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

        return torch.from_numpy(outputs[0])
