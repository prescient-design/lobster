
from torch import Size, Tensor

from prescient_plm.model.hyena import PrescientHyenaCLM


class TestPrescientHyenaCLM:
    def test_sequences_to_latents(self):
        model = PrescientHyenaCLM(model_name="hyena_mini")

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 10
        assert isinstance(outputs[0], Tensor)
        assert outputs[-1].shape == Size([1, 1024, 64])  # B, L, d_model
        assert outputs[0].device == model.device
