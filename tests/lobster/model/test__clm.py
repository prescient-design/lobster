from torch import Size, Tensor

from lobster.model import PrescientPCLM


class TestPrescientPMLM:
    def test_sequences_to_latents(self):
        model = PrescientPCLM(model_name="CLM_mini")

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 32])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device
