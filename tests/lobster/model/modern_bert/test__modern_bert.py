from lobster.model.modern_bert import FlexBERT
from torch import Size, Tensor


class TestFlexBERT:
    def test_sequences_to_latents(self):
        hidden_size = 252
        model = FlexBERT(embedding_layer="linear_pos", hidden_size=hidden_size).cuda()

        inputs = ["ACDAC", "ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 2
        assert isinstance(outputs[0], Tensor)
        assert outputs[-1].shape == Size([512, 252])  # L, d_model
        assert outputs[0].device == model.device
