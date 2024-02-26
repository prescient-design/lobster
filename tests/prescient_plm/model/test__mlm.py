from torch import Size, Tensor

from prescient_plm.model import PrescientPMLM


class TestPrescientPMLM:
    def test_sequences_to_latents(self):
        model = PrescientPMLM(model_name="MLM_small")

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 72])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    def test_dynamic_masking(self):
        model = PrescientPMLM(
            model_name="MLM_small", mask_percentage=0.1, initial_mask_percentage=0.8
        )

        assert model._initial_mask_percentage is not None

    def test_load_from_checkpoint(self):
        model = PrescientPMLM.load_from_checkpoint("s3://prescient-pcluster-data/freyn6/models/pmlm/prod/2023-10-30T15-23-25.795635/last.ckpt")

        assert model.config.hidden_size == 384
