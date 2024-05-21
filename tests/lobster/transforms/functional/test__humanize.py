import pytest
import torch
from lobster.transforms.functional import humanize


class TestHumanize:
    def test__humanize(self):
        fv_heavy = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYAADFKRRFTFSLETSVDTMSTSTVYMELSSLRSEDTAVYYCARGYRSYAMDYWGQGTSVTVSS"
        fv_light = "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"

        # random_tensor = torch.rand(149)
        # fv_heavy_mask = random_tensor > 0.5

        # random_tensor = torch.rand(148)
        # fv_light_mask = random_tensor > 0.5

        best_fv_heavy, best_fv_light = humanize(
            fv_heavy, fv_light, model_name="CLM_mini", return_sequences=True, smoke=True
        )
        df = humanize(
            fv_heavy,
            fv_light,
            model_name="CLM_mini",
            return_sequences=False,
            smoke=True,
        )

        assert df is not None
        assert best_fv_heavy is not None
        assert best_fv_light is not None

    @pytest.mark.skip(reason="Naturalness scoring is slow")
    def test__naturalness_scoring(self):
        fv_heavy = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYAADFKRRFTFSLETSVDTMSTSTVYMELSSLRSEDTAVYYCARGYRSYAMDYWGQGTSVTVSS"
        fv_light = "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"

        random_tensor = torch.rand(149)
        fv_heavy_mask = random_tensor > 0.5

        random_tensor = torch.rand(148)
        fv_light_mask = random_tensor > 0.5

        df = humanize(
            fv_heavy,
            fv_light,
            fv_heavy_mask,
            fv_light_mask,
            model_type="LobsterPMLM",
            model_name="esm2_t6_8M_UR50D",
            return_sequences=False,
            naturalness=True,
            smoke=True,
        )

        assert df is not None
        assert df["fv_heavy_naturalness"] is not None
