import pytest
import torch
from lobster.transforms.functional import get_fv_aho_mask, get_lobster_model, humanize


class TestHumanize:
    def test__humanize(self):
        fv_heavy_aho = "EVQLLES-GGGLVQPGGSLRLSCAVSG-FSLTT-----YAMGWVRQAPGKGLEWIGIILS----SDNTYYASWVNGRFTISKDSS-TTVYLQMNSLRAEDTAVYFCARDAGVID------------------YYAFNIWGPGTLVTVSS"
        fv_light_aho = "DVQMTQSPSSLSASVGDRVTITCQSS--QSIS------TALAWYQQKPGKPPKLLIYK--------ASTLASGVPSRFSGSGSG--TDFTLTISSLQPEDVATYYCQCTDYDS--------------------SYVFPFGGGTKVEIK"

        # random_tensor = torch.rand(149)
        # fv_heavy_mask = random_tensor > 0.5

        # random_tensor = torch.rand(148)
        # fv_light_mask = random_tensor > 0.5

        clm_model = get_lobster_model(model_name="CLM_mini", model_type="LobsterPCLM")
        df = humanize(clm_model, fv_heavy_aho, chain="H", return_sequences=True, naturalness=True, smoke=True)
        best_fv_heavy = humanize(clm_model, fv_heavy_aho, chain="H", return_sequences=False, smoke=True)
        best_fv_light = humanize(clm_model, fv_light_aho, chain="L_lambda", return_sequences=False, smoke=True)

        assert df is not None
        assert best_fv_heavy is not None
        assert best_fv_light is not None

    def test__custom_humanize(self):
        # EGFR-N032 example
        fv_heavy = "EVKLQQSGDETMRPGASVRMSCKAYGYTFTDYSVHWIRQRPGQGLEWIGIIIPLIDTTRYNQKFKGKAVLTADTSSDTAYMELSRLTFEDSAVYYCARSYGSSGDDWFAYWGQGTLVTVSS"
        fv_heavy_aho = "EVKLQQS-GDETMRPGASVRMSCKAYG-YTFTD-----YSVHWIRQRPGQGLEWIGIIIPL---IDTTRYNQKFKGKAVLTADTSSDTAYMELSRLTFEDSAVYYCARSYGSSG------------------DDWFAYWGQGTLVTVSS"

        keep_constant = [
            "GYTFTDYSVH",
            "IIIPLIDTTRYNQKFKG",
            "ARSYGSSGDDWFAY",
        ]  # constant regions, from AbGrafter boundary definitions

        mask = get_fv_aho_mask(fv_heavy, fv_heavy_aho, keep_constant)

        clm_model = get_lobster_model(model_name="CLM_mini", model_type="LobsterPCLM")
        best_fv_heavy = humanize(
            clm_model, fv_heavy_aho, chain="H", fv_aho_mask=mask, return_sequences=True, smoke=True
        )

        for region in keep_constant:
            assert region in best_fv_heavy

    @pytest.mark.skip(reason="Naturalness scoring is slow")
    def test__naturalness_scoring(self):
        fv_heavy_aho = "EVQLLES-GGGLVQPGGSLRLSCAVSG-FSLTT-----YAMGWVRQAPGKGLEWIGIILS----SDNTYYASWVNGRFTISKDSS-TTVYLQMNSLRAEDTAVYFCARDAGVID------------------YYAFNIWGPGTLVTVSS"
        fv_light_aho = "DVQMTQSPSSLSASVGDRVTITCQSS--QSIS------TALAWYQQKPGKPPKLLIYK--------ASTLASGVPSRFSGSGSG--TDFTLTISSLQPEDVATYYCQCTDYDS--------------------SYVFPFGGGTKVEIK"

        random_tensor = torch.rand(149)
        fv_heavy_mask = random_tensor > 0.5

        random_tensor = torch.rand(148)
        fv_light_mask = random_tensor > 0.5

        mlm_model = get_lobster_model(model_name="esm2_t6_8M_UR50D", model_type="LobsterPMLM")

        df = humanize(
            mlm_model,
            fv_heavy_aho,
            chain="H",
            fv_aho_mask=fv_heavy_mask,
            return_sequences=False,
            naturalness=True,
            smoke=True,
        )

        df = humanize(
            mlm_model,
            fv_light_aho,
            chain="L_lambda",
            fv_aho_mask=fv_light_mask,
            return_sequences=False,
            naturalness=True,
            smoke=True,
        )

        assert df is not None
        assert df["fv_heavy_naturalness"] is not None
