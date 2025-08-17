import os
import shutil
import tempfile

import pytest
import torch
from torch import Size, Tensor

from lobster.model import LobsterCBMPMLM


@pytest.fixture(scope="module", autouse=True)
def manage_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    yield temp_dir  # provide the fixture value

    # After test session: remove the temporary directory and all its contents
    shutil.rmtree(temp_dir)


class TestLobsterCBMPMLM:
    def test_sequences_to_latents(self):
        model = LobsterCBMPMLM(model_name="MLM_mini")
        model.eval()

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 72])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    @pytest.mark.skip("Requires s3 access")
    def test_load_from_s3(self):
        model = LobsterCBMPMLM.load_from_checkpoint(
            "s3://prescient-pcluster-data/prescient_plm/models_to_test_4/CBM_24.ckpt"
        )

        assert model.config.hidden_size == 408

    def test_concept_names_property(self):
        """Test that concept_names property returns expected concept names."""
        model = LobsterCBMPMLM(model_name="MLM_mini")

        concepts_name = model.concepts_name

        assert hasattr(model, "concepts_name")
        assert isinstance(concepts_name, (list, tuple))
        assert len(concepts_name) > 0

        assert concepts_name == model._concepts_name

        assert concepts_name is model._concepts_name


def test_cbmlm_checkpoint(tmp_path):
    print(f"{tmp_path=}")
    model = LobsterCBMPMLM("MLM_mini")

    for k, v in model.named_parameters():
        torch.nn.init.normal_(v)

    model.save_pretrained(tmp_path / "checkpoint")

    model2 = LobsterCBMPMLM(str(tmp_path / "checkpoint"))

    for (k1, v1), (k2, v2) in zip(model.named_parameters(), model2.named_parameters()):
        assert k1 == k2
        assert torch.equal(v1, v2)
        assert not torch.equal(v2, torch.zeros_like(v2)), f"{k1=}, {k2=}"

    assert torch.equal(model.model.lm_head.bias, model2.model.lm_head.bias)

    input = torch.randn(2, 56)
    output = model.model.lm_head.decoder(input)
    output2 = model2.model.lm_head.decoder(input)

    diff = output - output2
    print(f"{diff.abs().max()=}")

    torch.testing.assert_close(output, output2)


def test_manual_positions_validation():
    model = LobsterCBMPMLM(model_name="MLM_mini")
    model.eval()

    sequences = ["ACDAC"]
    concept = model.concepts_name[0]
    edits = 1
    intervention_type = "positive"

    # Test case 1: positions within bounds
    manual_positions = [1, 2, 3]
    result = model.intervene_on_sequences(
        sequences=sequences,
        concept=concept,
        edits=edits,
        intervention_type=intervention_type,
        manual_positions=manual_positions,
    )
    assert isinstance(result, list)

    # Test case 2: position exceeds sequence length
    input_ids = model.transform_fn_inf(sequences)[0]["input_ids"]
    seq_len = input_ids.shape[-1]

    manual_positions = [0, seq_len]  # seq_len is definitely out of bounds
    with pytest.raises((IndexError, RuntimeError)):
        model.intervene_on_sequences(
            sequences=sequences,
            concept=concept,
            edits=edits,
            intervention_type=intervention_type,
            manual_positions=manual_positions,
        )
