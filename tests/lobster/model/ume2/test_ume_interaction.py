from unittest.mock import patch
from lobster.model.ume2 import UMEInteraction, UMEInteractionLightningModule, SpecialTokenIds
from lobster.constants import Modality
import torch
import lightning as L
import pytest
from lobster.tokenization import get_ume_tokenizer_transforms
from lobster.transforms import TokenizerTransform
from torch import Tensor

BATCH_SIZE = 3
VOCAB_SIZE_AMINO_ACID = 25
VOCAB_SIZE_SMILES = 13
MAX_LENGTH_AMINO_ACID = 5
MAX_LENGTH_SMILES = 4
MAX_LENGTH_REALISTIC_BATCH = 10


@pytest.fixture
def batch():
    "Random batch"
    return {
        "input_ids1": torch.randint(0, VOCAB_SIZE_AMINO_ACID, (BATCH_SIZE, MAX_LENGTH_AMINO_ACID)),
        "attention_mask1": torch.ones((BATCH_SIZE, MAX_LENGTH_AMINO_ACID)),
        "modality1": [Modality.AMINO_ACID] * BATCH_SIZE,
        "input_ids2": torch.randint(0, VOCAB_SIZE_SMILES, (BATCH_SIZE, MAX_LENGTH_SMILES)),
        "attention_mask2": torch.ones((BATCH_SIZE, MAX_LENGTH_SMILES)),
        "modality2": [Modality.SMILES] * BATCH_SIZE,
    }


@pytest.fixture
def realistic_batch() -> tuple[dict[str, Tensor], TokenizerTransform, TokenizerTransform]:
    "Realistic batch with sequence pair: MVYKL and CC(=O) repeated BATCH_SIZE times"
    tokenizer_transforms = get_ume_tokenizer_transforms(
        max_length=MAX_LENGTH_REALISTIC_BATCH, use_shared_tokenizer=False
    )

    aa_tok_transform, smiles_tok_transform = (
        tokenizer_transforms[Modality.AMINO_ACID],
        tokenizer_transforms[Modality.SMILES],
    )

    aa_sequence = "MVYKL"
    smiles_sequence = "CC(=O)"

    aa_inputs = aa_tok_transform(aa_sequence)
    smiles_inputs = smiles_tok_transform(smiles_sequence)

    batch = {
        "input_ids1": aa_inputs["input_ids"].repeat(BATCH_SIZE, 1),
        "attention_mask1": aa_inputs["attention_mask"].repeat(BATCH_SIZE, 1),
        "modality1": [Modality.AMINO_ACID] * BATCH_SIZE,
        "input_ids2": smiles_inputs["input_ids"].repeat(BATCH_SIZE, 1),
        "attention_mask2": smiles_inputs["attention_mask"].repeat(BATCH_SIZE, 1),
        "modality2": [Modality.SMILES] * BATCH_SIZE,
    }

    return batch, aa_tok_transform, smiles_tok_transform


class TestUMEInteractionLightningModule:
    @pytest.fixture
    def model(self):
        L.seed_everything(1)

        return UMEInteractionLightningModule(
            special_token_mappings={
                Modality.AMINO_ACID: SpecialTokenIds(mask_token_id=6, pad_token_id=0, special_token_ids=[6, 0]),
                Modality.SMILES: SpecialTokenIds(mask_token_id=0, pad_token_id=1, special_token_ids=[0, 1]),
            },
            mask_probability=0.15,
            encoder_kwargs={
                Modality.AMINO_ACID: {
                    "max_length": MAX_LENGTH_AMINO_ACID,
                    "vocab_size": VOCAB_SIZE_AMINO_ACID,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 2,
                },
                Modality.SMILES: {
                    "max_length": MAX_LENGTH_SMILES,
                    "vocab_size": VOCAB_SIZE_SMILES,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 2,
                },
            },
        )

    def test__init__(self, model):
        assert isinstance(model.encoder, UMEInteraction)
        assert model.mask_probability == 0.15
        assert len(model.encoder.molecular_encoders) == 2
        assert model.encoder.molecular_encoders[Modality.AMINO_ACID].neobert.config.max_length == MAX_LENGTH_AMINO_ACID
        assert model.encoder.molecular_encoders[Modality.AMINO_ACID].neobert.config.vocab_size == VOCAB_SIZE_AMINO_ACID
        assert model.encoder.molecular_encoders[Modality.SMILES].neobert.config.max_length == MAX_LENGTH_SMILES
        assert model.encoder.molecular_encoders[Modality.SMILES].neobert.config.vocab_size == VOCAB_SIZE_SMILES

    def test_compute_mlm_loss(self, model, batch):
        loss = model.compute_mlm_loss(batch, stage="val")
        assert loss is not None
        assert loss.shape == ()

        assert torch.isclose(loss, torch.tensor(3.1852), atol=1e-3)

    @pytest.mark.integration
    def test_integration_compute_mlm_loss_inputs(self, realistic_batch):
        L.seed_everything(1)

        batch, aa_tok_transform, smiles_tok_transform = realistic_batch

        aa_mask_token_id = aa_tok_transform.tokenizer.mask_token_id
        smiles_mask_token_id = smiles_tok_transform.tokenizer.mask_token_id

        model = UMEInteractionLightningModule(
            special_token_mappings={
                Modality.AMINO_ACID: SpecialTokenIds(
                    mask_token_id=aa_mask_token_id,
                    pad_token_id=aa_tok_transform.tokenizer.pad_token_id,
                    special_token_ids=[
                        aa_tok_transform.tokenizer.cls_token_id,
                        aa_tok_transform.tokenizer.pad_token_id,
                    ],
                ),
                Modality.SMILES: SpecialTokenIds(
                    mask_token_id=smiles_mask_token_id,
                    pad_token_id=smiles_tok_transform.tokenizer.pad_token_id,
                    special_token_ids=[
                        smiles_tok_transform.tokenizer.cls_token_id,
                        smiles_tok_transform.tokenizer.pad_token_id,
                    ],
                ),
            },
            mask_probability=0.5,
            encoder_kwargs={
                Modality.AMINO_ACID: {
                    "max_length": MAX_LENGTH_REALISTIC_BATCH,
                    "vocab_size": aa_tok_transform.tokenizer.vocab_size,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 2,
                },
                Modality.SMILES: {
                    "max_length": MAX_LENGTH_REALISTIC_BATCH,
                    "vocab_size": smiles_tok_transform.tokenizer.vocab_size,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 2,
                },
            },
        )

        mock_logits1 = torch.randn(BATCH_SIZE, MAX_LENGTH_REALISTIC_BATCH, aa_tok_transform.tokenizer.vocab_size)
        mock_logits2 = torch.randn(BATCH_SIZE, MAX_LENGTH_REALISTIC_BATCH, smiles_tok_transform.tokenizer.vocab_size)

        with patch.object(model.encoder, "get_logits", return_value=(mock_logits1, mock_logits2)) as mock_get_logits:
            loss = model.compute_mlm_loss(batch, stage="train")
            mock_get_logits.assert_called_once()

            # check the inputs to get_logits
            call_args = mock_get_logits.call_args
            inputs1 = call_args.kwargs["inputs1"]
            inputs2 = call_args.kwargs["inputs2"]

            # decode back first sequence pair of the batch
            aa_decoded = aa_tok_transform.tokenizer.decode(inputs1["input_ids"][0])
            smiles_decoded = smiles_tok_transform.tokenizer.decode(inputs2["input_ids"][0])

            aa_decoded_expected = "<cls> M <mask> <mask> <mask> L <mask> <pad> <pad> <pad>"
            smiles_decoded_expected = "<cls> C <mask> <mask> <mask> O <mask> <eos> <pad> <pad>"

            assert aa_decoded == aa_decoded_expected
            assert smiles_decoded == smiles_decoded_expected

            assert inputs1["attention_mask"][0].int().tolist() == [1] * 7 + [0] * 3
            assert inputs2["attention_mask"][0].int().tolist() == [1] * 8 + [0] * 2

            assert loss is not None
            assert torch.isclose(loss, torch.tensor(5.5167), atol=1e-3)
