from lobster.model.ume2 import UMEInteraction, UMEInteractionLightningModule, SpecialTokenIds
from lobster.constants import Modality
import torch
import lightning as L
import pytest

BATCH_SIZE = 3
VOCAB_SIZE_AMINO_ACID = 25
VOCAB_SIZE_SMILES = 13
MAX_LENGTH_AMINO_ACID = 5
MAX_LENGTH_SMILES = 4


@pytest.fixture
def batch():
    return {
        "input_ids1": torch.randint(0, VOCAB_SIZE_AMINO_ACID, (BATCH_SIZE, MAX_LENGTH_AMINO_ACID)),
        "attention_mask1": torch.ones((BATCH_SIZE, MAX_LENGTH_AMINO_ACID)),
        "modality1": [Modality.AMINO_ACID] * BATCH_SIZE,
        "input_ids2": torch.randint(0, VOCAB_SIZE_SMILES, (BATCH_SIZE, MAX_LENGTH_SMILES)),
        "attention_mask2": torch.ones((BATCH_SIZE, MAX_LENGTH_SMILES)),
        "modality2": [Modality.SMILES] * BATCH_SIZE,
    }


class TestUMEInteraction:
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
