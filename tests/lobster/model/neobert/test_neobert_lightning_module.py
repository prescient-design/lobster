import torch
import pytest
import lightning as L
from lobster.model.neobert.neobert_lightning_module import NeoBERTLightningModule


class TestNeoBERTLightningModule:
    @pytest.fixture
    def model(self):
        L.seed_everything(1)

        model = NeoBERTLightningModule(
            mask_token_id=103,
            pad_token_id=0,
            special_token_ids=[0, 1, 2],
            mask_probability=0.15,
            seed=1,
            lr=1e-3,
            beta1=0.9,
            beta2=0.98,
            eps=1e-12,
            weight_decay=0.0,
            scheduler="constant_with_warmup",
            scheduler_kwargs={"num_warmup_steps": 1000, "num_training_steps": 10000},
            model_kwargs={"max_length": 512, "vocab_size": 1000},
            ckpt_path=None,
        )

        return model

    def test__init__(self, model):
        assert model.mask_token_id == 103
        assert model.pad_token_id == 0
        assert model.special_token_ids == [0, 1, 2]
        assert model.mask_probability == 0.15
        assert model.seed == 1
        assert model.lr == 1e-3
        assert model.beta1 == 0.9
        assert model.beta2 == 0.98
        assert model.eps == 1e-12
        assert model.weight_decay == 0.0
        assert model.scheduler == "constant_with_warmup"
        assert model.scheduler_kwargs == {"num_warmup_steps": 1000, "num_training_steps": 10000}

        assert model.model.config.max_length == 512
        assert model.model.config.pad_token_id == 0
        assert model.model.config.vocab_size == 1000
        assert model.model.config.max_length == 512

    @pytest.mark.integration
    def test_compute_mlm_loss(self, model):
        input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        attention_mask = torch.ones(2, 10)

        loss = model.compute_mlm_loss(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.is_tensor(loss)
        assert torch.isclose(loss, torch.tensor(6.5624), atol=1e-2)
