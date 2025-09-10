import lightning as L
import pytest
import torch
import torch.nn as nn

from lobster.constants import Modality
from lobster.model.ume2 import AuxiliaryRegressionTaskHead, AuxiliaryTask, UMESequenceEncoderLightningModule


class TestSequenceEncoderLightningModule:
    @pytest.fixture
    def model(self):
        L.seed_everything(1)

        model = UMESequenceEncoderLightningModule(
            auxiliary_tasks=[
                AuxiliaryTask(
                    name="aux-task-1",
                    output_dim=1,
                    task_type="regression",
                    pooling="cls",
                    hidden_size=2,
                    dropout=0.1,
                    num_layers=2,
                    loss_weight=1.0,
                ),
            ],
            mask_token_id=103,
            pad_token_id=0,
            special_token_ids=[0, 1, 2],
            mask_probability=1.0,
            seed=1,
            lr=1e-3,
            beta1=0.9,
            beta2=0.98,
            eps=1e-12,
            weight_decay=0.0,
            scheduler="constant_with_warmup",
            scheduler_kwargs={"num_warmup_steps": 1000, "num_training_steps": 10000},
            encoder_kwargs={
                "max_length": 4,
                "vocab_size": 1000,
                "hidden_size": 2,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": 2,
            },
            ckpt_path=None,
        )

        return model

    def test__init__(self, model):
        assert model.mask_token_id == 103
        assert model.pad_token_id == 0
        assert model.special_token_ids == [0, 1, 2]
        assert model.mask_probability == 1.0
        assert model.seed == 1
        assert model.lr == 1e-3
        assert model.beta1 == 0.9
        assert model.beta2 == 0.98
        assert model.eps == 1e-12
        assert model.weight_decay == 0.0
        assert model.scheduler == "constant_with_warmup"
        assert model.scheduler_kwargs == {"num_warmup_steps": 1000, "num_training_steps": 10000}

        assert model.encoder.neobert.config.max_length == 4
        assert model.encoder.neobert.config.pad_token_id == 0
        assert model.encoder.neobert.config.vocab_size == 1000
        assert model.encoder.neobert.config.max_length == 4
        assert model.encoder.neobert.config.hidden_size == 2
        assert model.encoder.neobert.config.num_hidden_layers == 1
        assert model.encoder.neobert.config.num_attention_heads == 1
        assert model.encoder.neobert.config.intermediate_size == 2

        assert len(model.encoder.auxiliary_tasks) == 1
        assert isinstance(model.encoder.auxiliary_tasks, nn.ModuleDict)
        assert isinstance(model.encoder.auxiliary_tasks["aux-task-1"], AuxiliaryRegressionTaskHead)

    def test_compute_mlm_loss(self, model):
        input_ids = torch.randint(3, 100, (1, 4))
        attention_mask = torch.ones(1, 4)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        loss = model.compute_mlm_loss(batch)

        assert torch.is_tensor(loss)
        assert torch.isclose(loss, torch.tensor(7.0376), atol=1e-3)

    def test_embed(self, model):
        ignore_padding = False
        aggregate = True

        inputs = {
            "input_ids": torch.randint(3, 100, (1, 4)),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }

        embeddings = model.embed(inputs, aggregate=aggregate, ignore_padding=ignore_padding)

        assert embeddings.shape == (1, 2)
        assert torch.allclose(embeddings, torch.tensor([[-0.3754, -1.0522]]), atol=1e-3)

    def test_embed_sequences(self, model):
        sequences = ["MYK"]
        embeddings = model.embed_sequences(sequences, modality=Modality.AMINO_ACID, aggregate=True)

        assert embeddings.shape == (1, 2)
        assert torch.allclose(embeddings, torch.tensor([[0.5312, 0.4737]]), atol=1e-3)
