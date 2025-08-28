import torch
import pytest
from lobster.model.neobert.neobert_module import NeoBERTModule


class TestNeoBERTModule:
    @pytest.fixture
    def model(self):
        model = NeoBERTModule(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=1000,
            max_length=512,
            pad_token_id=0,
            mask_token_id=103,
            mask_probability=0.15,
            special_token_ids=[0, 1, 2],
        )

        # Add required attributes for masking
        model.mask_token_id = 999
        model.mask_probability = 0.15
        model.special_token_ids = [0, 1, 2]  # pad, cls, sep tokens
        model.generator = torch.Generator()
        model.generator.manual_seed(42)

        return model

    def test_forward(self, model):
        input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        attention_mask = torch.ones(2, 10)

        output = model.forward(input_ids=input_ids, attention_mask=attention_mask)

        assert "last_hidden_state" in output
        assert output["last_hidden_state"].shape == (2, 10, 64)  # batch_size, seq_len, hidden_size

    def test_get_masked_logits_and_labels(self, model):
        input_ids = torch.randint(3, 1000, (2, 10))  # batch_size=2, seq_len=10, avoid special tokens
        attention_mask = torch.ones(2, 10)

        logits, labels = model.get_masked_logits_and_labels(input_ids=input_ids, attention_mask=attention_mask)

        assert logits.shape == (2, 10, 1000)  # batch_size, seq_len, vocab_size
        assert labels.shape == (2, 10)  # batch_size, seq_len
