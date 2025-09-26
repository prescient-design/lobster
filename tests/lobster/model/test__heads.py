import types
import torch
import torch.nn as nn

from lobster.model._heads import TaskConfig, TaskHead, MultiTaskHead, FlexibleEncoderWithHeads
from lobster.model._activations import get_recommended_activation
from lobster.model.losses._regression import MSELossWithSmoothing


def test_taskconfig_auto_and_validation():
    cfg = TaskConfig(name="reg1", output_dim=1, task_type="regression", pooling="mean")
    assert cfg.activation == get_recommended_activation("regression")
    assert cfg.loss_function == "mse"

    # invalid pooling
    try:
        TaskConfig(name="bad", output_dim=1, pooling="max")
        raise AssertionError("Expected ValueError for bad pooling")
    except ValueError:
        pass

    # binary classification must have output_dim=1
    try:
        TaskConfig(name="bin", output_dim=2, task_type="binary_classification")
        raise AssertionError("Expected ValueError for binary classification output_dim")
    except ValueError:
        pass

    # invalid activation
    try:
        TaskConfig(name="act", output_dim=1, activation="unknown")
        raise AssertionError("Expected ValueError for activation")
    except ValueError:
        pass

    # invalid loss for task type
    try:
        TaskConfig(name="loss", output_dim=1, loss_function="does_not_exist")
        raise AssertionError("Expected ValueError for loss function")
    except ValueError:
        pass


def test_taskhead_forward_shapes():
    batch_size, seq_len, hidden = 2, 5, 16
    hidden_states = torch.randn(batch_size, seq_len, hidden)
    attn = torch.ones(batch_size, seq_len)

    # multiclass classification
    cfg_cls = TaskConfig(name="cls", output_dim=3, task_type="multiclass_classification", pooling="mean")
    head_cls = TaskHead(input_dim=hidden, task_config=cfg_cls, encoder_config=types.SimpleNamespace(hidden_size=hidden))
    out_cls = head_cls(hidden_states, attn)
    assert out_cls.shape == (batch_size, 3)

    # binary classification squeezes last dim
    cfg_bin = TaskConfig(name="bin", output_dim=1, task_type="binary_classification", pooling="mean")
    head_bin = TaskHead(input_dim=hidden, task_config=cfg_bin, encoder_config=types.SimpleNamespace(hidden_size=hidden))
    out_bin = head_bin(hidden_states, attn)
    assert out_bin.shape == (batch_size,)


def test_multitaskhead_forward():
    batch_size, seq_len, hidden = 2, 7, 12
    hidden_states = torch.randn(batch_size, seq_len, hidden)
    attn = torch.ones(batch_size, seq_len)

    tasks = [
        TaskConfig(name="reg", output_dim=1, task_type="regression"),
        TaskConfig(name="mc", output_dim=4, task_type="multiclass_classification"),
    ]
    mt = MultiTaskHead(input_dim=hidden, task_configs=tasks, encoder_config=types.SimpleNamespace(hidden_size=hidden))
    outputs = mt(hidden_states, attn)
    assert set(outputs.keys()) == {"reg", "mc"}
    assert outputs["reg"].shape == (batch_size, 1)
    assert outputs["mc"].shape == (batch_size, 4)


class _DummyEncoderForward(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else 2
        seq = input_ids.shape[1] if input_ids is not None else 5
        hidden = self.config.hidden_size

        class Out:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return Out(torch.randn(batch, seq, hidden))


class _DummyEncoderEmbed(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden_size = hidden

    def embed(self, inputs, aggregate=False, **kwargs):
        batch = inputs["input_ids"].shape[0]
        seq = inputs["input_ids"].shape[1]
        return torch.randn(batch, seq, self.hidden_size)


def test_flexible_encoder_with_heads_forward_and_hidden_states():
    batch, seq, hidden = 3, 6, 10
    input_ids = torch.ones(batch, seq, dtype=torch.long)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)

    tasks = [TaskConfig(name="reg", output_dim=1, task_type="regression")]

    # forward() style encoder, like NeoBERT
    enc_fwd = _DummyEncoderForward(hidden)
    model_fwd = FlexibleEncoderWithHeads(encoder=enc_fwd, task_configs=tasks)
    outputs_fwd = model_fwd(input_ids=input_ids, attention_mask=attention_mask, return_hidden_states=True)
    assert "encoder_outputs" in outputs_fwd and "reg" in outputs_fwd and "hidden_states" in outputs_fwd
    assert outputs_fwd["hidden_states"].shape == (batch, seq, hidden)

    # embed() style encoder, like UME
    enc_emb = _DummyEncoderEmbed(hidden)
    model_emb = FlexibleEncoderWithHeads(encoder=enc_emb, task_configs=tasks, hidden_size=hidden)
    outputs_emb = model_emb(input_ids=input_ids, attention_mask=attention_mask, return_hidden_states=True)
    assert "encoder_outputs" in outputs_emb and "reg" in outputs_emb and "hidden_states" in outputs_emb
    assert outputs_emb["hidden_states"].shape == (batch, seq, hidden)


def test_add_task_and_get_loss_functions():
    hidden = 8
    enc = _DummyEncoderForward(hidden)
    wrapper = FlexibleEncoderWithHeads(encoder=enc, task_configs=[TaskConfig(name="reg", output_dim=1)])
    # Add a multiclass task
    wrapper.add_task(TaskConfig(name="mc", output_dim=5, task_type="multiclass_classification"))
    # Ensure predictions include both when requested
    batch, seq = 2, 4
    outs = wrapper(
        input_ids=torch.ones(batch, seq, dtype=torch.long), attention_mask=torch.ones(batch, seq, dtype=torch.long)
    )
    assert set(outs.keys()) - {"encoder_outputs"} == {"reg", "mc"}

    # Loss functions retrieved from registry
    losses = wrapper.get_loss_functions()
    assert isinstance(losses["reg"], MSELossWithSmoothing)
    assert "mc" in losses
