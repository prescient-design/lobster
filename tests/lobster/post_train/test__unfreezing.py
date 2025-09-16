import torch.nn as nn

from lobster.post_train.unfreezing import (
    set_unfrozen_layers,
    _freeze_all_parameters,
    _unfreeze_all_parameters,
    _unfreeze_last_n_layers,
    get_layer_wise_parameter_groups,
    progressive_unfreezing_schedule,
)


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


class _DummyEncoder(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList([_DummyLayer() for _ in range(num_layers)])


class _DummyInnerModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.encoder = _DummyEncoder(num_layers)
        self.pooler = None


class _DummyUME(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        # Simulate UME structure with registered submodules: model.encoder.layer
        self.model = _DummyInnerModel(num_layers)


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_freeze_and_unfreeze_all():
    m = _DummyUME(num_layers=4)
    _freeze_all_parameters(m)
    assert _count_trainable(m) == 0
    _unfreeze_all_parameters(m)
    assert _count_trainable(m) > 0


def test_set_unfrozen_layers_behavior():
    m = _DummyUME(num_layers=5)

    # Partial: unfreeze last 2
    _freeze_all_parameters(m)
    set_unfrozen_layers(m, 2)
    trainable_ids = {id(p) for p in m.parameters() if p.requires_grad}
    all_layers = list(m.model.encoder.layer)
    unfrozen_ids = {id(p) for lyr in all_layers[-2:] for p in lyr.parameters()}
    frozen_ids = {id(p) for lyr in all_layers[:-2] for p in lyr.parameters()}
    assert unfrozen_ids.issubset(trainable_ids)
    assert trainable_ids.isdisjoint(frozen_ids)

    # Freeze all
    set_unfrozen_layers(m, 0)
    assert _count_trainable(m) == 0

    # Unfreeze all via -1
    set_unfrozen_layers(m, -1)
    assert _count_trainable(m) > 0

    # Clamp: request more than available -> unfreeze all
    _freeze_all_parameters(m)
    set_unfrozen_layers(m, 10)
    trainable_layer_flags = [any(p.requires_grad for p in lyr.parameters()) for lyr in all_layers]
    assert all(trainable_layer_flags)

    # Explicit cases using the new API
    set_unfrozen_layers(m, 0)
    assert _count_trainable(m) == 0
    set_unfrozen_layers(m, -1)
    assert _count_trainable(m) > 0
    set_unfrozen_layers(m, 3)
    trainable_layer_flags = [any(p.requires_grad for p in lyr.parameters()) for lyr in all_layers]
    assert trainable_layer_flags[-3:] == [True, True, True]
    assert trainable_layer_flags[:-3] == [False] * (len(trainable_layer_flags) - 3)


def test_layer_wise_parameter_groups_and_progressive():
    m = _DummyUME(num_layers=3)
    groups = get_layer_wise_parameter_groups(m, base_lr=1e-4, decay_factor=0.5)
    # 3 layer groups + possibly one 'other' group if present
    assert len(groups) >= 3
    lrs = [g["lr"] for g in groups if isinstance(g.get("layer_id"), int)]
    assert lrs == sorted(lrs)  # earlier layers lower lr

    # progressive unfreezing: at epoch 5 with schedule [2,5] unfreezes 2 layers
    _freeze_all_parameters(m)
    progressive_unfreezing_schedule(m, current_epoch=5, unfreeze_schedule=[2, 5])
    trainable = _count_trainable(m)
    assert trainable > 0


