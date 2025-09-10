import torch

from lobster.model.losses._classification import FocalLoss


def test_focal_loss_shapes_and_reduction():
    torch.manual_seed(0)
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))

    # mean reduction
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0

    # sum reduction
    loss_fn_sum = FocalLoss(alpha=1.0, gamma=2.0, reduction="sum")
    loss_sum = loss_fn_sum(logits, targets)
    assert loss_sum.ndim == 0

    # none reduction returns vector
    loss_fn_none = FocalLoss(alpha=1.0, gamma=2.0, reduction="none")
    loss_vec = loss_fn_none(logits, targets)
    assert loss_vec.shape == (10,)


