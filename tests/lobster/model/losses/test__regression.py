import torch
import torch.nn.functional as F

from lobster.model.losses._regression import (
    MSELossWithSmoothing,
    HuberLossWithSmoothing,
    SmoothL1LossWithSmoothing,
    ExponentialParameterizedLoss,
    NaturalGaussianLoss,
)


def test_mse_with_smoothing_gaussian_and_moment():
    torch.manual_seed(0)
    pred = torch.randn(16, 1)
    target = torch.randn(16, 1)

    # No smoothing equals PyTorch MSE
    loss_pt = F.mse_loss(pred, target)
    loss = MSELossWithSmoothing(label_smoothing=0.0)(pred, target)
    assert torch.isclose(loss, loss_pt, atol=1e-6)

    # Gaussian smoothing equals MSE on (target + deterministic noise)
    torch.manual_seed(1234)
    loss_g = MSELossWithSmoothing(label_smoothing=0.1, smoothing_method="gaussian")(pred, target)
    torch.manual_seed(1234)
    noise = torch.randn_like(target) * 0.1
    expected_g = F.mse_loss(pred, target + noise)
    assert torch.isclose(loss_g, expected_g, atol=1e-6)

    # Moment averaging equals MSE on smoothed target
    alpha = 0.1
    loss_m = MSELossWithSmoothing(label_smoothing=alpha, smoothing_method="moment_average")(pred, target)
    pred_mean = pred.mean()
    smoothed_target = alpha * pred_mean + (1.0 - alpha) * target
    expected_m = F.mse_loss(pred, smoothed_target)
    assert torch.isclose(loss_m, expected_m, atol=1e-6)


def test_huber_and_smoothl1_with_smoothing():
    torch.manual_seed(0)
    pred = torch.randn(8, 1)
    target = torch.randn(8, 1)

    # With no smoothing, equals PyTorch huber
    loss_h_no = HuberLossWithSmoothing(delta=1.0, label_smoothing=0.0)(pred, target)
    expected_h = F.huber_loss(pred, target, delta=1.0)
    assert torch.isclose(loss_h_no, expected_h, atol=1e-6)

    # With smoothing, remains finite
    loss_h = HuberLossWithSmoothing(delta=1.0, label_smoothing=0.05)(pred, target)
    assert torch.isfinite(loss_h)

    # SmoothL1 equals PyTorch when no smoothing
    loss_s_no = SmoothL1LossWithSmoothing(beta=1.0, label_smoothing=0.0)(pred, target)
    expected_s = F.smooth_l1_loss(pred, target, beta=1.0)
    assert torch.isclose(loss_s_no, expected_s, atol=1e-6)

    loss_s = SmoothL1LossWithSmoothing(beta=1.0, label_smoothing=0.05)(pred, target)
    assert torch.isfinite(loss_s)


def test_exponential_parameterized_loss_and_gaussian_nll():
    torch.manual_seed(0)
    pred_log = torch.randn(6, 1)
    target_log = torch.randn(6, 1)

    # exp-param loss (mse base) equals MSE on exponentials
    loss_exp = ExponentialParameterizedLoss(base_loss="mse")(pred_log, target_log)
    expected_exp = F.mse_loss(torch.exp(pred_log), torch.exp(target_log))
    assert torch.isclose(loss_exp, expected_exp, atol=1e-6)

    # natural gaussian expects [mean, log_scale]
    pred_natural = torch.randn(6, 2)
    target = torch.randn(6, 1)
    nll = NaturalGaussianLoss()(pred_natural, target)
    assert torch.isfinite(nll)

    # Closed-form check for a simple case: target == mean
    mean = torch.zeros(4, 1)
    log_scale_small = torch.full((4, 1), -4.0)
    log_scale_large = torch.full((4, 1), 2.0)
    inp_small = torch.cat([mean, log_scale_small], dim=-1)
    inp_large = torch.cat([mean, log_scale_large], dim=-1)
    target_zero = torch.zeros(4, 1)

    loss_small = NaturalGaussianLoss()(inp_small, target_zero)
    loss_large = NaturalGaussianLoss()(inp_large, target_zero)
    # With target==mean, nll = log(scale) + const, so larger scale -> larger loss
    assert loss_small < loss_large

    # wrong shape raises
    try:
        NaturalGaussianLoss()(torch.randn(6, 3), target)
        assert False, "Expected ValueError on wrong shape"
    except ValueError:
        pass



