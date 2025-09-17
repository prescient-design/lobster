import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Note: For standard regression losses without additional features, use PyTorch's built-in functions:
- torch.nn.MSELoss (mean squared error)
- torch.nn.HuberLoss (robust to outliers)
- torch.nn.SmoothL1Loss (smooth L1 loss)
- torch.nn.L1Loss (mean absolute error)

This module provides enhanced versions with label smoothing, molecular-specific parameterizations,
and uncertainty quantification for specialized applications.
"""


class MSELossWithSmoothing(nn.Module):
    """Mean Squared Error loss with optional label smoothing.

    Supports both simple Gaussian noise smoothing and Cortex-style
    moment averaging for more sophisticated label smoothing.

    Parameters
    ----------
    label_smoothing : float, optional
        Label smoothing factor, by default 0.0
    smoothing_method : str, optional
        Method for label smoothing: "gaussian" or "moment_average", by default "gaussian"
    reduction : str, optional
        Reduction method, by default "mean"

    References
    ----------
    Label smoothing techniques: "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016)

    Examples
    --------
    >>> loss_fn = MSELossWithSmoothing(label_smoothing=0.1, smoothing_method="moment_average")
    >>> pred = torch.randn(32, 1)
    >>> target = torch.randn(32, 1)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(self, label_smoothing: float = 0.0, smoothing_method: str = "gaussian", reduction: str = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.smoothing_method = smoothing_method
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        if self.label_smoothing > 0.0:
            if self.smoothing_method == "gaussian":
                # Simple Gaussian noise smoothing
                noise = torch.randn_like(target) * self.label_smoothing
                target = target + noise
            elif self.smoothing_method == "moment_average":
                # Cortex-style moment averaging (simplified version)
                target = self._apply_moment_averaging(input, target)

        return F.mse_loss(input, target, reduction=self.reduction)

    def _apply_moment_averaging(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply moment averaging label smoothing (inspired by Cortex).

        This is a simplified version of Cortex's moment averaging.

        References
        ----------
        Moment averaging label smoothing: Cortex (https://github.com/prescient-design/cortex/blob/main/cortex/model/leaf/_regressor_leaf.py)
        Moment averaging theory: "Adaptive Approximate Inference in Bayesian Neural Networks" (Maddison et al., 2017)
        https://www.cs.toronto.edu/~cmaddis/pubs/aais.pdf
        """
        # Compute standard statistics
        pred_mean = pred.mean()
        pred.var()

        # Apply smoothing between prediction and target statistics
        smoothed_target = self.label_smoothing * pred_mean + (1.0 - self.label_smoothing) * target

        return smoothed_target


class HuberLossWithSmoothing(nn.Module):
    """Huber loss (smooth L1 loss) with optional label smoothing.

    More robust to outliers than MSE loss.

    Parameters
    ----------
    delta : float, optional
        Threshold for switching between L1 and L2 loss, by default 1.0
    label_smoothing : float, optional
        Label smoothing factor, by default 0.0
    reduction : str, optional
        Reduction method, by default "mean"

    Examples
    --------
    >>> loss_fn = HuberLossWithSmoothing(delta=0.5, label_smoothing=0.05)
    >>> pred = torch.randn(32, 1)
    >>> target = torch.randn(32, 1)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(self, delta: float = 1.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        if self.label_smoothing > 0.0:
            noise = torch.randn_like(target) * self.label_smoothing
            target = target + noise

        return F.huber_loss(input, target, delta=self.delta, reduction=self.reduction)


class SmoothL1LossWithSmoothing(nn.Module):
    """Smooth L1 loss with optional label smoothing.

    Equivalent to Huber loss with delta=1.0.

    Parameters
    ----------
    beta : float, optional
        Threshold for switching between L1 and L2 loss, by default 1.0
    label_smoothing : float, optional
        Label smoothing factor, by default 0.0
    reduction : str, optional
        Reduction method, by default "mean"

    Examples
    --------
    >>> loss_fn = SmoothL1LossWithSmoothing(beta=0.5)
    >>> pred = torch.randn(32, 1)
    >>> target = torch.randn(32, 1)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(self, beta: float = 1.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        if self.label_smoothing > 0.0:
            noise = torch.randn_like(target) * self.label_smoothing
            target = target + noise

        return F.smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)


class ExponentialParameterizedLoss(nn.Module):
    """Exponential parameterized regression loss.

    Useful for molecular property prediction where the target values
    might span several orders of magnitude.

    Parameters
    ----------
    base_loss : str, optional
        Base loss function to use, by default "mse"
    reduction : str, optional
        Reduction method, by default "mean"

    Examples
    --------
    >>> loss_fn = ExponentialParameterizedLoss(base_loss="huber")
    >>> pred = torch.randn(32, 1)
    >>> target = torch.randn(32, 1)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(self, base_loss: str = "mse", reduction: str = "mean"):
        super().__init__()
        self.base_loss = base_loss
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Predicted log values
        target : torch.Tensor
            Target log values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Convert from log space to linear space
        pred_linear = torch.exp(input)
        target_linear = torch.exp(target)

        # Apply base loss function
        if self.base_loss == "mse":
            loss = F.mse_loss(pred_linear, target_linear, reduction=self.reduction)
        elif self.base_loss == "huber":
            loss = F.huber_loss(pred_linear, target_linear, reduction=self.reduction)
        elif self.base_loss == "smooth_l1":
            loss = F.smooth_l1_loss(pred_linear, target_linear, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported base loss: {self.base_loss}")

        return loss


class NaturalGaussianLoss(nn.Module):
    """Natural Gaussian loss with uncertainty estimation (inspired by Cortex).

    This loss function predicts both mean and variance, allowing for
    uncertainty quantification in regression tasks. Particularly useful
    for molecular property prediction where uncertainty matters.

    References
    ----------
    Natural Gaussian parameterization: Cortex (https://github.com/prescient-design/cortex/blob/main/cortex/model/leaf/_regressor_leaf.py)

    Parameters
    ----------
    log_scale_min : float, optional
        Minimum log scale (for numerical stability), by default -8.0
    log_scale_max : float, optional
        Maximum log scale, by default 8.0
    label_smoothing : float, optional
        Label smoothing factor, by default 0.0
    reduction : str, optional
        Reduction method, by default "mean"

    Examples
    --------
    >>> loss_fn = NaturalGaussianLoss()
    >>> # Predictions should be [mean, log_scale] concatenated
    >>> pred = torch.randn(32, 2)  # mean and log_scale
    >>> target = torch.randn(32, 1)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        log_scale_min: float = -8.0,
        log_scale_max: float = 8.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values of shape (N, 2) where [:, 0] is mean and [:, 1] is log_scale
        target : torch.Tensor
            Target values of shape (N, 1)

        Returns
        -------
        torch.Tensor
            Negative log likelihood loss
        """
        if input.shape[-1] != 2:
            raise ValueError("Input must have shape (N, 2) for [mean, log_scale]")

        mean = input[..., 0:1]
        log_scale = input[..., 1:2]

        # Clamp log_scale for numerical stability
        log_scale = torch.clamp(log_scale, self.log_scale_min, self.log_scale_max)
        scale = torch.exp(log_scale)

        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            noise = torch.randn_like(target) * self.label_smoothing
            target = target + noise

        # Compute NLL and apply requested reduction
        dist = torch.distributions.Normal(mean, scale)
        nll = -dist.log_prob(target)

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll


class MixtureGaussianNLLLoss(nn.Module):
    """Negative log-likelihood loss for Mixture Density Networks (diagonal Gaussians).

    Expects predictions that parameterize a K-component diagonal Gaussian mixture over
    a D-dimensional target.

    Input format per sample: [logits_K, means_{K x D}, log_scales_{K x D}].

    Parameters
    ----------
    min_log_scale : float
        Lower clamp for log-scales for numerical stability
    max_log_scale : float
        Upper clamp for log-scales for numerical stability
    reduction : str
        One of {"mean", "sum", "none"}
    """

    def __init__(
        self,
        min_log_scale: float = -8.0,
        max_log_scale: float = 8.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MDN negative log-likelihood.

        Parameters
        ----------
        input : Tensor
            Shape (N, P) where P = K * (2*D + 1)
        target : Tensor
            Shape (N,) or (N, 1) or (N, D) where D is target dimension
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1)

        N, P = input.shape
        D = target.shape[-1]

        # Infer K from P and D: P = K*(2D+1)
        two_d_plus_1 = 2 * D + 1
        if P % two_d_plus_1 != 0:
            raise ValueError(f"Invalid MDN param size: got input dim {P} that is not divisible by 2*D+1={two_d_plus_1}")
        K = P // two_d_plus_1

        # Split parameters
        logits = input[:, :K]  # (N, K)
        means = input[:, K : K + K * D]  # (N, K*D)
        log_scales = input[:, K + K * D :]  # (N, K*D)

        means = means.view(N, K, D)
        log_scales = log_scales.view(N, K, D)

        # Clamp and exponentiate to get scales
        log_scales = torch.clamp(log_scales, self.min_log_scale, self.max_log_scale)
        scales = torch.exp(log_scales)

        # Expand target to (N, K, D)
        target_exp = target.unsqueeze(1).expand(-1, K, -1)

        # Diagonal Gaussian log-prob summed over D
        var = scales * scales
        log_det = torch.sum(torch.log(var + 1e-12), dim=-1) * 0.5  # (N, K)
        sq_mahalanobis = torch.sum(((target_exp - means) ** 2) / (var + 1e-12), dim=-1) * 0.5  # (N, K)
        log_two_pi_d = 0.5 * D * torch.log(torch.tensor(2 * torch.pi, device=input.device, dtype=input.dtype))
        log_prob = -(log_two_pi_d + log_det + sq_mahalanobis)  # (N, K)

        # Mixture log-sum-exp
        logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # stable softmax in log-space
        # But we want logsumexp(logits + log_prob) directly for stability
        nll = -torch.logsumexp(logits + log_prob, dim=-1)  # (N,)

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll
