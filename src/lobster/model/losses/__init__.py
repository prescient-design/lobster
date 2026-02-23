from ._infonce_loss import InfoNCELoss
from ._symile_loss import SymileLoss
from ._classification import FocalLoss
from ._regression import (
    MSELossWithSmoothing,
    HuberLossWithSmoothing,
    SmoothL1LossWithSmoothing,
    ExponentialParameterizedLoss,
    NaturalGaussianLoss,
    MixtureGaussianNLLLoss,
)
from ._diffusion_loss import (
    DiffusionLoss,
    SimpleMLPAdaLN,
    create_diffusion_loss,
)

# Import registry data
from ._registry import (
    AVAILABLE_LOSS_FUNCTIONS,
    DEFAULT_LOSS_FUNCTIONS,
    get_loss_function,
)

__all__ = [
    # Loss function classes
    "InfoNCELoss",
    "SymileLoss",
    "FocalLoss",
    "MSELossWithSmoothing",
    "HuberLossWithSmoothing",
    "SmoothL1LossWithSmoothing",
    "ExponentialParameterizedLoss",
    "NaturalGaussianLoss",
    "MixtureGaussianNLLLoss",
    # Diffusion loss for continuous tokens (MAR-style)
    "DiffusionLoss",
    "SimpleMLPAdaLN",
    "create_diffusion_loss",
    # Registry constants and function
    "AVAILABLE_LOSS_FUNCTIONS",
    "DEFAULT_LOSS_FUNCTIONS",
    "get_loss_function",
]
