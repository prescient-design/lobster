from typing import Dict, Union
import torch.nn as nn

from ._classification import FocalLoss
from ._regression import (
    MSELossWithSmoothing,
    HuberLossWithSmoothing,
    SmoothL1LossWithSmoothing,
    ExponentialParameterizedLoss,
    NaturalGaussianLoss,
)


# Available loss functions for each task type
AVAILABLE_LOSS_FUNCTIONS = {
    "regression": {
        "mse": MSELossWithSmoothing,
        "huber": HuberLossWithSmoothing, 
        "smooth_l1": SmoothL1LossWithSmoothing,
        "exponential": ExponentialParameterizedLoss,
        "gaussian": NaturalGaussianLoss,
        "mse_pytorch": nn.MSELoss,
        "huber_pytorch": nn.HuberLoss,
        "smooth_l1_pytorch": nn.SmoothL1Loss,
        "l1": nn.L1Loss,
    },
    "binary_classification": {
        "bce_logits": nn.BCEWithLogitsLoss,
        "bce": nn.BCELoss,
        "focal": FocalLoss,
    },
    "multiclass_classification": {
        "cross_entropy": nn.CrossEntropyLoss,
        "focal": FocalLoss,
        "nll": nn.NLLLoss,
    }
}

# Default loss functions for each task type
DEFAULT_LOSS_FUNCTIONS = {
    "regression": "mse",
    "binary_classification": "bce_logits", 
    "multiclass_classification": "focal",
}


def get_loss_function(task_type: str, loss_name: str):
    """Get a loss function instance for a task type and name.
    
    Parameters
    ----------
    task_type : str
        Type of task ("regression", "binary_classification", "multiclass_classification")
    loss_name : str
        Name of loss function. Use "auto" for task-specific defaults.
        
    Returns
    -------
    nn.Module
        Instantiated loss function
    """
    # Handle auto selection
    if loss_name == "auto":
        loss_name = DEFAULT_LOSS_FUNCTIONS[task_type]
    
    # Get loss class from registry and instantiate
    loss_class = AVAILABLE_LOSS_FUNCTIONS[task_type][loss_name]
    return loss_class()
