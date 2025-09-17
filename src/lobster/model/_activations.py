from collections.abc import Callable
import torch.nn as nn

from .neobert._swiglu import SwiGLU


ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,  # Alias for SiLU
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "swiglu": None,  # Special case - handled separately
}


def get_activation_function(
    activation_name: str, input_dim: int | None = None, hidden_dim: int | None = None, **kwargs
) -> nn.Module | Callable:
    """Get activation function by name."""
    activation_name = activation_name.lower()

    if activation_name == "swiglu":
        if input_dim is None or hidden_dim is None:
            raise ValueError("SwiGLU requires both input_dim and hidden_dim")
        return lambda out_dim: SwiGLU(input_dim, hidden_dim, out_dim)

    if activation_name not in ACTIVATION_FUNCTIONS:
        available = [k for k in ACTIVATION_FUNCTIONS.keys() if k != "swiglu"]
        raise ValueError(f"Unknown activation function: {activation_name}. Available: {available}")

    activation_class = ACTIVATION_FUNCTIONS[activation_name]

    # Handle activations that accept parameters
    if activation_name == "leaky_relu":
        return activation_class(negative_slope=kwargs.get("negative_slope", 0.01))
    elif activation_name == "prelu":
        return activation_class(num_parameters=kwargs.get("num_parameters", 1))
    elif activation_name == "elu":
        return activation_class(alpha=kwargs.get("alpha", 1.0))
    else:
        return activation_class()


def get_recommended_activation(task_type: str) -> str:
    """Get recommended activation function for a task type."""
    recommendations = {"regression": "silu", "binary_classification": "gelu", "multiclass_classification": "gelu"}
    return recommendations.get(task_type, "gelu")
