from dataclasses import dataclass
from typing import Any, Literal
import logging

import torch.nn as nn
from torch import Tensor

from ._utils import POOLERS
from ._activations import get_activation_function, get_recommended_activation, ACTIVATION_FUNCTIONS
from .losses._registry import get_loss_function, DEFAULT_LOSS_FUNCTIONS, AVAILABLE_LOSS_FUNCTIONS

logger = logging.getLogger(__name__)


"""Flexible task heads for molecular property prediction.

This module provides a flexible architecture for adding task-specific heads
to _ume (modernBERT) and _neo (NeoBERT) models. It supports multiple task types and uses the existing
pooling infrastructure from _pooler.py and activation functions from _activations.py.
"""


@dataclass
class TaskConfig:
    """Configuration for a prediction task.

    Parameters
    ----------
    name : str
        Unique identifier for the task
    output_dim : int
        Output dimension (1 for regression, num_classes for classification)
    task_type : Literal["regression", "binary_classification", "multiclass_classification"]
        Type of prediction task
    pooling : Literal["cls", "mean", "attn", "weighted_mean"]
        How to pool sequence representations (uses existing poolers from _pooler.py)
    hidden_sizes : List[int], optional
        Hidden layer dimensions. If None, uses [input_dim // 2]
    dropout : float, default=0.1
        Dropout probability
    activation : str, default="auto"
        Activation function name. Use "auto" for task-specific recommendations.
        Available: "relu", "gelu", "silu", "swish", "tanh", "leaky_relu", "elu", "mish", "swiglu", etc.
        See _activations.py for full list and recommendations.
    loss_function : str, default="auto"
        Loss function name. Use "auto" for task-specific defaults.
        Available options depend on task_type - see AVAILABLE_LOSS_FUNCTIONS.
    loss_weight : float, default=1.0
        Weight for this task's loss in multi-task learning
    mixture_components : int, optional
        If using an MDN loss (e.g., 'mdn_gaussian'), number of mixture components K.
    """

    name: str
    output_dim: int
    task_type: Literal["regression", "binary_classification", "multiclass_classification"] = "regression"
    pooling: Literal["cls", "mean", "attn", "weighted_mean"] = "mean"
    hidden_sizes: list[int] | None = None
    dropout: float = 0.1
    activation: str = "auto"
    loss_function: str = "auto"
    loss_weight: float = 1.0
    mixture_components: int | None = None

    def __post_init__(self):
        if self.pooling not in POOLERS:
            raise ValueError(f"Unsupported pooling type: {self.pooling}. Available: {list(POOLERS.keys())}")

        if self.loss_weight <= 0 or self.loss_weight > 1:
            raise ValueError(f"Loss weight must be between 0 and 1: {self.loss_weight}")

        if self.task_type not in {"regression", "binary_classification", "multiclass_classification"}:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        if self.task_type == "binary_classification" and self.output_dim != 1:
            raise ValueError("Binary classification should have output_dim=1")

        # Handle auto activation selection
        if self.activation == "auto":
            self.activation = get_recommended_activation(self.task_type)

        # Validate activation function
        if self.activation.lower() not in ACTIVATION_FUNCTIONS and self.activation.lower() != "swiglu":
            available = list(ACTIVATION_FUNCTIONS.keys())
            raise ValueError(f"Unknown activation function: {self.activation}. Available: {available}")

        # Handle auto loss function selection and validation
        if self.loss_function == "auto":
            self.loss_function = DEFAULT_LOSS_FUNCTIONS[self.task_type]

        # Validate loss function
        available_losses = AVAILABLE_LOSS_FUNCTIONS.get(self.task_type, {})
        if self.loss_function not in available_losses:
            raise ValueError(
                f"Unknown loss function '{self.loss_function}' for task type '{self.task_type}'. "
                f"Available: {list(available_losses.keys())}"
            )

        # MDN specific validation
        if self.task_type == "regression" and self.loss_function == "mdn_gaussian":
            if self.mixture_components is None or self.mixture_components <= 0:
                raise ValueError("For 'mdn_gaussian' loss, 'mixture_components' must be a positive integer")


class TaskHead(nn.Module):
    """Generic task head for molecular property prediction."""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        encoder_config: Any | None = None,
    ):
        super().__init__()
        self.task_config = task_config

        # Create pooler using existing infrastructure
        pooler_class = POOLERS[task_config.pooling]

        # Create a minimal config object if encoder_config is not provided
        if encoder_config is None:
            encoder_config = type("Config", (), {"hidden_size": input_dim})()

        self.pooler = pooler_class(encoder_config)

        # Build MLP layers
        hidden_sizes = task_config.hidden_sizes or [input_dim // 2]
        current_dim = input_dim

        # Determine output dimensionality. For MDN regression with diagonal Gaussians,
        # output size = K * (2*D + 1), where D = task_config.output_dim
        is_mdn = task_config.task_type == "regression" and task_config.loss_function == "mdn_gaussian"
        if is_mdn:
            if task_config.output_dim <= 0:
                raise ValueError("TaskConfig.output_dim must be >= 1 for MDN regression")
            if task_config.mixture_components is None:
                raise ValueError("TaskConfig.mixture_components must be set for MDN regression")
            computed_output_dim = task_config.mixture_components * (2 * task_config.output_dim + 1)
        else:
            computed_output_dim = task_config.output_dim

        # Handle SwiGLU specially since it replaces both linear + activation
        if task_config.activation.lower() == "swiglu":
            layers = []
            for i, hidden_size in enumerate(hidden_sizes):
                # SwiGLU combines linear transformation and activation
                swiglu_factory = get_activation_function("swiglu", input_dim=current_dim, hidden_dim=hidden_size * 2)
                layers.extend(
                    [
                        swiglu_factory(hidden_size),  # SwiGLU layer
                        nn.Dropout(task_config.dropout),
                    ]
                )
                current_dim = hidden_size

            # Output layer
            layers.append(nn.Linear(current_dim, computed_output_dim))
            self.mlp = nn.Sequential(*layers)

        else:
            # Standard MLP with regular activations
            layers = []
            activation_fn = get_activation_function(task_config.activation)

            # Hidden layers
            for hidden_size in hidden_sizes:
                layers.extend([nn.Linear(current_dim, hidden_size), activation_fn, nn.Dropout(task_config.dropout)])
                current_dim = hidden_size

            # Output layer
            layers.append(nn.Linear(current_dim, computed_output_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        hidden_states : Tensor
            Shape (batch_size, seq_len, hidden_size)
        attention_mask : Tensor, optional
            Shape (batch_size, seq_len)
        """
        # Normalize attention_mask to expected shape (batch_size, seq_len)
        if attention_mask is not None and attention_mask.dim() == 3:
            # Incoming masks may be (batch_size, 1, seq_len); squeeze the singleton dim
            attention_mask = attention_mask.squeeze(1)

        # Pool sequence representations using existing poolers
        pooled = self.pooler(hidden_states, input_mask=attention_mask)

        # Pass through MLP
        output = self.mlp(pooled)

        # For binary classification, ensure single output
        if self.task_config.task_type == "binary_classification":
            output = output.squeeze(-1)

        return output


class MultiTaskHead(nn.Module):
    """Multi-task head that can handle multiple prediction tasks simultaneously."""

    def __init__(
        self,
        input_dim: int,
        task_configs: list[TaskConfig],
        encoder_config: Any | None = None,
    ):
        super().__init__()
        self.task_configs = {config.name: config for config in task_configs}

        # Create task-specific heads
        self.task_heads = nn.ModuleDict(
            {config.name: TaskHead(input_dim, config, encoder_config) for config in task_configs}
        )

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None, task_names: list[str] | None = None
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        hidden_states : Tensor
            Shape (batch_size, seq_len, hidden_size)
        attention_mask : Tensor, optional
            Shape (batch_size, seq_len)
        task_names : List[str], optional
            Which tasks to compute. If None, computes all tasks.
        """
        if task_names is None:
            task_names = list(self.task_heads.keys())

        outputs = {}
        for task_name in task_names:
            if task_name in self.task_heads:
                outputs[task_name] = self.task_heads[task_name](hidden_states, attention_mask)
            else:
                logger.warning(f"Task '{task_name}' not found in available tasks: {list(self.task_heads.keys())}")

        return outputs


class FlexibleEncoderWithHeads(nn.Module):
    """Flexible encoder wrapper that can work with any base model and task heads.

    This class provides a generic interface for combining any encoder model
    with flexible task-specific heads for molecular property prediction.
    """

    def __init__(
        self,
        encoder: nn.Module,
        task_configs: list[TaskConfig] | None = None,
        encoder_output_key: str = "last_hidden_state",
        hidden_size: int | None = None,
    ):
        """
        Parameters
        ----------
        encoder : nn.Module
            Base encoder model (e.g., BERT, ESM, UME, etc.)
        task_configs : List[TaskConfig], optional
            Configuration for prediction tasks
        encoder_output_key : str, default="last_hidden_state"
            Key to extract hidden states from encoder output
        hidden_size : int, optional
            Hidden size of encoder. If None, tries to infer from encoder config
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_output_key = encoder_output_key

        # Try to infer hidden size from encoder config
        if hidden_size is None:
            if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
                hidden_size = encoder.config.hidden_size
            elif hasattr(encoder, "hidden_size"):
                hidden_size = encoder.hidden_size
            else:
                raise ValueError("Could not infer hidden_size from encoder. Please provide it explicitly.")

        self.hidden_size = hidden_size

        # Create task heads if provided
        if task_configs is not None:
            self.task_head = MultiTaskHead(
                hidden_size, task_configs, encoder.config if hasattr(encoder, "config") else None
            )
            self.task_configs = {config.name: config for config in task_configs}
        else:
            self.task_head = None
            self.task_configs = {}

    def add_task(self, task_config: TaskConfig):
        """Add a new task to the model."""
        encoder_config = self.encoder.config if hasattr(self.encoder, "config") else None

        if self.task_head is None:
            self.task_head = MultiTaskHead(self.hidden_size, [task_config], encoder_config)
        else:
            # Add new task head
            self.task_head.task_heads[task_config.name] = TaskHead(self.hidden_size, task_config, encoder_config)
            self.task_head.task_configs[task_config.name] = task_config

        self.task_configs[task_config.name] = task_config

    def _get_hidden_states(
        self, input_ids: Tensor | None, attention_mask: Tensor | None, **kwargs
    ) -> tuple[Tensor, Any]:
        """
        Get token-level hidden states from encoder.

        Handles the specific encoder types in our codebase:
        - UME: Uses .embed(aggregate=False)
        - NeoBERT/FlexBERT: Uses .forward() -> .last_hidden_state

        Returns
        -------
        tuple[Tensor, Any]
            (hidden_states, encoder_outputs) where hidden_states are token-level embeddings
        """
        # UME models: Use .embed() method with aggregate=False for token-level embeddings
        if hasattr(self.encoder, "embed"):
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            hidden_states = self.encoder.embed(inputs, aggregate=False, **kwargs)
            # For UME, encoder_outputs is the same as hidden_states since .embed() is the main interface
            encoder_outputs = hidden_states
            return hidden_states, encoder_outputs

        # NeoBERT/FlexBERT/LMBase models: Use standard forward() method
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Extract last_hidden_state from output
        if hasattr(encoder_outputs, "last_hidden_state"):
            # NeoBERT/FlexBERT: BaseModelOutput.last_hidden_state
            hidden_states = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, dict) and "last_hidden_state" in encoder_outputs:
            # Dict output with last_hidden_state key
            hidden_states = encoder_outputs["last_hidden_state"]
        else:
            raise ValueError(
                f"Unsupported encoder output format: {type(encoder_outputs)}. "
                f"Expected BaseModelOutput with .last_hidden_state or dict with 'last_hidden_state' key."
            )

        return hidden_states, encoder_outputs

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        task_names: list[str] | None = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        input_ids : Tensor, optional
            Input token IDs
        attention_mask : Tensor, optional
            Attention mask
        task_names : List[str], optional
            Which tasks to compute predictions for
        return_hidden_states : bool, default=False
            Whether to return the raw hidden states
        **kwargs
            Additional arguments passed to the encoder
        """
        # Smart encoder interface detection and calling
        hidden_states, encoder_outputs = self._get_hidden_states(input_ids, attention_mask, **kwargs)

        outputs = {"encoder_outputs": encoder_outputs}

        if return_hidden_states:
            outputs["hidden_states"] = hidden_states

        # Compute task predictions if task heads are available
        if self.task_head is not None:
            task_outputs = self.task_head(hidden_states, attention_mask, task_names)
            outputs.update(task_outputs)

        return outputs

    def get_loss_functions(self) -> dict[str, nn.Module]:
        """Get loss functions for each task based on their configurations."""
        loss_functions = {}
        for task_name, config in self.task_configs.items():
            loss_functions[task_name] = get_loss_function(config.task_type, config.loss_function)

        return loss_functions
