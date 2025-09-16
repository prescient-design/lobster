"""Layer unfreezing control for fine-tuning UME models.

Single-parameter API:
- num_layers = -1  => unfreeze all
- num_layers = 0   => freeze all
- num_layers = n>0 => unfreeze last n encoder layers
"""

import logging

from lobster.model import UME

logger = logging.getLogger(__name__)


def set_unfrozen_layers(model: UME, num_layers: int) -> None:
    """Set how many encoder layers are unfrozen.

    Parameters
    ----------
    model : UME
        The UME model to modify.
    num_layers : int
        -1 => unfreeze all parameters
         0 => freeze all parameters
         n > 0 => unfreeze last n encoder layers
    """
    if num_layers < 0:
        _unfreeze_all_parameters(model)
        strategy = "full"
    elif num_layers == 0:
        _freeze_all_parameters(model)
        strategy = "freeze_all"
    else:
        _freeze_all_parameters(model)
        _unfreeze_last_n_layers(model, num_layers)
        strategy = f"partial(n={num_layers})"

    _log_parameter_status(model, strategy)


def _freeze_all_parameters(model: UME) -> None:
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_all_parameters(model: UME) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def _unfreeze_last_n_layers(model: UME, n: int) -> None:
    """Unfreeze the last N transformer layers.

    Parameters
    ----------
    model : UME
        The UME model
    n : int
        Number of layers to unfreeze from the end
    """
    # First freeze all parameters
    _freeze_all_parameters(model)

    # Get the transformer layers from the underlying model (be robust to nesting)
    encoder = None
    inner = getattr(model, "model", model)
    if hasattr(inner, "encoder"):
        encoder = inner.encoder
    elif hasattr(inner, "model") and hasattr(inner.model, "encoder"):
        encoder = inner.model.encoder

    if encoder is None:
        logger.warning("Could not access encoder layers in UME model")
        return

    if hasattr(encoder, "layers"):
        layers = encoder.layers
    elif hasattr(encoder, "layer"):
        layers = encoder.layer
    elif hasattr(encoder, "blocks"):
        layers = encoder.blocks
    else:
        logger.warning("Could not find transformer layers attribute on encoder (expected 'layers' or 'layer')")
        return

    # Unfreeze the last n layers
    total_layers = len(layers)
    layers_to_unfreeze = min(n, total_layers)

    for i in range(total_layers - layers_to_unfreeze, total_layers):
        for param in layers[i].parameters():
            param.requires_grad = True

    # Also unfreeze the final layer norm and pooler if they exist
    if hasattr(model.model, "pooler") and model.model.pooler is not None:
        for param in model.model.pooler.parameters():
            param.requires_grad = True

    if hasattr(model.model, "LayerNorm"):
        for param in model.model.LayerNorm.parameters():
            param.requires_grad = True

    trainable = sum(any(p.requires_grad for p in l.parameters()) for l in layers)
    logger.info(
        f"Unfreezing: requested n={n}, applied={layers_to_unfreeze}, encoder_layers_trainable={trainable}/{total_layers}"
    )


def get_layer_wise_parameter_groups(model: UME, base_lr: float = 2e-4, decay_factor: float = 0.9) -> list[dict]:
    """Get parameter groups with layer-wise learning rate decay.

    This implements layer-wise learning rate decay (LLRD) where earlier
    layers get lower learning rates than later layers.

    References
    ----------
    Layer-wise learning rate decay: "ULMFiT: Universal Language Model Fine-tuning" (Howard & Ruder, 2018)

    Parameters
    ----------
    model : UME
        The UME model
    base_lr : float, optional
        Base learning rate for the last layer, by default 2e-4
    decay_factor : float, optional
        Decay factor for each earlier layer, by default 0.9

    Returns
    -------
    List[dict]
        Parameter groups with different learning rates

    Examples
    --------
    >>> from lobster.model import UME
    >>> model = UME(model_name="UME_mini")
    >>> param_groups = get_layer_wise_parameter_groups(model, base_lr=1e-4)
    >>> optimizer = torch.optim.AdamW(param_groups)
    """
    param_groups = []

    # Get the transformer layers
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        encoder = model.model.encoder
        if hasattr(encoder, "layer"):
            layers = encoder.layer
        elif hasattr(encoder, "layers"):
            layers = encoder.layers
        else:
            # Fallback to single group
            return [{"params": model.parameters(), "lr": base_lr}]
    else:
        # Fallback to single group
        return [{"params": model.parameters(), "lr": base_lr}]

    # Create parameter groups for each layer
    num_layers = len(layers)

    for i, layer in enumerate(layers):
        # Calculate learning rate for this layer
        # Earlier layers get lower learning rates
        layer_lr = base_lr * (decay_factor ** (num_layers - 1 - i))

        param_groups.append(
            {
                "params": list(layer.parameters()),
                "lr": layer_lr,
                "layer_id": i,
            }
        )

    # Add other parameters (embeddings, pooler, etc.) with base learning rate
    other_params = []
    layer_param_ids = set()

    # Collect all layer parameter IDs
    for layer in layers:
        for param in layer.parameters():
            layer_param_ids.add(id(param))

    # Find parameters not in layers
    for name, param in model.named_parameters():
        if id(param) not in layer_param_ids:
            other_params.append(param)

    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": base_lr,
                "layer_id": "other",
            }
        )

    logger.info(f"Created {len(param_groups)} parameter groups with layer-wise learning rates")
    return param_groups


def progressive_unfreezing_schedule(model: UME, current_epoch: int, unfreeze_schedule: list[int]) -> None:
    """Apply progressive unfreezing based on training epoch.

    References
    ----------
    Progressive unfreezing: "ULMFiT: Universal Language Model Fine-tuning" (Howard & Ruder, 2018)

    Parameters
    ----------
    model : UME
        The UME model
    current_epoch : int
        Current training epoch
    unfreeze_schedule : List[int]
        List of epochs at which to unfreeze additional layers

    Examples
    --------
    >>> # Unfreeze 1 layer at epoch 2, 2 more at epoch 5, etc.
    >>> schedule = [2, 5, 8, 12]
    >>> progressive_unfreezing_schedule(model, current_epoch=5, unfreeze_schedule=schedule)
    """
    # Count how many unfreeze events should have happened by now
    layers_to_unfreeze = sum(1 for epoch in unfreeze_schedule if current_epoch >= epoch)

    if layers_to_unfreeze > 0:
        _unfreeze_last_n_layers(model, layers_to_unfreeze)
        logger.info(f"Progressive unfreezing: unfroze {layers_to_unfreeze} layers at epoch {current_epoch}")


def _log_parameter_status(model: UME, strategy: str) -> None:
    """Log the parameter freezing status."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Applied unfreezing strategy: {strategy}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    logger.info(f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
