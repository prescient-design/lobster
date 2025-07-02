# Lobster Models

## [Universal Molecular Encoder](_ume.py)
* UME is trained with `flash-attn`. It implements a custom `load_from_checkpoint` method that will dynamically configure the model to correctly run on GPU with `flash-attn`, or on CPU without it.
