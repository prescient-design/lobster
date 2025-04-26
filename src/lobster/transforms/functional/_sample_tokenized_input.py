import torch
from torch import Tensor


def sample_tokenized_input(x: dict[str, Tensor]):
    """Samples a single tokenized input from a batch of tokenized inputs.

    Meant to be used with tokenized inputs from 3D coordinates latent generator
    dataset (e.g. LatentGeneratorPinderIterableDataset) which contains 4 poses
    for each input.

    Example
    ```python
        x = {
            "input_ids": torch.randint(0, 100, (4, 100)), # shape (4, 100)
            "attention_mask": torch.randint(0, 2, (4, 100)), # shape (4, 100)
        }

        sample_tokenized_input(x)

        # Output
        {
            "input_ids": tensor of shape (100,),
            "attention_mask": tensor of shape (100,),
        }

    Parameters
    ----------
    x : dict[str, Tensor]
        Dictionary containing tokenized inputs with shape (k, N) where N is the sequence length.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing a single sampled tokenized input with shape (N,).
    """
    return {
        key: value[torch.randint(0, value.size(0), (1,))] if isinstance(value, (Tensor)) else value
        for key, value in x.items()
    }
