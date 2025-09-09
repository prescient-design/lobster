import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20, device="cuda"):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, include_noise=True):
    """Draw a sample from the Gumbel-Softmax distribution."""
    if include_noise:
        y = logits + sample_gumbel(logits.size(), device=logits.device)
    else:
        y = logits
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, include_noise=True):
    """ST-gumple-softmax.

    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector.
    """
    y = gumbel_softmax_sample(logits, temperature, include_noise=include_noise)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard
