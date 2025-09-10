from torch import nn
from lobster.model.latent_generator.tokenizer import TokenizerLoss


class LossFactory(nn.Module):
    def __init__(self, losses, weight_dict: dict[str, float] = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weight_dict = weight_dict
        if self.weight_dict is None:
            self.weight_dict = {loss_name: 1.0 for loss_name in self.losses.keys()}

    @classmethod
    def from_mapping(cls, loss_mapping: dict[str, TokenizerLoss], **kwargs):
        losses = {loss_name: loss for loss_name, loss in loss_mapping.items()}
        return cls(losses, **kwargs)

    def list_losses(self):
        return self.losses.keys()

    def forward(self, loss_name, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        return self.losses[loss_name](ground_truth, predictions, mask, eps=eps, **kwargs)
