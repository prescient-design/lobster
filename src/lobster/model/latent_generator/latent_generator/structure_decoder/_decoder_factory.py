from typing import Dict

import torch
from torch import nn

from lobster.model.latent_generator.latent_generator.structure_decoder import BaseDecoder


class DecoderFactory(nn.Module):
    def __init__(self, decoders, decoder2loss_dict):
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)
        self.decoder2loss_dict = decoder2loss_dict

    @classmethod
    def from_mapping(cls, decoder_mapping: Dict[str, BaseDecoder], **kwargs):
        decoders = {decoder_name: decoder for decoder_name, decoder in decoder_mapping.items()}
        return cls(decoders, **kwargs)

    def list_decoders(self):
        return self.decoders.keys()

    def get_loss(self, decoder_name):
        return self.decoder2loss_dict[decoder_name]

    def forward(self, decoder_name: str, x_noise: torch.Tensor, x_quant: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, **kwargs):
        return self.decoders[decoder_name](x_noise, x_quant, t, mask, **kwargs)
