import copy

import lightning.pytorch as pl
import torch
from torch import nn

from lobster.model import RLM_CONFIG_ARGS, PrescientPMLM
from lobster.model.lm_base import LMBaseForMaskedLM, PMLMConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PLM(pl.LightningModule):
    def __init__(
        self,
        model_type=None,
        load_pretrained=True,
        pretrained_path="",
        simple_mlp=True,
        output_size=1,
        hidden_dropout_prob=0.5,
        use_layernorm=False,
    ):

        super().__init__()
        self.model_type = model_type
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_layernorm = use_layernorm

        if load_pretrained and pretrained_path != "":
            print("Loading pretrained backbone model...")
            model_checkpoint = PrescientPMLM.load_from_checkpoint(pretrained_path)
            self.model_antibody = copy.deepcopy(model_checkpoint.model)
            self.model_antigen = copy.deepcopy(model_checkpoint.model)
            self.d_embed = model_checkpoint.config.hidden_size * 2

        else:
            plm_config = RLM_CONFIG_ARGS[self.model_type]
            config = PMLMConfig(
                num_labels=plm_config["num_labels"],
                problem_type=plm_config["problem_type"],
                num_hidden_layers=plm_config["num_hidden_layers"],
                num_attention_heads=plm_config["num_attention_heads"],
                intermediate_size=plm_config["intermediate_size"],
                hidden_size=plm_config["hidden_size"],
                attention_probs_dropout_prob=plm_config["attention_probs_dropout_prob"],
                mask_token_id=plm_config["mask_token_id"],
                pad_token_id=plm_config["pad_token_id"],
                token_dropout=plm_config["token_dropout"],
                position_embedding_type=plm_config["position_embedding_type"],
                vocab_size=plm_config["vocab_size"],
                layer_norm_eps=plm_config["layer_norm_eps"],
            )
            self.model_antibody = LMBaseForMaskedLM(config)
            self.model_antigen = LMBaseForMaskedLM(config)
            self.d_embed = plm_config["hidden_size"] * 2

        self.simple_mlp = simple_mlp

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.d_embed)

        if self.simple_mlp:
            self.mlp = nn.Linear(self.d_embed, output_size)
        else:
            self.d_mlp = 64 * 2  # jwp FIXME: hardcoded, TODO: turn residual
            self.mlp = nn.Sequential(
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.d_embed, self.d_mlp),
                nn.LayerNorm(self.d_mlp),
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.d_mlp, self.d_mlp),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.d_mlp, self.d_mlp),
                nn.LayerNorm(self.d_mlp),
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.d_mlp, output_size),
            )

    def forward(self, x1, x2, x1_mask=None, x2_mask=None):

        outputs_1 = self.model_antibody.LMBase(input_ids=x1, attention_mask=x1_mask)

        features_1 = outputs_1[0]
        shared_x1 = features_1.mean(1)

        outputs_2 = self.model_antigen.LMBase(input_ids=x2, attention_mask=x2_mask)
        features_2 = outputs_2[0]
        shared_x2 = features_2.mean(1)

        x = torch.cat((shared_x1, shared_x2), 1)

        if self.use_layernorm:
            x = self.layernorm(x)

        y = self.mlp(x)

        return y
