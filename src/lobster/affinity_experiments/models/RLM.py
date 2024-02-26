import lightning.pytorch as pl
import torch
from torch import nn

from lobster.model._rlm_configuration import RLM_CONFIG_ARGS
from lobster.model.lm_base import PMLMConfig
from lobster.model.lm_base._lm_base import LMBaseForMaskedLMRelative

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLM(pl.LightningModule):
    def __init__(
        self,
        model_type=None,
        load_pretrained=True,
        pretrained_path="",
        simple_mlp=True,
        output_size=1,
        hidden_dropout_prob=0.5,
        use_layernorm=False,
        feature_token="mean",
    ):
        """

        Parameters
        ----------
        simple_mlp : bool
            Whether to have a simple (single FC) regression MLP. Default: True
            (defaults to R2 setting)

        """
        super().__init__()

        self.model_type = model_type
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_layernorm = use_layernorm
        self.feature_token = feature_token

        rlm_config = RLM_CONFIG_ARGS[self.model_type]
        config = PMLMConfig(
            num_labels=rlm_config["num_labels"],
            problem_type=rlm_config["problem_type"],
            num_hidden_layers=rlm_config["num_hidden_layers"],
            num_attention_heads=rlm_config["num_attention_heads"],
            intermediate_size=rlm_config["intermediate_size"],
            hidden_size=rlm_config["hidden_size"],
            attention_probs_dropout_prob=rlm_config["attention_probs_dropout_prob"],
            mask_token_id=rlm_config["mask_token_id"],
            pad_token_id=rlm_config["pad_token_id"],
            token_dropout=rlm_config["token_dropout"],
            position_embedding_type=rlm_config["position_embedding_type"],
            vocab_size=rlm_config["vocab_size"],
            layer_norm_eps=rlm_config["layer_norm_eps"],
        )
        self.base_model = LMBaseForMaskedLMRelative(config)
        self.d_embed = rlm_config["hidden_size"]

        if load_pretrained and pretrained_path != "":
            # # checkpoint = torch.load(pretrained_path)
            # checkpoint= torch.load(pretrained_path)
            # print(checkpoint.keys())
            # PrescientPRLM.load_from_checkpoint(pretrained_path)
            # print(self.base_model)
            # self.d_embed =self.base_model.config.hidden_size*2
            # checkpoint=checkpoint['state_dict']

            print("Loading pretrained backbone model...")
            # self.base_model.load_state_dict(torch.load(pretrained_path))

            self.base_model.load_state_dict(torch.load(pretrained_path), strict=False)

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

    def forward(self, x1, x2, x1_mask=None, x2_mask=None, return_embedding=False):

        x = self.base_model(
            input_ids_a=x1, attention_mask_a=x1_mask, input_ids_b=x2, attention_mask_b=x2_mask
        )
        # print(x.shape)
        # print(stop)
        # features = outputs[0]
        # mean_embedding = features.mean(1)
        # cls_embedding = features[:, 0, :] # take <s> token (equiv. to [CLS])
        # if self.feature_token=="mean":
        #     x=mean_embedding
        # else:
        #     x=cls_embedding

        if return_embedding:
            return x
        if self.use_layernorm:
            x = self.layernorm(x)

        y = self.mlp(x)
        return y
