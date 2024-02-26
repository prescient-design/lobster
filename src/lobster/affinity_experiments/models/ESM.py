import esm
import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESM(pl.LightningModule):
    def __init__(
        self,
        model=None,
        use_mlp=False,
        output_size=1,
        hidden_dropout_prob=0.5,
        use_layernorm=False,
        use_moe=False,
        num_experts=3,
        gate_type="Linear",
        expert_type="Linear",
    ):

        super().__init__()
        self.model = model
        self.use_layernorm = use_layernorm
        self.use_moe = use_moe
        self.hidden_dropout_prob = hidden_dropout_prob

        if self.use_moe:
            self.num_experts = num_experts
            self.expert_type = expert_type
            self.gate_type = gate_type

        if self.model == "esm1b_t33":
            self.model_antibody, _ = esm.pretrained.esm1b_t33_650M_UR50S()
            self.model_antigen, _ = esm.pretrained.esm1b_t33_650M_UR50S()
            self.final_layer = 33
            self.d_embed = 1280 * 2

        elif self.model == "esm1_t12_85M_UR50S":
            self.model_antibody, _ = esm.pretrained.esm1_t12_85M_UR50S()
            self.model_antigen, _ = esm.pretrained.esm1_t12_85M_UR50S()
            self.final_layer = 12
            self.d_embed = 768 * 2

        elif self.model == "esm2_t30_150M_UR50D":
            self.model_antibody, _ = esm.pretrained.esm2_t30_150M_UR50D()
            self.model_antigen, _ = esm.pretrained.esm2_t30_150M_UR50D()
            self.final_layer = 30
            self.d_embed = 640 * 2

        elif self.model == "esm2_t33_650M_UR50D":
            self.model_antibody, _ = esm.pretrained.esm2_t33_650M_UR50D()
            self.model_antigen, _ = esm.pretrained.esm2_t33_650M_UR50D()
            self.final_layer = 33
            self.d_embed = 1280 * 2

        elif self.model == "esm2_t36_3B_UR50D":
            self.model_antibody, _ = esm.pretrained.esm2_t36_3B_UR50D()
            self.model_antigen, _ = esm.pretrained.esm2_t36_3B_UR50D()
            self.final_layer = 36
            self.d_embed = 2560 * 2
        else:
            self.model_antibody, _ = esm.pretrained.esm1_t6_43M_UR50S()
            self.model_antigen, _ = esm.pretrained.esm1_t6_43M_UR50S()
            self.final_layer = 6
            self.d_embed = 768 * 2

        self.use_mlp = use_mlp

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.d_embed)

        if self.use_moe:

            self.y_dim = output_size  # assumed in the original code

            self.out_dim = self.y_dim
            if self.expert_type == "Linear":
                self.experts = nn.ModuleList(
                    [nn.Linear(self.d_embed, self.out_dim) for i in range(self.num_experts)]
                )
            elif self.expert_type == "MLP":
                self.d_mlp = 64 * 2
                self.experts = nn.ModuleList(
                    [
                        nn.Sequential(
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
                            nn.Linear(self.d_mlp, self.out_dim),
                        )
                        for i in range(self.num_experts)
                    ]
                )
            else:
                raise NotImplementedError

            if self.gate_type == "Linear":
                self.gate_network = nn.Linear(self.d_embed, self.num_experts)
            elif self.gate_type == "MLP":
                self.d_mlp = 64  # jwp FIXME: hardcoded, TODO: turn residual
                self.gate_network = nn.Sequential(
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
                    nn.Linear(self.d_mlp, self.num_experts),
                )
            else:
                raise NotImplementedError

        else:
            if not self.use_mlp:
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
        results_x1 = self.model_antibody(x1, repr_layers=[self.final_layer])
        results_x2 = self.model_antigen(x2, repr_layers=[self.final_layer])

        shared_x1 = results_x1["representations"][self.final_layer].mean(1)
        shared_x2 = results_x2["representations"][self.final_layer].mean(1)
        x = torch.cat((shared_x1, shared_x2), 1)

        if self.use_layernorm:
            x = self.layernorm(x)

        if self.use_moe:
            assignment_logits = self.gate_network(x)  # [B, num_experts]
            weights = F.softmax(assignment_logits, dim=1)  # [B, num_experts]
            # weights = self.sparsemax(assignment_logits)

            expert_outputs_list = [self.experts[i](x).unsqueeze(2) for i in range(self.num_experts)]
            # Concatenate the outputs of different experts
            expert_outputs = torch.cat(expert_outputs_list, dim=2)
            return expert_outputs, weights

        else:
            y = self.mlp(x)
            return y
