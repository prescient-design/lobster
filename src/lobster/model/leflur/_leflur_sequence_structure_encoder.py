import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


from ..neobert import NeoBERTModule
from ..ume2 import AuxiliaryRegressionTaskHead
from ..ume2._checkpoint_utils import load_checkpoint_from_s3_uri_or_local_path, map_checkpoint_keys

logger = logging.getLogger(__name__)


def _map_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map checkpoint keys to match current model structure.

    The checkpoint contains keys with an extra 'model.' prefix that needs to be removed.
    For example: 'model.model.encoder.weight' -> 'model.encoder.weight'

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The original state dict from the checkpoint

    Returns
    -------
    dict[str, torch.Tensor]
        The mapped state dict with corrected keys
    """
    mapped_state_dict = {}

    for key, value in state_dict.items():
        # Remove the extra 'model.' prefix if it exists
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "model.", 1)
            mapped_state_dict[new_key] = value
            logger.debug(f"Mapped key: {key} -> {new_key}")

        elif key.startswith("model.decoder."):
            new_key = key.replace("model.decoder.", "decoder.", 1)
            mapped_state_dict[new_key] = value
            logger.debug(f"Mapped key: {key} -> {new_key}")

        else:
            mapped_state_dict[key] = value

    return mapped_state_dict


@dataclass
class AuxiliaryTask:
    name: str
    output_dim: int
    task_type: Literal["regression"] = "regression"
    pooling: Literal["cls", "mean"] = "mean"
    hidden_size: int | None = None
    dropout: float = 0.1
    num_layers: int = 2
    loss_weight: float = 1.0

    def __post_init__(self):
        if self.pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        if self.loss_weight <= 0 or self.loss_weight > 1:
            raise ValueError(f"Loss weight must be between 0 and 1: {self.loss_weight}")

        if self.task_type not in {"regression"}:
            raise ValueError(f"Unsupported task type: {self.task_type}")


class LeFlurSequenceStructureEncoderModule(nn.Module):
    def __init__(
        self,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        model_ckpt: str | None = None,
        cache_dir: str | None = None,
        sequence_token_vocab_size: int = 33,
        structure_token_vocab_size: int = 258,
        sequence_token_pad_token_id: int = 1,
        structure_token_pad_token_id: int = 257,
        conditioning_input_dim: int = 1,
        # Per-residue chain-id embedding. `max_num_chains + 1` rows: class 0
        # is reserved as `padding_idx` (frozen at zero, no gradient). Classes
        # 1..max_num_chains carry learned chain-identity signal for dimers /
        # higher-order complexes. Defaults to 2 chains (Pinder dimers). Set
        # `max_num_chains=0` to disable (no layer instantiated) for back-
        # compatibility with checkpoints that never had chain embedding.
        max_num_chains: int = 2,
        # Per-residue TEMPLATE structure-token conditioning: give the model the
        # (leak-free, per-chain) structure tokens of one/both chains as context.
        # Extra vocab row (= structure_token_vocab_size) is the "no template"
        # index (padding_idx, frozen 0). Gated so legacy checkpoints without this
        # layer still load; zero-init so a warm-start is initially a no-op.
        use_template_conditioning: bool = False,
        # Per-design SCALAR conditioning (categorical): rg_ratio (compactness), iface_frac (normalized
        # interface size), interface SS composition (helix/sheet/coil bins), and interface aromatic
        # fraction. Each is a zero-init additive bin embedding (bin 0 = NULL/padding -> no-op), broadcast
        # over binder residues. Gated so legacy checkpoints load unchanged; zero-init => warm-start no-op.
        use_scalar_conditioning: bool = False,
        # Auxiliary AF3-style DISTOGRAM head: predict the binned pairwise Cb-Cb
        # distance map (full intra + inter chain) from the last hidden state, to
        # sharpen the model's geometric / docking signal during training. Gated so
        # legacy checkpoints without this head still load; the head is train-only
        # (never used at inference/decoding). ``distogram_pair_dim`` is the width of
        # the pair projection, ``distogram_num_bins`` the number of distance bins.
        use_distogram_head: bool = False,
        distogram_num_bins: int = 64,
        distogram_pair_dim: int = 64,
        # Pair-bias attention (AF3/Proteina-style): bias every attention layer with a per-head term
        # projected (per-layer, zero-init) from a shared (B,L,L,pair_dim) pair representation. The pair
        # rep is a FIXED-layout concat of one-hot features:
        #   [ Xt-distogram(distogram_num_bins) | relpos(pair_bias_relpos_bins) | chain-diff(1) |
        #     hotspot(1) | self-cond distogram(distogram_num_bins, STUB zeros) ].
        # Gated so legacy checkpoints load; zero-init to_bias makes warm-start a no-op. hotspot is ACTIVE
        # in v1; self-cond is reserved (zeros) for a later recycling pass.
        use_pair_bias_attention: bool = False,
        pair_bias_relpos_bins: int = 129,  # relative seq separation one-hot, clamped to ±64 -> 2*64+1
        use_pair_bias_hotspot: bool = True,
        use_pair_bias_self_cond: bool = False,
        pair_bias_hidden: int = 64,  # width of the shared pair rep fed to the per-layer to_bias
        **neobert_kwargs,
    ) -> None:
        super().__init__()

        # Pair-bias attention: the raw fixed-layout one-hot features (pair_bias_raw_dim) are projected
        # ONCE by a shared Linear to a small pair_bias_hidden rep (memory: avoids carrying/normalizing a
        # ~259-wide tensor per layer across 66 layers). NeoBERT builds per-layer to_bias on pair_hidden.
        self.use_pair_bias_attention = use_pair_bias_attention
        self.pair_bias_relpos_bins = pair_bias_relpos_bins
        self.pair_dist_bins = distogram_num_bins
        self.use_pair_bias_hotspot = use_pair_bias_hotspot
        self.use_pair_bias_self_cond = use_pair_bias_self_cond
        self.pair_bias_raw_dim = 0
        self.pair_bias_hidden = pair_bias_hidden
        self.pair_input_proj = None
        if use_pair_bias_attention:
            # 64 (Xt distogram) + relpos + 1 (chain) + 1 (hotspot) + 64 (self-cond stub)
            self.pair_bias_raw_dim = distogram_num_bins + pair_bias_relpos_bins + 1 + 1 + distogram_num_bins
            self.pair_input_proj = nn.Linear(self.pair_bias_raw_dim, pair_bias_hidden)
            neobert_kwargs["use_pair_bias"] = True
            neobert_kwargs["pair_dim"] = pair_bias_hidden

        self.neobert = NeoBERTModule(**neobert_kwargs)

        if auxiliary_tasks is not None:
            if not all(task.task_type == "regression" for task in auxiliary_tasks):
                raise NotImplementedError("Only regression tasks are currently supported for auxiliary tasks in UME-2")

            self.auxiliary_tasks = nn.ModuleDict(
                {
                    task.name: AuxiliaryRegressionTaskHead(
                        input_dim=self.neobert.config.hidden_size,
                        output_dim=task.output_dim,
                        task_name=task.name,
                        hidden_size=task.hidden_size,
                        dropout=task.dropout,
                        num_layers=task.num_layers,
                        pooling=task.pooling,
                    )
                    for task in auxiliary_tasks
                }
            )

        else:
            self.auxiliary_tasks = None

        # embedding for sequence and structure tokens
        self.sequence_embedding = nn.Embedding(
            sequence_token_vocab_size, self.neobert.config.hidden_size, padding_idx=sequence_token_pad_token_id
        )
        self.structure_embedding = nn.Embedding(
            structure_token_vocab_size, self.neobert.config.hidden_size, padding_idx=structure_token_pad_token_id
        )
        self.conditioning_embedding = nn.Linear(conditioning_input_dim, self.neobert.config.hidden_size, bias=False)
        # Per-residue chain-id embedding (additive into the conditioning path).
        # Disabled (None) when max_num_chains == 0 so legacy leflur-base style
        # checkpoints can still instantiate this class unchanged.
        self.max_num_chains = max_num_chains
        if max_num_chains > 0:
            self.chain_embedding = nn.Embedding(
                num_embeddings=max_num_chains + 1,
                embedding_dim=self.neobert.config.hidden_size,
                padding_idx=0,
            )
        else:
            self.chain_embedding = None
        # Template structure-token embedding (additive into the conditioning path,
        # like chain_embedding). Row `structure_token_vocab_size` = "no template".
        self.use_template_conditioning = use_template_conditioning
        self.no_template_idx = structure_token_vocab_size
        if use_template_conditioning:
            self.template_structure_embedding = nn.Embedding(
                num_embeddings=structure_token_vocab_size + 1,
                embedding_dim=self.neobert.config.hidden_size,
                padding_idx=self.no_template_idx,
            )
            nn.init.zeros_(self.template_structure_embedding.weight)  # warm-start no-op
        else:
            self.template_structure_embedding = None
        # Per-design scalar (categorical) conditioning embeddings. num_embeddings = K value bins + 1 NULL
        # (row 0 = padding_idx, frozen zero). Zero-init => warm-start no-op; additive into conditioning path,
        # broadcast over binder residues (the transform emits 0/NULL on target residues).
        self.use_scalar_conditioning = use_scalar_conditioning
        if use_scalar_conditioning:
            _sc_bins = {
                "rg_ratio": 6,
                "iface_frac": 5,
                "iface_helix": 5,
                "iface_sheet": 5,
                "iface_coil": 5,
                "frac_arom": 5,
            }  # num_embeddings incl. NULL row 0
            self.scalar_cond_emb = nn.ModuleDict(
                {k: nn.Embedding(n, self.neobert.config.hidden_size, padding_idx=0) for k, n in _sc_bins.items()}
            )
            for _emb in self.scalar_cond_emb.values():
                nn.init.zeros_(_emb.weight)
        else:
            self.scalar_cond_emb = None
        self.combine_embedding = nn.Linear(self.neobert.config.hidden_size * 3, self.neobert.config.hidden_size)

        # output for sequence and structure tokens
        self.sequence_output = nn.Linear(self.neobert.config.hidden_size, sequence_token_vocab_size)
        self.structure_output = nn.Linear(self.neobert.config.hidden_size, structure_token_vocab_size)

        # Auxiliary distogram head. neobert is a single-sequence encoder (no pair
        # track), so we build pair activations from the single hidden state via an
        # outer sum of two projections, then an AF3-style symmetric linear -> bins
        # (logits = half + half^T guarantees a symmetric distogram). The final layer
        # is zero-initialized so the head starts by predicting a uniform distribution
        # (stable warm-start when this head is added to an existing checkpoint).
        self.use_distogram_head = use_distogram_head
        self.distogram_num_bins = distogram_num_bins
        if use_distogram_head:
            hidden = self.neobert.config.hidden_size
            self.distogram_left = nn.Linear(hidden, distogram_pair_dim)
            self.distogram_right = nn.Linear(hidden, distogram_pair_dim)
            self.distogram_act = nn.GELU()
            self.distogram_out = nn.Linear(distogram_pair_dim, distogram_num_bins)
            nn.init.zeros_(self.distogram_out.weight)
            nn.init.zeros_(self.distogram_out.bias)
        else:
            self.distogram_left = None
            self.distogram_right = None
            self.distogram_out = None

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, *, device: str | None = None, cache_dir: str | None = None, **kwargs
    ) -> "LeFlurSequenceStructureEncoderModule":
        """Utility function to load state_dict and hyper_parameters from LeFlurSequenceStructureEncoderLightningModule checkpoint."""

        device = device or get_device()

        checkpoint = load_checkpoint_from_s3_uri_or_local_path(checkpoint_path, device=device, cache_dir=cache_dir)

        # Get and update hyper_parameters
        hyper_parameters = checkpoint["hyper_parameters"] or {}
        keys = ["auxiliary_tasks", "encoder_kwargs", "pad_token_id", "use_shared_tokenizer"]
        hyper_parameters = {key: value for key, value in hyper_parameters.items() if key in keys}
        hyper_parameters.update(kwargs)

        state_dict = checkpoint["state_dict"]

        encoder_kwargs = hyper_parameters.pop("encoder_kwargs", {})

        # Initialize encoder
        encoder = cls(**hyper_parameters, **encoder_kwargs)
        encoder.to(device)

        # Load state_dict
        state_dict = map_checkpoint_keys(state_dict, original_prefix="encoder.neobert.", new_prefix="")
        encoder.neobert.load_state_dict(state_dict)

        return encoder

    def forward(
        self,
        sequence_input_ids: Tensor,
        structure_input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        conditioning_tensor: Tensor | None = None,
        chain_ids: Tensor | None = None,
        return_auxiliary_tasks: bool = False,
        timesteps: Tensor | None = None,
        template_structure_tokens: Tensor | None = None,
        # Per-design scalar conditioning bins: dict {name -> (B,L) Long bin ids, 0=NULL}. Added additively
        # into the conditioning path via zero-init embeddings. No-op when None / all-NULL / layer disabled.
        scalar_cond_bins: dict | None = None,
        # Pair-bias attention inputs (built in the lightning forward): per-pair distance-bin ids from the
        # decoded current structure, relpos ids, chain-diff, hotspot, and the valid-pair mask. All
        # (B,L,L). When present + use_pair_bias_attention, assembled into the shared one-hot pair rep.
        pair_bin_ids: Tensor | None = None,
        pair_relpos_ids: Tensor | None = None,
        pair_chain_diff: Tensor | None = None,
        pair_hotspot: Tensor | None = None,
        pair_valid: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        sequence_output = self.sequence_embedding(sequence_input_ids)
        structure_output = self.structure_embedding(structure_input_ids)
        conditioning_output = self.conditioning_embedding(conditioning_tensor)
        # Additive chain-id signal -- a second conditioning channel on top
        # of the binary epitope one (`conditioning_tensor`). For monomer
        # batches (or any time chain_ids is missing / all-zero) the embedding
        # contributes a zero vector per residue because `padding_idx=0` pins
        # row 0 at zero and excludes it from gradient. So this is a no-op on
        # monomers and active only on multi-chain inputs.
        if self.chain_embedding is not None and chain_ids is not None:
            conditioning_output = conditioning_output + self.chain_embedding(chain_ids.long())
        # Additive template structure-token signal (leak-free per-chain templates).
        if self.template_structure_embedding is not None and template_structure_tokens is not None:
            conditioning_output = conditioning_output + self.template_structure_embedding(
                template_structure_tokens.long()
            )
        # Additive per-design scalar (categorical) conditioning signals. Each zero-init bin embedding is a
        # no-op at NULL (bin 0) / warm-start; broadcast over binder residues.
        if self.scalar_cond_emb is not None and scalar_cond_bins:
            for _name, _bins in scalar_cond_bins.items():
                if _bins is not None and _name in self.scalar_cond_emb:
                    conditioning_output = conditioning_output + self.scalar_cond_emb[_name](_bins.long())
        combined_output = self.combine_embedding(
            torch.cat([sequence_output, structure_output, conditioning_output], dim=-1)
        )

        # Assemble the shared pair representation once (fixed column layout: Xt-distogram | relpos |
        # chain-diff | hotspot | self-cond-stub). Each NeoBERT layer projects it with its own zero-init
        # to_bias. Built only when pair-bias is on and the geometry inputs were supplied.
        pair_feat = None
        if self.use_pair_bias_attention and pair_bin_ids is not None:
            oh = torch.nn.functional.one_hot
            B, L, _ = pair_bin_ids.shape
            dev, dt = combined_output.device, combined_output.dtype
            valid = pair_valid if pair_valid is not None else torch.ones(B, L, L, dtype=torch.bool, device=dev)
            chain_diff = pair_chain_diff if pair_chain_diff is not None else torch.zeros(B, L, L, device=dev, dtype=dt)
            # Relpos is a DIRECTIONAL (signed i-j) one-hot for SAME-chain pairs only; cross-chain pairs
            # are zeroed (Proteina's cross-sequence relpos returns zeros) -> cross-chain features stay
            # symmetric (distance sym + relpos zero + chain-diff sym + hotspot sym => binder<->target
            # pair_feat[i,j] == pair_feat[j,i], no explicit pair symmetrization needed).
            same_chain = (1.0 - chain_diff).unsqueeze(-1).to(dt)
            relpos_oh = oh(pair_relpos_ids.clamp(0, self.pair_bias_relpos_bins - 1), self.pair_bias_relpos_bins).to(dt)
            relpos_oh = relpos_oh * same_chain  # zero cross-chain sequence separation
            hotspot = (
                pair_hotspot
                if (pair_hotspot is not None and self.use_pair_bias_hotspot)
                else torch.zeros(B, L, L, device=dev, dtype=dt)
            )
            cols = [
                oh(pair_bin_ids.clamp(0, self.pair_dist_bins - 1), self.pair_dist_bins).to(dt)
                * valid.unsqueeze(-1).to(dt),  # Cb-distance distogram (symmetric)
                relpos_oh,  # relpos: directional intra, zeroed inter
                chain_diff.unsqueeze(-1).to(dt),  # different-chain indicator (symmetric)
                hotspot.unsqueeze(-1).to(dt),  # hotspot: either residue (symmetric)
                torch.zeros(B, L, L, self.pair_dist_bins, device=dev, dtype=dt),  # self-cond STUB (zeros)
            ]
            pair_feat_raw = torch.cat(cols, dim=-1)  # (B, L, L, pair_bias_raw_dim)
            # project the wide one-hot rep ONCE to a small hidden dim (per-layer to_bias operates on this)
            pair_feat = self.pair_input_proj(pair_feat_raw)  # (B, L, L, pair_bias_hidden)

        # removinf position_ids becuase not properly formulated for current neo architecture
        position_ids = None
        output = self.neobert(
            input_ids=None,
            inputs_embeds=combined_output,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pair_feat=pair_feat,
            pair_valid=pair_valid,
            **kwargs,
        )

        sequence_output = self.sequence_output(output["last_hidden_state"])
        structure_output = self.structure_output(output["last_hidden_state"])
        output["sequence_logits"] = sequence_output
        output["structure_logits"] = structure_output

        if self.use_distogram_head:
            h = output["last_hidden_state"]  # (B, L, hidden)
            pair = self.distogram_left(h).unsqueeze(2) + self.distogram_right(h).unsqueeze(1)  # (B, L, L, P)
            half = self.distogram_out(self.distogram_act(pair))  # (B, L, L, num_bins)
            output["distogram_logits"] = half + half.transpose(1, 2)  # symmetric

        if self.auxiliary_tasks is not None and return_auxiliary_tasks:
            for task_name, task_head in self.auxiliary_tasks.items():
                embeddings = output["last_hidden_state"]
                output[task_name] = task_head(embeddings)

        return output


def get_device() -> str:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
