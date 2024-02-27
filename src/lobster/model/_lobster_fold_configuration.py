from dataclasses import asdict, dataclass
from typing import Optional

from transformers.configuration_utils import PretrainedConfig

PPLMFOLD_CONFIG_ARGS = {
    "PPLM": {},
}


class PPLMFoldConfig(PretrainedConfig):
    r"""
    This is the configuration class for a Prescient Protein Language Model for Folding.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
    """

    model_type = "lobster_fold"

    def __init__(
        self,
        vocab_size=None,
        mask_token_id=None,
        pad_token_id=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="rotary",
        use_cache=True,
        classifier_dropout=None,
        emb_layer_norm_before=None,
        token_dropout=False,
        is_folding_model=True,
        fold_config=None,
        vocab_list=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.is_folding_model = is_folding_model

        if is_folding_model and fold_config is None:
            fold_config = FoldConfig()
        else:
            fold_config = FoldConfig(**fold_config)
        self.fold_config = fold_config

        self.vocab_list = vocab_list or get_default_vocab_list()

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        if isinstance(self.fold_config, FoldConfig):
            output["fold_config"] = self.fold_config.to_dict()
        return output


@dataclass
class FoldConfig:
    model_type: str = None
    fp16: bool = False
    use_attn_map: bool = False
    ablate_pairwise: bool = False
    ablate_sequence: bool = False
    input_dropout: float = 0

    embed_aa: bool = True
    bypass_lm: bool = False

    lddt_head_hid_dim: int = 128
    trunk: "TrunkConfig" = None

    def __post_init__(self):
        if self.trunk is None:
            self.trunk = TrunkConfig()
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = asdict(self)
        output["trunk"] = self.trunk.to_dict()
        return output


@dataclass
class TrunkConfig:
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False
    max_recycles: int = 4
    chunk_size: Optional[int] = 128
    structure_module: "StructureModuleConfig" = None

    def __post_init__(self):
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)

        if self.max_recycles <= 0:
            raise ValueError(
                f"`max_recycles` should be positive, got {self.max_recycles}."
            )
        if self.sequence_state_dim % self.sequence_state_dim != 0:
            raise ValueError(
                "`sequence_state_dim` should be a round multiple of `sequence_state_dim`, got"
                f" {self.sequence_state_dim} and {self.sequence_state_dim}."
            )
        if self.pairwise_state_dim % self.pairwise_state_dim != 0:
            raise ValueError(
                "`pairwise_state_dim` should be a round multiple of `pairwise_state_dim`, got"
                f" {self.pairwise_state_dim} and {self.pairwise_state_dim}."
            )

        sequence_num_heads = self.sequence_state_dim // self.sequence_head_width
        pairwise_num_heads = self.pairwise_state_dim // self.pairwise_head_width

        if self.sequence_state_dim != sequence_num_heads * self.sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {self.sequence_state_dim} != {sequence_num_heads} * {self.sequence_head_width}."
            )
        if self.pairwise_state_dim != pairwise_num_heads * self.pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {self.pairwise_state_dim} != {pairwise_num_heads} * {self.pairwise_head_width}."
            )
        if self.pairwise_state_dim % 2 != 0:
            raise ValueError(
                f"`pairwise_state_dim` should be even, got {self.pairwise_state_dim}."
            )

        if self.dropout >= 0.4:
            raise ValueError(
                f"`dropout` should not be greater than 0.4, got {self.dropout}."
            )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = asdict(self)
        output["structure_module"] = self.structure_module.to_dict()
        return output


@dataclass
class StructureModuleConfig:
    """
    Args:
        sequence_dim:
            Single representation channel dimension
        pairwise_dim:
            Pair representation channel dimension
        ipa_dim:
            IPA hidden channel dimension
        resnet_dim:
            Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
        num_heads_ipa:
            Number of IPA heads
        num_qk_points:
            Number of query/key points to generate during IPA
        num_v_points:
            Number of value points to generate during IPA
        dropout_rate:
            Dropout rate used throughout the layer
        num_blocks:
            Number of structure module blocks
        num_transition_layers:
            Number of layers in the single representation transition (Alg. 23 lines 8-9)
        num_resnet_blocks:
            Number of blocks in the angle resnet
        num_angles:
            Number of angles to generate in the angle resnet
        trans_scale_factor:
            Scale of single representation transition hidden dimension
        epsilon:
            Small number used in angle resnet normalization
        inf:
            Large number used for attention masking
    """

    sequence_dim: int = 384
    pairwise_dim: int = 128
    ipa_dim: int = 16
    resnet_dim: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_transition_layers: int = 1
    num_resnet_blocks: int = 2
    num_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5

    def to_dict(self):
        return asdict(self)


def get_default_vocab_list():
    return (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
    )
