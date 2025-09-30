# File: encode_latents.py

import argparse
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import boto3
import hydra
from hydra.core.global_hydra import GlobalHydra
import torch
from botocore.exceptions import NoCredentialsError
from omegaconf import DictConfig, OmegaConf

from lobster.model.latent_generator.io import load_ligand, load_pdb, writepdb, writepdb_ligand_complex
from lobster.model.latent_generator.tokenizer import TokenizerMulti

py_logger = logging.getLogger(__name__)


# Pre-configured model configurations
# These configurations include all necessary overrides and settings
# No need to manually specify overrides when using these configurations
@dataclass
class ModelConfig:
    """Configuration for a model checkpoint."""

    checkpoint: str
    config_path: str
    config_name: str
    overrides: list[str]


@dataclass
class ModelInfo:
    """Information about a model including its configuration."""

    description: str
    features: list[str]
    model_config: ModelConfig


methods = {
    # Ligand Models
    # These models are optimized for ligand structure analysis
    "LG Ligand 20A": ModelInfo(
        description="Ligand only model with 20Å spatial attention",
        features=["256-dim embeddings", "20Å spatial attention", "Ligand only decoder", "512 ligand tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_Ligand_20A.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer.structure_encoder.encode_ligand=true",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=512",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=512",
                "tokenizer.quantizer.ligand_n_tokens=512",
                "tokenizer/quantizer=slq_quantizer_ligand",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
            ],
        ),
    ),
    "LG Ligand 20A 512 1024": ModelInfo(
        description="Ligand only model with 20Å spatial attention",
        features=["512-dim embeddings", "20Å spatial attention", "Ligand only decoder", "1024 ligand tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_Ligand_20A_512_1024.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=512",
                "tokenizer.structure_encoder.embed_dim_hidden=512",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer/quantizer=slq_quantizer_ligand",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=1024",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=1024",
                "tokenizer.quantizer.ligand_embed_dim=512",
                "tokenizer.quantizer.ligand_n_tokens=1024",
            ],
        ),
    ),
    "LG Ligand 20A 512 1024 element": ModelInfo(
        description="Ligand only model with 20Å spatial attention",
        features=["512-dim embeddings", "20Å spatial attention", "Ligand only decoder", "1024 ligand tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_Ligand_20A_512_1024_element.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=512",
                "tokenizer.structure_encoder.embed_dim_hidden=512",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer/quantizer=slq_quantizer_ligand",
                "tokenizer/decoder_factory=struc_decoder_ligand_element",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.structure_encoder.ligand_atom_embedding=true",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=1024",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=1024",
                "tokenizer.decoder_factory.decoder_mapping.element_decoder.struc_token_codebook_size=512",
                "tokenizer.quantizer.ligand_embed_dim=512",
                "tokenizer.quantizer.ligand_n_tokens=1024",
            ],
        ),
    ),
    "LG Ligand 20A continuous": ModelInfo(
        description="Ligand only model with 20Å spatial attention",
        features=["512-dim embeddings", "20Å spatial attention", "Ligand only decoder", "Continuous ligand encoding"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_Ligand_20A_continuous.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim_hidden=512",
                "tokenizer.structure_encoder.embed_dim=4",
                "tokenizer.structure_encoder.encode_ligand=true",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=4",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=512",
                "tokenizer.quantizer=null",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
            ],
        ),
    ),
    # Protein-Ligand Models
    # These models can handle both protein and ligand structures
    "LG Ligand 20A seq 3di Aux": ModelInfo(
        description="Protein-ligand model with sequence and 3Di awareness",
        features=[
            "256-dim embeddings",
            "20Å spatial attention",
            "Sequence and 3Di decoder",
            "Ligand encoding support",
            "512 ligand tokens",
            "512 protein tokens",
        ],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_Ligand_20A_seq_3di_Aux.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer.structure_encoder.encode_ligand=true",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=512",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=512",
                "tokenizer.quantizer.ligand_n_tokens=512",
                "tokenizer/quantizer=slq_quantizer_ligand",
                "tokenizer/decoder_factory=struc_decoder_ligand_3di_sequence",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
            ],
        ),
    ),
    # Protein-Only Models
    # These models are optimized for protein structure analysis
    "LG 20A seq Aux": ModelInfo(
        description="Sequence-aware protein model",
        features=["256-dim embeddings", "20Å spatial attention", "Sequence decoder", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A_seq_Aux.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer/decoder_factory=struc_decoder_sequence",
            ],
        ),
    ),
    "LG 20A seq 3di c6d Aux": ModelInfo(
        description="Sequence, 3Di and C6D-aware protein model",
        features=["256-dim embeddings", "20Å spatial attention", "Sequence + 3Di + C6D decoder", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A_seq_3di_c6d_Aux.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer/decoder_factory=struc_decoder_3di_c6d_sequence",
            ],
        ),
    ),
    "LG 20A seq 3di c6d Aux Pinder": ModelInfo(
        description="Sequence, 3Di and C6D-aware protein model",
        features=["256-dim embeddings", "20Å spatial attention", "Sequence + 3Di + C6D decoder", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A_seq_3di_c6d_Aux_Pinder.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer/decoder_factory=struc_decoder_3di_c6d_sequence",
            ],
        ),
    ),
    "LG 20A seq 3di c6d Aux PDB": ModelInfo(
        description="Sequence, 3Di and C6D-aware protein model",
        features=["256-dim embeddings", "20Å spatial attention", "Sequence + 3Di + C6D decoder", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A_seq_3di_c6d_Aux_PDB.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer/decoder_factory=struc_decoder_3di_c6d_sequence",
            ],
        ),
    ),
    "LG 20A seq 3di c6d Aux PDB Pinder": ModelInfo(
        description="Sequence, 3Di and C6D-aware protein model",
        features=["256-dim embeddings", "20Å spatial attention", "Sequence + 3Di + C6D decoder", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A_seq_3di_c6d_Aux_PDB_Pinder.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer/decoder_factory=struc_decoder_3di_c6d_sequence",
            ],
        ),
    ),
    "LG 20A seq 3di c6d Aux PDB Pinder Finetune": ModelInfo(
        description="Sequence, 3Di and C6D-aware protein model",
        features=["256-dim embeddings", "20Å spatial attention", "Sequence + 3Di + C6D decoder", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A_seq_3di_c6d_Aux_PDB_Pinder_Finetune.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1",
                "+tokenizer.structure_encoder.dropout=0.1",
                "+tokenizer.structure_encoder.attention_dropout=0.1",
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.quantizer.embed_dim=256",
                "tokenizer/decoder_factory=struc_decoder_3di_c6d_sequence",
            ],
        ),
    ),
    "LG 20A": ModelInfo(
        description="Basic protein model with 20Å cutoff",
        features=["Standard configuration", "20Å spatial attention", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_20A.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=20.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0",
            ],
        ),
    ),
    "LG 10A": ModelInfo(
        description="Basic protein model with 10Å cutoff",
        features=["Standard configuration", "10Å spatial attention", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_10A.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "+tokenizer.structure_encoder.spatial_attention_mask=true",
                "+tokenizer.structure_encoder.angstrom_cutoff=10.0",
                "+tokenizer.structure_encoder.angstrom_cutoff_spatial=10.0",
            ],
        ),
    ),
    "LG full attention": ModelInfo(
        description="Full attention model without spatial masking",
        features=["Standard configuration", "Full attention (no spatial masking)", "256 protein tokens"],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_full_attention.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[],
        ),
    ),
}


def format_resolver(x, pattern):
    """Format `x` using `pattern`.

    Can be registered as an OmegaConf resolver:
    ```
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("format", format_resolver)
    ```
    to enable formatted interpolations in hydra config.
    """
    return f"{x:{pattern}}"


def instantiate_dict_cfg(cfg: DictConfig | None, verbose=False):
    """Instantiate each value in a dictionary and return a list of the instantiated objects."""
    out = []

    if not cfg:
        return out

    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be a DictConfig")

    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            if "_target_" in v:
                if verbose:
                    print(f"instantiating <{v._target_}>")
                out.append(hydra.utils.instantiate(v))
            else:
                out.extend(instantiate_dict_cfg(v, verbose=verbose))

    return out


OmegaConf.register_new_resolver("format", format_resolver, replace=True)


def load_config(config_path: str, config_name: str, overrides: list[str] | None = None) -> DictConfig:
    # Check if Hydra is already initialized
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra with the configuration path
    with hydra.initialize(config_path=config_path, version_base=None):
        # Compose the configuration object from the specified config name
        if overrides:
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
        else:
            cfg = hydra.compose(config_name=config_name)
    return cfg


class LatentEncoderDecoder:
    """A utility class for loading a TokenizerMulti model and encoding inputs to latent vectors."""

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(
        self, checkpoint_path: str, cfg_path: str, cfg_name: str, overrides: list[str] | None = None
    ) -> None:
        """Load a TokenizerMulti model from a checkpoint path.

        Args:
            checkpoint_path: Path to the model checkpoint
            config_path: Optional path to the model configuration file
        """
        if checkpoint_path.startswith("s3://"):
            # Handle S3 path
            try:
                s3 = boto3.client("s3")
                bucket_name, key = checkpoint_path[5:].split("/", 1)  # Extract bucket and key
                local_checkpoint_path = "/tmp/" + os.path.basename(key)  # Temporary local path
                s3.download_file(bucket_name, key, local_checkpoint_path)
                checkpoint_path = local_checkpoint_path  # Update checkpoint_path to the local file
            except NoCredentialsError as e:
                raise RuntimeError("AWS credentials not found. Ensure they are configured properly.") from e
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint from S3: {e}") from e
        elif checkpoint_path.startswith("https://huggingface.co/"):
            # Handle Hugging Face path
            try:
                # Convert Hugging Face blob URL to raw URL
                if "/blob/" in checkpoint_path:
                    checkpoint_path = checkpoint_path.replace("/blob/", "/resolve/")

                # Extract filename from URL
                filename = os.path.basename(urllib.parse.urlparse(checkpoint_path).path)
                local_checkpoint_path = "/tmp/" + filename

                py_logger.info(f"Downloading checkpoint from Hugging Face: {checkpoint_path}")
                urllib.request.urlretrieve(checkpoint_path, local_checkpoint_path)
                checkpoint_path = local_checkpoint_path  # Update checkpoint_path to the local file
                py_logger.info(f"Checkpoint downloaded to: {checkpoint_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint from Hugging Face: {e}") from e
        else:
            # Handle local path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        py_logger.info(f"Loading model from {checkpoint_path}")

        cfg = load_config(cfg_path, cfg_name, overrides)

        # If config path is provided, load the model with the config
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)

        tokenizer = TokenizerMulti.load_from_checkpoint(
            checkpoint_path,
            structure_encoder=tokenizer.encoder,
            decoder_factory=tokenizer.decoder_factory,
            loss_factory=tokenizer.loss_factory,
            optim=tokenizer.optim_factory,
            lr_scheduler=tokenizer.lr_scheduler,
            quantizer=tokenizer.quantizer,
            freeze_decoder=tokenizer.freeze_decoder,
            freeze_encoder=tokenizer.freeze_encoder,
            freeze_quantizer=tokenizer.freeze_quantizer,
            strict=True,
        )

        self.model = tokenizer

        self.model = self.model.to(self.device)
        py_logger.info(f"Model loaded successfully and moved to {self.device}")

    def encode(
        self,
        inputs: torch.Tensor | dict[str, Any],
        discrete: bool = True,
        return_embeddings: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Encode inputs to latent vectors.

        Args:
            inputs: Input data to encode
            discrete: Whether to use discrete encoding
            **kwargs: Additional arguments to pass to the model

        Returns:
            Latent vectors

        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        if self.model.quantizer is None:
            discrete = False

        # Move inputs to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Ensure model is in eval mode
        self.model.eval()

        # Generate latents
        with torch.no_grad():
            latents = self.model(inputs, return_embeddings=return_embeddings, **kwargs)

        if return_embeddings:
            latents, embeddings = latents
            if discrete:
                if isinstance(latents, dict):
                    if "protein_tokens" in latents:
                        _, _, n_tokens = latents["protein_tokens"].shape
                        latents_protein = torch.argmax(latents["protein_tokens"], dim=-1)
                        latents["protein_tokens"] = torch.nn.functional.one_hot(
                            latents_protein, num_classes=n_tokens
                        ).float()
                    _, _, n_tokens_ligand = latents["ligand_tokens"].shape
                    latents_ligand = torch.argmax(latents["ligand_tokens"], dim=-1)
                    latents["ligand_tokens"] = torch.nn.functional.one_hot(
                        latents_ligand, num_classes=n_tokens_ligand
                    ).float()
                    return latents, embeddings
                else:
                    _, _, n_tokens = latents.shape
                    latents = torch.argmax(latents, dim=-1)
                    latents = torch.nn.functional.one_hot(latents, num_classes=n_tokens).float()
                    return latents, embeddings
            else:
                return latents, embeddings

        if discrete:
            if isinstance(latents, dict):
                if "protein_tokens" in latents:
                    _, _, n_tokens = latents["protein_tokens"].shape
                    latents_protein = torch.argmax(latents["protein_tokens"], dim=-1)
                    latents["protein_tokens"] = torch.nn.functional.one_hot(
                        latents_protein, num_classes=n_tokens
                    ).float()
                _, _, n_tokens_ligand = latents["ligand_tokens"].shape
                latents_ligand = torch.argmax(latents["ligand_tokens"], dim=-1)
                latents["ligand_tokens"] = torch.nn.functional.one_hot(
                    latents_ligand, num_classes=n_tokens_ligand
                ).float()
                return latents
            else:
                _, _, n_tokens = latents.shape
                latents = torch.argmax(latents, dim=-1)
                latents = torch.nn.functional.one_hot(latents, num_classes=n_tokens).float()

        return latents

    def decode(self, latents: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Decode latent vectors back to original inputs.

        Args:
            latents: Latent vectors to decode
            **kwargs: Additional arguments to pass to the model

        Returns:
            Decoded outputs
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        # Move latents to device
        if isinstance(latents, dict):
            latents = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in latents.items()}
        else:
            latents = latents.to(self.device)

        # Ensure model is in eval mode
        self.model.eval()

        # Decode latents
        with torch.no_grad():
            decoded_outputs = self.model.decode(latents, **kwargs)
        decoded_structure = decoded_outputs["vit_decoder"]

        if "sequence_decoder" in decoded_outputs:
            decoded_sequence = decoded_outputs["sequence_decoder"]
        else:
            decoded_sequence = None

        return decoded_structure, decoded_sequence


# Create a global instance for easy importing
encoder_decoder = LatentEncoderDecoder()
load_model = encoder_decoder.load_model
encode = encoder_decoder.encode
decode = encoder_decoder.decode


def main():
    """
    Main function for loading a model and encoding inputs.
    """
    # Use argparse instead of hydra for better handling of paths with '=' in them
    parser = argparse.ArgumentParser(
        description="Load a TokenizerMulti model and encode inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
----------------

Ligand Models:
--------------
LG Ligand 20A
    Description: Ligand only model with 20Å spatial attention
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Ligand only decoder
    - 512 ligand tokens

LG Ligand 20A 512 1024
    Description: Ligand only model with 20Å spatial attention
    Features:
    - 512-dim embeddings
    - 20Å spatial attention
    - Ligand only decoder
    - 1024 ligand tokens

LG Ligand 20A 512 1024 element
    Description: Ligand only model with 20Å spatial attention
    Features:
    - 512-dim embeddings
    - 20Å spatial attention
    - Ligand only decoder with element awareness
    - 1024 ligand tokens

LG Ligand 20A continuous
    Description: Ligand only model with 20Å spatial attention
    Features:
    - 512-dim embeddings
    - 20Å spatial attention
    - Ligand only decoder
    - Continuous ligand encoding

Protein-Ligand Models:
---------------------
LG Ligand 20A seq 3di Aux
    Description: Protein-ligand model with sequence and 3Di awareness
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence and 3Di decoder
    - Ligand encoding support
    - 512 ligand tokens
    - 512 protein tokens

Protein-Only Models:
------------------
LG 20A seq Aux
    Description: Sequence-aware protein model
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence decoder
    - 256 protein tokens

LG 20A seq 3di c6d Aux
    Description: Sequence, 3Di and C6D-aware protein model
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence + 3Di + C6D decoder
    - 256 protein tokens

LG 20A seq 3di c6d Aux Pinder
    Description: Sequence, 3Di and C6D-aware protein model
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence + 3Di + C6D decoder
    - 256 protein tokens

LG 20A seq 3di c6d Aux PDB
    Description: Sequence, 3Di and C6D-aware protein model
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence + 3Di + C6D decoder
    - 256 protein tokens

LG 20A seq 3di c6d Aux PDB Pinder
    Description: Sequence, 3Di and C6D-aware protein model
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence + 3Di + C6D decoder
    - 256 protein tokens

LG 20A seq 3di c6d Aux PDB Pinder Finetune
    Description: Sequence, 3Di and C6D-aware protein model
    Features:
    - 256-dim embeddings
    - 20Å spatial attention
    - Sequence + 3Di + C6D decoder
    - 256 protein tokens

Vanilla Models:
------------
LG 20A
    Description: Basic protein model with 20Å cutoff
    Features:
    - Standard configuration
    - 20Å spatial attention
    - 256 protein tokens

LG 10A
    Description: Basic protein model with 10Å cutoff
    Features:
    - Standard configuration
    - 10Å spatial attention
    - 256 protein tokens

LG full attention
    Description: Full attention model without spatial masking
    Features:
    - Standard configuration
    - Full attention (no spatial masking)
    - 256 protein tokens
""",
    )
    parser.add_argument(
        "--model_name", type=str, required=False, help=f"Name of the model to load. Options: {list(methods.keys())}"
    )
    parser.add_argument("--ckpt_path", type=str, required=False, help="Path to the checkpoint file")
    parser.add_argument("--cfg_path", type=str, required=False, help="Path to the configuration file")
    parser.add_argument("--cfg_name", type=str, default="train_multi", help="Name of the configuration to load")
    parser.add_argument("--pdb_path", type=str, help="Path to a PDB file to encode")
    parser.add_argument("--ligand_path", type=str, help="Path to a sdf file to encode")
    parser.add_argument("--decode", action="store_true", help="Decode latents back to original inputs")
    parser.add_argument("--decode_latents", type=str, help="Path to latents to decode")
    parser.add_argument("--batch_file", type=str, help="Path to a batch file to encode")
    parser.add_argument(
        "--output_file_encode", type=str, default="encoded_latents.pt", help="Path to save encoded outputs"
    )
    parser.add_argument(
        "--output_file_decode", type=str, default="decoded_outputs.pt", help="Path to save decoded outputs"
    )
    parser.add_argument("--overrides", type=str, nargs="+", help="Configuration overrides in the format key=value")

    args = parser.parse_args()

    # Set up logging
    py_logger.info(f"Current directory: {os.getcwd()}")
    py_logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        py_logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load the model with overrides if provided
    if (
        args.model_name != "LG Ligand 20A seq 3di Aux"
        and args.model_name != "LG Ligand 20A"
        and args.model_name != "LG Ligand 20A continuous"
    ) and args.ligand_path is not None:
        raise ValueError(
            "Ligand path is only supported for LG Ligand 20A seq 3di Aux model, LG Ligand 20A model or LG Ligand 20A continuous model"
        )

    if args.model_name in methods:
        load_model(
            methods[args.model_name].model_config.checkpoint,
            methods[args.model_name].model_config.config_path,
            methods[args.model_name].model_config.config_name,
            overrides=methods[args.model_name].model_config.overrides,
        )
    elif args.ckpt_path and args.cfg_path:
        py_logger.info(f"Loading model from {args.ckpt_path}")
        load_model(args.ckpt_path, args.cfg_path, args.cfg_name, overrides=args.overrides)
    else:
        raise ValueError(f"Model name {args.model_name} not found. Please choose from {list(methods.keys())}")
    py_logger.info("Model loaded successfully")

    # encode pdb
    if (args.pdb_path and os.path.exists(args.pdb_path)) or (args.ligand_path and os.path.exists(args.ligand_path)):
        if args.pdb_path and os.path.exists(args.pdb_path):
            py_logger.info(f"Encoding PDB from {args.pdb_path}")
            pdb_data = load_pdb(args.pdb_path)
        else:
            py_logger.info("No PDB path provided, using empty protein data")
            pdb_data = {"protein_coords": None, "protein_mask": None, "protein_seq": None}

        if args.ligand_path and os.path.exists(args.ligand_path):
            py_logger.info(f"Encoding ligand from {args.ligand_path}")
            ligand_data = load_ligand(args.ligand_path)
            pdb_data["ligand_coords"] = ligand_data["atom_coords"]
            pdb_data["ligand_mask"] = ligand_data["mask"]
            pdb_data["ligand_residue_index"] = ligand_data["atom_indices"]
            pdb_data["ligand_atom_names"] = ligand_data["atom_names"]
            pdb_data["ligand_indices"] = ligand_data["atom_indices"]

        if args.model_name in [
            "LG ESMC 300M 256 cont",
            "LG ESMC 300M 512 cont",
            "LG ESMC 300M 960 cont",
            "LG ESMC 300M 960 full cont",
        ]:
            from lobster.model.latent_generator.latent_generator.datasets._transforms import ESMEmbeddingTransform

            esm_transform = ESMEmbeddingTransform(model_name="esmc_300m", device="auto")
            pdb_data = esm_transform(pdb_data)
            pdb_data["plm_embeddings"] = pdb_data["esm_c_embeddings"]

        pdb_latents, pdb_embeddings = encode(pdb_data, return_embeddings=True)

        # Save encoded latents
        torch.save(pdb_latents, args.output_file_encode)
        py_logger.info(f"PDB encoding completed. Results saved to {args.output_file_encode}")

        if args.decode:
            py_logger.info(f"Decoding latents from {args.output_file_encode}")
            decoded_outputs, sequence_outputs = decode(pdb_latents, x_emb=pdb_embeddings)
            if isinstance(decoded_outputs, dict):
                x_recon_ligand = decoded_outputs["ligand_coords"]
                x_recon_xyz = decoded_outputs["protein_coords"]
                filename = f"{args.output_file_decode.split('.')[0]}_ligand_decoded.pdb"
                if x_recon_xyz is not None:
                    if sequence_outputs is not None:
                        seq = sequence_outputs.argmax(dim=-1)
                        seq[seq == 22] = 21
                    else:
                        seq = torch.zeros(x_recon_xyz.shape[1], dtype=torch.long)[None]
                else:
                    seq = None
                    filename = f"{args.output_file_decode.split('.')[0]}_ligand_only_decoded.pdb"
                if x_recon_ligand is not None:
                    ligand_atoms = x_recon_ligand[0]
                    ligand_atom_names = None
                    ligand_chain = "L"
                    ligand_resname = "LIG"
                    if x_recon_xyz is not None:
                        writepdb_ligand_complex(
                            filename,
                            ligand_atoms=ligand_atoms,
                            ligand_atom_names=ligand_atom_names,
                            ligand_chain=ligand_chain,
                            ligand_resname=ligand_resname,
                            protein_atoms=x_recon_xyz[0],
                            protein_seq=seq[0],
                        )
                    else:
                        writepdb_ligand_complex(
                            filename,
                            ligand_atoms=ligand_atoms,
                            ligand_atom_names=ligand_atom_names,
                            ligand_chain=ligand_chain,
                            ligand_resname=ligand_resname,
                        )
            else:
                if sequence_outputs is not None:
                    seq = sequence_outputs.argmax(dim=-1)
                    seq[seq == 22] = 21
                else:
                    seq = torch.zeros(decoded_outputs.shape[1], dtype=torch.long)[None]
                filename = f"{args.output_file_decode.split('.')[0]}_decoded.pdb"
                writepdb(filename, decoded_outputs[0], seq[0])
                writepdb(filename, decoded_outputs[0], seq[0])

            # Save decoded outputs
            torch.save(decoded_outputs, args.output_file_decode)
            py_logger.info(f"Decoding completed. Results saved to {args.output_file_decode}")

    # If batch file is provided, encode all inputs in the batch
    if args.batch_file and os.path.exists(args.batch_file):
        py_logger.info(f"Encoding batch from {args.batch_file}")
        batch_data = torch.load(args.batch_file)
        batch_latents = encode(batch_data)

        # Save encoded latents
        torch.save(batch_latents, args.output_file_encode)
        py_logger.info(f"Batch encoding completed. Results saved to {args.output_file_encode}")

    # If decode_latents is provided, decode the latents
    if args.decode_latents and os.path.exists(args.decode_latents):
        py_logger.info(f"Decoding latents from {args.decode_latents}")
        latents = torch.load(args.decode_latents)
        decoded_outputs = decode(latents)

        # Save decoded outputs
        torch.save(decoded_outputs, args.output_file)
        py_logger.info(f"Decoding completed. Results saved to {args.output_file_decode}")


if __name__ == "__main__":
    main()
