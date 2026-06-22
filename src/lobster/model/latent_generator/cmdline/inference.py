# File: encode_latents.py

import argparse
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import hydra
from hydra.core.global_hydra import GlobalHydra
import torch
from botocore.exceptions import NoCredentialsError
from omegaconf import DictConfig, OmegaConf

from lobster.model.latent_generator.io import load_ligand, load_pdb, writepdb, writepdb_ligand_complex
from lobster.model.latent_generator.tokenizer import TokenizerMulti
from lobster.model.latent_generator.utils import get_ligand_energy, minimize_ligand_structure

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
    # Optional override for the model class. Defaults to "TokenizerMulti"
    # (the LG codec autoencoder) which the existing pipeline expects.
    # Set to "Tokenizer3diInputFlow" for the 3Di-conditioned flow decoder
    # — it doesn't have an encoder/quantizer pair and decodes via SDE
    # flow-matching sampling instead, so load_model + decode dispatch
    # to a different code path.
    model_class: str = "TokenizerMulti"


@dataclass
class ModelInfo:
    """Information about a model including its configuration."""

    description: str
    features: list[str]
    model_config: ModelConfig


methods = {
    # Ligand Models
    # These models are optimized for ligand structure analysis
    "LG Ligand": ModelInfo(
        description="Ligand only model with",
        features=["256-dim embeddings", "Ligand only decoder", "512 ligand tokens"],
        model_config=ModelConfig(
            # S3 backup: s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Ligand_2025-11-09.ckpt
            checkpoint="/cv/data/ai4dd/data2/ume/latent_generator_/runs//2025-11-09T14-23-55/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
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
    "LG Protein Ligand": ModelInfo(
        description="Protein-ligand model with structure-only encoding",
        features=[
            "256-dim embeddings",
            "Ligand encoding support",
            "512 ligand tokens",
            "512 protein tokens",
        ],
        model_config=ModelConfig(
            checkpoint="/cv/data/ai4dd/data2/ume/latent_generator_/runs//2025-12-07T22-38-42/epoch=830-step=88917-val_loss=16.5010.ckpt",  # "/cv/data/ai4dd/data2/ume/latent_generator_/runs//2025-11-26T15-51-49/last.ckpt", #"/cv/data/ai4dd/data2/ume/latent_generator_/runs//2025-11-25T14-42-33/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=4",
                "tokenizer.quantizer.embed_dim=4",
                "tokenizer.quantizer.ligand_embed_dim=4",
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
    "LG Protein Ligand 4096": ModelInfo(
        description="Protein-ligand model with structure-only encoding",
        features=[
            "256-dim embeddings",
            "Ligand encoding support",
            "4096 ligand tokens",
            "4096 protein tokens",
        ],
        model_config=ModelConfig(
            # S3 backup: s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_4096_2026-01-05.ckpt
            checkpoint="/cv/data/ai4dd/data2/ume/latent_generator_/runs//2026-01-05T16-48-02/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=4",
                "tokenizer.quantizer.embed_dim=4",
                "tokenizer.quantizer.ligand_embed_dim=4",
                "tokenizer.structure_encoder.encode_ligand=true",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=4096",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=4096",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=512",
                "tokenizer.quantizer.ligand_n_tokens=4096",
                "tokenizer.quantizer.n_tokens=4096",
                "tokenizer/quantizer=slq_quantizer_ligand",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
            ],
        ),
    ),
    "LG Protein Ligand fsq 4375": ModelInfo(
        description="Protein-ligand model with FSQ quantization (4375 tokens)",
        features=[
            "5-dim embeddings",
            "FSQ quantization",
            "Ligand encoding support",
            "4375 ligand tokens",
            "4375 protein tokens",
        ],
        model_config=ModelConfig(
            # S3 backup: s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt
            # data2: /cv/data/ai4dd/data2/ume/latent_generator_/runs//2026-01-05T16-13-19/last.ckpt
            checkpoint="s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=5",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer/quantizer=fsq_quantizer_ligand",
                "tokenizer.quantizer.protein_levels=[7,5,5,5,5]",
                "tokenizer.quantizer.ligand_levels=[7,5,5,5,5]",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "tokenizer/loss_factory=structure_losses_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=4375",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=4375",
            ],
        ),
    ),
    "LG Protein Ligand fsq 4375 15360": ModelInfo(
        description="Protein-ligand model with FSQ quantization (4375 protein tokens, 15360 ligand tokens)",
        features=[
            "5-dim embeddings",
            "FSQ quantization",
            "Ligand encoding support",
            "15360 ligand tokens",
            "4375 protein tokens",
        ],
        model_config=ModelConfig(
            # S3 backup: s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_15360_2026-01-07.ckpt
            checkpoint="/cv/data/ai4dd/data2/ume/latent_generator_/runs//2026-01-07T02-17-14/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=5",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer/quantizer=fsq_quantizer_ligand",
                "tokenizer.quantizer.protein_levels=[7,5,5,5,5]",
                "tokenizer.quantizer.ligand_levels=[8,8,8,6,5]",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "tokenizer/loss_factory=structure_losses_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=4375",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=15360",
            ],
        ),
    ),
    "LG Protein Ligand fsq 240 4375": ModelInfo(
        description="Protein-ligand model with FSQ quantization (240 protein tokens, 4375 ligand tokens)",
        features=[
            "5-dim embeddings",
            "FSQ quantization",
            "Ligand encoding support",
            "4375 ligand tokens",
            "240 protein tokens",
        ],
        model_config=ModelConfig(
            # Trained with different FSQ levels for smaller protein codebook (240 vs 4375)
            checkpoint="/cv/scratch/u/lisanzas/latent_generator_fsq_sair_240_4375/runs//2026-02-02T21-26-19/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=5",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer/quantizer=fsq_quantizer_ligand",
                "tokenizer.quantizer.protein_levels=[5,4,3,2,2]",
                "tokenizer.quantizer.ligand_levels=[7,5,5,5,5]",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "tokenizer/loss_factory=structure_losses_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=240",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=4375",
            ],
        ),
    ),
    "LG Protein Ligand fsq 4375 15360 bond": ModelInfo(
        description="Protein-ligand model with FSQ quantization, bond matrix embedding, and extended element vocabulary",
        features=[
            "5-dim embeddings",
            "FSQ quantization",
            "Ligand encoding support",
            "Bond matrix embedding",
            "Extended element vocabulary (25 tokens)",
            "15360 ligand tokens",
            "4375 protein tokens",
        ],
        model_config=ModelConfig(
            # Trained with: slurm/scripts/train_latent_generator_protein_ligand_fsq_bond_element.sh
            checkpoint="/cv/scratch/u/lisanzas/latent_generator_bond_element/runs/2026-01-24T20-54-23/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=5",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer.structure_encoder.ligand_atom_embedding=true",
                "tokenizer.structure_encoder.use_ligand_bond_embedding=true",
                "tokenizer.structure_encoder.use_extended_element_vocab=true",
                "tokenizer/quantizer=fsq_quantizer_ligand",
                "tokenizer.quantizer.protein_levels=[7,5,5,5,5]",
                "tokenizer.quantizer.ligand_levels=[8,8,8,6,5]",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "tokenizer/loss_factory=structure_losses_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=4375",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=15360",
            ],
        ),
    ),
    "LG Protein Ligand cont": ModelInfo(
        description="Protein-ligand model with CONTINUOUS embeddings (no quantization)",
        features=[
            "256-dim continuous embeddings",
            "No quantization (quantizer=null)",
            "Ligand encoding support",
            "Bond matrix embedding",
            "Extended element vocabulary (25 tokens)",
            "For use with DiffusionLoss in Gen-UME",
        ],
        model_config=ModelConfig(
            # Trained with: slurm/scripts/train_latent_generator_protein_ligand_continuous_bond_element.sh
            checkpoint="/cv/scratch/u/lisanzas/latent_generator_continuous_bond_element/runs/2026-01-24T21-03-23/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=256",
                "tokenizer.structure_encoder.embed_dim_hidden=512",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer.structure_encoder.ligand_atom_embedding=true",
                "tokenizer.structure_encoder.use_ligand_bond_embedding=true",
                "tokenizer.structure_encoder.use_extended_element_vocab=true",
                "tokenizer.quantizer=null",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "tokenizer/loss_factory=structure_losses_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=256",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=256",
            ],
        ),
    ),
    "LG Protein Ligand fsq 1000": ModelInfo(
        description="Protein-ligand model with FSQ quantization (1000 tokens)",
        features=[
            "4-dim embeddings",
            "FSQ quantization",
            "Ligand encoding support",
            "1000 ligand tokens",
            "1000 protein tokens",
        ],
        model_config=ModelConfig(
            checkpoint="/cv/data/ai4dd/data2/ume/latent_generator_/runs//2025-12-13T14-57-53/epoch=210-step=22577-val_loss=17.2066.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=4",
                "tokenizer.structure_encoder.encode_ligand=true",
                "tokenizer/quantizer=fsq_quantizer_ligand",
                "tokenizer.quantizer.protein_levels=[8,5,5,5]",
                "tokenizer.quantizer.ligand_levels=[8,5,5,5]",
                "tokenizer/decoder_factory=struc_decoder_ligand",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=1000",
                "+tokenizer.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=1000",
            ],
        ),
    ),
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
    "LG 3Di Flow Decoder": ModelInfo(
        description=(
            "3Di-conditioned flow-matching backbone decoder. Reconstructs "
            "N/Cα/C coordinates from per-residue Foldseek 3Di tokens via a "
            "200-step power-2 SDE trajectory. CFG scale w=2.0."
        ),
        features=[
            "U-ViT 768d / 12L / 12h (~110M params)",
            "Velocity-prediction continuous flow matching",
            "Foldseek 3Di alphabet input (20 states)",
            "No encoder, no quantizer -- pure decoder",
            "Tokenizer3diInputFlow (separate class from TokenizerMulti)",
        ],
        model_config=ModelConfig(
            checkpoint="https://huggingface.co/Sidney-Lisanza/latent_generator/resolve/main/checkpoints_for_lg/LG_3Di_Flow_Decoder.ckpt",
            # Hydra config below points at the project-wide hydra tree
            # (`src/lobster/hydra_config/`) instead of the LG-internal one,
            # because Tokenizer3diInputFlow's architecture/dependency yaml
            # (`latent_generator_3di_input_flow_nokabsch_velocity_base`)
            # lives there. `config_path` is resolved relative to this
            # file via the special "@project" prefix handled by load_model.
            config_path="@project",
            config_name="latent_generator_3di_input_flow_nokabsch_velocity_base",
            # Match the published checkpoint's training config. The base
            # yaml doesn't carry these keys, so the `+` prefix tells
            # hydra to add (not override) them. Setting either weight>0
            # or `track_aux_3di_coord_ce=true` causes the constructor to
            # build the frozen mini3di buffers so the state_dict's
            # `mini3di_torch.*` keys land cleanly.
            overrides=[
                "+aux_3di_coord_ce_weight=0.1",
                "+aux_3di_coord_ce_t_lim=0.5",
                "+aux_3di_coord_ce_temperature=1.0",
            ],
            model_class="Tokenizer3diInputFlow",
        ),
    ),
    "LG full attention 2": ModelInfo(
        description="Full attention model without spatial masking",
        features=["Standard configuration", "Full attention (no spatial masking)", "256 protein tokens"],
        model_config=ModelConfig(
            # S3 backup: s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_full_attention_2_2025-11-06.ckpt
            checkpoint="/cv/data/ai4dd/data2/ume/latent_generator_/runs//2025-11-06T00-40-11/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[],
        ),
    ),
    "LG full attention 512 PDB Pinder FSQ": ModelInfo(
        description="Full attention model with 512 protein tokens and FSQ quantization",
        features=["240 protein tokens", "FSQ quantization"],
        model_config=ModelConfig(
            checkpoint="/cv/data/ai4dd/data2/lisanzas/latent_generator/studies/outputs/train/dev/runs/2025-11-09_22-19-12/checkpoints/last.ckpt",
            config_path="../../latent_generator/hydra_config/",
            config_name="train_multi",
            overrides=[
                "tokenizer.structure_encoder.embed_dim=3",
                "tokenizer/quantizer=fsq_quantizer",
                "tokenizer.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=240",
            ],
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
        # Set by `load_model` so `encode` / `decode` know which code
        # path to take. Defaults to the canonical TokenizerMulti
        # autoencoder; "Tokenizer3diInputFlow" routes through the
        # SDE flow-matching decoder instead.
        self.model_class = "TokenizerMulti"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(
        self,
        checkpoint_path: str,
        cfg_path: str,
        cfg_name: str,
        overrides: list[str] | None = None,
        model_class: str = "TokenizerMulti",
    ) -> None:
        """Load a model from a checkpoint path.

        Args:
            checkpoint_path: Path to the model checkpoint
            cfg_path: Path to the model configuration tree (relative to
                this file). The literal "@project" routes to the
                project-wide ``src/lobster/hydra_config/`` tree --
                used by the 3Di flow decoder whose architecture yaml
                lives there.
            cfg_name: Hydra config name. For TokenizerMulti this is
                ``train_multi``; for Tokenizer3diInputFlow it's the
                model yaml itself (e.g.
                ``latent_generator_3di_input_flow_nokabsch_velocity_base``).
            overrides: Hydra-style key=value overrides applied at compose.
            model_class: ``"TokenizerMulti"`` (default, LG autoencoder)
                or ``"Tokenizer3diInputFlow"`` (3Di flow decoder).
        """
        if checkpoint_path.startswith("s3://"):
            # Handle S3 path
            try:
                s3 = boto3.client("s3")
                bucket_name, key = checkpoint_path[5:].split("/", 1)  # Extract bucket and key
                cache_dir = os.path.expanduser("~/.cache/lobster")
                os.makedirs(cache_dir, exist_ok=True)
                filename = os.path.basename(key)
                local_checkpoint_path = os.path.join(cache_dir, filename)
                if not os.path.exists(local_checkpoint_path):
                    s3.download_file(bucket_name, key, local_checkpoint_path)
                else:
                    py_logger.info(f"Checkpoint already exists at {local_checkpoint_path}")
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
                cache_dir = os.path.expanduser("~/.cache/lobster")
                os.makedirs(cache_dir, exist_ok=True)
                local_checkpoint_path = os.path.join(cache_dir, filename)
                if not os.path.exists(local_checkpoint_path):
                    py_logger.info(f"Downloading checkpoint from Hugging Face: {checkpoint_path}")
                    urllib.request.urlretrieve(checkpoint_path, local_checkpoint_path)
                else:
                    py_logger.info(f"Checkpoint already exists at {local_checkpoint_path}")
                checkpoint_path = local_checkpoint_path  # Update checkpoint_path to the local file
                py_logger.info(f"Checkpoint downloaded to: {checkpoint_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint from Hugging Face: {e}") from e
        else:
            # Handle local path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        py_logger.info(f"Loading model from {checkpoint_path}")

        # 3Di flow decoder: separate model class with no encoder/quantizer.
        # Doesn't go through `train_multi.yaml`'s tokenizer factory; instead
        # we hydra-compose the project-wide model yaml directly and load
        # the state_dict over an instantiated module.
        if model_class == "Tokenizer3diInputFlow":
            self._load_3di_flow_model(checkpoint_path, cfg_path, cfg_name, overrides)
            self.model_class = "Tokenizer3diInputFlow"
            return

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

    # Published inference recipe for the 3Di flow decoder. Same as
    # WINNER_W2 in the eval scripts and in the metadata sidecar shipped
    # alongside the HuggingFace ckpt.
    _WINNER_W2_3DI_FLOW = {
        "n_steps": 200,
        "sampling_mode": "sde",
        "schedule_type": "power",
        "schedule_exponent": 2.0,
        "guidance_scale": 2.0,
        "sc_scale_noise": 0.3,
        "sc_scale_score": 1.0,
        "gt_mode": "us",
        "gt_p": 1.0,
        "t_lim_ode": 0.99,
        "init_temperature": 1.0,
        "min_t": 0.0,
    }

    def _load_3di_flow_model(
        self,
        checkpoint_path: str,
        cfg_path: str,
        cfg_name: str,
        overrides: list[str] | None,
    ) -> None:
        """Hydra-compose the project-wide 3Di-flow model yaml + load state.

        The 3Di flow decoder's architecture yaml lives under the project's
        `src/lobster/hydra_config/model/` tree (not the LG-internal
        `cmdline/hydra_config/`), so we route via `@project` here.
        """
        import lobster  # noqa: F401  -- ensures package is importable

        if cfg_path == "@project":
            # Resolve to <repo_root>/src/lobster/hydra_config absolute.
            project_cfg_dir = Path(__file__).resolve().parents[3] / "hydra_config"
            if not project_cfg_dir.is_dir():
                raise FileNotFoundError(f"Could not find project hydra_config dir at {project_cfg_dir}")
            with hydra.initialize_config_dir(config_dir=str(project_cfg_dir / "model"), version_base=None):
                cfg = hydra.compose(config_name=cfg_name, overrides=list(overrides or []))
        else:
            cfg = load_config(cfg_path, cfg_name, overrides)

        # `cfg` here is the model node directly (no `tokenizer.` wrapper).
        if "ckpt_path" in cfg:
            cfg.ckpt_path = None
        py_logger.info(f"Instantiating {cfg._target_}")
        model = hydra.utils.instantiate(cfg, _recursive_=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            py_logger.warning("load_state_dict missing %d keys (e.g. %s)", len(missing), missing[:3])
        if unexpected:
            py_logger.warning("load_state_dict unexpected %d keys (e.g. %s)", len(unexpected), unexpected[:3])
        self.model = model.to(self.device).eval()
        py_logger.info(f"3Di flow decoder loaded successfully on {self.device}")

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

        # 3Di flow decoder has no learned encoder, so the "encoded
        # form" is just the per-residue Foldseek 3Di tokens.
        # `Structure3diTransform` derives them from the input
        # backbone here; `decode` then consumes them directly and
        # samples coords without re-deriving anything. Returns
        # ``({"3di_states", "mask", "indices"}, embeddings_or_None)``
        # to match the LG-codec encode signature.
        if self.model_class == "Tokenizer3diInputFlow":
            tokens = self._encode_3di_flow(inputs)
            if return_embeddings:
                # The encoded form has no continuous embedding (3Di
                # tokens themselves are the bottleneck); return an
                # empty tensor to keep the (latents, embeddings)
                # return shape stable across model_class branches.
                return tokens, torch.empty(0, device=self.device)
            return tokens

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

        # 3Di flow decoder: the "latents" passed in are actually the
        # `pdb_data` dict carrying ``coords_res`` + ``mask`` + ``indices``
        # (since `encode` is a no-op for this model). We compute the 3Di
        # tokens via Structure3diTransform here and call `model.sample`.
        if self.model_class == "Tokenizer3diInputFlow":
            return self._decode_3di_flow(latents, **kwargs)

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

    def _encode_3di_flow(self, pdb_data: dict) -> dict:
        """3Di-flow encode path: backbone coords -> per-residue 3Di tokens.

        Runs :class:`Structure3diTransform` once on each protein in the
        input batch and returns the canonical 3Di-flow latent dict
        (``{"3di_states", "mask", "indices"}``) that
        :meth:`_decode_3di_flow` consumes verbatim. Decoupling the
        transform from `decode` lets callers persist / reuse the
        tokens (e.g. write them to disk via ``--output_file_encode``)
        without re-running the transform on every decode call.
        """
        from lobster.transforms._structure_transforms import Structure3diTransform

        if not isinstance(pdb_data, dict) or "coords_res" not in pdb_data:
            raise ValueError("3Di flow encoder expects a `pdb_data` dict with `coords_res`; got " + str(type(pdb_data)))

        coords = pdb_data["coords_res"]
        if coords.dim() == 3:
            coords = coords.unsqueeze(0)
        B, L = coords.shape[0], coords.shape[1]
        mask = pdb_data.get("mask")
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.float32)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        indices = pdb_data.get("indices")
        if indices is None:
            indices = torch.arange(L).unsqueeze(0).expand(B, -1)
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)

        transform = Structure3diTransform()
        three_di = torch.full((B, L), 20, dtype=torch.long)
        for b in range(B):
            Lb = int(mask[b].sum().item())
            if Lb < 3:
                continue
            td = transform({"coords_res": coords[b, :Lb]})
            three_di[b, :Lb] = td["3di_states"].long()

        return {
            "3di_states": three_di.to(self.device),
            "mask": mask.to(self.device).float(),
            "indices": indices.to(self.device).long(),
        }

    def _decode_3di_flow(self, latents: dict, **_unused_kwargs) -> tuple[torch.Tensor, None]:
        """3Di-flow code path: 3Di tokens (+ mask, indices) -> backbone coords.

        Consumes the dict produced by :meth:`_encode_3di_flow` (the
        ``{"3di_states", "mask", "indices"}`` triple) and integrates a
        200-step SDE flow trajectory at guidance ``w=2.0``. Mirrors the
        LG-codec protein-only return shape so the existing ``main()``
        writepdb branch handles us without changes:
        ``(coords (B, L, 3, 3) tensor, None)``.
        """
        if not isinstance(latents, dict) or "3di_states" not in latents:
            raise ValueError(
                "3Di flow decoder expects an encoded dict with `3di_states`; "
                "did you skip the `encode(...)` step? Got " + str(type(latents))
            )
        states_in = latents["3di_states"].to(self.device).long()
        mask = latents.get("mask")
        if mask is None:
            mask = torch.ones_like(states_in, dtype=torch.float32)
        mask = mask.to(self.device).float()
        indices = latents.get("indices")
        if indices is None:
            indices = torch.arange(states_in.shape[1], device=self.device).unsqueeze(0).expand(states_in.shape[0], -1)
        indices = indices.to(self.device).long()

        states, seq_mask, residue_index = self.model.featurize(
            {"3di_states": states_in, "mask": mask, "indices": indices}
        )

        B, L = states_in.shape
        py_logger.info("3Di flow decode: B=%d, L=%d, sampling with WINNER_W2 recipe", B, L)
        with torch.no_grad():
            samp = self.model.sample(
                states=states,
                seq_mask=seq_mask,
                residue_index=residue_index,
                **self._WINNER_W2_3DI_FLOW,
            )
        return (samp.detach(), None)


# Create a global instance for easy importing
encoder_decoder = LatentEncoderDecoder()
load_model = encoder_decoder.load_model
encode = encoder_decoder.encode
decode = encoder_decoder.decode

# Minimization functions are imported from utils and can be accessed via:
# from lobster.model.latent_generator.cmdline import minimize_ligand_structure, get_ligand_energy
# or directly from: from lobster.model.latent_generator.utils import minimize_ligand_structure, get_ligand_energy


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
LG Protein Ligand
    Description: Protein-ligand model with structure-only encoding
    Features:
    - 256-dim embeddings
    - Ligand encoding support
    - 512 ligand tokens
    - 512 protein tokens

LG Protein Ligand fsq 4375
    Description: Protein-ligand model with FSQ quantization (4375 tokens)
    Features:
    - 5-dim embeddings
    - FSQ quantization
    - Ligand encoding support
    - 4375 ligand tokens
    - 4375 protein tokens

LG Protein Ligand fsq 1000
    Description: Protein-ligand model with FSQ quantization (1000 tokens)
    Features:
    - 4-dim embeddings
    - FSQ quantization
    - Ligand encoding support
    - 1000 ligand tokens
    - 1000 protein tokens

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
    parser.add_argument(
        "--output_pdb",
        type=str,
        default=None,
        help="Path to save decoded structure as PDB (optional, auto-generated if not provided)",
    )
    parser.add_argument("--overrides", type=str, nargs="+", help="Configuration overrides in the format key=value")

    # Minimization options
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize ligand structure after decoding using Open Babel force field",
    )
    parser.add_argument(
        "--minimize_steps",
        type=int,
        default=500,
        help="Maximum number of minimization steps (default: 500)",
    )
    parser.add_argument(
        "--force_field",
        type=str,
        default="MMFF94",
        choices=["MMFF94", "MMFF94s", "UFF", "GAFF", "Ghemical"],
        help="Force field for minimization (default: MMFF94)",
    )
    parser.add_argument(
        "--minimize_method",
        type=str,
        default="cg",
        choices=["cg", "sd"],
        help="Optimization method: cg (conjugate gradients) or sd (steepest descent)",
    )
    parser.add_argument(
        "--minimize_mode",
        type=str,
        default="bonds_and_angles",
        choices=["bonds_only", "bonds_and_angles"],
        help="Minimization mode: 'bonds_only' (ideal bond lengths), 'bonds_and_angles' (ideal bonds + angles, recommended)",
    )

    args = parser.parse_args()

    # Set up logging
    py_logger.info(f"Current directory: {os.getcwd()}")
    py_logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        py_logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load the model with overrides if provided
    ligand_supported_models = [
        "LG Protein Ligand",
        "LG Protein Ligand 4096",
        "LG Protein Ligand fsq 4375",
        "LG Protein Ligand fsq 4375 15360",
        "LG Protein Ligand fsq 240 4375",
        "LG Protein Ligand fsq 4375 15360 bond",
        "LG Protein Ligand fsq 1000",
        "LG Protein Ligand cont",
        "LG Ligand 20A seq 3di Aux",
        "LG Ligand 20A",
        "LG Ligand 20A continuous",
    ]
    if args.model_name not in ligand_supported_models and args.ligand_path is not None:
        raise ValueError(
            f"Ligand path is only supported for the following models: {', '.join(ligand_supported_models)}"
        )

    if args.model_name in methods:
        mc = methods[args.model_name].model_config
        load_model(
            mc.checkpoint,
            mc.config_path,
            mc.config_name,
            overrides=mc.overrides,
            model_class=mc.model_class,
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
            # Include bond matrix for CONECT records and minimization
            if "bond_matrix" in ligand_data:
                pdb_data["ligand_bond_matrix"] = ligand_data["bond_matrix"]

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

                # Get bond matrix and atom names for CONECT records (always retrieve these)
                bond_matrix = decoded_outputs.get("ligand_bond_matrix", None)
                if bond_matrix is None:
                    bond_matrix = pdb_data.get("ligand_bond_matrix", None)
                if bond_matrix is None:
                    bond_matrix = pdb_data.get("bond_matrix", None)

                ligand_atom_names = pdb_data.get("ligand_atom_names", None)
                if ligand_atom_names is None:
                    ligand_atom_names = pdb_data.get("atom_names", None)

                # Apply ligand minimization if requested
                if args.minimize and x_recon_ligand is not None:
                    py_logger.info(
                        f"Minimizing ligand structure (mode={args.minimize_mode}, force_field={args.force_field}, "
                        f"steps={args.minimize_steps}, method={args.minimize_method})"
                    )
                    # Get atom types from input data if available
                    ligand_atom_types = ligand_atom_names
                    if ligand_atom_types is None:
                        # Default to carbon for unknown atoms
                        num_atoms = x_recon_ligand.shape[1]
                        ligand_atom_types = ["C"] * num_atoms
                        py_logger.warning(f"No atom types provided, defaulting to Carbon for {num_atoms} atoms")

                    try:
                        # Calculate energy before minimization
                        energy_before = get_ligand_energy(
                            x_recon_ligand[0], ligand_atom_types, bond_matrix, args.force_field
                        )
                        py_logger.info(f"Energy before minimization: {energy_before:.2f} kcal/mol")

                        # Minimize
                        x_recon_ligand_minimized = minimize_ligand_structure(
                            x_recon_ligand[0],
                            ligand_atom_types,
                            bond_matrix=bond_matrix,
                            steps=args.minimize_steps,
                            force_field=args.force_field,
                            method=args.minimize_method,
                            mode=args.minimize_mode,
                        )

                        # Calculate energy after minimization
                        energy_after = get_ligand_energy(
                            x_recon_ligand_minimized, ligand_atom_types, bond_matrix, args.force_field
                        )
                        py_logger.info(f"Energy after minimization: {energy_after:.2f} kcal/mol")
                        py_logger.info(f"Energy reduction: {energy_before - energy_after:.2f} kcal/mol")

                        # Update ligand coordinates
                        x_recon_ligand = x_recon_ligand_minimized.unsqueeze(0)
                        decoded_outputs["ligand_coords"] = x_recon_ligand
                        py_logger.info("Ligand minimization completed successfully")
                    except Exception as e:
                        py_logger.warning(f"Ligand minimization failed: {e}. Using original coordinates.")

                # Determine PDB output filename
                if args.output_pdb:
                    filename = args.output_pdb
                else:
                    filename = f"{args.output_file_decode.split('.')[0]}_ligand_decoded.pdb"

                if x_recon_xyz is not None:
                    if sequence_outputs is not None:
                        seq = sequence_outputs.argmax(dim=-1)
                        seq[seq == 22] = 21
                    else:
                        seq = torch.zeros(x_recon_xyz.shape[1], dtype=torch.long)[None]
                else:
                    seq = None
                    if not args.output_pdb:
                        filename = f"{args.output_file_decode.split('.')[0]}_ligand_only_decoded.pdb"
                if x_recon_ligand is not None:
                    ligand_atoms = x_recon_ligand[0]
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
                            ligand_bond_matrix=bond_matrix,
                        )
                    else:
                        writepdb_ligand_complex(
                            filename,
                            ligand_atoms=ligand_atoms,
                            ligand_atom_names=ligand_atom_names,
                            ligand_chain=ligand_chain,
                            ligand_resname=ligand_resname,
                            ligand_bond_matrix=bond_matrix,
                        )
                    py_logger.info(f"PDB structure saved to {filename}")
            else:
                if sequence_outputs is not None:
                    seq = sequence_outputs.argmax(dim=-1)
                    seq[seq == 22] = 21
                else:
                    seq = torch.zeros(decoded_outputs.shape[1], dtype=torch.long)[None]
                if args.output_pdb:
                    filename = args.output_pdb
                else:
                    filename = f"{args.output_file_decode.split('.')[0]}_decoded.pdb"
                writepdb(filename, decoded_outputs[0], seq[0])
                py_logger.info(f"PDB structure saved to {filename}")

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
