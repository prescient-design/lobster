import os
import pathlib

import dotenv
import hydra
import lightning
import torch
import wandb
from icecream import ic
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger as py_logger
from omegaconf import OmegaConf
from omegaconf import DictConfig
from typing import Optional

import lobster.model.latent_generator as latent_generator
from lobster.model.latent_generator.tokenizer import TokenizerMulti

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

def instantiate_dict_cfg(cfg: Optional[DictConfig], verbose=False):
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

dotenv.load_dotenv(".env")


OmegaConf.register_new_resolver("format", format_resolver, replace=True)


@hydra.main(version_base=None, config_path="../../latent_generator/hydra_config/", config_name="train_multi")
def main(cfg):  # noqa: D103
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    if rank_zero_only.rank == 0:
        py_logger.info(f"{OmegaConf.to_yaml(log_cfg)}")
        py_logger.info(f"{os.getcwd()=}")
        py_logger.info(f"{torch.__config__.parallel_info()}")
        py_logger.info(f"{os.cpu_count()=}")
        py_logger.info(f"{os.getpid()=}")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if matmul_prec := cfg.get("float32_matmul_precision"):
        py_logger.info(f"setting float_32_matmul_precision to {matmul_prec}")
        torch.set_float32_matmul_precision(matmul_prec)

    loggers = instantiate_dict_cfg(cfg.get("logger"), verbose=(rank_zero_only.rank == 0))
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, lightning.pytorch.loggers.WandbLogger):
            wandb_logger = logger

    if wandb_logger:
        py_logger.info(f"{wandb_logger.experiment.name=}")

    callbacks = instantiate_dict_cfg(cfg.get("callbacks"), verbose=(rank_zero_only.rank == 0))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)  # noqa: F841

    if rank_zero_only.rank == 0 and wandb_logger: #latent_generator.__version__
        wandb_logger.experiment.config.update(
            {"cfg": log_cfg, "version": "1.0.0", "cwd": os.getcwd()}
        )

    if rank_zero_only.rank == 0 and wandb_logger:
        wandb_logger.watch(tokenizer, log="all")

    if cfg.get("ckpt_path") is not None and not cfg.get("continue_training", False):
        py_logger.info(f"loading checkpoint from {cfg.get('ckpt_path')}")
        
        # New configuration options for component loading
        load_encoder = cfg.get("load_encoder", True)
        load_encoder_strict = cfg.get("load_encoder_strict", True)
        load_quantizer = cfg.get("load_quantizer", True)
        load_quantizer_strict = cfg.get("load_quantizer_strict", True)
        load_decoder = cfg.get("load_decoder", True)
        load_decoder_strict = cfg.get("load_decoder_strict", True)
        
        if cfg.get("strict", True):
            # Option 1: Component-specific loading when any component is set to false
            py_logger.info("Using component-specific loading from checkpoint")
            checkpoint = torch.load(cfg.get("ckpt_path"))
            checkpoint_state_dict = checkpoint["state_dict"]
            
            # Load encoder weights if requested
            if load_encoder:
                py_logger.info("Loading encoder weights from checkpoint")
                # Extract encoder weights and remove the "encoder." prefix for loading
                encoder_state_dict = {}
                for k, v in checkpoint_state_dict.items():
                    if k.startswith("encoder."):
                        # Remove "encoder." prefix to match the model's expected keys
                        new_key = k.replace("encoder.", "", 1)  # Remove first occurrence of "encoder."
                        encoder_state_dict[new_key] = v
                
                if encoder_state_dict:
                    tokenizer.encoder.load_state_dict(encoder_state_dict, strict=load_encoder_strict)
                    py_logger.info(f"Successfully loaded {len(encoder_state_dict)} encoder parameters")
                else:
                    py_logger.warning("No encoder weights found in checkpoint")
            else:
                py_logger.info("Keeping randomly initialized encoder")
            
            # Load quantizer weights if requested
            if load_quantizer:
                py_logger.info("Loading quantizer weights from checkpoint")
                # Extract quantizer weights and remove the "quantizer." prefix for loading
                quantizer_state_dict = {}
                for k, v in checkpoint_state_dict.items():
                    if k.startswith("quantizer."):
                        # Remove "quantizer." prefix to match the model's expected keys
                        new_key = k.replace("quantizer.", "", 1)  # Remove first occurrence of "quantizer."
                        quantizer_state_dict[new_key] = v
                
                if quantizer_state_dict:
                    tokenizer.quantizer.load_state_dict(quantizer_state_dict, strict=load_quantizer_strict)
                    py_logger.info(f"Successfully loaded {len(quantizer_state_dict)} quantizer parameters")
                else:
                    py_logger.warning("No quantizer weights found in checkpoint")
            else:
                py_logger.info("Keeping randomly initialized quantizer")
            
            # Load decoder weights if requested
            if load_decoder:
                py_logger.info("Loading decoder weights from checkpoint")
                # Extract decoder weights and remove the "decoder_factory." prefix for loading
                decoder_state_dict = {}
                for k, v in checkpoint_state_dict.items():
                    if k.startswith("decoder_factory."):
                        # Remove "decoder_factory." prefix to match the model's expected keys
                        new_key = k.replace("decoder_factory.", "", 1)  # Remove first occurrence of "decoder_factory."
                        decoder_state_dict[new_key] = v
                
                if decoder_state_dict:
                    tokenizer.decoder_factory.load_state_dict(decoder_state_dict, strict=load_decoder_strict)
                    py_logger.info(f"Successfully loaded {len(decoder_state_dict)} decoder parameters")
                else:
                    py_logger.warning("No decoder weights found in checkpoint")
            else:
                py_logger.info("Keeping randomly initialized decoder")
        else:
            checkpoint = torch.load(cfg.get("ckpt_path"))
            checkpoint_state_dict = checkpoint["state_dict"]
            tokenizer.encoder.load_state_dict(checkpoint_state_dict, strict=False)
        trainer.fit(tokenizer, datamodule=datamodule)
    else:
        if cfg.get("ckpt_path") is not None:
            py_logger.info(f"continuing training from checkpoint {cfg.get('ckpt_path')}")
        trainer.fit(tokenizer, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if wandb_logger and isinstance(trainer.profiler, lightning.pytorch.profilers.PyTorchProfiler):
        profile_art = wandb.Artifact("trace", type="profile")
        for trace in pathlib.Path(trainer.profiler.dirpath).glob("*.pt.trace.json"):
            profile_art.add_file(trace)
        profile_art.save()

    if rank_zero_only.rank == 0:
        py_logger.info(f"{torch.cuda.max_memory_allocated()=:0.2e}")

    if wandb_logger:
        wandb.finish()
