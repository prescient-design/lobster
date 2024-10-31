from typing import Literal, Optional

import lightning.pytorch as pl
from peft import LoraConfig, get_peft_model

from lobster.model import LobsterPMLM

from ._utils import model_typer


class LobsterPEFT(pl.LightningModule):
    def __init__(
        self,
        model_type: Literal["LobsterPMLM", "LobsterPCLM"] = "LobsterPMLM",
        ckpt_path: Optional[str] = None,
        lora_rank: int = 64,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        freeze_encoder: bool = True,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        seed: int = 0,
    ):
        """Add trainable weights to the query and value layers of each attention block in the encoder."""
        super().__init__()
        self._model_type = model_type
        self._ckpt_path = ckpt_path
        self._freeze_encoder = freeze_encoder
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._seed = seed

        model_cls = model_typer[model_type]

        if ckpt_path is not None:
            model = model_cls.load_from_checkpoint(
                ckpt_path,
            )
        else:
            model = LobsterPMLM(model_name="MLM_mini")  # random init

        lora_config = LoraConfig(
            r=lora_rank,  # Lora rank
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            inference_mode=False,
        )
        self._lora_config = lora_config
        self._peft_model = get_peft_model(model.model, lora_config)
        self.model = self._peft_model  # for compatibility with LobsterPMLM and LobsterPCLM
        self.config = self._lora_config
        self.save_hyperparameters(logger=False)
