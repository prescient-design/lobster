from typing import Literal

from transformers import PretrainedConfig


class UMEConfig(PretrainedConfig):
    model_type = "ume"

    def __init__(
        self,
        model_name: Literal[
            "ume-mini-base-12M", "ume-small-base-90M", "ume-medium-base-480M", "ume-large-base-740M"
        ] = "ume-mini-base-12M",
        **kwargs,
    ):
        self.model_name = model_name
        super().__init__(**kwargs)
