from transformers import PretrainedConfig

from lobster.constants import UMEModelVersion


class UMEConfig(PretrainedConfig):
    model_type = "ume"

    def __init__(
        self,
        model_name: UMEModelVersion | str = UMEModelVersion.MEDIUM,
        **kwargs,
    ):
        model_name = UMEModelVersion(model_name) if isinstance(model_name, str) else model_name

        if "mini" in model_name.value:
            model_name = UMEModelVersion.MINI
        elif "small" in model_name.value:
            model_name = UMEModelVersion.SMALL
        elif "medium" in model_name.value:
            model_name = UMEModelVersion.MEDIUM
        elif "large" in model_name.value:
            model_name = UMEModelVersion.LARGE
        else:
            raise ValueError(f"Unknown model name: {model_name}. Must contain 'mini', 'small', 'medium', or 'large'.")

        self.model_name = model_name.value
        super().__init__(**kwargs)
