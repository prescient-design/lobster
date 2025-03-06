from importlib.util import find_spec
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

_FLASH_ATTN_AVAILABLE = False

if find_spec("flash_attn"):
    from lobster.model.modern_bert import FlexBERT

    _FLASH_ATTN_AVAILABLE = True


class TestFlexBERT:
    # def test_sequences_to_latents(self):
    #     if _FLASH_ATTN_AVAILABLE and torch.cuda.is_available():
    #         model = FlexBERT(model_name="UME_mini").cuda()

    #         inputs = ["ACDAC", "ACDAC"]
    #         outputs = model.sequences_to_latents(inputs)

    #         assert len(outputs) == 2
    #         assert isinstance(outputs[0], Tensor)
    #         assert outputs[-1].shape == Size([512, 252])  # L, d_model
    #         assert outputs[0].device == model.device

    def test_hydra_instantiate(self):
        if not _FLASH_ATTN_AVAILABLE:
            import pytest

            pytest.skip("flash_attn not available")

        # Define path to the config file
        config_path = config_path = (
            Path(__file__).parents[4] / "src" / "lobster" / "hydra_config" / "model" / "modern_bert.yaml"
        )

        # Load the config directly from YAML
        config = OmegaConf.load(config_path)

        # Need to resolve the trainer.max_steps variable for testing
        config.num_training_steps = 10_000  # Set to a fixed value for testing

        # Add missing required parameters for proper instantiation
        config.vocab_size = 30522  # Standard BERT vocab size
        config.pad_token_id = 0
        config.mask_token_id = 103
        config.cls_token_id = 101
        config.eos_token_id = 102

        # Instantiate the model using the loaded config
        model = instantiate(config)

        # Test basic model properties
        assert isinstance(model, FlexBERT)
        assert model._model_name == "UME_mini"
        assert model._lr == 1e-3
        assert model._beta1 == 0.9  # Default value
        assert model._beta2 == 0.98  # Default value
        assert model._eps == 1e-12  # Default value
        assert model._num_training_steps == 10_000
        assert model._num_warmup_steps == 10_000  # From the YAML
        assert model._mask_percentage == 0.25
        assert model.max_length == 512  # Note: Updated attribute name
        assert model.scheduler == "constant_with_warmup"  # Note: Updated attribute name

        # Test that model_kwargs were correctly passed to the config
        assert model.config.embedding_layer == "linear_pos"
        assert model.config.hidden_act == "gelu"

        # Test scheduler_kwargs is initialized correctly
        assert hasattr(model, "scheduler_kwargs")
