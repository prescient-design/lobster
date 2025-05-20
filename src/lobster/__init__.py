import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("lbster")
except PackageNotFoundError:
    __version__ = None

# Define modules that are part of the public API
__all__ = ["callbacks", "cmdline", "data", "evaluation", "hydra_config", "model"]

# Import submodules to make them available through the package
from . import callbacks, cmdline, data, evaluation, hydra_config, model

# from . import cmdline, data, hydra_config, model, utils
