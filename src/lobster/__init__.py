import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("lbster")
except PackageNotFoundError:
    __version__ = None


from ._ensure_package import ensure_package
from . import callbacks, cmdline, data, evaluation, hydra_config, model

__all__ = ["callbacks", "cmdline", "data", "evaluation", "hydra_config", "model", "ensure_package"]
