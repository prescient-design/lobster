import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("prescient-plm")
except PackageNotFoundError:
    __version__ = None

# from . import cmdline, data, hydra_config, model, utils
