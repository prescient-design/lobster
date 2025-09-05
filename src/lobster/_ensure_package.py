"""Utility for ensuring optional dependencies are available."""

import importlib


def ensure_package(
    package_name: str,
    *,
    group: str | None = None,
) -> None:
    """Ensure that an optional package is available, with helpful error message.

    Parameters
    ----------
    package_name : str
        Name of the package to check for availability
    group : str | None, optional
        Optional dependency group name (e.g., 'mgm', 'flash', 'mcp'). If provided,
        the error message will suggest installing with `uv sync --extra {group}`.
        Defaults to None.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError as e:
        error_msg = f"{package_name} is not installed."

        if group:
            error_msg += f" Please install it with `uv sync --extra {group}`."
        else:
            error_msg += f" Please install {package_name}."

        raise ImportError(error_msg) from e
