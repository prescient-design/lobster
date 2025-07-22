#!/usr/bin/env python3
"""
Legacy Lobster MCP Inference Server

DEPRECATED: This file has been refactored into a modular structure.
Please use the new `server.py` file instead.

This file is kept for backward compatibility but will be removed in future versions.
"""

import warnings

# Import the new modular server
from .server import main as new_main

warnings.warn(
    "inference_server.py is deprecated. Please use 'server.py' instead. This file will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)


def main():
    """Legacy main entry point - redirects to new server."""
    return new_main()


if __name__ == "__main__":
    main()
