"""
Lobster MCP (Model Context Protocol) Integration

This module provides MCP servers that expose Lobster's pretrained models
for sequence representation, concept analysis, and interventions.

Available when the 'mcp' extra is installed:
    uv sync --extra mcp
"""

try:
    from .inference_server import LobsterInferenceServer

    __all__ = ["LobsterInferenceServer"]
except ImportError:
    # MCP dependencies not installed
    __all__ = []
