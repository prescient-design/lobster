"""
Lobster MCP (Model Context Protocol) Integration

This module provides MCP servers that expose Lobster's pretrained models
for sequence representation, concept analysis, and interventions.

Available when the 'mcp' extra is installed:
    uv sync --extra mcp
"""

try:
    # Import new modular components
    from .models import ModelManager, AVAILABLE_MODELS
    from .server import app as mcp_app

    __all__ = ["ModelManager", "AVAILABLE_MODELS", "mcp_app"]
except ImportError:
    # MCP dependencies not installed
    __all__ = []
