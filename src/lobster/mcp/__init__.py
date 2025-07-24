"""Lobster MCP Server Package

This package provides a FastMCP server for Lobster model inference and interventions.
"""

from .models import AVAILABLE_MODELS
from .server import app as mcp_app

__all__ = ["AVAILABLE_MODELS", "mcp_app"]
