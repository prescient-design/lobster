"""Tool factory for automatically registering Lobster MCP tools with FastMCP.

This module provides a factory pattern to automatically discover and register
all available tools with the FastMCP server, eliminating code duplication.
"""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from fastmcp import FastMCP

from .tools import (
    compute_naturalness,
    get_sequence_concepts,
    get_sequence_representations,
    get_supported_concepts,
    intervene_on_sequence,
    list_available_models,
)


class ToolFactory:
    """Factory for automatically registering Lobster tools with FastMCP."""

    def __init__(self, app: FastMCP):
        """Initialize the tool factory.

        Parameters
        ----------
        app : FastMCP
            The FastMCP application instance
        """
        self.app = app
        self._registered_tools = {}

    def register_all_tools(self) -> None:
        """Register all available Lobster tools with FastMCP."""
        # Define tools to register with their actual function names
        tools_to_register = [
            (list_available_models, "list_available_models"),
            (get_sequence_representations, "get_sequence_representations"),
            (get_sequence_concepts, "get_sequence_concepts"),
            (intervene_on_sequence, "intervene_on_sequence"),
            (get_supported_concepts, "get_supported_concepts"),
            (compute_naturalness, "compute_naturalness"),
        ]

        # Register each tool directly
        for func, name in tools_to_register:
            self.app.tool(func)
            self._registered_tools[name] = func

    def get_registered_tools(self) -> dict[str, Callable]:
        """Get all registered tools.

        Returns
        -------
        Dict[str, Callable]
            Dictionary mapping tool names to their wrapper functions
        """
        return self._registered_tools.copy()

    def get_tool_info(self) -> dict[str, dict[str, Any]]:
        """Get detailed information about all registered tools.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping tool names to their signature information
        """
        tool_info = {}
        for tool_name, func in self._registered_tools.items():
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            params = {}
            for name, param in sig.parameters.items():
                param_type = type_hints.get(name, Any)
                if param.default is not inspect.Parameter.empty:
                    params[name] = {"type": param_type, "default": param.default, "has_default": True}
                else:
                    params[name] = {"type": param_type, "has_default": False}

            tool_info[tool_name] = {"parameters": params, "doc": func.__doc__ or "", "name": func.__name__}

        return tool_info


def create_and_register_tools(app: FastMCP) -> ToolFactory:
    """Create a tool factory and register all tools.

    Parameters
    ----------
    app : FastMCP
        The FastMCP application instance

    Returns
    -------
    ToolFactory
        The configured tool factory
    """
    factory = ToolFactory(app)
    factory.register_all_tools()
    return factory
