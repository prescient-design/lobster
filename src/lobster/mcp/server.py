#!/usr/bin/env python3
"""
FastMCP Server for Lobster Model Inference and Interventions

This server provides access to pretrained Lobster models for:
1. Getting sequence representations (embeddings)
2. Performing concept interventions on sequences
3. Computing sequence likelihoods/naturalness

Available models:
- LobsterPMLM: Masked Language Models (24M, 150M)
- LobsterCBMPMLM: Concept Bottleneck Models (24M, 150M, 650M, 3B)

This version uses FastMCP best practices with native type hints and a tool factory.
"""

import logging

from fastmcp import FastMCP

from lobster.mcp.tool_factory import create_and_register_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lobster-fastmcp-server")

# Initialize FastMCP server
app = FastMCP("Lobster Inference")

# Register all tools using the factory
tool_factory = create_and_register_tools(app)


def main():
    """Main entry point for FastMCP server."""
    app.run()


if __name__ == "__main__":
    main()
