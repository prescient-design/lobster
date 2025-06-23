
# Lobster MCP Integration

This document describes how to use Lobster's Model Context Protocol (MCP) integration to expose pretrained models through MCP servers.

## Overview

Lobster provides MCP servers that expose pretrained models for:

- **Sequence Representations**: Get embeddings from protein sequences using various Lobster models
- **Concept Analysis**: Extract biological concepts from sequences using concept bottleneck models
- **Concept Interventions**: Modify sequences based on specific biological concepts
- **Naturalness Scoring**: Compute likelihood/naturalness scores for sequences

### Available Models

**Masked Language Models (MLM):**
- `lobster_24M`: 24M parameter model trained on UniRef50
- `lobster_150M`: 150M parameter model trained on UniRef50

**Concept Bottleneck Models (CBM):**
- `cb_lobster_24M`: 24M parameter model with 718 biological concepts
- `cb_lobster_150M`: 150M parameter model with 718 biological concepts  
- `cb_lobster_650M`: 650M parameter model with 718 biological concepts
- `cb_lobster_3B`: 3B parameter model with 718 biological concepts

## Installation

### Quick Start with uv (Recommended)

1. Install Lobster with MCP support:
```bash
uv sync --extra mcp
```

2. Run the setup script to configure Claude Desktop:
```bash
uv run lobster_mcp_setup
```

### Alternative Installation

If you prefer pip:
```bash
pip install -e .[mcp]
```

### GPU Support

For GPU acceleration, install with both MCP and flash attention:
```bash
uv sync --extra mcp --extra flash
```

## Usage with Claude Desktop

The setup script will automatically configure Claude Desktop:

```bash
uv run lobster_mcp_setup
```

This will:
1. Verify all dependencies are installed (including FastMCP)
2. Create the proper Claude Desktop configuration
3. Set up the MCP server to run via `uv`

After running setup, restart Claude Desktop and you can use commands like:
- "What Lobster models are available?"
- "Get embeddings for protein sequence MKTVRQ using lobster_24M"
- "What concepts are supported by the cb_lobster_24M model?"
- "Can you intervene on the sequence MKTVRQERLKSIVRIL to reduce hydrophobicity?"

✅ **Confirmed working in Claude Desktop** with FastMCP implementation!

## Usage with Cursor

### Automated Setup (Recommended)

Use the setup script to automatically configure Cursor:

```bash
uv run lobster_mcp_setup
```

When prompted, choose option "2" for Cursor or "3" for both Claude Desktop and Cursor.

### Manual Setup

If you prefer to configure manually, create or edit the file `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "lobster-inference": {
      "command": "uv",
      "args": [
        "run",
        "--project", "/path/to/lobster",
        "--extra", "mcp",
        "lobster_mcp_server"
      ]
    }
  }
}
```

Replace `/path/to/lobster` with the actual path to your Lobster repository.

### Using the Server in Cursor

After setup and restarting Cursor:

1. **Command Palette**: Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)
2. **MCP Commands**: Type "MCP" to see available MCP-related commands
3. **Chat Integration**: Use `@lobster-inference` in chat to reference the server
4. **Available Tools**: The server provides the same tools as Claude Desktop:
   - `list_available_models` - List all available Lobster models
   - `get_sequence_representations` - Get embeddings for protein sequences
   - `get_sequence_concepts` - Extract biological concepts from sequences
   - `intervene_on_sequence` - Modify sequences based on concepts
   - `get_supported_concepts` - List supported concepts for CBM models
   - `compute_naturalness` - Calculate sequence naturalness scores

### Example Usage in Cursor

Once configured, you can use natural language commands in Cursor:

```
@lobster-inference What models are available?

@lobster-inference Get embeddings for the sequence MKTVRQERLKSIVRIL using lobster_24M

@lobster-inference What concepts are supported by cb_lobster_24M?

@lobster-inference Intervene on MKTVRQERLKSIVRIL to reduce hydrophobicity using cb_lobster_24M
```

✅ **Confirmed working with FastMCP implementation!**

## Usage with MCP CLI

```bash
# Test the MCP server directly
uv run lobster_mcp_server

# Use MCP CLI dev mode for testing (if compatible)
uv run mcp dev src/lobster/mcp/inference_server.py:app --with-editable .
```

## Available Tools

### get_sequence_representations
Get embedding representations for protein sequences.

**Parameters:**
- `sequences`: List of protein sequences
- `model_name`: Name of the model to use
- `model_type`: "masked_lm" or "concept_bottleneck"  
- `representation_type`: "cls", "pooled", or "full"

**Example:**
```json
{
  "sequences": ["MKTVRQERLKSIVRIL"],
  "model_name": "lobster_24M",
  "model_type": "masked_lm",
  "representation_type": "pooled"
}
```

### get_sequence_concepts
Get concept predictions for protein sequences.

**Parameters:**
- `sequences`: List of protein sequences
- `model_name`: Name of the concept bottleneck model

### intervene_on_sequence
Perform concept interventions on protein sequences.

**Parameters:**
- `sequence`: Protein sequence to modify
- `concept`: Concept to intervene on (e.g., "gravy", "hydrophobicity")
- `model_name`: Name of the concept bottleneck model
- `edits`: Number of edits to make (default: 5)
- `intervention_type`: "positive" or "negative" (default: "negative")

### get_supported_concepts
Get list of supported concepts for a concept bottleneck model.

**Parameters:**
- `model_name`: Name of the concept bottleneck model

### compute_naturalness
Compute naturalness/likelihood scores for protein sequences.

**Parameters:**
- `sequences`: List of protein sequences
- `model_name`: Name of the model
- `model_type`: "masked_lm" or "concept_bottleneck"

### list_available_models
List all available pretrained Lobster models and current device.

## Example Usage

Once configured with Claude Desktop or MCP CLI, you can use the tools like:

```
Could you get embeddings for the protein sequence "MKTVRQERLKSIVRIL" using the lobster_24M model?
```

```
What concepts are supported by the cb_lobster_24M model?
```

```
Can you intervene on the sequence "MKTVRQERLKSIVRIL" to reduce hydrophobicity using the cb_lobster_24M model?
```

## GPU Support

The server automatically detects and uses CUDA if available. Models are cached after first load for efficiency.

## Troubleshooting

1. **Import errors**: Make sure Lobster is installed with MCP support: `uv sync --extra mcp`
2. **CUDA errors**: Check that PyTorch is installed with CUDA support if using GPU
3. **Model loading errors**: Ensure you have internet connectivity for downloading models from HuggingFace
4. **Memory issues**: Use smaller models (24M, 150M) if running out of memory
5. **Claude Desktop not connecting**: Check that the setup script ran successfully and restart Claude Desktop

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/lobster/mcp/

# Run linting and formatting
uv run ruff check src/lobster/mcp/
uv run ruff format src/lobster/mcp/

# Type checking
uv run mypy src/lobster/mcp/
```

### Adding New Tools

To add new tools or models:

1. Add the tool definition to `handle_list_tools()` in `src/lobster/mcp/inference_server.py`
2. Add the implementation to `handle_call_tool()`
3. Update the `LobsterInferenceServer` class with new methods
4. Add tests in `tests/lobster/mcp/`
5. Test with: `uv run lobster_mcp_server`

### Package Structure

```
src/lobster/mcp/
├── __init__.py                         # MCP module init
├── inference_server.py                 # FastMCP-based MCP server
├── setup.py                           # Setup script for Claude Desktop
├── example_server.py                  # Usage examples
└── claude_desktop_config.json         # Config template

tests/lobster/mcp/
├── __init__.py
├── test_inference_server.py           # Unit tests
└── simple_test.py                     # Basic functionality tests
```

### MCP Server Implementation

The package provides a FastMCP-based server implementation:

- **lobster_mcp_server** - FastMCP-based server
  - Better performance and simpler debugging
  - Uses Pydantic models for input validation
  - Compatible with modern MCP tooling

## Technical Details

### MCP Server Architecture

The Lobster MCP server is built using FastMCP and follows these principles:

- **Lazy Loading**: Models are only loaded when first requested
- **Caching**: Models remain in memory once loaded for efficiency
- **Error Handling**: Comprehensive error handling with informative messages
- **Type Safety**: Full type hints and validation using Pydantic
- **Testing**: Comprehensive unit tests with mocking for CI/CD

### Model Loading Strategy

- Models are downloaded from HuggingFace Hub on first use
- GPU/CPU selection is automatic based on availability
- Memory management through model caching and cleanup
- Support for different model types (MLM, CBM) with unified interface

### Integration with Claude

The MCP server integrates seamlessly with Claude Desktop and other MCP clients:
- Automatic tool discovery and schema validation
- Structured input/output with JSON schemas
- Async/await support for non-blocking operations
- Proper error propagation to the client