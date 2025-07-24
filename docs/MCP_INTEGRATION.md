
# Lobster MCP Integration

This document describes how to use Lobster's Model Context Protocol (MCP) integration to expose pretrained models through MCP servers.

## Overview

Lobster provides MCP servers that expose pretrained models for:

- **Sequence Representations**: Get embeddings from protein sequences using various Lobster models
- **Concept Analysis**: Extract biological concepts from sequences using concept bottleneck models
- **Concept Interventions**: Modify sequences based on specific biological concepts
- **Naturalness Scoring**: Compute likelihood/naturalness scores for sequences

> 💡 **Quick Start Guide**: For a more concise setup and usage guide, see the [MCP README](../src/lobster/mcp/README.md) in the MCP directory.

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

### 🚀 One-Click Install (Recommended)
Download and install the DXT package for automatic setup:

[📦 Download Lobster DXT Package](../src/lobster/mcp/lobster-inference.dxt)

After downloading, double-click the `.dxt` file to install in Claude Desktop, then restart Claude.

### Manual Setup
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

#### Option 1: One-Click Install (Recommended)

[![Add Lobster to Cursor](https://img.shields.io/badge/Add%20to%20Cursor-MCP%20Server-blue?style=for-the-badge&logo=cursor)](cursor://anysphere.cursor-deeplink/mcp/install?name=lobster-inference&config=eyJjb21tYW5kIjogInV2IiwgImFyZ3MiOiBbInJ1biIsICItLXByb2plY3QiLCAiLiIsICItLWV4dHJhIiwgIm1jcCIsICJsb2JzdGVyX21jcF9zZXJ2ZXIiXSwgImVudiI6IHt9LCAiY3dkIjogIiR7d29ya3NwYWNlRm9sZGVyfSJ9Cg==)

Click the button above to automatically add the Lobster MCP server to Cursor.

**Requirements:**
- [Cursor](https://cursor.com/) installed
- [uv](https://docs.astral.sh/uv/) package manager available in PATH  
- Lobster repository cloned locally with all dependencies installed (`uv sync --all-extras`)

#### Option 2: Automated Setup Script

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
        "--all-extras",
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
   - `list_models` - List all available Lobster models
   - `get_representations` - Get embeddings for protein sequences
   - `get_concepts` - Extract biological concepts from sequences
   - `intervene_sequence` - Modify sequences based on concepts
   - `get_supported_concepts_list` - List supported concepts for CBM models
   - `compute_sequence_naturalness` - Calculate sequence naturalness scores

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
uv run mcp dev src/lobster/mcp/server.py:app --with-editable .
```

## Available Tools

### get_representations
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

### get_concepts
Get concept predictions for protein sequences.

**Parameters:**
- `sequences`: List of protein sequences
- `model_name`: Name of the concept bottleneck model

### intervene_sequence
Perform concept interventions on protein sequences.

**Parameters:**
- `sequence`: Protein sequence to modify
- `concept`: Concept to intervene on (e.g., "gravy", "hydrophobicity")
- `model_name`: Name of the concept bottleneck model
- `edits`: Number of edits to make (default: 5)
- `intervention_type`: "positive" or "negative" (default: "negative")

### get_supported_concepts_list
Get list of supported concepts for a concept bottleneck model.

**Parameters:**
- `model_name`: Name of the concept bottleneck model

### compute_sequence_naturalness
Compute naturalness/likelihood scores for protein sequences.

**Parameters:**
- `sequences`: List of protein sequences
- `model_name`: Name of the model
- `model_type`: "masked_lm" or "concept_bottleneck"

### list_models
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

1. Create tool functions in `src/lobster/mcp/tools/` directory
2. Add request/response schemas in `src/lobster/mcp/schemas/`
3. Register tools in `src/lobster/mcp/server.py` with `@app.tool()` decorator
4. Add tests in `tests/lobster/mcp/`
5. Test with: `uv run lobster_mcp_server`

### Package Structure

```
src/lobster/mcp/
├── __init__.py                         # MCP module init
├── server.py                          # Main FastMCP server
├── models/                            # Model management
│   ├── __init__.py
│   ├── config.py                      # Model configurations
│   └── manager.py                     # Model loading and caching
├── schemas/                           # Request/response validation
│   ├── __init__.py
│   └── requests.py                    # Pydantic schemas
├── tools/                             # MCP tool implementations
│   ├── __init__.py
│   ├── representations.py            # Sequence embeddings
│   ├── concepts.py                    # Concept predictions
│   ├── interventions.py              # Sequence modifications
│   └── utils.py                       # Utility tools
├── setup.py                           # Setup script for clients
├── example_server.py                  # Functional testing script
├── inference_server.py                # Legacy file (backward compatibility)
├── README.md                          # Comprehensive usage guide
└── claude_desktop_config.json         # Example configuration

tests/lobster/mcp/
├── __init__.py
├── test_inference_server.py           # Legacy server tests
├── test_modular_components.py         # Modular component tests
└── test_simple_server.py              # Basic functionality tests
```

### MCP Server Implementation

The package provides a modular FastMCP-based server implementation:

- **server.py** - Main FastMCP server with clean modular architecture
- **models/** - Model management with caching and loading strategies
- **tools/** - Individual MCP tool implementations grouped by functionality
- **schemas/** - Pydantic models for type-safe request/response validation

Key benefits of the modular design:
- **Separation of concerns** - Each module has a single responsibility
- **Easy testing** - Individual components can be unit tested in isolation
- **Better maintainability** - Changes to one area don't affect others
- **Scalability** - Easy to add new tools or model types

## Technical Details

### MCP Server Architecture

The Lobster MCP server follows a clean modular architecture using FastMCP:

- **ModelManager**: Handles lazy loading, caching, and GPU/CPU management
- **Tool Functions**: Individual functions for each MCP tool (representations, concepts, etc.)
- **Schema Validation**: Pydantic models ensure type safety and input validation
- **FastMCP Integration**: Modern MCP framework with automatic tool registration
- **Comprehensive Testing**: Unit tests for each modular component

### Model Loading Strategy

- Models are downloaded from HuggingFace Hub on first use
- GPU/CPU selection is automatic based on availability  
- Memory management through model caching and cleanup
- Support for different model types (MLM, CBM) with unified interface

### Modular Design Pattern

The refactored architecture follows these design principles:

**models/** - Model Management Layer
- `config.py`: Centralized model configurations and constants
- `manager.py`: ModelManager class for loading, caching, and device management

**schemas/** - Data Validation Layer  
- `requests.py`: Pydantic models for all request/response schemas
- Type-safe validation with clear error messages

**tools/** - Business Logic Layer
- `representations.py`: Sequence embedding tools
- `concepts.py`: Concept prediction and analysis tools
- `interventions.py`: Sequence modification tools
- `utils.py`: Utility tools (model listing, naturalness scoring)

**server.py** - Presentation Layer
- FastMCP server setup and tool registration
- Minimal orchestration code that delegates to tool functions
- Clean separation from business logic

### Development Workflow

Adding a new MCP tool involves these steps:

1. **Define schemas** in `schemas/requests.py`:
   ```python
   class NewToolRequest(BaseModel):
       sequence: str = Field(..., description="Protein sequence")
       parameter: int = Field(default=5, description="Tool parameter")
   ```

2. **Implement tool function** in appropriate `tools/` file:
   ```python
   def new_tool(request: NewToolRequest, model_manager: ModelManager) -> dict[str, Any]:
       # Implementation here
       return {"result": "success"}
   ```

3. **Register tool** in `server.py`:
   ```python
   @app.tool()
   def new_tool_endpoint(request: NewToolRequest):
       return new_tool(request, model_manager)
   ```

4. **Add tests** in `tests/lobster/mcp/test_modular_components.py`

### Integration with Claude

The MCP server integrates seamlessly with Claude Desktop and other MCP clients:
- Automatic tool discovery and schema validation
- Structured input/output with JSON schemas
- Clean error handling and propagation
- Modern FastMCP framework compatibility