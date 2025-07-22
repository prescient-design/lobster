# Lobster MCP Server - Desktop Extension (DXT)

A comprehensive Model Context Protocol (MCP) server providing access to Lobster protein language models as a Desktop Extension for Claude Desktop. This DXT package enables seamless protein sequence analysis, embeddings generation, concept prediction, and biological interventions.

## 🚀 Quick Start (DXT Installation)

### Automatic Installation
1. **Download**: Get the `lobster-inference.dxt` file from this directory
2. **Install**: Double-click the `.dxt` file in Claude Desktop
3. **Restart**: Restart Claude Desktop to activate the extension
4. **Test**: Ask Claude "What Lobster models are available?"

### Verification
Once installed, you can verify the installation by asking Claude:
- "List available Lobster models"
- "Get embeddings for protein sequence MKLLVVV"
- "What biological concepts are supported?"

### Alternative Installation (Manual)
```bash
# Install with MCP dependencies
uv sync --extra mcp

# Set up MCP client configuration  
python -m lobster.mcp.setup
```

## Architecture

The MCP integration follows a modular design pattern:

```
src/lobster/mcp/
├── models/           # Model management
│   ├── config.py     # Available model configurations
│   └── manager.py    # Model loading and caching
├── schemas/          # Request/response validation
│   └── requests.py   # Pydantic schemas for all endpoints
├── tools/            # MCP tool implementations
│   ├── representations.py  # Sequence embeddings
│   ├── concepts.py          # Concept predictions
│   ├── interventions.py     # Sequence modifications
│   └── utils.py            # Utility tools
├── server.py         # Main FastMCP server (recommended)
├── setup.py          # Client configuration helper
└── example_server.py # Local testing script
```

## Available Models

### Masked Language Models (MLM)
- `lobster_24M` - 24M parameter BERT-style encoder
- `lobster_150M` - 150M parameter BERT-style encoder

### Concept Bottleneck Models (CBM)
- `cb_lobster_24M` - 24M parameter model with 718 biological concepts
- `cb_lobster_150M` - 150M parameter concept bottleneck model
- `cb_lobster_650M` - 650M parameter concept bottleneck model
- `cb_lobster_3B` - 3B parameter concept bottleneck model

## MCP Tools

### 1. Sequence Representations
Get vector embeddings for protein sequences.

**Example:** "Get embeddings for protein sequence MKLLKN using lobster_24M"

### 2. Concept Analysis
Analyze biological concepts present in sequences (CBM models only).

**Example:** "What concepts are predicted for sequence MKLLKN using cb_lobster_24M?"

### 3. Concept Interventions
Modify sequences to increase/decrease specific biological concepts.

**Example:** "Modify sequence MKLLKN to decrease hydrophobicity using cb_lobster_24M"

### 4. Available Models
List all available models and current device information.

**Example:** "What Lobster models are available?"

### 5. Sequence Naturalness
Compute likelihood/naturalness scores for protein sequences.

**Example:** "How natural is the sequence MKLLKN according to lobster_24M?"

## Usage Examples

### Claude Desktop
After running setup, you can interact with Lobster models:

```
You: "Get embeddings for protein sequence MKTVRQ using lobster_24M"
Claude: [Uses MCP tool to get 408-dimensional embeddings]

You: "What biological concepts are present in sequence ACDEFG?"
Claude: [Analyzes sequence using concept bottleneck model]

You: "Modify MKTVRQ to increase stability"
Claude: [Performs concept intervention to modify sequence]
```

### Cursor IDE
Use `@lobster-inference` to access MCP tools:

```
@lobster-inference get_representations with sequence MKTVRQ
```

### Programmatic Usage
```python
from lobster.mcp.models import ModelManager
from lobster.mcp.tools import get_sequence_representations
from lobster.mcp.schemas import SequenceRepresentationRequest

# Initialize model manager
model_manager = ModelManager()

# Create request
request = SequenceRepresentationRequest(
    sequences=["MKTVRQ"],
    model_name="lobster_24M", 
    model_type="masked_lm"
)

# Get embeddings
result = get_sequence_representations(request, model_manager)
```

## Setup for Different Clients

### Claude Desktop

#### 🚀 One-Click Install (Recommended)
Download and install the DXT package for automatic setup:

[📦 Download Lobster DXT Package](./lobster-inference.dxt)

After downloading, double-click the `.dxt` file to install in Claude Desktop.

#### Manual Setup
```bash
python -m lobster.mcp.setup
# Choose option 1 (Claude Desktop)
# Restart Claude Desktop
```

### Cursor

#### 🚀 One-Click Install (Recommended)
Click the button below to automatically install the Lobster MCP server in Cursor:

[![Add Lobster to Cursor](https://img.shields.io/badge/Add%20to%20Cursor-MCP%20Server-blue?style=for-the-badge&logo=cursor)](cursor://anysphere.cursor-deeplink/mcp/install?name=lobster-inference&config=eyJjb21tYW5kIjogInV2IiwgImFyZ3MiOiBbInJ1biIsICItLWV4dHJhIiwgIm1jcCIsICJsb2JzdGVyX21jcF9zZXJ2ZXIiXX0=)

After clicking, restart Cursor and use `@lobster-inference` to access MCP tools.

#### Manual Setup
```bash
python -m lobster.mcp.setup  
# Choose option 2 (Cursor)
# Restart Cursor
```

### Custom MCP Client
Use the configuration from `claude_desktop_config.json` as a template:

```json
{
  "mcpServers": {
    "lobster-inference": {
      "command": "uv",
      "args": ["run", "--project", ".", "--extra", "mcp", "lobster_mcp_server"]
    }
  }
}
```

## Testing

```bash
# Run local tests
python src/lobster/mcp/example_server.py

# Run unit tests  
python -m pytest tests/lobster/mcp/ -v

# Test MCP server directly
uv run --extra mcp lobster_mcp_server
```

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'mcp'"**
```bash
uv sync --extra mcp
```

**"Device: cpu" (want GPU)**
- Ensure PyTorch with CUDA support is installed
- Models will automatically use GPU if available

**"Server not responding in Claude Desktop"**
- Restart Claude Desktop after configuration
- Check that `uv` is in your PATH
- Verify configuration with: `python -m lobster.mcp.setup`

**"Model loading errors"**
- Ensure internet connectivity for model downloads
- Models are cached locally after first download

### Performance Notes

- First model load requires downloading (~100MB per model)
- Subsequent loads use cached models
- GPU acceleration automatically used when available
- Models remain loaded in memory for fast inference

## Development

To extend the MCP server:

1. **Add new tools:** Create functions in `tools/` directory
2. **Add schemas:** Define request/response models in `schemas/`
3. **Register tools:** Add to `server.py` with `@app.tool()` decorator
4. **Test:** Add tests in `tests/lobster/mcp/`

The modular architecture makes it easy to add new functionality while maintaining clean separation of concerns.