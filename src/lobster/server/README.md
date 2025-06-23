# UMEServer

The `UMEServer` class provides a FastMCP server implementation for serving UME (Universal Model Embedding) functionality. It allows you to expose embedding capabilities through a standardized MCP (Model Control Protocol) interface.

## Overview

`UMEServer` wraps a UME model instance and exposes its embedding functionality through a FastMCP server. This enables standardized access to sequence embedding capabilities across different modalities.

## Usage

### Initialization

```python
from lobster.model import UME
from lobster.server import UMEServer

# Initialize your UME model
model = UME(...)

# Create the server instance
server = UMEServer(model)
```

### Available Tools

The server exposes the following tool:

#### embed_sequences

Embeds sequences into a latent space.

Parameters:
- `sequences`: The input sequences to embed
- `modality`: The modality of the sequences
- `aggregate` (optional): Whether to aggregate the embeddings (default: True)

Returns:
- The embedded sequences in the latent space

### Getting the Server Instance

To get the configured FastMCP server instance:

```python
mcp_server = server.get_server()
```

## Example

```python
from lobster.model import UME
from lobster.server import UMEServer

# Initialize model and server
model = UME(...)
server = UMEServer(model)

# Get the MCP server instance
mcp_server = server.get_server()

# Example amino acid sequences to embed
sequences = [
    "MLLAVLYCLAVFALSSRAG",  # Example protein sequence 1
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  # Example protein sequence 2
]
modality = "amino_acid"

# Embed sequences using the server's tool
embeddings = server.embed_sequences(
    sequences=sequences,
    modality=modality,
    aggregate=True  # optional, defaults to True
)

# The embeddings can now be used for downstream tasks
print(f"Generated embeddings shape: {embeddings.shape}")
```

## Dependencies

- FastMCP: For the MCP server implementation
- UME: The underlying embedding model
