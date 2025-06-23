from mcp.server.fastmcp import FastMCP

from lobster.model import UME


class UMEServer:
    def __init__(self, model: UME):
        """Initialize the UME MCP server with a model.

        Args:
            model: A UME model instance to use for embeddings
        """
        self.model = model
        self.server = FastMCP()
        self._register_tools()

    def _register_tools(self):
        """Register the embedding tool with the MCP server."""

        @self.server.tool(description="Embed sequences into a latent space")
        def embed_sequences(sequences, modality, aggregate=True):
            return self.model.embed_sequences(sequences, modality, aggregate=aggregate)

        # Store the tool function as an instance method
        self.embed_sequences = embed_sequences

    def get_server(self):
        """Get the FastMCP server instance.

        Returns:
            FastMCP: The configured MCP server instance
        """
        return self.server
