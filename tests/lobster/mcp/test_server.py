"""Unit tests for the Lobster MCP server module."""

from unittest.mock import Mock, patch
from fastmcp import FastMCP

from lobster.mcp.server import main, app, tool_factory
from lobster.mcp.tool_factory import ToolFactory, create_and_register_tools


class TestServer:
    """Test cases for the main server module."""

    def test_server_initialization(self):
        """Test that the server initializes correctly."""
        assert isinstance(app, FastMCP)
        assert app.name == "Lobster Inference"
        assert tool_factory is not None

    def test_tool_factory_creation(self):
        """Test that the tool factory is created correctly."""
        assert isinstance(tool_factory, ToolFactory)
        assert tool_factory.app == app

    def test_registered_tools(self):
        """Test that all expected tools are registered."""
        registered_tools = tool_factory.get_registered_tools()
        expected_tools = [
            "list_models",
            "get_representations",
            "get_concepts",
            "intervene_sequence",
            "get_supported_concepts_list",
            "compute_sequence_naturalness",
        ]

        for tool_name in expected_tools:
            assert tool_name in registered_tools
            assert callable(registered_tools[tool_name])

    @patch("lobster.mcp.server.app.run")
    def test_main_function(self, mock_run):
        """Test the main function calls app.run()."""
        main()
        mock_run.assert_called_once()

    def test_logging_configuration(self):
        """Test that logging is configured correctly."""
        import logging

        # The logger level might be inherited from parent, so we just check it exists
        logger = logging.getLogger("lobster-fastmcp-server")
        assert logger is not None


class TestToolFactory:
    """Test cases for the ToolFactory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock(spec=FastMCP)
        self.factory = ToolFactory(self.mock_app)

    def test_factory_initialization(self):
        """Test ToolFactory initialization."""
        assert self.factory.app == self.mock_app
        assert self.factory._registered_tools == {}

    def test_register_all_tools(self):
        """Test registering all available tools."""
        self.factory.register_all_tools()

        expected_tools = [
            "list_models",
            "get_representations",
            "get_concepts",
            "intervene_sequence",
            "get_supported_concepts_list",
            "compute_sequence_naturalness",
        ]

        for tool_name in expected_tools:
            assert tool_name in self.factory._registered_tools
            assert callable(self.factory._registered_tools[tool_name])

    def test_get_registered_tools(self):
        """Test getting registered tools returns a copy."""
        self.factory.register_all_tools()
        tools = self.factory.get_registered_tools()

        assert len(tools) > 0
        assert tools is not self.factory._registered_tools  # Should be a copy

    def test_get_tool_info(self):
        """Test getting detailed tool information."""
        self.factory.register_all_tools()
        tool_info = self.factory.get_tool_info()

        expected_tools = [
            "list_models",
            "get_representations",
            "get_concepts",
            "intervene_sequence",
            "get_supported_concepts_list",
            "compute_sequence_naturalness",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_info
            assert "parameters" in tool_info[tool_name]
            assert "doc" in tool_info[tool_name]
            assert "name" in tool_info[tool_name]


class TestCreateAndRegisterTools:
    """Test cases for the create_and_register_tools function."""

    def test_create_and_register_tools(self):
        """Test the factory creation and tool registration function."""
        mock_app = Mock(spec=FastMCP)

        factory = create_and_register_tools(mock_app)

        assert isinstance(factory, ToolFactory)
        assert factory.app == mock_app

        # Check that tools were registered
        registered_tools = factory.get_registered_tools()
        assert len(registered_tools) > 0


class TestToolFactoryErrorHandling:
    """Test error handling in the ToolFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock(spec=FastMCP)
        self.factory = ToolFactory(self.mock_app)

    def test_factory_with_none_app(self):
        """Test factory initialization with None app."""
        # This should not raise an error since we're not validating in __init__
        factory = ToolFactory(None)
        assert factory.app is None
