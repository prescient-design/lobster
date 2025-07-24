"""Unit tests for the Lobster MCP tool factory."""

import inspect
from unittest.mock import Mock
import pytest

try:
    from lobster.mcp.tool_factory import ToolFactory, create_and_register_tools
    from lobster.mcp.tools import (
        list_available_models,
        get_sequence_representations,
        get_sequence_concepts,
        intervene_on_sequence,
        get_supported_concepts,
        compute_naturalness,
    )
    from fastmcp import FastMCP

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestToolFactory:
    """Test the ToolFactory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock(spec=FastMCP)
        self.factory = ToolFactory(self.mock_app)

    def test_tool_factory_initialization(self):
        """Test ToolFactory initialization."""
        assert self.factory.app == self.mock_app
        assert self.factory._registered_tools == {}

    def test_register_all_tools(self):
        """Test that all tools are registered correctly."""
        self.factory.register_all_tools()

        # Verify that app.tool was called for each tool
        assert self.mock_app.tool.call_count == 6

        # Verify all expected tools are registered
        expected_tools = {
            "list_models": list_available_models,
            "get_representations": get_sequence_representations,
            "get_concepts": get_sequence_concepts,
            "intervene_sequence": intervene_on_sequence,
            "get_supported_concepts_list": get_supported_concepts,
            "compute_sequence_naturalness": compute_naturalness,
        }

        assert len(self.factory._registered_tools) == 6
        for name, func in expected_tools.items():
            assert name in self.factory._registered_tools
            assert self.factory._registered_tools[name] == func

    def test_register_all_tools_calls_app_tool(self):
        """Test that app.tool is called with the correct functions."""
        self.factory.register_all_tools()

        # Get all the calls to app.tool
        tool_calls = [call[0][0] for call in self.mock_app.tool.call_args_list]

        # Verify each expected function was passed to app.tool
        expected_functions = [
            list_available_models,
            get_sequence_representations,
            get_sequence_concepts,
            intervene_on_sequence,
            get_supported_concepts,
            compute_naturalness,
        ]

        for func in expected_functions:
            assert func in tool_calls

    def test_get_registered_tools(self):
        """Test get_registered_tools returns a copy of registered tools."""
        self.factory.register_all_tools()

        tools = self.factory.get_registered_tools()

        # Should be a copy, not the same object
        assert tools is not self.factory._registered_tools
        assert tools == self.factory._registered_tools

        # Should have all expected tools
        assert len(tools) == 6
        assert "list_models" in tools
        assert "get_representations" in tools
        assert "get_concepts" in tools
        assert "intervene_sequence" in tools
        assert "get_supported_concepts_list" in tools
        assert "compute_sequence_naturalness" in tools

    def test_get_tool_info(self):
        """Test get_tool_info returns correct information."""
        self.factory.register_all_tools()

        tool_info = self.factory.get_tool_info()

        # Should have info for all tools
        assert len(tool_info) == 6

        # Test structure of tool info for one tool
        list_models_info = tool_info["list_models"]
        assert "parameters" in list_models_info
        assert "doc" in list_models_info
        assert "name" in list_models_info
        assert list_models_info["name"] == "list_available_models"
        assert "List all available pretrained Lobster models" in list_models_info["doc"]

    def test_get_tool_info_parameters(self):
        """Test that tool info includes correct parameter information."""
        self.factory.register_all_tools()

        tool_info = self.factory.get_tool_info()

        # Test parameters for list_available_models (no parameters)
        list_models_params = tool_info["list_models"]["parameters"]
        assert len(list_models_params) == 0

        # Test parameters for get_sequence_representations (has parameters)
        representations_params = tool_info["get_representations"]["parameters"]
        assert "model_name" in representations_params
        assert "sequences" in representations_params
        assert "model_type" in representations_params
        assert "representation_type" in representations_params

        # Check that representation_type has a default
        rep_type_param = representations_params["representation_type"]
        assert rep_type_param["has_default"] is True
        assert rep_type_param["default"] == "pooled"

    def test_get_tool_info_function_signatures(self):
        """Test that tool info correctly captures function signatures."""
        self.factory.register_all_tools()

        tool_info = self.factory.get_tool_info()

        # Test that we can reconstruct function signatures
        for tool_name, info in tool_info.items():
            func = self.factory._registered_tools[tool_name]
            sig = inspect.signature(func)

            # Verify parameter count matches
            assert len(info["parameters"]) == len(sig.parameters)

            # Verify parameter names match
            for param_name in info["parameters"]:
                assert param_name in sig.parameters

    def test_empty_tool_info_before_registration(self):
        """Test that tool info is empty before registration."""
        tool_info = self.factory.get_tool_info()
        assert tool_info == {}

    def test_empty_registered_tools_before_registration(self):
        """Test that registered tools is empty before registration."""
        tools = self.factory.get_registered_tools()
        assert tools == {}


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestCreateAndRegisterTools:
    """Test the create_and_register_tools function."""

    def test_create_and_register_tools(self):
        """Test create_and_register_tools function."""
        mock_app = Mock(spec=FastMCP)

        factory = create_and_register_tools(mock_app)

        # Verify factory was created correctly
        assert isinstance(factory, ToolFactory)
        assert factory.app == mock_app

        # Verify tools were registered
        assert len(factory._registered_tools) == 6
        assert mock_app.tool.call_count == 6

    def test_create_and_register_tools_returns_factory(self):
        """Test that create_and_register_tools returns a usable factory."""
        mock_app = Mock(spec=FastMCP)

        factory = create_and_register_tools(mock_app)

        # Should be able to get tool info
        tool_info = factory.get_tool_info()
        assert len(tool_info) == 6

        # Should be able to get registered tools
        tools = factory.get_registered_tools()
        assert len(tools) == 6


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestToolFactoryIntegration:
    """Integration tests for ToolFactory."""

    def test_tool_factory_with_real_fastmcp_app(self):
        """Test ToolFactory with a real FastMCP app instance."""
        # Create a real FastMCP app
        app = FastMCP()

        # Create factory and register tools
        factory = ToolFactory(app)
        factory.register_all_tools()

        # Verify tools were registered
        assert len(factory._registered_tools) == 6

        # Verify we can get tool info
        tool_info = factory.get_tool_info()
        assert len(tool_info) == 6

    def test_tool_factory_tool_functionality(self):
        """Test that registered tools maintain their functionality."""
        app = FastMCP()
        factory = ToolFactory(app)
        factory.register_all_tools()

        # Get the registered tools
        tools = factory.get_registered_tools()

        # Test that list_available_models still works
        list_models_func = tools["list_models"]
        result = list_models_func()

        # Should return a response with available models
        assert isinstance(result, dict)
        assert "available_models" in result
        assert "device" in result
        assert "device_type" in result

    def test_tool_factory_error_handling(self):
        """Test error handling in ToolFactory."""
        # Test that ToolFactory can be created with any object
        # (it doesn't validate the app during initialization)
        invalid_app = Mock()
        del invalid_app.tool
        factory = ToolFactory(invalid_app)

        # But calling register_all_tools should fail
        with pytest.raises(AttributeError):
            factory.register_all_tools()

    def test_tool_factory_registration_order(self):
        """Test that tools are registered in the expected order."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)
        factory.register_all_tools()

        # Get the order of tool registrations
        tool_calls = [call[0][0] for call in mock_app.tool.call_args_list]

        expected_order = [
            list_available_models,
            get_sequence_representations,
            get_sequence_concepts,
            intervene_on_sequence,
            get_supported_concepts,
            compute_naturalness,
        ]

        assert tool_calls == expected_order


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestToolFactoryEdgeCases:
    """Test edge cases and error conditions."""

    def test_tool_factory_with_empty_tools_list(self):
        """Test ToolFactory behavior with no tools to register."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)

        # Manually set empty tools list
        factory._registered_tools = {}

        # Should handle empty tools gracefully
        tools = factory.get_registered_tools()
        assert tools == {}

        tool_info = factory.get_tool_info()
        assert tool_info == {}

    def test_tool_factory_with_missing_tool_attributes(self):
        """Test ToolFactory with tools that have missing attributes."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)

        # Create a real function without __doc__ to avoid Mock issues
        def test_func():
            pass

        del test_func.__doc__
        test_func.__name__ = "test_func"

        # Add to registered tools
        factory._registered_tools["test_tool"] = test_func

        # Should handle missing __doc__ gracefully
        tool_info = factory.get_tool_info()
        assert "test_tool" in tool_info
        assert tool_info["test_tool"]["doc"] == ""

    def test_tool_factory_with_complex_function_signatures(self):
        """Test ToolFactory with functions that have complex signatures."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)

        # Test with a function that has various parameter types
        def complex_func(required_param: str, optional_param: int = 42, *args, kw_only_param: bool = True, **kwargs):
            pass

        factory._registered_tools["complex_tool"] = complex_func

        # Should handle complex signatures
        tool_info = factory.get_tool_info()
        assert "complex_tool" in tool_info

        params = tool_info["complex_tool"]["parameters"]
        assert "required_param" in params
        assert "optional_param" in params
        assert "kw_only_param" in params

        # Check default values
        assert params["optional_param"]["has_default"] is True
        assert params["optional_param"]["default"] == 42
        assert params["kw_only_param"]["has_default"] is True
        assert params["kw_only_param"]["default"] is True

    def test_tool_factory_memory_efficiency(self):
        """Test that ToolFactory doesn't create unnecessary copies."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)
        factory.register_all_tools()

        # Get tools multiple times
        tools1 = factory.get_registered_tools()
        tools2 = factory.get_registered_tools()

        # Should be different objects (copies)
        assert tools1 is not tools2
        assert tools1 == tools2

        # But the original should remain unchanged
        assert factory._registered_tools is not tools1
        assert factory._registered_tools is not tools2


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestToolFactoryDocumentation:
    """Test documentation and docstring handling."""

    def test_tool_factory_docstrings(self):
        """Test that ToolFactory has proper docstrings."""
        assert ToolFactory.__doc__ is not None
        assert "Factory for automatically registering Lobster tools" in ToolFactory.__doc__

        assert ToolFactory.__init__.__doc__ is not None
        assert ToolFactory.register_all_tools.__doc__ is not None
        assert ToolFactory.get_registered_tools.__doc__ is not None
        assert ToolFactory.get_tool_info.__doc__ is not None

    def test_tool_docstrings_preserved(self):
        """Test that tool docstrings are preserved in tool info."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)
        factory.register_all_tools()

        tool_info = factory.get_tool_info()

        # Check that docstrings are preserved
        list_models_doc = tool_info["list_models"]["doc"]
        assert "List all available pretrained Lobster models" in list_models_doc

        representations_doc = tool_info["get_representations"]["doc"]
        assert "Get sequence representations from a model" in representations_doc

    def test_tool_factory_function_names_preserved(self):
        """Test that function names are preserved in tool info."""
        mock_app = Mock(spec=FastMCP)
        factory = ToolFactory(mock_app)
        factory.register_all_tools()

        tool_info = factory.get_tool_info()

        # Check that function names are preserved
        assert tool_info["list_models"]["name"] == "list_available_models"
        assert tool_info["get_representations"]["name"] == "get_sequence_representations"
        assert tool_info["get_concepts"]["name"] == "get_sequence_concepts"
        assert tool_info["intervene_sequence"]["name"] == "intervene_on_sequence"
        assert tool_info["get_supported_concepts_list"]["name"] == "get_supported_concepts"
        assert tool_info["compute_sequence_naturalness"]["name"] == "compute_naturalness"
