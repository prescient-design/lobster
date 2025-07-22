#!/usr/bin/env python3
"""
Simple test for Lobster MCP server without full dependencies
"""

import os
import sys

# Add the Lobster root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def test_imports():
    """Test if we can import the core components"""
    print("🧪 Testing core imports...")

    # Test PyTorch import
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError as e:
        print("❌ PyTorch not available")
        raise AssertionError("PyTorch not available") from e

    # Test Lobster MCP modular imports
    try:
        from lobster.mcp.models import ModelManager, AVAILABLE_MODELS  # noqa: F401
        from lobster.mcp.server import app  # noqa: F401

        print("✅ Lobster MCP modular structure available")
    except ImportError as e:
        print(f"❌ Lobster MCP modules not available: {e}")
        raise AssertionError(f"Lobster MCP modules not available: {e}") from e


def test_device():
    """Test device detection"""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    if device == "cuda":
        print(f"📋 GPU: {torch.cuda.get_device_name()}")
        print(f"📋 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def test_main():
    """Main test function for pytest"""
    test_imports()
    test_device()


def _check_imports():
    """Helper function that returns boolean for script usage"""
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("❌ PyTorch not available")
        return False

    try:
        from lobster.mcp.models import ModelManager, AVAILABLE_MODELS  # noqa: F401
        from lobster.mcp.server import app  # noqa: F401

        print("✅ Lobster MCP modular structure available")
        return True
    except ImportError as e:
        print(f"❌ Lobster MCP modules not available: {e}")
        return False


if __name__ == "__main__":
    print("🦞 Lobster MCP Server - Simple Test")
    print("=" * 40)

    if _check_imports():
        test_device()
        print("\n✅ Basic functionality test passed!")
        print("📋 Ready to set up MCP server")
    else:
        print("\n❌ Test failed - check dependencies")
        print("💡 Install lobster with: uv sync --extra mcp")
