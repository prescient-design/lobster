#!/usr/bin/env python3
"""
Simple test for Lobster MCP server without full dependencies
"""

import os
import sys

# Add the Lobster root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_imports():
    """Test if we can import the core components"""
    print("🧪 Testing core imports...")

    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("❌ PyTorch not available")
        return False

    try:
        # Import from the actual source location
        import os
        import sys

        # Add the src directory to the path
        lobster_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "src")
        sys.path.insert(0, lobster_src)

        from lobster.model import LobsterCBMPMLM, LobsterPMLM  # noqa: F401

        print("✅ Lobster model classes available")
    except ImportError as e:
        print(f"❌ Lobster models not available: {e}")
        return False

    return True


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
    assert test_imports()
    test_device()


if __name__ == "__main__":
    print("🦞 Lobster MCP Server - Simple Test")
    print("=" * 40)

    if test_imports():
        test_device()
        print("\n✅ Basic functionality test passed!")
        print("📋 Ready to set up MCP server")
    else:
        print("\n❌ Test failed - check dependencies")
        print("💡 Install lobster with: uv sync --extra mcp")
