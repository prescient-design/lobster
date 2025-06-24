#!/usr/bin/env python3
"""
Setup script for Lobster MCP Server

This script helps set up the MCP server for use with Claude Desktop
or other MCP clients.
"""

import json
import shutil
import sys
from pathlib import Path


def get_lobster_path():
    """Get the absolute path to the Lobster repository"""
    # From src/lobster/mcp/setup.py -> lobster root
    return Path(__file__).resolve().parents[3]


def get_claude_desktop_config_path():
    """Get the path to Claude Desktop config file"""
    home = Path.home()

    # Try different possible locations
    possible_paths = [
        home / ".config" / "claude-desktop" / "claude_desktop_config.json",
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
    ]

    for path in possible_paths:
        if path.parent.exists():
            return path

    # Default to first option and create directory
    default_path = possible_paths[0]
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


def get_cursor_mcp_config_path():
    """Get the path to Cursor MCP config file"""
    home = Path.home()
    cursor_config_path = home / ".cursor" / "mcp.json"

    # Create .cursor directory if it doesn't exist
    cursor_config_path.parent.mkdir(parents=True, exist_ok=True)
    return cursor_config_path


def setup_claude_desktop():
    """Set up Claude Desktop configuration"""
    print("🔧 Setting up Claude Desktop configuration...")

    lobster_path = get_lobster_path()
    config_path = get_claude_desktop_config_path()

    # Get the full path to uv for GUI applications
    uv_path = shutil.which("uv")
    if not uv_path:
        raise RuntimeError("uv not found in PATH. Please install uv.")

    # Create the configuration
    config = {
        "mcpServers": {
            "lobster-inference": {
                "command": uv_path,
                "args": ["run", "--project", str(lobster_path), "--extra", "mcp", "lobster_mcp_server"],
            }
        }
    }

    # Load existing config if it exists
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing_config = json.load(f)

            # Merge configurations
            if "mcpServers" not in existing_config:
                existing_config["mcpServers"] = {}
            existing_config["mcpServers"]["lobster-inference"] = config["mcpServers"]["lobster-inference"]
            config = existing_config

        except json.JSONDecodeError:
            print("⚠️  Existing config file is invalid JSON, creating new one...")

    # Write the configuration
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration written to: {config_path}")
    print("📋 Please restart Claude Desktop to load the new server.")


def setup_cursor():
    """Set up Cursor MCP configuration"""
    print("🔧 Setting up Cursor MCP configuration...")

    lobster_path = get_lobster_path()
    config_path = get_cursor_mcp_config_path()

    # Get the full path to uv for GUI applications
    uv_path = shutil.which("uv")
    if not uv_path:
        raise RuntimeError("uv not found in PATH. Please install uv.")

    # Create the configuration
    config = {
        "mcpServers": {
            "lobster-inference": {
                "command": uv_path,
                "args": ["run", "--project", str(lobster_path), "--extra", "mcp", "lobster_mcp_server"],
            }
        }
    }

    # Load existing config if it exists
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing_config = json.load(f)

            # Merge configurations
            if "mcpServers" not in existing_config:
                existing_config["mcpServers"] = {}
            existing_config["mcpServers"]["lobster-inference"] = config["mcpServers"]["lobster-inference"]
            config = existing_config

        except json.JSONDecodeError:
            print("⚠️  Existing Cursor MCP config file is invalid JSON, creating new one...")

    # Write the configuration
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Cursor MCP configuration written to: {config_path}")
    print("📋 Please restart Cursor to load the new server.")


def test_installation():
    """Test that dependencies are available"""
    print("🧪 Testing installation...")

    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("❌ PyTorch not found. Please install with: pip install torch")
        return False

    try:
        import mcp  # noqa: F401

        print("✅ MCP package available")
    except ImportError:
        print("❌ MCP package not found. Please install with: pip install mcp")
        return False

    try:
        import fastmcp  # noqa: F401

        print("✅ FastMCP package available")
    except ImportError:
        print("❌ FastMCP package not found. Please install with: uv sync --extra mcp")
        return False

    try:
        from lobster.model import LobsterPMLM  # noqa: F401

        print("✅ Lobster package available")
    except ImportError as e:
        print(f"❌ Lobster package not found: {e}")
        print("Please install Lobster with: uv sync")
        return False

    return True


def main():
    """Main setup function"""
    print("🦞 Lobster MCP Server Setup")
    print("=" * 40)

    # Test installation
    if not test_installation():
        print("\n❌ Setup failed due to missing dependencies.")
        print("Please install the required packages with:")
        print("  uv sync --extra mcp")
        return 1

    # Ask user which client(s) to configure
    print("\nWhich MCP client(s) would you like to configure?")
    print("1. Claude Desktop")
    print("2. Cursor")
    print("3. Both")

    try:
        choice = input("Enter your choice (1-3): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Setup cancelled by user.")
        return 1

    setup_claude = choice in ["1", "3"]
    setup_cursor_client = choice in ["2", "3"]

    if not (setup_claude or setup_cursor_client):
        print("❌ Invalid choice. Please run the setup again.")
        return 1

    # Set up Claude Desktop
    if setup_claude:
        try:
            setup_claude_desktop()
        except Exception as e:
            print(f"❌ Failed to set up Claude Desktop: {e}")
            return 1

    # Set up Cursor
    if setup_cursor_client:
        try:
            setup_cursor()
        except Exception as e:
            print(f"❌ Failed to set up Cursor: {e}")
            return 1

    print("\n🎉 Setup complete!")
    print("\nNext steps:")

    if setup_claude:
        print("📋 Claude Desktop:")
        print("  1. Restart Claude Desktop")
        print("  2. Try asking Claude: 'What Lobster models are available?'")
        print("  3. Test with: 'Get embeddings for protein sequence MKTVRQ using lobster_24M'")

    if setup_cursor_client:
        print("📋 Cursor:")
        print("  1. Restart Cursor")
        print("  2. Open the Command Palette (Cmd+Shift+P)")
        print("  3. Type 'MCP' to see available MCP commands")
        print("  4. Use '@lobster-inference' to interact with the server")

    print("\nAlternatively, test locally with:")
    print("  uv run lobster_mcp_server")

    return 0


if __name__ == "__main__":
    sys.exit(main())
