# DXT Extension Distribution

This document explains how to build and distribute the Lobster DXT extension for Claude Desktop.

## What is DXT?

DXT (Desktop Extension Toolkit) is Anthropic's system for distributing MCP servers as local desktop extensions for Claude Desktop. DXT extensions are packaged as `.dxt` files that contain:

- The MCP server code and dependencies
- A manifest file describing the extension
- Any additional resources needed

## Building DXT Extensions

### Prerequisites

- Python 3.11+
- `uv` package manager
- Lobster repository with MCP dependencies installed

### Local Development Build

For testing and development, build the DXT extension locally:

```bash
# Install dependencies
uv sync --extra mcp

# Build DXT extension
python scripts/build_dxt.py

# This creates lobster.dxt in the current directory
```

### Automated Builds

DXT extensions are automatically built and released when you push version tags:

```bash
# Create and push a version tag
git tag v0.1.0
git push origin v0.1.0

# This triggers the GitHub Actions workflow that:
# 1. Builds the DXT extension
# 2. Creates a GitHub Release
# 3. Attaches the .dxt file to the release
```

### Testing Without Version Tags

To test the build workflow without pushing version tags:

#### Method 1: Manual GitHub Actions Trigger (Recommended)
1. Go to your GitHub repository → **Actions** tab
2. Select "Build and Release DXT Extension" workflow
3. Click **"Run workflow"** button
4. Enter a test version like `v0.1.0-test`
5. This creates a prerelease without affecting your main version tags

#### Method 2: Local Build Testing
```bash
# Test the build script locally
python scripts/build_dxt.py

# This creates lobster.dxt in the current directory (~339 MB)
# Test installation in Claude Desktop before releasing
```

#### Method 3: Branch Testing
```bash
# Create a separate branch for testing
git checkout -b test-dxt-workflow
git push origin test-dxt-workflow

# Test changes without affecting main branch
```

## Distribution Strategy

The recommended approach is to distribute .dxt files through **GitHub Releases**:

1. **Automatic builds**: GitHub Actions builds .dxt files on version tags
2. **Versioned releases**: Each release corresponds to a specific version
3. **Easy downloads**: Users can download directly from GitHub
4. **Release notes**: Include changelog and installation instructions

### Release Process

1. **Update version** in `manifest.json` and `pyproject.toml`
2. **Create tag**: `git tag v0.1.0`
3. **Push tag**: `git push origin v0.1.0`
4. **GitHub Actions** automatically:
   - Builds the DXT extension
   - Creates a release
   - Attaches the .dxt file
5. **Users download** from the GitHub release page

## Installation Instructions

### For Users

1. **Download**: Get the latest `.dxt` file from [GitHub Releases](https://github.com/prescient-design/lobster/releases)
2. **Install**: Double-click the `.dxt` file or drag it into Claude Desktop
3. **Verify**: The extension should appear in Claude Desktop's extension manager

### For Developers

1. **Build locally**: `python scripts/build_dxt.py`
2. **Test**: Install the local `.dxt` file in Claude Desktop
3. **Iterate**: Make changes and rebuild as needed

## File Structure

The DXT extension contains:

```
lobster.dxt (zip archive)
├── manifest.json          # Extension metadata
├── server/
│   └── lib/              # Python dependencies
│       ├── lobster/      # Lobster package
│       ├── torch/        # PyTorch
│       ├── transformers/ # HuggingFace Transformers
│       └── ...           # Other dependencies
└── src/                  # Source code
    └── lobster/
        └── mcp/
            └── inference_server.py
```

## Troubleshooting

### Build Issues

- **Missing dependencies**: Run `uv sync --extra mcp`
- **Permission errors**: Ensure write permissions in the output directory
- **Large file size**: The extension includes all dependencies (~339 MB)
- **GitHub Actions timeout**: Large dependency downloads may timeout; consider caching

### Installation Issues

- **Claude Desktop not recognizing**: Restart Claude Desktop after installation
- **Import errors**: Check that all dependencies are included in `server/lib`
- **Version conflicts**: Ensure compatibility with your Claude Desktop version

### Distribution Issues

- **GitHub release failed**: Check GitHub Actions logs for build errors
- **File too large**: Consider excluding unnecessary dependencies
- **Download issues**: Verify the release is public and accessible

## Best Practices

### Development

1. **Test locally** before releasing using `python scripts/build_dxt.py`
2. **Use manual workflow triggers** for testing before version tags
3. **Version appropriately** following semantic versioning
4. **Include release notes** with changes and known issues
5. **Test installation** on clean systems

### Distribution

1. **Use GitHub Releases** for all distribution
2. **Never commit** .dxt files to Git
3. **Include installation instructions** in release notes
4. **Monitor download metrics** and user feedback

### Maintenance

1. **Regular updates** to keep dependencies current
2. **Security patches** for critical vulnerabilities
3. **Compatibility testing** with new Claude Desktop versions
4. **User feedback** collection and response

### Building with Claude Code
Build [extensions directly with Claude Code](https://www.anthropic.com/engineering/desktop-extensions). For example, describe what the extension should do and add this prompt:
```
I want to build this as a Desktop Extension, abbreviated as "DXT". Please follow these steps:

1. **Read the specifications thoroughly:**
   - https://github.com/anthropics/dxt/blob/main/README.md - DXT architecture overview, capabilities, and integration patterns
   - https://github.com/anthropics/dxt/blob/main/MANIFEST.md - Complete extension manifest structure and field definitions
   - https://github.com/anthropics/dxt/tree/main/examples - Reference implementations including a "Hello World" example

2. **Create a proper extension structure:**
   - Generate a valid manifest.json following the MANIFEST.md spec
   - Implement an MCP server using @modelcontextprotocol/sdk with proper tool definitions
   - Include proper error handling and timeout management

3. **Follow best development practices:**
   - Implement proper MCP protocol communication via stdio transport
   - Structure tools with clear schemas, validation, and consistent JSON responses
   - Make use of the fact that this extension will be running locally
   - Add appropriate logging and debugging capabilities
   - Include proper documentation and setup instructions

4. **Test considerations:**
   - Validate that all tool calls return properly structured responses
   - Verify manifest loads correctly and host integration works

Generate complete, production-ready code that can be immediately tested. Focus on defensive programming, clear error messages, and following the exact DXT specifications to ensure compatibility with the ecosystem.
```