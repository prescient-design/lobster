# DXT Extension Distribution

This document explains how to build and distribute the Lobster DXT extension for Claude Desktop.

## Building DXT Extensions

### Prerequisites

- Python 3.11+
- `uv` package manager
- Node.js 18+ (for DXT CLI)
- Lobster repository with MCP dependencies installed

### Local Development Build

For testing and development, build the DXT extension locally:

```bash
# Install DXT CLI
npm install -g @anthropic-ai/dxt

# Install dependencies
uv sync --extra mcp

# Build DXT extension using official CLI
dxt pack

# This creates lobster.dxt in the current directory (~327 MB)
```

### Automated Builds

DXT extensions are automatically built and released via GitHub Actions workflow (`.github/workflows/build-dxt.yml`):

#### Automatic Trigger on Version Tags
```bash
# Create and push a version tag
git tag v0.1.0
git push origin v0.1.0

# This triggers the GitHub Actions workflow that:
# 1. Sets up Node.js 18 and Python 3.11
# 2. Installs DXT CLI and dependencies
# 3. Builds the DXT extension using `dxt pack`
# 4. Creates a GitHub Release
# 5. Attaches the .dxt file to the release
```

#### Manual Trigger for Testing
You can also manually trigger the workflow without creating version tags:

1. Go to your GitHub repository → **Actions** tab
2. Select **"Build and Release DXT Extension"** workflow
3. Click **"Run workflow"** button
4. Enter a test version (e.g., `v0.1.0-test`)
5. Click **"Run workflow"** to start the build

This creates a prerelease without affecting your main version tags.

### Testing Methods

#### Method 1: Manual GitHub Actions Trigger (Recommended)
The workflow includes `workflow_dispatch` for manual testing:

1. Go to your GitHub repository → **Actions** tab
2. Select **"Build and Release DXT Extension"** workflow
3. Click **"Run workflow"** button
4. Enter a test version like `v0.1.0-test`
5. This creates a prerelease without affecting your main version tags

#### Method 2: Local Build Testing
```bash
# Install DXT CLI globally
npm install -g @anthropic-ai/dxt

# Install dependencies
uv sync --extra mcp

# Build DXT extension locally
dxt pack

# This creates lobster.dxt in the current directory (~327 MB)
# Test installation in Claude Desktop before releasing
```

#### Method 3: Test with act (GitHub Actions locally)
```bash
# Install act (requires Docker)
brew install act

# Run the workflow locally
act workflow_dispatch -W .github/workflows/build-dxt.yml --input version=v0.1.0-test
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

1. **Build locally**: `dxt pack`
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
- **DXT CLI not found**: Install with `npm install -g @anthropic-ai/dxt`
- **Permission errors**: Ensure write permissions in the output directory
- **Large file size**: The extension includes all dependencies (~327 MB)
- **GitHub Actions timeout**: Large dependency downloads may timeout; consider caching

### Installation Issues

- **Claude Desktop not recognizing**: Restart Claude Desktop after installation
- **Import errors**: Check that all dependencies are included in `server/lib`
- **Version conflicts**: Ensure compatibility with your Claude Desktop version

### Distribution Issues

- **GitHub release failed**: Check GitHub Actions logs for build errors
- **File too large**: Consider excluding unnecessary dependencies
- **Download issues**: Verify the release is public and accessible