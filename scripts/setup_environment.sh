#!/bin/bash
# Environment Setup Script for Lobster
# This script sets up the Python environment and installs dependencies

set -e  # Exit on error

echo "==================================="
echo "Lobster Environment Setup"
echo "==================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

# Sync dependencies
echo ""
echo "Step 1: Syncing dependencies with uv..."
uv sync --all-extras

echo ""
echo "Step 2: Installing additional dependencies for notebooks..."
uv pip install ipywidgets jupyter matplotlib seaborn biopython python-Levenshtein

echo ""
echo "==================================="
echo "Environment setup complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run interactive notebooks:"
echo "  jupyter notebook notebooks/"
echo ""
echo "To run the automated workflow script:"
echo "  python scripts/protein_inference_workflow.py"
echo "==================================="
