# Contributing to LBSTER 🦞

We welcome contributions to LBSTER! This guide will help you get started with contributing to our library for biological sequence language models.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Review Process](#code-review-process)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Fork and Clone the Repository

1. **Fork the repository** by clicking the "Fork" button on the [main repository page](https://github.com/prescient-design/lobster)
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lobster.git
   cd lobster
   ```
3. **Add the upstream remote** to keep your fork in sync:
   ```bash
   git remote add upstream https://github.com/prescient-design/lobster.git
   ```

### Keep Your Fork Updated

Before starting new work, sync your fork with the upstream repository:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Commit and push** your changes
5. **Submit a pull request** from your fork to the main repository

### Branch Naming Conventions

Use descriptive branch names with prefixes:
- `feature/` - New features or enhancements
- `fix/` - Bug fixes  
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding or improving tests

Examples:
- `feature/add-protein-folding-model`
- `fix/training-memory-leak`
- `docs/update-installation-guide`

## Development Setup

### Prerequisites

- Python 3.11 or 3.12 (3.13 not yet supported)
- [uv](https://github.com/astral-sh/uv)
- Git

### Installation

We recommend using `uv` for development:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/lobster.git
cd lobster

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install in development mode with all extras
uv sync --all-extras
```

**Using conda/mamba:**
```bash
mamba env create -f env.yml
pip install -e .
```

### Verify Installation

Test that everything is working:

```bash
# Run basic tests
pytest tests/ -v

# Test CLI commands
uv run lobster_embed --help
uv run lobster_train --help
```

## Code Quality Standards

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality. **You must set up pre-commit hooks** before making contributions:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files (optional, to test setup)
pre-commit run --all-files
```





## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure: `src/lobster/model.py` → `tests/test_model.py`
- Use descriptive test names: `test_embed_sequences_with_valid_input`
- Test both success and failure cases
- Use fixtures for common test data

Example test:

```python
import pytest
import torch
from unittest.mock import Mock, patch
from lobster.rl_training.reward_functions import UmeRewardFunction


class TestUmeRewardFunction:
    @patch('lobster.rl_training.reward_functions.UmeRewardFunction')
    def test_compute_rewards_single_batch(self, mock_ume_class):
        """Test computing UME rewards for a small batch of sequences."""
        # Mock the UME reward function
        mock_reward_fn = Mock()
        mock_reward_fn.return_value = torch.tensor([0.8, 0.6, 0.9])
        mock_ume_class.return_value = mock_reward_fn
        
        sequences = [
            "CCO",  # ethanol SMILES
            "MKTVRQERLKSIVRILERSKEPVSGAQ",  # protein sequence
            "ATCGATCGATCG"  # DNA sequence
        ]
        
        reward_fn = UmeRewardFunction("dummy_path.ckpt")
        rewards = reward_fn(sequences)
        
        assert isinstance(rewards, torch.Tensor)
        assert len(rewards) == 3
        assert all(0.0 <= r <= 1.0 for r in rewards.tolist())
        
    def test_empty_sequences_raises_error(self):
        """Test that empty sequence list raises ValueError."""
        from lobster.rl_training.reward_functions import compute_ume_rewards
        
        with pytest.raises(ValueError, match="sequences cannot be empty"):
            compute_ume_rewards([], "dummy_path.ckpt")
            
    def test_invalid_model_path_raises_error(self):
        """Test that invalid model path raises FileNotFoundError."""
        from lobster.rl_training.reward_functions import compute_ume_rewards
        
        sequences = ["CCO", "MKTVRQERLK"]
        with pytest.raises(FileNotFoundError):
            compute_ume_rewards(sequences, "nonexistent_model.ckpt")
```

### Test Categories

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows (e.g., training pipeline)
- **Performance tests**: Test memory usage and inference speed for large models


### Adding New Documentation

- Add new `.md` files to the `docs/` directory
- Update the main README.md if you add major features
- Include usage examples in docstrings
- Add notebook tutorials for complex workflows

### CLAUDE.md Updates

If you change project structure, commands, or major features, remember to update `CLAUDE.md` as prompted by the pre-commit hook.

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date** with upstream main
2. **Run tests locally** to make sure everything passes
3. **Push your branch** to your fork
4. **Create a pull request** from your branch to upstream main
5. **Fill out the PR template** with details about your changes

### Pull Request Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what changes you made and why
- **Link issues**: Reference related issues with "Fixes #123" or "Closes #123"
- **Screenshots**: Include screenshots for UI changes
- **Breaking changes**: Clearly mark any breaking changes

### PR Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes made

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated existing tests if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)
```



### Recognition

Please add your name to the `CONTRIBUTORS.md` file

Thank you for contributing to LBSTER! Your contributions help advance biological sequence modeling for everyone. 🦞✨