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

## Documentation

### Docstring Guidelines

Use Google-style docstrings:

```python
def train_ume_grpo(
    dataset_path: str, 
    ume_model_path: str,
    output_dir: str,
    config: dict[str, Any] | None = None
) -> None:
    """Train a language model using UME-based GRPO rewards.
    
    Args:
        dataset_path: Path to the training dataset (HuggingFace dataset format)
        ume_model_path: Path to the UME model checkpoint for reward computation
        output_dir: Directory to save training artifacts and checkpoints
        config: Optional training configuration overrides. If None, uses defaults
            
    Returns:
        None. Saves trained model and logs to the output directory
        
    Raises:
        FileNotFoundError: If dataset or UME model paths don't exist
        ValueError: If config contains invalid training parameters
        RuntimeError: If training fails due to memory or device issues
        
    Example:
        >>> # Train with default configuration
        >>> train_ume_grpo(
        ...     dataset_path="synthetic_molecular_dataset/",
        ...     ume_model_path="models/ume_checkpoint.ckpt", 
        ...     output_dir="./grpo_training_output"
        ... )
        
        >>> # Train with custom configuration
        >>> custom_config = {
        ...     "learning_rate": 1e-5,
        ...     "batch_size": 16,
        ...     "num_train_epochs": 5,
        ...     "gradient_accumulation_steps": 4
        ... }
        >>> train_ume_grpo(
        ...     dataset_path="data/molecules.json",
        ...     ume_model_path="models/ume_v2.ckpt",
        ...     output_dir="./custom_training",
        ...     config=custom_config
        ... )
    """
```

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



## Code Review Process

### What to Expect

- **Initial response**: Maintainers will respond within 1-2 business days
- **Review process**: May involve multiple rounds of feedback
- **CI checks**: All automated checks must pass
- **Approval**: At least one maintainer approval required for merge

### Addressing Feedback

- **Be responsive**: Address feedback promptly
- **Ask questions**: If feedback is unclear, ask for clarification
- **Make focused changes**: Address each piece of feedback individually
- **Test changes**: Re-run tests after making changes

### Review Criteria

Reviewers will check for:
- Code correctness and logic
- Test coverage for new functionality
- Documentation completeness
- Performance implications
- Security considerations
- Consistency with existing codebase

## Community Guidelines

### Code of Conduct

We follow a code of conduct that promotes:
- **Respectful communication** with all community members
- **Constructive feedback** in code reviews and discussions
- **Inclusive environment** welcoming contributors of all backgrounds
- **Focus on technical merit** in discussions and decisions

### Getting Help

- **GitHub Issues**: For bugs, feature requests, and questions
- **Discussions**: For general questions and community interaction
- **Documentation**: Check existing docs first
- **Code examples**: Look at the `examples/` directory

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file (if it exists)
- Release notes for significant contributions
- Commit history and GitHub contributors page

## Additional Resources

### Useful Commands

```bash
# Development workflow
git checkout main && git pull upstream main
git checkout -b feature/my-feature
# ... make changes ...
git add . && git commit -m "descriptive message"
git push origin feature/my-feature

# Testing and quality
pytest tests/ --cov=src/lobster
ruff check . --fix
ruff format .
pre-commit run --all-files

# Documentation
# Build docs locally (if doc building is set up)
# python -m sphinx docs/ docs/_build/
```

### Project Structure

```
lobster/
├── src/lobster/          # Main package source
├── tests/                # Test files
├── docs/                 # Documentation
├── examples/             # Usage examples
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter tutorials
├── pyproject.toml        # Project configuration
├── .pre-commit-config.yaml
└── README.md
```

### Key Dependencies

- **PyTorch**: Deep learning framework
- **Lightning**: Training framework
- **Transformers**: HuggingFace transformers
- **Biopython**: Biological sequence processing
- **uv**: Fast Python package manager
- **Ruff**: Fast Python linter/formatter

---

Thank you for contributing to LBSTER! Your contributions help advance biological sequence modeling for everyone. 🦞✨