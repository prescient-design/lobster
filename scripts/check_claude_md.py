#!/usr/bin/env python3
"""Script to check if CLAUDE.md might need updating based on recent changes."""

import subprocess
import sys
from pathlib import Path

def get_recent_changes():
    """Get files changed in recent commits."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~5..HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def should_update_claude_md(changed_files):
    """Check if changes suggest CLAUDE.md should be updated."""
    sensitive_paths = [
        "pyproject.toml",
        "src/lobster/cmdline/",
        "src/lobster/model/__init__.py",
        "src/lobster/hydra_config/",
        "README.md",
        ".pre-commit-config.yaml"
    ]
    
    for file in changed_files:
        for path in sensitive_paths:
            if path in file:
                return True
    return False

def main():
    changed_files = get_recent_changes()
    if should_update_claude_md(changed_files):
        print("⚠️  Recent changes suggest CLAUDE.md might need updating:")
        for file in changed_files:
            print(f"   - {file}")
        print("\n   Consider updating CLAUDE.md if you changed:")
        print("   - CLI commands, project structure, dependencies, or core architecture")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())