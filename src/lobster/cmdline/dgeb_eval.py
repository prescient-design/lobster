"""Command-line interface for DGEB evaluation of UME models."""

import sys
from pathlib import Path

# Add the evaluation module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobster.evaluation.dgeb_runner import main

if __name__ == "__main__":
    main()
