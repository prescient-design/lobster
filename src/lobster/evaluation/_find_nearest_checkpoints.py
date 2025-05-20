import os

from upath import UPath

from ._extract_step import extract_step

CACHE_DIR = os.environ.get("LOBSTER_CACHE_DIR", "/data2/lobster/ume/data")


def find_nearest_checkpoints(model_paths: dict[str, str]) -> dict[str, str]:
    """
    Find the nearest checkpoints across all models to ensure fair comparison.

    Parameters
    ----------
    model_paths : Dict[str, str]
        Dictionary mapping model names to checkpoint paths or directories

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model names to their selected checkpoint paths
    """
    model_checkpoints = {}
    model_steps = {}

    # Extract steps for each model
    for model_name, model_path in model_paths.items():
        # Convert to UPath for cloud storage compatibility
        path = UPath(model_path)
        if path.is_file():
            # If the config points directly to a file, use its parent directory
            directory = path.parent
            # Add this specific checkpoint
            step = extract_step(path.name)
            if step is not None:
                model_steps[model_name] = [step]
                model_checkpoints[model_name] = {step: str(path)}
        else:
            # If the config points to a directory, find all checkpoints
            directory = path

            # Get all checkpoints in the directory
            checkpoint_files = list(directory.glob("*.ckpt"))

            # Extract steps and build mapping
            steps = []
            checkpoints = {}

            for ckpt in checkpoint_files:
                step = extract_step(ckpt.name)
                if step is not None:
                    steps.append(step)
                    checkpoints[step] = str(ckpt)

            if steps:  # Only update if we found checkpoints
                model_steps[model_name] = steps
                model_checkpoints[model_name] = checkpoints

    # Find the smallest maximum step across all models
    max_steps = {model: max(steps) for model, steps in model_steps.items()}
    smallest_max_step = min(max_steps.values())

    # Find the nearest checkpoint to the smallest max step for each model
    selected_checkpoints = {}
    for model, checkpoints in model_checkpoints.items():
        available_steps = list(checkpoints.keys())
        nearest_step = min(available_steps, key=lambda x: abs(x - smallest_max_step))
        selected_checkpoints[model] = checkpoints[nearest_step]

    return selected_checkpoints
