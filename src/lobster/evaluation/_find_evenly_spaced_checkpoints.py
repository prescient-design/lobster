from upath import UPath

from ._extract_step import extract_step


def find_evenly_spaced_checkpoints(model_path: str, n: int = 5) -> dict[str, str]:
    """
    Find n evenly spaced checkpoints in a directory to evaluate model changes over time.

    Parameters
    ----------
    model_path : str
        Path to the model directory containing checkpoints
    n : int, default=5
        Number of checkpoints to select

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model names (with step numbers) to checkpoint paths
    """
    path = UPath(model_path)

    if not path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {model_path}")

    checkpoint_files = list(path.glob("*.ckpt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in directory: {model_path}")

    step_to_ckpt = {}
    for ckpt in checkpoint_files:
        step = extract_step(ckpt.name)
        if step is not None:
            step_to_ckpt[step] = str(ckpt)

    if not step_to_ckpt:
        raise ValueError(f"No valid checkpoints with step numbers found in: {model_path}")

    steps = sorted(step_to_ckpt.keys())
    min_step = steps[0]
    max_step = steps[-1]

    if n >= len(steps):
        result = {}
        for step in steps:
            result[f"{path.name}-step-{step}"] = step_to_ckpt[step]
        return result

    step_size = (max_step - min_step) / (n - 1) if n > 1 else 0

    result = {}
    for i in range(n):
        if i == 0:
            target_step = min_step
        elif i == n - 1:
            target_step = max_step
        else:
            target_step = min_step + i * step_size

        closest_step = min(steps, key=lambda x: abs(x - target_step))

        if closest_step not in [int(name.split("-step-")[1]) for name in result.keys()]:
            result[f"{path.name}-step-{closest_step}"] = step_to_ckpt[closest_step]

    return result
