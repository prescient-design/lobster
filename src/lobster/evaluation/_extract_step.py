import re


def extract_step(filename: str) -> int | None:
    """
    Extract the step number from a checkpoint filename.

    Parameters
    ----------
    filename : str
        Checkpoint filename

    Returns
    -------
    int or None
        The step number, or None if not found
    """
    # Match patterns like 'step=10000' or 'step-10000'
    step_match = re.search(r"step[=\-](\d+)", filename)
    if step_match:
        return int(step_match.group(1))
    return None
