import os
from pathlib import Path
from subprocess import Popen
from typing import Tuple


def download(data: Tuple[str, str], output_dir: str) -> None:
    """
    Download a file from a URL to a specified location.

    Parameters
    ----------
    data : tuple
        A tuple containing (filepath, url)
    output_dir : str
        The directory to save the downloaded file
    """
    filepath, url = data
    output_filepath = os.path.join(output_dir, filepath)
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    try:
        Popen(f"wget -q {url} -O {output_filepath}", shell=True).wait()
    except Exception as e:
        print(f"Download error: {e}")
