"""Upload UME vocabulary files to HuggingFace Hub.

This script uploads vocabulary files from the lobster assets
to the HuggingFace repository under a vocabs/ folder.

Usage:
------
    python upload_vocabs.py

Or from Python:
    from upload_vocabs import upload_vocabulary_files
    upload_vocabulary_files()
"""

import importlib.resources
from pathlib import Path

from huggingface_hub import HfApi

from lobster.constants import HF_UME_REPO_ID


def upload_vocabulary_files(repo_id: str = HF_UME_REPO_ID, create_repo: bool = False) -> None:
    """
    Upload vocabulary files to HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID
    create_repo : bool, default=False
        Whether to create the repository if it doesn't exist
    """
    api = HfApi()

    if create_repo:
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")

    assets_path = importlib.resources.files("lobster") / "assets"

    vocab_files = [
        {
            "local_path": assets_path / "ume_tokenizers" / "amino_acid_tokenizer" / "vocab.txt",
            "hub_path": "vocabs/amino_acid_vocab.txt",
        },
        {
            "local_path": assets_path / "ume_tokenizers" / "nucleotide_tokenizer" / "vocab.txt",
            "hub_path": "vocabs/nucleotide_vocab.txt",
        },
        {
            "local_path": assets_path / "smiles_tokenizer" / "vocab.txt",
            "hub_path": "vocabs/smiles_vocab.txt",
        },
    ]

    for vocab_file in vocab_files:
        local_path = Path(vocab_file["local_path"])
        hub_path = vocab_file["hub_path"]

        if not local_path.exists():
            print(f"File not found: {local_path}")
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hub_path,
                repo_id=repo_id,
                commit_message=f"Upload {hub_path}",
            )
            print(f"Uploaded: {hub_path}")

        except Exception as e:
            print(f"Failed to upload {hub_path}: {e}")


if __name__ == "__main__":
    upload_vocabulary_files(create_repo=True)
