from pathlib import Path


def load_vocab_file(vocab_file: str | Path) -> list[str]:
    with open(vocab_file) as f:
        lines = f.read().splitlines()

    return [ll.strip() for ll in lines]
