def load_vocab_file(vocab_file: str) -> list[str]:
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()

    return [ll.strip() for ll in lines]
