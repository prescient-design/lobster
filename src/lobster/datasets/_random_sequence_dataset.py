import random

from torch.utils.data import Dataset


class RandomSequenceDataset(Dataset):
    """Dataset that generates random sequences from a vocabulary."""

    def __init__(
        self,
        vocab: set[str],
        num_sequences: int,
        min_length: int = 50,
        max_length: int = 500,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        vocab : set[str]
            Set of characters/tokens to use for generating random sequences
        num_sequences : int
            Number of random sequences to generate
        min_length : int
            Minimum length of generated sequences
        max_length : int
            Maximum length of generated sequences
        seed : int
            Random seed for reproducibility
        """
        self.vocab = list(vocab)
        self.num_sequences = num_sequences
        self.min_length = min_length
        self.max_length = max_length

        # Generate all sequences at initialization for consistency
        random.seed(seed)
        self.sequences = []
        for _ in range(num_sequences):
            length = random.randint(min_length, max_length)
            sequence = "".join(random.choices(self.vocab, k=length))
            self.sequences.append(sequence)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]
