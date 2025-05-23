from collections.abc import Callable, Sequence
from typing import Literal

import pandas as pd
from torch.utils.data import Dataset

INTERACTION_MODALITY_TYPES = [
    "protein-dna",
    "protein-protein",
    "protein-rna",
    "protein-small_molecule",
    "small_molecule-small_molecule",
]


class AtomicaDataset(Dataset):
    """ATOMICA contains various molecular interaction types (protein-DNA, protein-protein, protein-RNA, etc.)
    with sequences from different modalities (amino_acid, nucleotide, smiles, etc.) and their interaction metadata.
    This dataset is a processed version that contains a subset of the original ATOMICA dataset,
    and sequence only information.

    Reference:
        @article{Fang2025ATOMICA,
            author = {Fang, Ada and Zhang, Zaixi and Zhou, Andrew and Zitnik, Marinka},
            title = {ATOMICA: Learning Universal Representations of Intermolecular Interactions},
            year = {2025},
            journal = {bioRxiv},
            doi = {10.1101/2025.04.02.646906}
        }

    Each item in the dataset represents a molecular interaction with up to 5 different sequences
    (referred to as sequence1, sequence2, etc.) and their corresponding modalities.

    Example of dataset structure:
    ```
    sequence1              | modality1  | sequence2              | modality2   | ... | interaction_type
    ---------------------- | ---------- | ---------------------- | ----------- | --- | ----------------
    MAGVKNSIIW...          | amino_acid | CAGCGGTTGC...          | nucleotide  | ... | protein-dna
    GPLGSPEFGR...          | amino_acid | GPLGSPEFGR...          | amino_acid  | ... | protein-protein
    MHHHHHHENLY...         | amino_acid | MAAETRNVAG...          | amino_acid  | ... | protein-rna
    ```

    When max_modalities=3, an item might look like:
    (
        'MHHHHHHENLYFQGSGMAGSVGLALCGQTL...', 'amino_acid',  # sequence1: A protein sequence
        'MAAETRNVAGAEAPPQKRYYQRAHSNPM...', 'amino_acid',   # sequence2: Another protein sequence
        'GCCCGGAUAGCUCAGUCGGUAGAGCAUC...', 'nucleotide',   # sequence3: An RNA sequence
    )

    Dataset size:
    | Interaction Type                  |   Count |
    |-----------------------------------|--------:|
    | small_molecule-small_molecule     | 246,728 |
    | protein-protein                   |  60,624 |
    | protein-small_molecule            |  32,120 |
    | protein-dna                       |   2,087 |
    | protein-rna                       |   1,785 |

    Parameters
    ----------
    modalities : Sequence[str] or None, optional
        Filter dataset to only include specified interaction modality types.
        If None, all interaction types are included. Valid types are defined
        in INTERACTION_MODALITY_TYPES. Default is None.
    split : {'train', 'val', 'test'}, optional
        Dataset split to use. Default is 'train'.
    max_modalities : int, optional
        Maximum number of modalities to include per item (2-5). This affects how
        many sequence/modality pairs are included in each returned item. For example,
        if max_modalities=3, each returned item will have up to 3 sequence/modality pairs.
        Default is 2.
    transform : callable or None, optional
        Optional transform to apply to each item. Default is None.

    Raises
    ------
    ValueError
        If max_modalities is not between 2 and 5
        If split is not one of ['train', 'val', 'test']
        If any specified modality is not in INTERACTION_MODALITY_TYPES
    """

    def __init__(
        self,
        modalities: Sequence[str] | None = None,
        split: Literal["train", "val", "test"] = "train",
        max_modalities: int = 2,
        transform: Callable | None = None,
    ):
        super().__init__()

        if max_modalities < 2 or max_modalities > 5:
            raise ValueError("max_modalities must be between 2 and 5")

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of ['train', 'val', 'test']")

        # Define column names based on max_modalities
        # For each modality i, include both sequence{i} and modality{i} columns
        self.columns = [item for i in range(1, max_modalities + 1) for item in (f"sequence{i}", f"modality{i}")]

        self.split = split
        self.transform = transform

        fpath = "hf://datasets/karina-zadorozhny/ATOMICA"
        fpath = f"{fpath}/split={split}"

        data = pd.read_parquet(fpath)

        if modalities is not None:
            if any(mod not in INTERACTION_MODALITY_TYPES for mod in modalities):
                raise ValueError(f"Invalid interaction modality. Choose from {INTERACTION_MODALITY_TYPES}")
            data = data[data["interaction_type"].isin(modalities)]

        # Keep only items where there is at most max_modalities of sequences
        # For example, if max_modalities=3, we only keep rows where sequence4 is NA
        # This ensures we only include interactions with the specified max number of modalities
        data = data[data[f"sequence{max_modalities + 1}"].isna()]

        self.data = data.dropna(subset=self.columns)

        self._x = list(self.data[self.columns].apply(tuple, axis=1))

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """Get an item from the dataset at the specified index.

        Returns a tuple of sequence-modality pairs. The number of pairs depends on max_modalities.

        Parameters
        ----------
        index : int
            Index of the item to retrieve

        Returns
        -------
        tuple
            A tuple containing sequence-modality pairs
            (sequence1, modality1, sequence2, modality2, ...) based on max_modalities
        """
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self.data)
