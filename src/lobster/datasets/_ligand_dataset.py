import logging
import os
import pathlib
from collections.abc import Callable

import numpy as np
import torch

try:
    from torch_geometric.data import Dataset

except ImportError:
    Dataset = None

logger = logging.getLogger(__name__)


class LigandDataset(Dataset):
    """Dataset class for ligand atom coordinates.
    Expects .pt files with a 'coords' key for atom coordinates.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        transform_protein: Callable | None = None,
        transform_ligand: Callable | None = None,
        pre_transform: Callable | None = None,
        min_len: int = 1,
        testing: bool = False,
    ):
        import lobster

        lobster.ensure_package("torch_geometric", group="struct-gpu (or --extra struct-cpu)")

        self.root = pathlib.Path(root)
        self.transform_protein = transform_protein
        self.transform_ligand = transform_ligand
        self.pre_transform = pre_transform
        self.min_len = min_len
        self.testing = testing
        self._load_data()
        logger.info("Loaded ligand data points.")
        super().__init__(root, transform_protein, transform_ligand, pre_transform)

    def _load_data(self):
        processed_files_ligand = []
        processed_files_protein = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith("ligand.pt") or file.startswith("ligand"):
                    processed_files_ligand.append(os.path.join(root, file))
                elif file.endswith("protein.pt"):
                    processed_files_protein.append(os.path.join(root, file))
        self.dataset_filenames_ligand = processed_files_ligand
        self.dataset_filenames_protein = processed_files_protein
        self.dataset_filenames_ligand.sort()
        self.dataset_filenames_protein.sort()
        logger.info(f"Loaded {len(self.dataset_filenames_ligand)} ligand data points.")
        logger.info(f"Loaded {len(self.dataset_filenames_protein)} protein data points.")
        # make tuple of ligand and protein if pdb_id is the same
        self.dataset_filenames = []

        # Create dictionaries for faster lookup
        ligand_dict = {}
        protein_dict = {}

        for ligand_file in self.dataset_filenames_ligand:
            ligand_id = ligand_file.split("/")[-1].split("_")[0]
            ligand_dict[ligand_id] = ligand_file

        for protein_file in self.dataset_filenames_protein:
            protein_id = protein_file.split("/")[-1].split("_")[0]
            protein_dict[protein_id] = protein_file

        if len(self.dataset_filenames_protein) == 0:
            self.dataset_filenames = self.dataset_filenames_ligand
            logger.info("Only ligand data points loaded.")
            return

        # Find matching pairs
        for pdb_id in ligand_dict.keys():
            if pdb_id in protein_dict:
                self.dataset_filenames.append((ligand_dict[pdb_id], protein_dict[pdb_id]))

        logger.info(f"Found {len(self.dataset_filenames)} matching ligand-protein pairs.")
        logger.info(f"Unmatched ligands: {len(self.dataset_filenames_ligand) - len(self.dataset_filenames)}")
        logger.info(f"Unmatched proteins: {len(self.dataset_filenames_protein) - len(self.dataset_filenames)}")

    def len(self) -> int:
        return len(self.dataset_filenames)

    def __getitem__(self, idx: int):
        if isinstance(self.dataset_filenames[idx], tuple):
            x_ligand = torch.load(self.dataset_filenames[idx][0])
            x_protein = torch.load(self.dataset_filenames[idx][1])
            if self.transform_protein:
                x_protein = self.transform_protein(x_protein)
        else:
            x_protein = None
            x_ligand = torch.load(self.dataset_filenames[idx])
        # pick a random 'conformer' in 'conformers' list
        if "conformers" in x_ligand:
            x_ligand = x_ligand["conformers"][np.random.randint(0, len(x_ligand["conformers"]))]

        if self.transform_ligand:
            x_ligand = self.transform_ligand(x_ligand)

        return {"protein": x_protein, "ligand": x_ligand}
