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

    Parameters
    ----------
    root : str | os.PathLike
        Root directory containing ligand/protein .pt files.
    cluster_file : str | os.PathLike, optional
        Path to cluster file (.pt) mapping sample IDs to cluster IDs.
        If provided, samples are grouped by cluster for balanced sampling.
        If None, each sample is treated as its own cluster.
    transform_protein : Callable, optional
        Transform to apply to protein data.
    transform_ligand : Callable, optional
        Transform to apply to ligand data.
    pre_transform : Callable, optional
        Pre-transform to apply.
    min_len : int
        Minimum length filter (default: 1).
    testing : bool
        If True, limit dataset size for testing (default: False).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        cluster_file: str | os.PathLike | None = None,
        transform_protein: Callable | None = None,
        transform_ligand: Callable | None = None,
        pre_transform: Callable | None = None,
        min_len: int = 1,
        testing: bool = False,
    ):
        import lobster

        lobster.ensure_package("torch_geometric", group="struct-gpu (or --extra struct-cpu)")

        self.root = pathlib.Path(root)
        self.cluster_file = cluster_file
        self.transform_protein = transform_protein
        self.transform_ligand = transform_ligand
        self.pre_transform = pre_transform
        self.min_len = min_len
        self.testing = testing
        self._load_data()
        self._build_cluster_dict()
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

    def _get_sample_id(self, idx: int) -> str:
        """Extract sample ID from filename for cluster lookup."""
        filename = self.dataset_filenames[idx]
        if isinstance(filename, tuple):
            # For protein-ligand pairs, use ligand file's ID
            filename = filename[0]
        # Extract ID: typically the first part before underscore
        return pathlib.Path(filename).stem.split("_")[0]

    def _build_cluster_dict(self):
        """Build cluster dictionary from cluster file or default to individual clusters."""
        if self.cluster_file is not None:
            # Load cluster file: expects dict mapping sample_id -> cluster_id
            cluster_mapping = torch.load(self.cluster_file)
            logger.info(f"Loaded cluster file {self.cluster_file} with {len(cluster_mapping)} entries.")

            # Build cluster_dict as list of lists (indices grouped by cluster)
            cluster_to_indices = {}
            for idx in range(len(self.dataset_filenames)):
                sample_id = self._get_sample_id(idx)
                cluster_id = cluster_mapping.get(sample_id)
                if cluster_id is not None:
                    if cluster_id not in cluster_to_indices:
                        cluster_to_indices[cluster_id] = []
                    cluster_to_indices[cluster_id].append(idx)

            self.cluster_dict = list(cluster_to_indices.values())
            logger.info(f"Built {len(self.cluster_dict)} clusters from cluster file.")
        else:
            # No cluster file: each sample is its own cluster
            self.cluster_dict = [[i] for i in range(len(self.dataset_filenames))]
            logger.info(f"No cluster file: {len(self.cluster_dict)} samples (each as own cluster).")

    @property
    def get_cluster_dict(self):
        """Return cluster dict for compatibility with RandomizedMinorityUpsampler."""
        return self.cluster_dict

    def __getitem__(self, idx: int, _retry_count: int = 0):
        max_retries = 5
        try:
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
        except (EOFError, RuntimeError, Exception) as e:
            # Handle corrupted files by trying a different random sample
            filename = (
                self.dataset_filenames[idx]
                if not isinstance(self.dataset_filenames[idx], tuple)
                else self.dataset_filenames[idx][0]
            )
            logger.warning(f"Failed to load file {filename}: {e}. Trying another sample.")
            if _retry_count < max_retries:
                # Pick a random different index
                new_idx = np.random.randint(0, len(self.dataset_filenames))
                while new_idx == idx:
                    new_idx = np.random.randint(0, len(self.dataset_filenames))
                return self.__getitem__(new_idx, _retry_count + 1)
            else:
                raise RuntimeError(f"Failed to load data after {max_retries} retries. Last error: {e}") from e
