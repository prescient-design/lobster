"""Base dataset class for biomolecules."""

import glob
import logging
import multiprocessing as mp
import os
import pathlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

try:
    from torch_geometric.data import Dataset
except ImportError:
    Dataset = None

try:
    from icecream import ic
except ImportError:
    ic = None

logger = logging.getLogger(__name__)


def merge_small_lists(list_of_lists, min_size=100):
    # Identify lists with less than min_size entries
    small_lists = [sublist for sublist in list_of_lists if len(sublist) < min_size]

    # Merge all small lists into a single list
    merged_list = [item for small_list in small_lists for item in small_list]

    # Create result by replacing small lists with the single merged list
    result = [sublist for sublist in list_of_lists if len(sublist) >= min_size]
    if len(merged_list) > 0:
        result.append(merged_list)

    return result


def make_struc_dict(cluster_file, processed_dir):
    # E>G /data/lisanzas/latent_generator/studies/data/pinder/pinder.parquet
    df = pd.read_parquet(cluster_file, engine="pyarrow")
    cluster_dict = {}
    # make "id" column in df key and "cluster_id" column in df value
    cluster_dict = df.set_index("id")["cluster_id"].to_dict()
    # save cluster_dict to file as pt
    torch.save(cluster_dict, pathlib.Path(processed_dir) / "cluster_dict.pt")


def process_file(file_info):
    """Process a single file and return relevant information."""
    file_path, files_to_keep, cluster_dict = file_info

    # Quick filter for .pt files
    if not file_path.endswith(".pt") or any(x in file_path for x in ["cluster", "filter", "transform"]):
        return None, None

    fname = Path(file_path).stem

    # Check files_to_keep
    if files_to_keep is not None and fname not in files_to_keep:
        return file_path, None

    # Check file size
    try:
        if Path(file_path).stat().st_size == 0:
            return file_path, None
    except OSError:
        return file_path, None

    # Get cluster info if needed
    cluster_info = None
    if cluster_dict is not None:
        cluster_info = (fname, cluster_dict.get(fname))

    return file_path, cluster_info


class StructureDataset(Dataset):
    """Base dataset class for protein dataset datasets.

    This class is a subclass of the PyTorch Geometric Dataset class.

    Parameters
    ----------
    root : str | os.PathLike]
        The root directory of the dataset.

    cluster_file : str | os.PathLike, optional
        Path to the cluster file containing cluster assignments per training example.

    transform : callable, optional
        Transform to apply to the data.

    pre_transform : callable, optional
        Transform to apply to the data before processing.

    overwrite : bool, optional
        Whether to overwrite existing processed files, by default False.

    num_cores : int, optional
        Number of CPU cores to use for processing, by default 1.

    min_len : int, optional
        Minimum length for merging small clusters, by default 100.

    testing : bool, optional
        Whether to run in testing mode (limited data), by default False.

    files_to_keep : str | os.PathLike, optional
        Path to pickle file containing list of files to keep.

    use_mmap : bool, optional
        Whether to use memory mapping for loading large datasets and cluster files.
        This can significantly reduce memory usage for large datasets by loading
        data on-demand rather than all at once, by default False.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        cluster_file: str | os.PathLike = None,
        transform=None,
        pre_transform=None,
        overwrite: bool = False,
        num_cores: int = 1,
        min_len: int = 100,
        testing: bool = False,
        files_to_keep: str | os.PathLike = None,
        use_mmap: bool = False,
    ):
        import lobster

        lobster.ensure_package("torch_geometric", group="struct-gpu (or --extra struct-cpu)")
        lobster.ensure_package("icecream", group="struct-gpu (or --extra struct-cpu)")

        self.root = pathlib.Path(root)
        self.processed_dir = self.root
        # check if self.processed_dir is a file
        if os.path.isfile(self.processed_dir):
            self.load_to_disk = True
            self.load_to_disk_file = self.root
            self.processed_dir = self.root.parent
        else:
            self.load_to_disk = False
        self.transform = transform
        self.pre_transform = pre_transform
        self.cluster_file = cluster_file
        self.files_to_keep = files_to_keep

        self.overwrite = overwrite
        self.num_cores = num_cores
        self.min_len = min_len
        self.testing = testing
        self.use_mmap = use_mmap
        logger.info(f"Loading data from {self.root}")
        self._load_data()
        logger.info("Loaded data points.")
        # breakpoint()
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        """Return path to the raw datasets."""
        return str(self.dataset_dir)

    @property
    def raw_file_names(self) -> list[str]:
        """Return list of raw file names."""
        return self.dataset_filenames

    @property
    def processed_dir(self):
        return self._processed_dir

    @processed_dir.setter
    def processed_dir(self, value):
        self._processed_dir = value

    @property
    def get_cluster_dict(self):
        return self.cluster_dict

    @property
    def processed_file_names(self) -> list[str]:
        """Return list of processed files (ending with `.pt`)."""
        # use both dataset_filenames and identifiers to create processed file names assums .cif or .pdb ending for strucs
        return [f"{self.dataset_filenames[i]}" for i, f in enumerate(self.dataset_filenames)]

    def len(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset_filenames)

    def process(self):
        # Process datasets into pt files
        if self.load_to_disk:
            return
        for idx, dataset_file in enumerate(self.dataset_filenames):
            logger.info(f"Processing {dataset_file}...")
            file_exists = os.path.exists(pathlib.Path(self.processed_dir) / self.processed_file_names[idx])
            if not file_exists or self.overwrite:
                raise NotImplementedError
            else:
                logger.info(f"Skipping {dataset_file} as it already exists.")

        logger.info("Finished processing datasets.")

    def _load_data(self):
        """Load the dataset from the processed files."""
        # Load cluster file
        if self.cluster_file is not None:
            self.cluster_dict = torch.load(self.cluster_file)
            logger.info(f"Loaded cluster file {self.cluster_file} with {len(self.cluster_dict)} clusters.")

        # Load files to keep
        files_to_keep = None
        if self.files_to_keep is not None:
            with open(self.files_to_keep, "rb") as f:
                files_to_keep = pickle.load(f)
            logger.info(f"Using files_to_keep with currently {len(files_to_keep)} files to keep")

        # Get all .pt files recursively using glob
        all_files = glob.glob(str(Path(self.processed_dir) / "**/*.pt"), recursive=True)

        # Prepare arguments for parallel processing
        process_args = [
            (f, files_to_keep, self.cluster_dict if self.cluster_file is not None else None) for f in all_files
        ]

        # Process files in parallel
        processed_files = []
        skip_files = []
        cluster_dict = {}

        if not self.load_to_disk:
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=min(32, mp.cpu_count() * 2)) as executor:
                results = list(
                    tqdm(executor.map(process_file, process_args), total=len(process_args), desc="Processing files")
                )

            # Process results
            for i, (file_path, cluster_info) in enumerate(results):
                if file_path is None:
                    continue

                if cluster_info is None and self.cluster_file is not None:
                    skip_files.append(file_path)
                    continue

                processed_files.append(file_path)

                if self.cluster_file is not None and cluster_info[1] is not None:  # If we have cluster info
                    cluster_id = cluster_info[1]
                    if cluster_id not in cluster_dict:
                        cluster_dict[cluster_id] = []
                    cluster_dict[cluster_id].append(i)

                if self.testing and len(processed_files) > 500:
                    break
        else:
            # Handle in-memory loading case
            logger.info("Loading dataset into memory...")
            if self.use_mmap:
                # Use memory mapping for large dataset files
                self.preloaded_dataset = torch.load(self.load_to_disk_file, map_location="cpu", mmap=True)
                logger.info("Loaded dataset with memory mapping")
            else:
                self.preloaded_dataset = torch.load(self.load_to_disk_file)
                logger.info("Loaded dataset into memory")

            logger.info("Turning to df...")
            self.preloaded_dataset = pd.DataFrame(self.preloaded_dataset)

            for i, p_file in tqdm(self.preloaded_dataset.iterrows(), desc="Processing files"):
                if self.files_to_keep is not None and p_file["name"] not in files_to_keep:
                    skip_files.append(p_file["name"])
                    continue

                processed_files.append(p_file["name"])

                if self.cluster_file is not None:
                    cluster_id = self.cluster_dict[p_file["name"]]
                    if cluster_id not in cluster_dict:
                        cluster_dict[cluster_id] = []
                    cluster_dict[cluster_id].append(i)

                if self.testing and len(processed_files) > 500:
                    break

        self.dataset_filenames = processed_files
        logger.info(f"Loaded {len(self.dataset_filenames)} data points.")
        logger.info(f"Skipped {len(skip_files)} data points.")

        if self.cluster_file is not None:
            min_size = 1
            self.cluster_dict = cluster_dict
            self.cluster_dict = list(self.cluster_dict.values())
            logger.info(f"dataset has prior to removing <{min_size} frequent cluster {len(self.cluster_dict)} clusters")
            self.cluster_dict = merge_small_lists(self.cluster_dict, min_size=min_size)
            logger.info(f"dataset has after removing <{min_size} frequent cluster {len(self.cluster_dict)} clusters")
        else:
            self.cluster_dict = {0: list(range(len(self.dataset_filenames)))}
            self.cluster_dict = list(self.cluster_dict.values())
            logger.info(f"No cluster file provided: dataset has {len(self.cluster_dict)} clusters")

    def __getitem__(self, idx: int) -> tuple:
        """Return the dataset at the given index."""
        if not self.load_to_disk:
            try:
                x = torch.load(self.processed_paths[idx])
            except Exception as e:
                # ic(f"Error loading {self.processed_paths[idx]}: {e}")
                # load the next file if it exists
                if idx + 1 < len(self.processed_paths):
                    return self.__getitem__(idx + 1)
                elif idx - 1 >= 0:
                    return self.__getitem__(idx - 1)
                else:
                    raise e
        else:
            x = self.preloaded_dataset.iloc[idx]
            # If using mmap, ensure the data is properly loaded into memory when accessed
            if self.use_mmap and hasattr(x, "to"):
                x = x.to("cpu")

        if self.transform:
            x = self.transform(x)

        return x
