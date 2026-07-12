"""Base dataset class for biomolecules."""

import glob
import logging
import multiprocessing as mp
import os
import pathlib
import pickle
import time
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


def _collect_file_paths_recursive(root_path: Path, exclude_patterns: list[str], handle_errors: bool = True):
    """
    Recursively collect file paths using os.scandir (fast, single-threaded I/O).

    This only collects paths, deferring stat calls for parallel processing.

    Parameters
    ----------
    root_path : Path
        Directory to scan
    exclude_patterns : list[str]
        Patterns to exclude from filenames
    handle_errors : bool
        If True, log errors and continue. If False, let errors propagate.

    Yields
    ------
    str
        File path that matches criteria
    """
    dir_excludes = {"__pycache__", "cache"}

    try:
        with os.scandir(root_path) as entries:
            for entry in entries:
                try:
                    # Check if it's a directory (don't follow symlinks)
                    if entry.is_dir(follow_symlinks=False):
                        if not entry.name.startswith(".") and entry.name not in dir_excludes:
                            # Recursively scan subdirectory
                            yield from _collect_file_paths_recursive(Path(entry.path), exclude_patterns, handle_errors)
                        continue

                    # Check if it's a file
                    if not entry.is_file(follow_symlinks=False):
                        continue

                    # Filter by extension
                    if not entry.name.endswith(".pt"):
                        continue

                    # Exclude certain patterns
                    if any(pattern in entry.name for pattern in exclude_patterns):
                        continue

                    # Just yield the path, defer stat() call for parallel processing
                    yield entry.path

                except OSError as e:
                    if handle_errors:
                        logger.warning(f"Error accessing {entry.path if hasattr(entry, 'path') else root_path}: {e}")
                        continue
                    else:
                        raise

    except OSError as e:
        if handle_errors:
            logger.warning(f"Error scanning directory {root_path}: {e}")
        else:
            raise


def _get_file_metadata(file_path: str) -> dict | None:
    """
    Worker function to get metadata for a single file path.

    This is designed to be called in parallel via ThreadPoolExecutor.

    Parameters
    ----------
    file_path : str
        Path to the file

    Returns
    -------
    dict | None
        File metadata or None if stat fails
    """
    try:
        stat_info = os.stat(file_path)
        return {
            "path": file_path,
            "size_bytes": stat_info.st_size,
            "mtime": stat_info.st_mtime,
            "stem": Path(file_path).stem,
        }
    except OSError as e:
        logger.warning(f"Could not stat {file_path}: {e}")
        return None


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
    file_path, files_to_keep, cluster_dict, file_metadata = file_info

    # Quick filter for .pt files (skip if metadata provided, as it's pre-filtered)
    if file_metadata is None:
        if not file_path.endswith(".pt") or any(x in file_path for x in ["cluster", "filter", "transform"]):
            return None, None

    fname = Path(file_path).stem

    # Check files_to_keep
    if files_to_keep is not None and fname not in files_to_keep:
        return file_path, None

    # Check file size - use metadata if available to avoid stat call
    if file_metadata is not None:
        if "size_bytes" in file_metadata and file_metadata["size_bytes"] is not None:
            if file_metadata["size_bytes"] == 0:
                return file_path, None
    else:
        try:
            if Path(file_path).stat().st_size == 0:
                return file_path, None
        except OSError:
            return file_path, None

    # Get cluster info if needed
    # Always return cluster_info tuple (fname, cluster_id) when cluster_dict is provided
    # This allows us to distinguish between "file not checked" vs "file not in cluster"
    cluster_info = None
    if cluster_dict is not None:
        cluster_id = cluster_dict.get(fname)  # Can be None if not in cluster
        cluster_info = (fname, cluster_id)

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

    cache_file : str | os.PathLike, optional
        Path to cache file for storing file listings. If None, auto-generates path
        as {processed_dir}/.cache/file_listing_cache.parquet, by default None.

    use_cache : bool, optional
        Whether to use file listing cache to speed up initialization, by default True.

    rebuild_cache : bool, optional
        Whether to force rebuild of cache file, by default False.

    cache_max_age_hours : float, optional
        Maximum age of cache in hours before auto-rebuild. If None, cache never
        expires based on age, by default None.

    skip_stat : bool, optional
        Whether to skip stat calls during cache building (assumes all files exist
        and are non-zero). Dramatically speeds up cache building on slow filesystems.
        Files are validated on first access instead, by default False.

    stat_workers : int, optional
        Number of workers for parallel stat operations. If None, uses cpu_count() * 4.
        Reduce this for network filesystems (try 8-32), by default None.
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
        cache_file: str | os.PathLike = None,
        use_cache: bool = True,
        rebuild_cache: bool = False,
        cache_max_age_hours: float = None,
        skip_stat: bool = True,
        stat_workers: int = None,
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
        self.cache_file = cache_file
        self.use_cache = use_cache
        self.rebuild_cache = rebuild_cache
        self.cache_max_age_hours = cache_max_age_hours
        self.skip_stat = skip_stat
        self.stat_workers = stat_workers
        logger.info(f"Loading data from {self.root}")
        self._load_data()
        logger.info("Loaded data points.")

        # For large datasets, skip PyG's expensive __init__ operations
        if len(self.dataset_filenames) > 100000:
            logger.info(
                f"Large dataset detected ({len(self.dataset_filenames)} files), using lightweight initialization"
            )
            # Call object.__init__ directly to skip PyG's validation/processing logic
            # This bypasses all the expensive PyG checks for huge datasets
            object.__init__(self)
            self._transform = transform
            self._pre_transform = pre_transform
            # Set PyG internal attributes that are normally set in Dataset.__init__
            self._indices = None
            self.__dict__["root"] = str(root)  # Ensure root is set
            logger.info("Initialization complete (bypassed PyG overhead)")
        else:
            # Normal PyG initialization for small datasets.
            # In DDP, PyG's _process() races to read/write `pre_transform.pt` and
            # `pre_filter.pt` under `root`. Concurrent torch.save from one rank +
            # torch.load from another can leave readers seeing a partial file,
            # surfacing as: "<path>/pre_transform.pt is a zip archive (did you
            # mean to use torch.jit.load()?)". We avoid this by letting rank 0
            # initialize first (writing/validating the cache), then barriering,
            # then letting the remaining ranks proceed against the now-stable
            # cache. Falls back to plain init when distributed isn't up.
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                if rank == 0:
                    super().__init__(root, transform, pre_transform)
                dist.barrier()
                if rank != 0:
                    super().__init__(root, transform, pre_transform)
                dist.barrier()
            else:
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
    def processed_paths(self):
        """Override PyG's processed_paths to return actual file paths from cache."""
        # For large datasets loaded from cache, dataset_filenames already contains full paths
        # Don't let PyG construct paths by joining processed_dir + filename
        return self.dataset_filenames

    @property
    def transform(self):
        """Handle transform for both PyG and lightweight init."""
        return getattr(self, "_transform", None)

    @transform.setter
    def transform(self, value):
        """Handle transform setter for both PyG and lightweight init."""
        self._transform = value

    @property
    def get_cluster_dict(self):
        return self.cluster_dict

    @property
    def processed_file_names(self) -> list[str]:
        """Return list of processed files (ending with `.pt`)."""
        if len(self.dataset_filenames) > 100000:  # Large dataset threshold
            return []  # PyG won't try to check files
        # use both dataset_filenames and identifiers to create processed file names assums .cif or .pdb ending for strucs
        return [f"{self.dataset_filenames[i]}" for i, f in enumerate(self.dataset_filenames)]

    def len(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset_filenames)

    def __len__(self) -> int:
        """Return the number of examples in the dataset. Required for PyTorch DataLoader."""
        return len(self.dataset_filenames)

    def indices(self):
        """Return indices for the dataset. Required for PyG compatibility."""
        # Handle both PyG and lightweight initialization
        _indices = getattr(self, "_indices", None)
        if _indices is None:
            return range(self.len())
        return _indices

    def process(self):
        # Process datasets into pt files
        return

    def _get_cache_path(self) -> Path:
        """Determine cache file path."""
        if self.cache_file is not None:
            return Path(self.cache_file)

        # Auto-generate cache path
        cache_dir = Path(self.processed_dir) / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "file_listing_cache.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is valid and should be used."""
        # Force rebuild if requested
        if self.rebuild_cache:
            logger.info("Cache rebuild requested, will rebuild cache.")
            return False

        # Check if cache exists
        if not cache_path.exists():
            logger.info("Cache file does not exist.")
            return False

        # Check age if max_age is set
        if self.cache_max_age_hours is not None:
            cache_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if cache_age_hours > self.cache_max_age_hours:
                logger.info(f"Cache is {cache_age_hours:.1f} hours old (max: {self.cache_max_age_hours}), rebuilding.")
                return False

        # Validate cache contents
        try:
            cache_data = pd.read_parquet(cache_path)
            if "metadata" not in cache_data.columns:
                logger.warning("Cache file missing metadata, rebuilding.")
                return False

            metadata = cache_data["metadata"].iloc[0]
            if metadata.get("processed_dir") != str(self.processed_dir):
                logger.warning(
                    f"Cache processed_dir mismatch (cached: {metadata.get('processed_dir')}, current: {self.processed_dir}), rebuilding."
                )
                return False

            logger.info(f"Cache is valid with {metadata.get('file_count', 0)} files.")
            return True
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}, rebuilding.")
            return False

    def _scan_files_from_disk(self) -> list[dict]:
        """
        Scan filesystem and return file metadata using parallel processing.

        Two-stage approach:
        1. Collect file paths (single-threaded, I/O bound)
        2. Get file metadata with parallel stat calls (multi-threaded) OR skip if skip_stat=True
        """
        logger.info(f"Scanning files from disk in {self.processed_dir}...")
        start_time = time.time()

        exclude_patterns = ["cluster", "filter", "transform"]

        # Stage 1: Collect file paths (fast directory traversal)
        logger.info("Stage 1: Discovering file paths...")
        stage1_start = time.time()

        # Use Python-based scanning
        file_path_generator = _collect_file_paths_recursive(
            Path(self.processed_dir), exclude_patterns, handle_errors=True
        )

        # Collect all paths into a list (needed for parallel processing)
        file_paths = []
        with tqdm(desc="Discovering paths", unit=" paths", mininterval=0.5) as pbar:
            for path in file_path_generator:
                file_paths.append(path)
                pbar.update(1)

        stage1_duration = time.time() - stage1_start
        logger.info(f"Stage 1 complete: Found {len(file_paths)} file paths in {stage1_duration:.2f}s")

        # Stage 2: Get file metadata
        if self.skip_stat:
            # Fast path: Skip stat calls, assume files exist and are valid
            logger.info("Stage 2: Skipping stat calls (skip_stat=True)")
            logger.warning("Files will be validated on first access. Invalid files may cause errors later.")
            files = [
                {
                    "path": path,
                    "size_bytes": None,  # Unknown, will be checked on access
                    "mtime": None,  # Unknown
                    "stem": Path(path).stem,
                }
                for path in file_paths
            ]
            stage2_duration = 0.0
        else:
            # Normal path: Parallel stat calls to get file metadata
            max_workers = self.stat_workers if self.stat_workers is not None else min(mp.cpu_count() * 4, 128)
            logger.info(f"Stage 2: Getting file metadata with {max_workers} workers...")

            if max_workers <= 32:
                logger.info(f"Using reduced worker count ({max_workers}) - optimized for network filesystems")

            stage2_start = time.time()

            files = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Use chunksize for better performance
                chunksize = max(1, len(file_paths) // (max_workers * 10))

                # Map file paths to metadata
                results = executor.map(_get_file_metadata, file_paths, chunksize=chunksize)

                # Collect results with progress bar
                with tqdm(
                    results, total=len(file_paths), desc="Processing metadata", unit=" files", mininterval=0.5
                ) as pbar:
                    for metadata in pbar:
                        if metadata is not None:  # Skip files that failed stat
                            files.append(metadata)

            stage2_duration = time.time() - stage2_start
            logger.info(f"Stage 2 complete: Processed {len(files)} files in {stage2_duration:.2f}s")

        total_duration = time.time() - start_time
        logger.info(f"Total scan time: {total_duration:.2f}s ({len(file_paths) / total_duration:.0f} files/sec)")
        return files

    def _save_cache(self, file_data: list[dict], cache_path: Path):
        """Save file listing to cache."""
        logger.info(f"Saving cache to {cache_path}...")
        start_time = time.time()

        try:
            # Create metadata
            metadata = {
                "created_at": time.time(),
                "processed_dir": str(self.processed_dir),
                "file_count": len(file_data),
                "total_size_bytes": sum(f["size_bytes"] for f in file_data if f["size_bytes"] is not None),
                "scan_duration_seconds": time.time() - start_time,
            }

            # Convert to DataFrame
            df = pd.DataFrame(file_data)
            # Add metadata as a column (store as dict in first row)
            df["metadata"] = None
            df.at[0, "metadata"] = metadata

            # Save to parquet
            df.to_parquet(cache_path, engine="pyarrow", compression="snappy")

            duration = time.time() - start_time
            logger.info(f"Cache saved with {len(file_data)} files in {duration:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self, cache_path: Path) -> list[dict]:
        """Load file listing from cache."""
        logger.info(f"Loading from cache: {cache_path}")
        start_time = time.time()

        try:
            df = pd.read_parquet(cache_path, engine="pyarrow")
            # Remove metadata column
            metadata = df["metadata"].iloc[0] if "metadata" in df.columns else {}
            df = df.drop(columns=["metadata"], errors="ignore")

            # Convert to list of dicts
            file_data = df.to_dict("records")

            duration = time.time() - start_time
            logger.info(
                f"Cache loaded: {len(file_data)} files in {duration:.2f}s (original scan took {metadata.get('scan_duration_seconds', 'unknown')}s)"
            )
            return file_data
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            raise

    def _load_data(self):
        """Load the dataset from the processed files."""
        # Load cluster file
        if self.cluster_file is not None:
            self.cluster_dict = torch.load(self.cluster_file)
            logger.info(f"Loaded cluster file {self.cluster_file} with {len(self.cluster_dict)} clusters.")

        # Load files to keep (convert to set for O(1) lookup)
        files_to_keep = None
        if self.files_to_keep is not None:
            with open(self.files_to_keep, "rb") as f:
                files_to_keep_list = pickle.load(f)
                files_to_keep = set(files_to_keep_list) if isinstance(files_to_keep_list, list) else files_to_keep_list
            logger.info(f"Using files_to_keep with currently {len(files_to_keep)} files to keep")

        # Get file listings - use cache if enabled and not loading to disk
        if not self.load_to_disk and self.use_cache:
            cache_path = self._get_cache_path()

            if self._is_cache_valid(cache_path):
                # Load from cache
                file_data = self._load_cache(cache_path)
            else:
                # Scan from disk and save to cache
                file_data = self._scan_files_from_disk()
                self._save_cache(file_data, cache_path)

            # Convert file_data list of dicts to list of paths for compatibility
            all_files = [f["path"] for f in file_data]
            # Store file metadata for potential future use
            self._file_metadata = {f["path"]: f for f in file_data}
        elif not self.load_to_disk:
            # Cache disabled, use traditional glob method
            logger.info("Cache disabled, using glob to find files...")
            all_files = glob.glob(str(Path(self.processed_dir) / "**/*.pt"), recursive=True)
        else:
            # For load_to_disk mode, we'll handle file loading separately
            all_files = []

        # Prepare arguments for parallel processing
        # Include file metadata if available (from cache)
        if hasattr(self, "_file_metadata"):
            logger.info(f"Using file metadata from cache with {len(self._file_metadata)} files")
            process_args = [
                (
                    f,
                    files_to_keep,
                    self.cluster_dict if self.cluster_file is not None else None,
                    self._file_metadata.get(f),
                )
                for f in all_files
            ]
        else:
            logger.info("No file metadata from cache, using None")
            process_args = [
                (f, files_to_keep, self.cluster_dict if self.cluster_file is not None else None, None)
                for f in all_files
            ]

        # Process files in parallel
        processed_files = []
        skip_files = []
        cluster_dict = {}

        if not self.load_to_disk:
            # Use ThreadPoolExecutor for I/O bound operations
            # Use stat_workers if set, otherwise use conservative defaults
            if self.stat_workers is not None:
                max_workers = self.stat_workers
            else:
                # Conservative defaults to avoid OOM on multi-node training
                max_workers = min(128, mp.cpu_count() * 4) if len(process_args) > 10000 else min(32, mp.cpu_count() * 2)
            logger.info(f"Using {max_workers} workers for parallel processing (available CPUs: {mp.cpu_count()})")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Use chunksize for better performance with large datasets
                chunksize = max(1, len(process_args) // (max_workers * 10))
                results = list(
                    tqdm(
                        executor.map(process_file, process_args, chunksize=chunksize),
                        total=len(process_args),
                        desc="Processing files",
                        mininterval=1.0,  # Reduce progress bar update frequency
                    )
                )

            # Process results
            for i, (file_path, cluster_info) in enumerate(results):
                if file_path is None:
                    continue

                # Only skip files that were checked against cluster_dict but not found in it
                # cluster_info is None means: file was filtered for other reasons (files_to_keep, size, etc)
                # cluster_info[1] is None means: file was checked but not in cluster_dict
                if cluster_info is not None and cluster_info[1] is None and self.cluster_file is not None:
                    skip_files.append(file_path)
                    continue

                processed_files.append(file_path)

                # Add to cluster if we have cluster info and the file is in a cluster
                if self.cluster_file is not None and cluster_info is not None and cluster_info[1] is not None:
                    cluster_id = cluster_info[1]
                    if cluster_id not in cluster_dict:
                        cluster_dict[cluster_id] = []
                    # index into processed_files (the filtered list that becomes
                    # dataset_filenames), NOT the full-results enumerate index `i`:
                    # skipped/None files make `i` overshoot len(processed_files),
                    # producing cluster indices past the dataset end (IndexError in
                    # ConcatDataset) for any source whose cluster file omits some files
                    # (e.g. afdb_homo: ~48k of 1.93M lack a seqid40 assignment).
                    cluster_dict[cluster_id].append(len(processed_files) - 1)

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
                    # index into processed_files (the filtered list that becomes
                    # dataset_filenames), NOT the full-results enumerate index `i`:
                    # skipped/None files make `i` overshoot len(processed_files),
                    # producing cluster indices past the dataset end (IndexError in
                    # ConcatDataset) for any source whose cluster file omits some files
                    # (e.g. afdb_homo: ~48k of 1.93M lack a seqid40 assignment).
                    cluster_dict[cluster_id].append(len(processed_files) - 1)

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
            # Use dataset_filenames directly (already full paths from cache)
            # instead of processed_paths which PyG constructs incorrectly
            try:
                file_path = self.dataset_filenames[idx]
                x = torch.load(file_path)
            except Exception as e:
                logger.error(
                    f"Error loading {self.dataset_filenames[idx] if idx < len(self.dataset_filenames) else 'unknown'}: {e}"
                )
                # load the next file if it exists
                if idx + 1 < len(self.dataset_filenames):
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

        # Handle transform (works with both PyG's and our lightweight init)
        if self.transform:
            x = self.transform(x)

        return x
