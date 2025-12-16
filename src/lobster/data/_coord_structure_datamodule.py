from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.model.latent_generator.datamodules._utils import collate_fn_backbone
from lobster.datasets._ligand_dataset import LigandDataset
from lobster.datasets._sampler import RandomizedMinorityUpsampler
from lobster.datasets._structure_dataset import StructureDataset
from lobster.datasets._structure_dataset_iterable import ShardedStructureDataset
from lobster.transforms._structure_transforms import StructureBackboneTransform, StructureLigandTransform

try:
    from torch_geometric.transforms import Compose
except ImportError:
    Compose = None

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StructureLightningDataModule(LightningDataModule):
    def __init__(
        self,
        path_to_datasets: str | list[str] = None,
        root: str | Path = None,
        datasets: Sequence[str] = None,
        *,
        transform_fn: Iterable[Callable] = None,
        ligand_transforms: Iterable[Callable] = None,
        lengths: Sequence[float] | None = (0.9, 0.05, 0.05),
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Iterable | Sampler | None = None,
        cluster_file: str | None = None,
        cluster_file_list: list[str] | None = None,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[T]], Any] | None = collate_fn_backbone,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
        is_relative_model: bool = False,
        mlm: bool = True,
        repeat_count: int = 1,
        testing: bool = False,
        files_to_keep: str | None = None,
        files_to_keep_list: list[str] | None = None,
        use_shards: bool = False,
        use_ligand_dataset: bool = False,
        buffer_size: int = 5,
    ) -> None:
        """:param path_to_datasets: path to data set directories

        :param model_name: name of esm model

        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.


        :param use_transform_fn: If ``True``, use transform_fn for dataset
            tokenization, else no transform.

        :param lengths: Fractions of splits to generate.

        :param generator: Generator used for the random permutation (default:
            ``None``).

        :param seed: Desired seed. Value must be within the inclusive range
            ``[-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]`` (default:
            ``0xDEADBEEF``). Otherwise, a ``RuntimeError`` is raised. Negative
            inputs are remapped to positive values with the formula
            ``0xFFFFFFFFFFFFFFFF + seed``.

        :param batch_size: Samples per batch (default: ``1``).

        :param shuffle: If ``True``, reshuffle datasets at every epoch (default:
            ``True``).

        :param sampler: Strategy to draw samples from the dataset (default:
            ``None``). Can be any ``Iterable`` with ``__len__`` implemented.
            If specified, :attr:`shuffle` must be ``False``.

        :param batch_sampler: :attr:`sampler`, but returns a batch of indices
            (default: ``None``). Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.

        :param num_workers: Subprocesses to use (default: ``0``). ``0`` means
            that the datasets will be loaded in the main process.

        :param collate_fn: Merges samples to form a mini-batch of Tensor(s)
            (default: ``None``).

        :param pin_memory: If ``True``, Tensors are copied to the device's
            (e.g., CUDA) pinned memory before returning them (default:
            ``True``).

        :param drop_last: If ``True``, drop the last incomplete batch, if the
            dataset size is not divisible by the batch size (default:
            ``False``). If ``False`` and the size of dataset is not divisible
            by the batch size, then the last batch will be smaller.


        :param is_relative_model: If ``True``, assumes training between two sequences
            and calls a relative representation data loader

        """
        transforms = transform_fn
        super().__init__()

        import lobster

        lobster.ensure_package("torch_geometric", group="struct-gpu (or --extra struct-cpu)")

        if lengths is None:
            lengths = [0.4, 0.4, 0.2]

        if generator is None:
            generator = Generator().manual_seed(seed)

        if path_to_datasets is None:
            path_to_datasets = [
                "/data/lisanzas/AF3_dataset/processed/pdb/train/",
                "/data/lisanzas/AF3_dataset/processed/pdb/val/",
                "/data/lisanzas/AF3_dataset/processed/pdb/test/",
            ]
        elif isinstance(path_to_datasets, str):
            path_to_datasets = [path_to_datasets]

        self._path_to_datasets = path_to_datasets
        self._root = root
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._shuffle = shuffle
        self._sampler = sampler
        if self._sampler is not None:
            self._shuffle = False
        self._cluster_file = cluster_file
        self._cluster_file_list = cluster_file_list
        self._files_to_keep = files_to_keep
        self._files_to_keep_list = files_to_keep_list
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        logger.info(f"using collate_fn: {collate_fn}")
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._is_relative_model = is_relative_model
        self._mlm = mlm
        self.repeat_count = repeat_count
        self.testing = testing
        if self.testing and not use_shards:
            self._path_to_datasets = [
                "/data/lisanzas/structure_tokenizer/studies/data/pinder_raw_pdbs_bb_coords/train_dummy.pt",
                "/data/lisanzas/structure_tokenizer/studies/data/pinder_raw_pdbs_bb_coords/val.pt",
                "/data/lisanzas/structure_tokenizer/studies/data/pinder_raw_pdbs_bb_coords/test.pt",
            ]
            self._cluster_file = None
            self._files_to_keep = None
        self.use_shards = use_shards
        self.buffer_size = buffer_size
        self.use_ligand_dataset = use_ligand_dataset
        if transforms is None:
            logger.info("No transform function provided. Using default transform function: StructureBackboneTransform")
            self._transform_fn = StructureBackboneTransform(max_length=max_length)
        else:
            logger.info("Using custom transform function.")
            transforms = list(transforms.values())
            self._transform_fn = self.compose_transforms(transforms)

        if ligand_transforms is None:
            logger.info(
                "No ligand transform function provided. Using default transform function: StructureLigandTransform"
            )
            self._ligand_transform_fn = StructureLigandTransform(max_length=max_length)
        else:
            logger.info("Using custom ligand transform function.")
            ligand_transforms = list(ligand_transforms.values())
            self._ligand_transform_fn = self.compose_transforms(ligand_transforms)

        logger.info(
            f"SequenceLightningDataModule: path_to_datasets={path_to_datasets}, root={root}, lengths={lengths}, seed={seed}, batch_size={batch_size}, max_length={max_length}, shuffle={shuffle}, sampler={sampler}, batch_sampler={batch_sampler}, num_workers={num_workers}, collate_fn={collate_fn}, use_shards={use_shards}"
        )

    def _create_dataset(
        self, path: str, is_train: bool = False, cluster_file: str | None = None, files_to_keep: str | None = None
    ) -> StructureDataset | ShardedStructureDataset:
        """Create either a regular or sharded dataset based on configuration."""
        if cluster_file is None:
            cluster_file = self._cluster_file
        if files_to_keep is None:
            files_to_keep = self._files_to_keep
        logger.info(f"Creating dataset from {path} with cluster_file {cluster_file} and files_to_keep {files_to_keep}")
        if self.use_shards:
            logger.info(f"Creating sharded dataset from {path}")
            return ShardedStructureDataset(
                root=path,
                buffer_size=self.buffer_size,
                transform=self._transform_fn,
                shuffle=self._shuffle,
                testing=self.testing,
            )
        else:
            logger.info(f"Creating regular dataset from {path}")
            if self.use_ligand_dataset:
                logger.info(f"Creating ligand dataset from {path}")
                return LigandDataset(
                    root=path,
                    transform_protein=self._transform_fn,
                    transform_ligand=self._ligand_transform_fn,
                )
            else:
                logger.info(f"Creating structure dataset from {path}")
                return StructureDataset(
                    root=path,
                    transform=self._transform_fn,
                    testing=self.testing,
                    cluster_file=cluster_file if is_train else None,
                    files_to_keep=files_to_keep,
                )

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            contains_train = (
                any("train" in path.split("/")[-2:] for path in self._path_to_datasets)
                or any("train" in path.split("/")[-1] for path in self._path_to_datasets)
                or any("train_shards" in path.split("/")[-2:] for path in self._path_to_datasets)
            )
            contains_val = (
                any("val" in path.split("/")[-2:] for path in self._path_to_datasets)
                or any("val" in path.split("/")[-1] for path in self._path_to_datasets)
                or any("val_shards" in path.split("/")[-2:] for path in self._path_to_datasets)
            )
            contains_test = (
                any("test" in path.split("/")[-2:] for path in self._path_to_datasets)
                or any("test" in path.split("/")[-1] for path in self._path_to_datasets)
                or any("test_shards" in path.split("/")[-2:] for path in self._path_to_datasets)
            )
            all_present = contains_train and contains_val and contains_test

            if all_present:  # pre computed splits
                logger.info("Using precomputed splits.")
                if self.use_shards:
                    # For sharded datasets, use the first matching path
                    train_path = next(p for p in self._path_to_datasets if "train" in p)
                    val_path = next(p for p in self._path_to_datasets if "val" in p)
                    test_path = next(p for p in self._path_to_datasets if "test" in p)

                    self._train_dataset = self._create_dataset(train_path, is_train=True)
                    self._val_dataset = self._create_dataset(val_path)
                    self._test_dataset = self._create_dataset(test_path)
                else:
                    # For regular datasets, use ConcatDataset
                    if self._cluster_file_list is not None:
                        self._train_dataset = torch.utils.data.ConcatDataset(
                            # [self._create_dataset(p, is_train=True) for p in self._path_to_datasets if "train" in p]
                            [
                                self._create_dataset(
                                    self._path_to_datasets[j],
                                    is_train=True,
                                    cluster_file=self._cluster_file_list[j],
                                    files_to_keep=self._files_to_keep_list[j],
                                )
                                for j in range(len(self._path_to_datasets))
                                if "train" in self._path_to_datasets[j]
                            ]
                        )
                    else:
                        self._train_dataset = torch.utils.data.ConcatDataset(
                            [self._create_dataset(p, is_train=True) for p in self._path_to_datasets if "train" in p]
                        )

                    self._val_dataset = torch.utils.data.ConcatDataset(
                        [self._create_dataset(p) for p in self._path_to_datasets if "val" in p]
                    )
                    self._test_dataset = torch.utils.data.ConcatDataset(
                        [self._create_dataset(p) for p in self._path_to_datasets if "test" in p]
                    )
            else:  # iid split
                logger.info("Using iid splits.")
                if self.use_shards:
                    # For sharded datasets, use the first path
                    dataset = self._create_dataset(self._path_to_datasets[0], is_train=True)
                    # Calculate split sizes
                    total_size = len(dataset)
                    train_size = int(total_size * self._lengths[0])
                    val_size = int(total_size * self._lengths[1])
                    test_size = total_size - train_size - val_size

                    self._train_dataset, self._val_dataset, self._test_dataset = torch.utils.data.random_split(
                        dataset,
                        [train_size, val_size, test_size],
                        generator=self._generator,
                    )
                else:
                    # For regular datasets, use ConcatDataset
                    datasets = [self._create_dataset(p, is_train=True) for p in self._path_to_datasets]
                    dataset = torch.utils.data.ConcatDataset(datasets)
                    (
                        self._train_dataset,
                        self._val_dataset,
                        self._test_dataset,
                    ) = torch.utils.data.random_split(
                        dataset,
                        lengths=self._lengths,
                        generator=self._generator,
                    )

        if stage == "predict":
            if self.use_shards:
                # For sharded datasets, use the first path
                self._predict_dataset = self._create_dataset(self._path_to_datasets[0])
            else:
                # For regular datasets, use ConcatDataset
                self._path_to_datasets = [self._path_to_datasets[0]] * self.repeat_count
                datasets = [self._create_dataset(p) for p in self._path_to_datasets]
                self._predict_dataset = torch.utils.data.ConcatDataset(datasets)

    def train_dataloader(self) -> DataLoader:
        if not self.use_shards and isinstance(self._sampler, (functools.partial, RandomizedMinorityUpsampler)):
            group_indices = []
            cumulative_size = 0
            for dataset in self._train_dataset.datasets:
                # convert local indices to global indices
                global_clusters = []
                for cluster in dataset.get_cluster_dict:
                    global_cluster = [idx + cumulative_size for idx in cluster]
                    global_clusters.append(global_cluster)
                # group_indices.extend(dataset.get_cluster_dict)
                # ic(dataset.get_cluster_dict)
                group_indices.extend(global_clusters)
                cumulative_size += len(dataset)

            if isinstance(self._sampler, functools.partial):
                self._sampler = self._sampler(group_indices)
            else:
                self._sampler = RandomizedMinorityUpsampler(group_indices)
            logger.info(f"Train dataloader using RandomizedMinorityUpsampler with {len(group_indices)} clusters")
        else:
            logger.info("Using standard sampling strategy")

        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle if not self.use_shards else False,  # Shuffle is handled by ShardedStructureDataset
            sampler=self._sampler if not self.use_shards else None,  # Sampler not used with sharded dataset
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        # Only log if we're actually in a validation step
        if hasattr(self, "trainer") and self.trainer.state.stage == "validate":
            if not self.use_shards and isinstance(self._sampler, (functools.partial, RandomizedMinorityUpsampler)):
                group_indices = []
                for dataset in self._val_dataset.datasets:
                    group_indices.extend(dataset.get_cluster_dict)
                if isinstance(self._sampler, functools.partial):
                    self._sampler = self._sampler(group_indices)
                else:
                    self._sampler = RandomizedMinorityUpsampler(group_indices)
                logger.info(f"Val dataloader using RandomizedMinorityUpsampler with {len(group_indices)} clusters")
            else:
                logger.info("Using standard sampling strategy")

        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler if not self.use_shards else None,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler if not self.use_shards else None,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._predict_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler if not self.use_shards else None,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

    def compose_transforms(self, transforms: Iterable[Callable]) -> Compose:
        """Compose an iterable of Transforms into a single transform.

        Parameters
        ----------
        transforms : Iterable[Callable]
            An iterable of transforms.

        Raises
        ------
        ValueError
            If ``transforms`` is not a list or dict.

        Returns
        -------
        T.Compose
            A single transform.
        """
        if isinstance(transforms, list):
            return Compose(transforms)
        elif isinstance(transforms, dict):
            return Compose(list(transforms.values()))
        else:
            raise ValueError("Transforms must be a list or dict")
