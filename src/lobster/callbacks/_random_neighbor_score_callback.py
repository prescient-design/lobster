from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import (
    AMPLIFYIterableDataset,
    CalmIterableDataset,
    M320MIterableDataset,
    OpenGenome2IterableDataset,
    RandomSequenceDataset,
    ZINCIterableDataset,
)
from ..metrics._random_neighbor_score import RandomNeighborScore


class RandomNeighborScoreCallback(Callback):
    """
    Callback for evaluating embedding models using Random Neighbor Score (RNS).

    RNS quantifies the fraction of non-biological (randomly shuffled) sequence
    embeddings among the k-nearest neighbours of a protein embedding.
    Higher RNS → greater representational uncertainty; lower is better.

    This version supports pre-loaded dataset options for biological embeddings
    and automatically generates random sequences from the extracted vocabulary.

    The model must implement an `embed_sequences` method that takes sequences
    and a modality parameter and returns embeddings. To use with other
    models, you may need to override `_get_embeddings`.

    Reference
    ---------
    Prabakaran R, Bromberg Y. "Quantifying uncertainty in Protein Representations
    Across Models and Task." bioRxiv (2025) doi:10.1101/2025.04.30.651545

    Examples
    --------
    Basic usage with a trainer:

    >>> from lobster.callbacks import RandomNeighborScoreCallback
    >>> from lobster.model import Ume
    >>>
    >>> # Initialize callback
    >>> callback = RandomNeighborScoreCallback(
    ...     dataset_name="AMPLIFY",
    ...     k=100,
    ...     biological_dataset_limit=500,
    ...     seed=42,
    ...     split="test"
    ... )
    >>>
    >>> # Use with trainer
    >>> trainer = L.Trainer(callbacks=[callback])
    >>> trainer.fit(model, datamodule)

    Standalone evaluation:

    >>> # Load your model
    >>> model = Ume(model_name="UME_mini", use_flash_attn=False)
    >>>
    >>> # Evaluate
    >>> metrics = callback.evaluate(model, trainer=None)
    >>> rns_score = metrics["random_neighbor_score"]
    >>> print(f"Random Neighbor Score: {rns_score:.4f}")
    """

    SUPPORTED_DATASETS = {
        "AMPLIFY": AMPLIFYIterableDataset,
        "Calm": CalmIterableDataset,
        "M320M": M320MIterableDataset,
        "ZINC": ZINCIterableDataset,
        "OpenGenome2": OpenGenome2IterableDataset,
    }

    def __init__(
        self,
        dataset_name: Literal["AMPLIFY", "Calm", "M320M", "ZINC", "OpenGenome2"],
        k: int = 500,
        distance_metric: Literal["cosine", "euclidean"] = "cosine",
        run_every_n_epochs: int | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        num_random_sequences: int = 1000,
        random_seq_min_length: int = 50,
        random_seq_max_length: int = 500,
        biological_dataset_limit: int | None = 1000,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "test",
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the biological dataset to use. Must be one of the supported datasets.
        k : int
            Number of nearest neighbors to consider for RNS calculation
        distance_metric : str
            Distance metric to use ("cosine" or "euclidean")
        run_every_n_epochs : int | None
            Run evaluation every N epochs. If None, runs every epoch.
        batch_size : int
            Batch size for dataloader
        num_workers : int
            Number of workers for dataloader
        num_random_sequences : int
            Number of random sequences to generate
        random_seq_min_length : int
            Minimum length for random sequences
        random_seq_max_length : int
            Maximum length for random sequences
        biological_dataset_limit : int | None
            Limit number of biological sequences to use. If None, uses all.
        root : str | Path | None
            Root directory for dataset storage
        split : Literal["train", "val", "test"]
            Split of the biological dataset to use.
        seed : int
            Random seed for reproducibility
        """
        super().__init__()

        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. Choose from: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.k = k
        self.distance_metric = distance_metric
        self.run_every_n_epochs = run_every_n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_random_sequences = num_random_sequences
        self.random_seq_min_length = random_seq_min_length
        self.random_seq_max_length = random_seq_max_length
        self.biological_dataset_limit = biological_dataset_limit
        self.root = root
        self.split = split
        self.seed = seed

        # Initialize dataloaders during construction
        self._create_dataloaders()

    def _extract_vocabulary_from_dataset(self, dataset) -> set[str]:
        """Extract unique characters from dataset sequences."""
        vocab = set()

        # Sample sequences to extract vocabulary
        sample_count = 0
        max_samples = 1000  # Limit sampling for efficiency

        for item in dataset:
            if sample_count >= max_samples:
                break

            # Handle different dataset output formats
            if isinstance(item, str):
                sequence = item
            elif isinstance(item, tuple) and len(item) > 0:
                sequence = item[0]  # Assume first element is sequence
            else:
                continue

            # Add characters to vocabulary
            vocab.update(set(sequence))
            sample_count += 1

        # Filter out common special characters and keep only valid sequence characters
        # This is a heuristic - adjust based on your specific datasets
        valid_chars = set()
        for char in vocab:
            if char.isalpha() or char in {"-", ".", "*"}:  # Common sequence characters
                valid_chars.add(char.upper())  # Normalize to uppercase

        return valid_chars

    def _get_sequence_key_for_dataset(self) -> str:
        """Get the appropriate sequence key/column name for each dataset."""
        dataset_class = self.SUPPORTED_DATASETS[self.dataset_name]
        return dataset_class.SEQUENCE_KEY

    def _get_modality_for_dataset(self) -> str:
        """Determine the appropriate modality based on the dataset type."""
        dataset_class = self.SUPPORTED_DATASETS[self.dataset_name]
        return dataset_class.MODALITY

    def _create_dataloaders(self):
        """Create biological and random dataloaders."""
        # Create biological dataset
        dataset_class = self.SUPPORTED_DATASETS[self.dataset_name]
        sequence_key = self._get_sequence_key_for_dataset()

        # Create dataset with only the sequence column using the dataset's SEQUENCE_KEY
        biological_dataset = dataset_class(
            root=self.root,
            download=True,
            shuffle=False,
            limit=self.biological_dataset_limit,
            split=self.split,
            keys=[sequence_key],
        )

        # Extract vocabulary from biological dataset
        vocab = self._extract_vocabulary_from_dataset(biological_dataset)

        if not vocab:
            raise ValueError("Could not extract vocabulary from biological dataset")

        # Create random sequence dataset
        random_dataset = RandomSequenceDataset(
            vocab=vocab,
            num_sequences=self.num_random_sequences,
            min_length=self.random_seq_min_length,
            max_length=self.random_seq_max_length,
            seed=self.seed,
        )

        # Create dataloaders
        self.biological_dataloader = DataLoader(
            biological_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.random_dataloader = DataLoader(
            random_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip validation this epoch."""
        if self.run_every_n_epochs is None:
            return False

        if trainer.global_rank != 0:
            return True

        return trainer.current_epoch % self.run_every_n_epochs != 0

    def _get_embeddings(self, model: L.LightningModule | torch.nn.Module, dataloader: DataLoader) -> Tensor:
        """Extract embeddings from the model for a given dataloader.

        The model must implement an `embed_sequences` method that accepts:
        - sequences: list of sequence strings
        - modality: string indicating the sequence type
        - aggregate: boolean for aggregation (set to True)

        Parameters
        ----------
        model : L.LightningModule | torch.nn.Module
            The model to extract embeddings from. Must have `embed_sequences` method.
        dataloader : DataLoader
            DataLoader for the data to extract embeddings for

        Returns
        -------
        Tensor
            Embeddings tensor of shape (n_samples, embedding_dim)

        Raises
        ------
        NotImplementedError
            If the model does not have an `embed_sequences` method
        """
        embeddings = []

        model.eval()

        with torch.no_grad():
            for sequences in tqdm(dataloader):
                # Skip empty batches
                if not sequences:
                    continue

                # Use the model's embed_sequences method
                if hasattr(model, "embed_sequences"):
                    # Determine modality based on dataset type
                    modality = self._get_modality_for_dataset()
                    batch_embeddings = model.embed_sequences(sequences, modality, aggregate=True)
                    embeddings.append(batch_embeddings.cpu())
                else:
                    # Fallback for other model types
                    raise NotImplementedError("Model embedding extraction not implemented for this model type")

        return torch.cat(embeddings) if embeddings else torch.empty(0, 0)

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, float]:
        """Evaluate the model using Random Neighbor Score.

        This method can be used both during training (with a trainer) and
        standalone (with just a model).

        Parameters
        ----------
        module : L.LightningModule
            The model to evaluate. Must implement `embed_sequences` method.
        trainer : L.Trainer | None
            Optional trainer for logging metrics

        Returns
        -------
        dict[str, float]
            Dictionary containing the RNS score

        Examples
        --------
        >>> from lobster.model import Ume
        >>> model = Ume(model_name="UME_mini")
        >>> callback = RandomNeighborScoreCallback(dataset_name="AMPLIFY")
        >>> metrics = callback.evaluate(model)
        >>> print(f"RNS: {metrics['random_neighbor_score']:.4f}")
        """
        biological_embeddings = self._get_embeddings(module, self.biological_dataloader)
        random_embeddings = self._get_embeddings(module, self.random_dataloader)

        # Check if we have enough embeddings
        if biological_embeddings.size(0) == 0 or random_embeddings.size(0) == 0:
            return {"random_neighbor_score": float("nan")}

        rns_metric = RandomNeighborScore(
            biological_embeddings=biological_embeddings,
            random_embeddings=random_embeddings,
            k=self.k,
            distance_metric=self.distance_metric,
        )

        query_embeddings = biological_embeddings
        rns_metric.update(query_embeddings)
        rns_score = rns_metric.compute()

        metrics = {"random_neighbor_score": rns_score.item()}

        if trainer is not None:
            trainer.logger.experiment.log(
                {
                    "val/random_neighbor_score": rns_score.item(),
                    "val/global_step": trainer.global_step,
                }
            )

        return metrics

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Evaluate Random Neighbor Score at specified epochs."""
        if self._skip(trainer):
            return

        self.evaluate(pl_module, trainer)
