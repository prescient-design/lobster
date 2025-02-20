from collections import defaultdict
from typing import Optional, Sequence, Tuple

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lobster.constants import CALM_TASKS
from lobster.datasets import CalmPropertyDataset
from lobster.tokenization import NucleotideTokenizerFast
from lobster.transforms import TokenizerTransform

from ._linear_probe_callback import LinearProbeCallback


class CalmLinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the CALM dataset collection."""

    def __init__(
        self,
        max_length: int,
        tasks: Optional[Sequence[str]] = None,
        species: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        run_every_n_epochs: Optional[int] = None,
        test_size: float = 0.2,
        max_samples: int = 3000,
        seed: int = 42,
    ):
        tokenizer_transform = TokenizerTransform(
            tokenizer=NucleotideTokenizerFast(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        super().__init__(
            transform_fn=tokenizer_transform,
            task_type="regression",
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
        )

        self.tasks = set(tasks) if tasks else set(CALM_TASKS.keys())
        self.species = set(species) if species else None

        self.test_size = test_size
        self.max_samples = max_samples
        self.seed = seed

        self.dataset_splits = {}
        self.aggregate_metrics = defaultdict(list)

    def _create_split_datasets(
        self,
        task: str,
        species: Optional[str] = None
    ) -> Tuple[CalmPropertyDataset, CalmPropertyDataset]:
        """Create train/test splits for a given task."""

        rng = np.random.RandomState(self.seed) # TODO - seed everything fn

        # Check cache for existing splits
        split_key = f"{task}_{species}" if species else task
        if split_key in self.dataset_splits:
            return self.dataset_splits[split_key]

        dataset = CalmPropertyDataset(task=task, species=species, transform_fn=self.transform_fn)

        indices = np.arange(len(dataset))

        # If dataset is too large, subsample it first
        if len(indices) > self.max_samples:
            indices = rng.choice(indices, size=self.max_samples, replace=False)

        # Create train/test split from (possibly subsampled) indices
        test_size = int(len(indices) * self.test_size)
        train_size = len(indices) - test_size
        shuffled_indices = rng.permutation(indices)
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        # Cache the splits
        self.dataset_splits[split_key] = (train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _evaluate_task(
        self,
        task_key: str,
        task: str,
        train_dataset,
        test_dataset,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        ):
        """Evaluate a single task."""

        task_type, num_classes = CALM_TASKS[task]

        self._set_metrics(task_type, num_classes)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        try:
            train_embeddings, train_targets = self._get_embeddings(pl_module, train_loader)
            test_embeddings, test_targets = self._get_embeddings(pl_module, test_loader)

            probe = self._train_probe(train_embeddings, train_targets)
            self.probes[task_key] = probe

            metrics = self._evaluate_probe(probe, test_embeddings, test_targets)

            # Log metrics and store for averaging
            for metric_name, value in metrics.items():
                metric_key = f"calm_linear_probe/{task_key}/{metric_name}"
                trainer.logger.log_metrics({metric_key: value}, step=trainer.current_epoch)
                self.aggregate_metrics[metric_name].append(value)

        except Exception as e:
            print(f"Error in _evaluate_task for {task_key}: {str(e)}")
            raise

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._skip(trainer):
            return

        self.device = pl_module.device
        self.aggregate_metrics.clear()  # Reset aggregation for new epoch

        for task in tqdm(self.tasks, desc=f"{self.__class__.__name__}"):
            # Handle species-specific tasks
            if task in ["protein_abundance", "transcript_abundance"]:
                if not self.species:
                    continue
                for species in self.species:
                    task_key = f"{task}_{species}"
                    try:
                        train_dataset, test_dataset = self._create_split_datasets(task, species)
                        self._evaluate_task(task_key, task, train_dataset, test_dataset, trainer, pl_module)
                    except Exception as e:
                        print(f"Error processing {task_key}: {str(e)}")
            else:
                try:
                    train_dataset, test_dataset = self._create_split_datasets(task)
                    self._evaluate_task(task, task, train_dataset, test_dataset, trainer, pl_module)
                except Exception as e:
                    print(f"Error processing {task}: {str(e)}")

        # Calculate and log aggregate metrics
        for metric_name, values in self.aggregate_metrics.items():
            if values:  # Only log if we have values
                avg_value = sum(values) / len(values)
                trainer.logger.log_metrics(
                    {f"calm_linear_probe/mean/{metric_name}": avg_value},
                    step=trainer.current_epoch
                )
