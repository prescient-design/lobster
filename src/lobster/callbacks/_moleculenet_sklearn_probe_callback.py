import logging
from collections.abc import Sequence

import lightning as L
import torch
import torch.utils.data
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from lobster.constants import MOLECULENET_TASK_NAMES, MOLECULENET_TASKS, SklearnProbeType
from lobster.datasets import MoleculeNetDataset

from ._sklearn_probe_callback import SklearnProbeCallback, SklearnProbeTaskConfig

logger = logging.getLogger(__name__)


class MoleculeNetSklearnProbeCallback(SklearnProbeCallback):
    """Callback for evaluating embedding models on MoleculeNet single-task datasets.

    Assesses how well a molecular embedding model captures property-relevant
    structure by training scikit-learn probes on frozen SMILES embeddings for
    binary classification (BBBP, BACE, HIV) and regression (ESOL, FreeSolv,
    Lipophilicity) tasks from MoleculeNet.

    Datasets do not ship with official train/test splits in this integration;
    this callback loads the full task dataset and creates a random split with
    ``seed`` / ``test_size`` (same approach as :class:`CalmSklearnProbeCallback`).

    Reference:
        Wu et al. (2018) "MoleculeNet: a benchmark for molecular machine learning"
        https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a

    Data source:
        DeepChem S3 — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/

    Parameters
    ----------
    tasks : Sequence[str] | None, default=None
        MoleculeNet tasks to evaluate. If None, all supported single-task sets
        are used: BBBP, BACE, HIV, ESOL, FreeSolv, Lipophilicity.
    batch_size : int, default=32
        Batch size for embedding extraction.
    probe_type : SklearnProbeType, default="linear"
        Type of probe to use. Options: "linear", "elastic", "svm", "gradient_boosting".
    test_size : float, default=0.2
        Fraction of data held out for testing.
    max_samples : int | None, default=None
        If set, randomly subsample each dataset to at most this many examples
        before splitting (useful for large sets such as HIV).
    ignore_errors : bool, default=False
        If True, log and skip tasks that fail during evaluation.
    seed : int, default=0
        Random seed for splitting and probing.
    """

    def __init__(
        self,
        tasks: Sequence[str] | None = None,
        batch_size: int = 32,
        probe_type: SklearnProbeType = "linear",
        test_size: float = 0.2,
        max_samples: int | None = None,
        ignore_errors: bool = False,
        seed: int = 0,
    ):
        super().__init__(batch_size=batch_size, seed=seed)

        self.probe_type = probe_type
        self.test_size = test_size
        self.max_samples = max_samples
        self.ignore_errors = ignore_errors

        self.tasks = set(tasks) if tasks is not None else set(MOLECULENET_TASK_NAMES)
        unknown = self.tasks - MOLECULENET_TASK_NAMES
        if unknown:
            raise ValueError(
                f"Unknown MoleculeNet tasks: {sorted(unknown)}. Available: {sorted(MOLECULENET_TASK_NAMES)}"
            )

        logger.info(f"MoleculeNet tasks to evaluate: {sorted(self.tasks)}")

    def _random_split_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Create a seeded train/test split for a dataset without official splits."""
        generator = torch.Generator().manual_seed(self.seed)
        total_size = len(dataset)
        test_count = int(total_size * self.test_size)
        train_count = total_size - test_count
        return torch.utils.data.random_split(dataset, [train_count, test_count], generator=generator)

    def _subsample_dataset(self, dataset: Dataset) -> Dataset:
        """Optionally subsample a large dataset before splitting."""
        if self.max_samples is None or len(dataset) <= self.max_samples:
            return dataset

        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(self.seed))[
            : self.max_samples
        ].tolist()
        return Subset(dataset, indices)

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model on MoleculeNet datasets using linear probes."""
        all_task_metrics: dict[str, dict[str, float]] = {}

        for task in tqdm(self.tasks, desc=self.__class__.__name__):
            task_type, num_classes, *_ = MOLECULENET_TASKS[task]
            logger.info(f"Evaluating MoleculeNet task: {task} ({task_type})")

            dataset = MoleculeNetDataset(task=task)
            dataset = self._subsample_dataset(dataset)
            train_dataset, test_dataset = self._random_split_dataset(dataset)
            logger.info(f"Created datasets for {task}: train={len(train_dataset)}, test={len(test_dataset)}")

            config = SklearnProbeTaskConfig(
                task_name=task,
                task_type=task_type,
                probe_type=self.probe_type,
                num_classes=num_classes,
                modality="SMILES",
            )

            try:
                result = self.train_and_evaluate_probe_on_task(
                    model=module,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    task_config=config,
                )
                metrics = result.metrics
                all_task_metrics[task] = metrics

                self.log_metrics(
                    metrics=metrics,
                    task_name=task,
                    probe_type=self.probe_type,
                    trainer=trainer,
                )
            except Exception as e:
                if self.ignore_errors:
                    logger.error(f"Error processing task {task}: {str(e)}. Skipping task.")
                    continue
                raise

        mean_metrics = self._compute_mean_metrics(all_task_metrics)
        self.log_metrics(
            metrics=mean_metrics,
            task_name="mean",
            probe_type=self.probe_type,
            is_mean=True,
            trainer=trainer,
        )
        all_task_metrics["mean"] = mean_metrics

        successful_tasks = [k for k in all_task_metrics.keys() if k != "mean"]
        logger.info(
            f"Evaluation completed. Successful tasks: (n={len(successful_tasks)}/{len(self.tasks)}) {successful_tasks}"
        )
        logger.info(f"Results: {all_task_metrics}")

        return all_task_metrics
