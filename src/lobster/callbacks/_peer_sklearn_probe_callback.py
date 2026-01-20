import logging
import math
from collections.abc import Sequence
from typing import override
import warnings

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from lobster.constants import (
    PEER_TASK_CATEGORIES,
    PEER_TASK_METRICS,
    PEER_TASKS,
    PEERTask,
    PEERTaskCategory,
    SklearnProbeType,
    Modality,
)
from lobster.datasets import PEERDataset

from ._sklearn_probe_callback import SklearnProbeCallback, SklearnProbeTaskConfig

logger = logging.getLogger(__name__)


class PEERSklearnProbeCallback(SklearnProbeCallback):
    """Simplified PEER evaluation callback using sklearn probes.

    This callback evaluates model embeddings on PEER benchmark tasks by leveraging
    the base SklearnProbeCallback infrastructure. It handles the PEER-specific
    complexities like paired sequences and structure prediction while reusing
    the base class for embedding extraction, probe training, and evaluation.

    By default, evaluates 16 out of 17 PEER tasks (excludes PROTEINNET due to memory issues).

    Parameters
    ----------
    tasks : Sequence[PEERTask | str] | None, default=None
        Specific PEER tasks to evaluate. If None, all tasks except PROTEINNET are used.
    batch_size : int, default=32
        Batch size for embedding extraction and evaluation.
    probe_type : SklearnProbeType, default="linear"
        Type of probe to use. Options: "linear", "elastic", "svm".
    ignore_errors : bool, default=False
        Whether to continue evaluation if individual tasks fail.
    seed : int, default=0
        Random seed for reproducibility.
    """

    def __init__(
        self,
        tasks: Sequence[PEERTask | str] | None = None,
        batch_size: int = 32,
        probe_type: SklearnProbeType = "linear",
        ignore_errors: bool = True,
        use_joint_embedding_for_pairs: bool = False,
        seed: int = 0,
    ):
        super().__init__(batch_size=batch_size, seed=seed)

        self.probe_type = probe_type
        self.ignore_errors = ignore_errors
        self.use_joint_embedding_for_pairs = use_joint_embedding_for_pairs

        # Convert string tasks to enum
        if tasks is not None:
            self.selected_tasks = {PEERTask(task) if isinstance(task, str) else task for task in tasks}
        else:
            # Default: all tasks except PROTEINNET (memory issues)
            self.selected_tasks = set(PEER_TASKS.keys()) - {PEERTask.PROTEINNET}

        logger.info(f"PEER tasks to evaluate: {sorted([task.value for task in self.selected_tasks])}")

    def _filter_metrics_for_task(self, task: PEERTask, metrics: dict[str, float]) -> dict[str, float]:
        """Filter metrics to return only the preferred metric for tasks shown in PEER benchmark image.

        For tasks not in PEER_TASK_METRICS, returns all metrics unchanged.
        For tasks in PEER_TASK_METRICS, returns only the single preferred metric.

        Parameters
        ----------
        task : PEERTask
            The PEER task being evaluated
        metrics : dict[str, float]
            Dictionary of all computed metrics for the task

        Returns
        -------
        dict[str, float]
            Filtered metrics dictionary
        """
        if task not in PEER_TASK_METRICS:
            return metrics

        preferred_metric = PEER_TASK_METRICS[task]

        if preferred_metric == "rmse":
            if "mse" in metrics:
                return {"rmse": math.sqrt(metrics["mse"])}
            else:
                logger.warning(f"MSE metric not found for task {task.value}, cannot compute RMSE")
                return metrics
        elif preferred_metric in metrics:
            return {preferred_metric: metrics[preferred_metric]}
        else:
            logger.warning(f"Preferred metric '{preferred_metric}' not found for task {task.value}")
            return metrics

    def _get_task_test_split(self, task: PEERTask) -> str:
        """Get the most relevant test split for each task."""
        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                return "cb513"
            case PEERTask.BINDINGDB:
                return "holdout_test"
            case PEERTask.FOLD:
                return "test_superfamily_holdout"
            case _:
                return "test"

    @override
    def get_embeddings(
        self,
        model: L.LightningModule | torch.nn.Module,
        dataset: Dataset,
        *,
        modality: str = None,
        aggregate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Override to handle PEER-specific data formats (paired sequences, structure tasks)."""

        # Check if this is a PEER dataset to determine task type
        if hasattr(dataset, "task"):
            task = dataset.task
            category = PEER_TASK_CATEGORIES[task]

            # Handle paired sequence tasks (protein-protein, protein-ligand)
            if category in {PEERTaskCategory.PROTEIN_PROTEIN_INTERACTION, PEERTaskCategory.PROTEIN_LIGAND_INTERACTION}:
                return self._get_paired_embeddings(model, dataset, task)

            # Handle structure prediction tasks (token-level predictions)
            elif category == PEERTaskCategory.STRUCTURE_PREDICTION:
                return self._get_structure_embeddings(model, dataset, task)

        # Default: use base class implementation for standard single-sequence tasks
        return super().get_embeddings(model, dataset, modality=modality, aggregate=aggregate)

    def _get_paired_embeddings(
        self, model: L.LightningModule | torch.nn.Module, dataset: Dataset, task: PEERTask
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for paired sequence tasks."""
        embeddings = []
        targets = []

        model.eval()

        for batch in torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
            inputs, y = batch

            # inputs should be a list of two sequences for paired tasks
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError(f"Expected paired inputs for task {task}, got: {type(inputs)}")

            seq1, seq2 = inputs

            with torch.no_grad():
                # Handle protein-protein interactions
                if task in {PEERTask.HUMANPPI, PEERTask.YEASTPPI, PEERTask.PPIAFFINITY}:
                    if not self.use_joint_embedding_for_pairs:
                        emb1 = model.embed_sequences(seq1, modality=Modality.AMINO_ACID, aggregate=True)
                        emb2 = model.embed_sequences(seq2, modality=Modality.AMINO_ACID, aggregate=True)
                    else:
                        emb1, emb2 = model.embed_sequences(
                            seq1, seq2, modality1=Modality.AMINO_ACID, modality2=Modality.AMINO_ACID, aggregate=True
                        )

                # Handle protein-ligand interactions
                elif task in {PEERTask.BINDINGDB, PEERTask.PDBBIND}:
                    warnings.warn(
                        f"Task {task} requires embeddings for SMILES modality. Please confirm model supports this.",
                        stacklevel=2,
                    )
                    if not self.use_joint_embedding_for_pairs:
                        emb1 = model.embed_sequences(seq1, modality=Modality.AMINO_ACID, aggregate=True)
                        emb2 = model.embed_sequences(seq2, modality=Modality.SMILES, aggregate=True)
                    else:
                        emb1, emb2 = model.embed_sequences(
                            seq1, seq2, modality1=Modality.AMINO_ACID, modality2=Modality.SMILES, aggregate=True
                        )

                else:
                    raise ValueError(f"Unknown paired task: {task}")

                # Concatenate embeddings
                batch_embeddings = torch.cat([emb1, emb2], dim=1)

            embeddings.append(batch_embeddings.cpu())
            targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _get_structure_embeddings(
        self, model: L.LightningModule | torch.nn.Module, dataset: Dataset, task: PEERTask
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for structure prediction tasks (token-level)."""
        embeddings = []
        targets = []

        model.eval()

        for batch in torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False
        ):  # batch_size=1 for structure tasks
            x, y = batch

            with torch.no_grad():
                batch_embeddings = model.embed_sequences(x, modality=Modality.AMINO_ACID, aggregate=False)
                batch_embeddings = batch_embeddings.squeeze(0)  # Remove batch dimension

                if task == PEERTask.SECONDARY_STRUCTURE:
                    raise NotImplementedError("Secondary structure prediction is temporarily disabled")

                elif task == PEERTask.FOLD:
                    embeddings.append(batch_embeddings.mean(dim=0, keepdim=True))
                    targets.append(y)

                else:
                    raise ValueError(f"Unknown structure task: {task}")

        if embeddings:
            return torch.cat(embeddings), torch.cat(targets)
        else:
            return torch.tensor([]), torch.tensor([])

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model on PEER datasets using sklearn probes."""
        all_task_metrics = {}

        for task in tqdm(sorted(self.selected_tasks), desc=self.__class__.__name__):
            logger.info(f"Evaluating task: {task.value}")

            try:
                # Get task configuration
                task_type, num_classes = PEER_TASKS[task]
                test_split = self._get_task_test_split(task)

                # Create datasets
                train_dataset = PEERDataset(task=task, split="train")
                test_dataset = PEERDataset(task=task, split=test_split)

                # Create task configuration
                config = SklearnProbeTaskConfig(
                    task_name=task.value,
                    task_type=task_type,
                    probe_type=self.probe_type,
                    num_classes=num_classes,
                    modality=Modality.AMINO_ACID,  # Default, overridden in get_embeddings if needed
                )

                # Train and evaluate probe using base class method
                result = self.train_and_evaluate_probe_on_task(
                    model=module,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    task_config=config,
                )

                metrics = result.metrics
                # Filter metrics based on PEER_TASK_METRICS configuration
                filtered_metrics = self._filter_metrics_for_task(task, metrics)
                all_task_metrics[task.value] = filtered_metrics

                # Log filtered metrics using base class method
                self.log_metrics(
                    metrics=filtered_metrics,
                    task_name=task.value,
                    probe_type=self.probe_type,
                    trainer=trainer,
                )

            except Exception as e:
                if self.ignore_errors:
                    logger.error(f"Error processing task {task.value}: {str(e)}. Skipping task.")
                    continue
                else:
                    raise e

        # Calculate mean metrics using base class method
        mean_metrics = self._compute_mean_metrics(all_task_metrics)
        all_task_metrics["mean"] = mean_metrics

        # Log mean metrics
        self.log_metrics(
            metrics=mean_metrics,
            task_name="mean",
            probe_type=self.probe_type,
            is_mean=True,
            trainer=trainer,
        )

        successful_tasks = [k for k in all_task_metrics.keys() if k != "mean"]
        logger.info(
            f"Evaluation completed. Successful tasks: (n={len(successful_tasks)}/{len(self.selected_tasks)}) {successful_tasks}"
        )
        logger.info(f"Results: {all_task_metrics}")

        return all_task_metrics
