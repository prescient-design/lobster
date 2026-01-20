import logging
import math
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import AUROC, Accuracy, F1Score, MeanSquaredError, PearsonCorrCoef, R2Score, SpearmanCorrCoef

from lobster.constants import SklearnProbeTaskType, SklearnProbeType
from lobster.model import predict_with_sklearn_probe, train_sklearn_probe

logger = logging.getLogger(__name__)


@dataclass
class SklearnProbeTaskConfig:
    """Configuration for a single sklearn probe task."""

    task_name: str
    task_type: SklearnProbeTaskType
    probe_type: SklearnProbeType
    num_classes: int | None = None
    modality: str | None = None
    dimensionality_reduction: bool = False
    reduced_dim: int = 128
    classification_threshold: float = 0.5


@dataclass
class SklearnProbeTaskResult:
    """Results from evaluating a single sklearn probe task."""

    config: SklearnProbeTaskConfig
    metrics: dict[str, float]
    probe: Any | None
    preprocessors: dict[str, Any] | None


class SklearnProbeCallback(Callback):
    """Callback for evaluating embedding models using scikit-learn model probes.

    Assumes the underlying model implements the following method:

    `def embed_sequences(sequences: Sequence[str], aggregate: bool = True, modality: str = None) -> Tensor`
    where the Tensor shape is (batch_size, hidden_size)
    or (batch_size, seq_len, hidden_size) if aggregate is False.

    Subclasses must implement the `evaluate` method to evaluate the model on
    specific tasks.
    """

    def __init__(self, batch_size: int = 32, seed: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.task_results: dict[str, SklearnProbeTaskResult] = {}

    def _create_metrics_for_task(self, task_config: SklearnProbeTaskConfig) -> dict[str, Any]:
        """Create metrics dictionary for a specific task type."""
        if task_config.task_type == "regression":
            return {
                "mse": MeanSquaredError(),
                "r2": R2Score(),
                "spearman": SpearmanCorrCoef(),
                "pearson": PearsonCorrCoef(),
            }
        elif task_config.task_type in {"binary", "multiclass", "multilabel"}:
            if task_config.task_type == "multilabel":
                return {
                    "accuracy": Accuracy(
                        task=task_config.task_type,
                        num_labels=task_config.num_classes,
                        threshold=task_config.classification_threshold,
                    ),
                    "f1": F1Score(
                        task=task_config.task_type,
                        num_labels=task_config.num_classes,
                        threshold=task_config.classification_threshold,
                    ),
                    "f1_weighted": F1Score(
                        task=task_config.task_type,
                        num_labels=task_config.num_classes,
                        average="weighted",
                        threshold=task_config.classification_threshold,
                    ),
                    "auroc": AUROC(task=task_config.task_type, num_labels=task_config.num_classes),
                }
            else:
                return {
                    "accuracy": Accuracy(task=task_config.task_type, num_classes=task_config.num_classes),
                    "f1": F1Score(task=task_config.task_type, num_classes=task_config.num_classes),
                    "f1_weighted": F1Score(
                        task=task_config.task_type, num_classes=task_config.num_classes, average="weighted"
                    ),
                    "auroc": AUROC(task=task_config.task_type, num_classes=task_config.num_classes),
                }
        else:
            raise ValueError(f"Task type {task_config.task_type} not supported. Must be one of {SklearnProbeTaskType}")

    def _compute_regression_metrics(
        self, predictions: Tensor, targets: Tensor, task_config: SklearnProbeTaskConfig
    ) -> dict[str, float]:
        """Compute regression metrics for probe evaluation.

        Parameters
        ----------
        predictions : Tensor
            Model predictions
        targets : Tensor
            Ground truth targets

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values
        """
        # Ensure predictions and targets have matching shapes
        if predictions.dim() != targets.dim():
            if targets.dim() == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            elif predictions.dim() == 1 and targets.dim() == 2:
                predictions = predictions.unsqueeze(1)

        metrics = self._create_metrics_for_task(task_config)

        results = {}
        results["mse"] = metrics["mse"](predictions, targets).item()
        results["r2"] = metrics["r2"](predictions, targets).item()

        spearman_val = metrics["spearman"](predictions.squeeze(), targets.squeeze()).item()
        results["spearman"] = 0.0 if math.isnan(spearman_val) else spearman_val

        pearson_val = metrics["pearson"](predictions.squeeze(), targets.squeeze()).item()
        results["pearson"] = 0.0 if math.isnan(pearson_val) else pearson_val

        return results

    def _compute_classification_metrics(
        self, predictions: Tensor, targets: Tensor, task_config: SklearnProbeTaskConfig
    ) -> dict[str, float]:
        """Compute classification metrics for probe evaluation."""
        metrics = self._create_metrics_for_task(task_config)

        return {metric_name: metrics[metric_name](predictions, targets).item() for metric_name in metrics}

    def _compute_mean_metrics(self, metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        """Compute mean metrics from all_task_metrics.

        Example metrics:
        {
            "task1": {
                "mse": 0.5,
                "r2": 0.6
            },
            "task2": {
                "mse": 0.7,
                "r2": 0.8
            },
        }
        Returns:
        {
            "mse": 0.6,
            "r2": 0.7
        }


        """
        mean_metrics = {}

        metric_names = set()

        for task_metrics in metrics.values():
            metric_names.update(task_metrics.keys())

        # Calculate mean for each metric
        for metric_name in metric_names:
            values = [task_metrics[metric_name] for task_metrics in metrics.values() if metric_name in task_metrics]
            if values:
                mean_metrics[metric_name] = sum(values) / len(values)

        return mean_metrics

    def get_embeddings(
        self,
        model: L.LightningModule | torch.nn.Module,
        dataset: Dataset,
        *,
        modality: str = None,
        aggregate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader.

        Parameters
        ----------
        model : Union[L.LightningModule, torch.nn.Module]
            The model to extract embeddings from
        dataset : Dataset
            DataLoader for the data to extract embeddings for
        modality : str, optional
            Explicit modality for embed_sequences. If None, falls back to embed() method.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Tuple of (embeddings, targets)
            Embeddings are of shape (batch_size, hidden_size) if aggregate is True,
            otherwise (batch_size, seq_len, hidden_size)
        """
        embeddings = []
        targets = []

        model.eval()

        for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
            x, y = batch

            if not isinstance(x, Sequence) and not all(isinstance(seq, str) for seq in x):
                raise ValueError(f"Expected the first element of the batch to be a sequence of strings, got {x}")

            with torch.no_grad():
                batch_embeddings = model.embed_sequences(list(x), modality=modality, aggregate=aggregate)

            if aggregate and not batch_embeddings.ndim == 2:
                raise ValueError(
                    f"Expected the embeddings to be of shape (batch_size, hidden_size), got {batch_embeddings.shape}"
                )
            elif not aggregate and not batch_embeddings.ndim == 3:
                raise ValueError(
                    f"Expected the embeddings to be of shape (batch_size, seq_len, hidden_size), got {batch_embeddings.shape}"
                )

            embeddings.append(batch_embeddings.cpu())
            targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def train_and_evaluate_probe_on_task(
        self,
        model: L.LightningModule | torch.nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        task_config: SklearnProbeTaskConfig,
    ) -> SklearnProbeTaskResult:
        """Train and evaluate a probe on a given task.

        Parameters
        ----------
        model : L.LightningModule | torch.nn.Module
            The model to extract embeddings from
        train_dataset : Dataset
            Training dataset
        test_dataset : Dataset
            Test dataset
        task_config : SklearnProbeTaskConfig
            Task configuration

        Returns
        -------
        SklearnProbeTaskResult
            Results containing metrics, probe, and preprocessors
        """
        train_embeddings, train_targets = self.get_embeddings(model, train_dataset, modality=task_config.modality)
        test_embeddings, test_targets = self.get_embeddings(model, test_dataset, modality=task_config.modality)

        probe, preprocessors = train_sklearn_probe(
            x=train_embeddings,
            y=train_targets,
            task_type=task_config.task_type,
            probe_type=task_config.probe_type,
            dimensionality_reduction=task_config.dimensionality_reduction,
            reduced_dim=task_config.reduced_dim,
            seed=self.seed,
        )

        predictions = predict_with_sklearn_probe(
            x=test_embeddings, probe=probe, preprocessors=preprocessors, task_type=task_config.task_type
        )

        if task_config.task_type == "regression":
            metrics = self._compute_regression_metrics(predictions, test_targets, task_config)
        else:
            metrics = self._compute_classification_metrics(predictions, test_targets, task_config)

        return SklearnProbeTaskResult(config=task_config, probe=probe, metrics=metrics, preprocessors=preprocessors)

    def train_and_evaluate_cv_probe_on_task(
        self,
        model: L.LightningModule | torch.nn.Module,
        dataset: Dataset,
        task_config: SklearnProbeTaskConfig,
        n_folds: int = 5,
    ) -> SklearnProbeTaskResult:
        """Train and evaluate a probe on a given task using cross-validation."""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        kfold_metrics = []
        indices = list(range(len(dataset)))

        for train_indices, val_indices in kfold.split(indices):
            fold_train_dataset = Subset(dataset, train_indices)
            fold_val_dataset = Subset(dataset, val_indices)

            result = self.train_and_evaluate_probe_on_task(model, fold_train_dataset, fold_val_dataset, task_config)

            kfold_metrics.append(result.metrics)

        # Average metrics across folds
        avg_metrics = {}
        if kfold_metrics:
            for metric_name in kfold_metrics[0].keys():
                values = [fold[metric_name] for fold in kfold_metrics if metric_name in fold]
                if values:
                    avg_metrics[metric_name] = sum(values) / len(values)

        return SklearnProbeTaskResult(config=task_config, probe=None, metrics=avg_metrics, preprocessors={})

    def log_metrics(
        self,
        metrics: dict[str, float],
        task_name: str,
        probe_type: str,
        is_mean: bool = False,
        trainer: L.Trainer | None = None,
    ) -> None:
        """Log the result of a task evaluation."""
        if trainer is not None:
            for metric_name, value in metrics.items():
                if is_mean:
                    metric_name = f"mean/{metric_name}"

                trainer.logger.log_metrics({f"{task_name}_{probe_type}_probe/{metric_name}": value})

        if task_name == "mean":
            logger.info(f"Mean scores: {metrics}")
        else:
            logger.info(f"Task `{task_name}` scores: {metrics}")

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model using linear probes.

        This method can be used both during training (with a trainer) and
        standalone (with just a model).

        Parameters
        ----------
        module : L.LightningModule
            The model to evaluate
        trainer : Optional[L.Trainer]
            Optional trainer for logging metrics

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of task_name -> metric_name -> value
            {"task_name_1": {"metric_name_1": 0.89,
                    "metric_name_2": 0.9,
                }
            }
        """
        raise NotImplementedError("Subclasses must implement evaluate")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.device = pl_module.device
        self.evaluate(pl_module, trainer)
