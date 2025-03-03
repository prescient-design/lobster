from typing import Callable, Dict, Literal, Optional, Tuple

import lightning as L
import numpy as np
import torch
from beignet.transforms import Transform
from lightning.pytorch.callbacks import Callback
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, MeanSquaredError, R2Score, SpearmanCorrCoef

TaskType = Literal["regression", "binary", "multiclass", "multilabel"]


class LinearProbeCallback(Callback):
    """Callback for evaluating embedding models using scikit-learn linear probes."""

    def __init__(
        self,
        task_type: TaskType = "regression",
        transform_fn: Transform | Callable | None = None,
        num_classes: Optional[int] = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
    ):
        super().__init__()
        self.transform_fn = transform_fn
        self.task_type = task_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.run_every_n_epochs = run_every_n_epochs

        # Initialize metrics based on task type
        self._set_metrics(task_type, num_classes)

        # Dictionary to store trained probes
        self.probes: Dict[str, LinearRegression | LogisticRegression] = {}

    def _set_metrics(self, task_type: TaskType, num_classes: Optional[int] = None) -> None:
        """Initialize metrics based on task type."""
        if task_type == "regression":
            self.mse = MeanSquaredError()
            self.r2 = R2Score()
            self.spearman = SpearmanCorrCoef()
            self.accuracy = None
            self.f1 = None
            self.auroc = None

        elif task_type in {"binary", "multiclass", "multilabel"}:
            # For multilabel, we use num_classes as num_labels
            metric_task = task_type
            self.accuracy = Accuracy(task=metric_task, num_labels=num_classes)
            self.f1 = F1Score(task=metric_task, num_labels=num_classes)
            self.auroc = AUROC(task=metric_task, num_labels=num_classes)
            self.mse = None
            self.r2 = None
            self.spearman = None

        else:
            raise ValueError("task_type must be: regression, binary, multiclass, or multilabel")

        self.task_type = task_type
        self.num_classes = num_classes

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip validation this epoch."""
        # Don't skip if run_every_n_epochs is not set
        if self.run_every_n_epochs is None:
            return False

        # Don't skip if global_rank=0 (main process)
        if trainer.global_rank == 0:
            return False
        else:
            return True

        return trainer.current_epoch % self.run_every_n_epochs != 0

    def _get_embeddings(self, module: L.LightningModule, dataloader: DataLoader) -> Tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader."""
        embeddings = []
        targets = []

        module.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = {k: v.to(module.device) for k, v in x.items()}

                # Get token-level embeddings
                batch_embeddings = module.tokens_to_latents(**x)

                # Reshape to (batch_size, seq_len, hidden_size)
                batch_size = len(y)
                seq_len = x["input_ids"].size(-1)
                batch_embeddings = batch_embeddings.view(batch_size, seq_len, -1)

                # Simple mean pooling over sequence length dimension
                seq_embeddings = batch_embeddings.mean(dim=1)

                embeddings.append(seq_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _train_probe(self, embeddings: Tensor, targets: Tensor):
        """Train a probe on the given embeddings and targets."""
        embeddings = embeddings.numpy()
        targets = targets.numpy()

        if self.task_type == "regression":
            probe = LinearRegression()
            probe.fit(embeddings, targets)

        elif self.task_type == "multilabel":
            base_classifier = LogisticRegression(random_state=42)
            probe = MultiOutputClassifier(base_classifier)
            probe.fit(embeddings, targets)

        else:  # binary or multiclass
            probe = LogisticRegression(
                multi_class="ovr" if self.task_type == "binary" else "multinomial",
                random_state=42,
            )
            probe.fit(embeddings, targets.ravel())

        return probe

    def _evaluate_probe(self, probe, embeddings: Tensor, targets: Tensor) -> Dict[str, float]:
        """Evaluate a trained probe using task-appropriate metrics."""
        embeddings_np = embeddings.numpy()  # Convert to numpy for probe prediction
        metrics = {}

        if self.task_type == "regression":
            predictions_np = probe.predict(embeddings_np)
            predictions = torch.from_numpy(predictions_np).float()

            metrics["mse"] = self.mse(predictions, targets).item()
            metrics["r2"] = self.r2(predictions, targets).item()
            metrics["spearman"] = self.spearman(predictions.squeeze(), targets.squeeze()).item()

        else:  # binary, multiclass, or multilabel
            if self.task_type == "multilabel":
                # Get probabilities for each label
                predictions_np = np.stack([est.predict_proba(embeddings_np)[:, 1] for est in probe.estimators_], axis=1)
            else:  # binary or multiclass
                predictions_np = probe.predict_proba(embeddings_np)
                if self.task_type == "binary":
                    predictions_np = predictions_np[:, 1]

            predictions = torch.from_numpy(predictions_np).float()
            metrics["accuracy"] = self.accuracy(predictions, targets).item()
            metrics["f1"] = self.f1(predictions, targets).item()
            metrics["auroc"] = self.auroc(predictions, targets).item()

        return metrics

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes, optionally at specified epochs."""
        raise NotImplementedError("Subclasses must implement on_validation_epoch_end")
