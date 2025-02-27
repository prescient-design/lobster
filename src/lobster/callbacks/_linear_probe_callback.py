from typing import Callable, Dict, Literal, Optional, Tuple

import lightning as L
import torch
from beignet.transforms import Transform
from lightning.pytorch.callbacks import Callback
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, MeanSquaredError, R2Score, SpearmanCorrCoef

TaskType = Literal["regression", "binary", "multiclass"]


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
        if task_type == "regression":
            self.mse = MeanSquaredError()
            self.r2 = R2Score()
            self.spearman = SpearmanCorrCoef()

        elif task_type in {"binary", "multiclass"}:
            self.accuracy = Accuracy(task=task_type, num_classes=num_classes)
            self.f1 = F1Score(task=task_type, num_classes=num_classes)
            self.auroc = AUROC(task_type=task_type, num_classes=num_classes)

        else:
            raise ValueError("task_type must be: regression, binary, or multiclass")

        # Dictionary to store trained probes
        self.probes: Dict[str, LinearRegression | LogisticRegression] = {}

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip validation this epoch."""
        if self.run_every_n_epochs is None:
            return False

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
        else:
            probe = LogisticRegression(
                multi_class="ovr" if self.task_type == "binary" else "multinomial",
            )

        probe.fit(embeddings, targets)

        return probe

    def _evaluate_probe(self, probe, embeddings: Tensor, targets: Tensor) -> Dict[str, float]:
        """Evaluate a trained probe using task-appropriate metrics."""
        metrics = {}

        if self.task_type == "regression":
            predictions = probe.predict(embeddings.numpy())
            predictions = torch.from_numpy(predictions).float()

            metrics["mse"] = self.mse(predictions, targets).item()
            metrics["r2"] = self.r2(predictions, targets).item()
            metrics["spearman"] = self.spearman(predictions.squeeze(), targets.squeeze()).item()

        else:  # binary or multiclass
            pred_probs = probe.predict_proba(embeddings.numpy())
            predictions = torch.from_numpy(pred_probs).float()

            if self.task_type == "binary":
                predictions = predictions[:, 1]

            metrics["accuracy"] = self.accuracy(predictions, targets).item()
            metrics["f1"] = self.f1(predictions, targets).item()
            metrics["auroc"] = self.auroc(predictions, targets).item()

        return metrics

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes, optionally at specified epochs."""
        raise NotImplementedError("Subclasses must implement on_validation_epoch_end")
