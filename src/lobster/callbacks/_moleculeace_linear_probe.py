from typing import Dict, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import torch
from beignet.transforms import Transform
from lightning.pytorch.callbacks import Callback
from sklearn.linear_model import LinearRegression
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef

from lobster.constants import MOLECULEACE_TASKS
from lobster.datasets import MoleculeACEDataset


class LinearProbeCallback(Callback):
    """Callback for evaluating BERT-like models using scikit-learn linear probes."""

    def __init__(
        self,
        tasks: Sequence[str] | None = None,
        batch_size: int = 32,
        transform_fn: Optional[Transform] = None,
    ):
        super().__init__()

        if tasks is None:
            tasks = MOLECULEACE_TASKS

        self.tasks = tasks
        self.transform_fn = transform_fn
        self.batch_size = batch_size

        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.spearman = SpearmanCorrCoef()

        self.probes: Dict[str, LinearRegression] = {}

    def _get_embeddings(self, module: L.LightningModule, dataloader: DataLoader) -> Tuple[Tensor, Tensor]:
        embeddings = []
        targets = []

        module.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch

                # TODO: Handle multiple modalities in ModernBERT
                batch_embeddings = module.sequences_to_latents(x)
                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _train_probe(self, embeddings: Tensor, targets: Tensor) -> LinearRegression:
        """Train a linear probe on the given embeddings."""
        embeddings = embeddings.numpy()
        targets = targets.numpy()

        probe = LinearRegression()
        probe.fit(embeddings, targets)
        return probe

    def _evaluate_probe(self, probe: LinearRegression, embeddings: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate a linear probe's performance."""
        predictions = torch.from_numpy(probe.predict(embeddings)).float()
        targets = torch.from_numpy(targets).float()

        return {
            "mse": self.mse(predictions, targets).item(),
            "r2": self.r2(predictions, targets).item(),
            "spearman": self.spearman(predictions.squeeze(), targets.squeeze()).item(),
        }

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes at the end of each validation epoch."""
        if trainer.sanity_checking:
            return

        for task in self.tasks:
            # Create train dataset
            train_dataset = MoleculeACEDataset(task=task, transform_fn=self.transform_fn, train=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            # Create test dataset
            test_dataset = MoleculeACEDataset(task=task, transform_fn=self.transform_fn, train=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # Get embeddings
            train_embeddings, train_targets = self._get_embeddings(pl_module, train_loader)
            test_embeddings, test_targets = self._get_embeddings(pl_module, test_loader)

            # Train probe
            probe = self._train_probe(train_embeddings, train_targets)
            self.probes[task] = probe

            # Evaluate
            metrics = self._evaluate_probe(probe, test_embeddings, test_targets)

            # Log metrics
            for metric_name, value in metrics.items():
                trainer.logger.log_metrics(
                    {f"moleculeace_linear_probe/{task}/{metric_name}": value}, step=trainer.global_step
                )
