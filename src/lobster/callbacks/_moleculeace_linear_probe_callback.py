from collections import defaultdict
from typing import Optional, Sequence

import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm

from lobster.constants import MOLECULEACE_TASKS
from lobster.datasets import MoleculeACEDataset
from lobster.tokenization import SmilesTokenizerFast
from lobster.transforms import TokenizerTransform

from ._linear_probe_callback import LinearProbeCallback


class MoleculeACELinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the Molecule Activity Cliff
    Estimation (MoleculeACE) dataset from Tilborg et al. (2022).

    This callback assesses how well a molecular embedding model captures activity
    cliffs - pairs of molecules that are structurally similar but show large
    differences in biological activity (potency). It does this by training linear
    probes on the frozen embeddings to predict pEC50/pKi values for 30 different
    protein targets from ChEMBL.

    Reference:
        van Tilborg et al. (2022) "Exposing the Limitations of Molecular Machine
        Learning with Activity Cliffs"
        https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073
    """

    def __init__(
        self,
        max_length: int,
        tasks: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
        log_individual_tasks: bool = False,
    ):
        tokenizer_transform = TokenizerTransform(
            tokenizer=SmilesTokenizerFast(),
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

        # Set tasks
        self.tasks = set(tasks) if tasks is not None else MOLECULEACE_TASKS
        self.log_individual_tasks = log_individual_tasks

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes at specified epochs."""
        if self._skip(trainer):
            return

        aggregate_metrics = defaultdict(list)

        for task in tqdm(self.tasks, desc=f"{self.__class__.__name__}"):
            # Create datasets
            train_dataset = MoleculeACEDataset(task=task, transform_fn=self.transform_fn, train=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
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

            # Log metrics and store for averaging
            for metric_name, value in metrics.items():
                if self.log_individual_tasks:
                    trainer.logger.log_metrics(
                        {f"moleculeace_linear_probe/{task}/{metric_name}": value}, step=trainer.current_epoch
                    )
                aggregate_metrics[metric_name].append(value)

        # Calculate and log aggregate metrics
        for metric_name, values in aggregate_metrics.items():
            avg_value = sum(values) / len(values)
            trainer.logger.log_metrics(
                {f"moleculeace_linear_probe/mean/{metric_name}": avg_value}, step=trainer.current_epoch
            )
