from collections import defaultdict
from collections.abc import Sequence

import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm

from lobster.constants import MOLECULEACE_TASKS
from lobster.datasets import MoleculeACEDataset

from ._linear_probe_callback import LinearProbeCallback


class MoleculeACELinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the Molecule Activity Cliff
    Estimation (MoleculeACE) dataset from Tilborg et al. (2022).

    Currently assumes models can embed raw SMILES sequences via
    ``embed_sequences(..., modality="SMILES")`` through the base callback's
    embedding helper.

    This callback assesses how well a molecular embedding model captures activity
    cliffs by training linear probes on frozen embeddings to predict pEC50/pKi
    values for 30 different protein targets from ChEMBL.

    Reference:
        van Tilborg et al. (2022) "Exposing the Limitations of Molecular Machine
        Learning with Activity Cliffs"
        https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073
    """

    def __init__(
        self,
        tasks: Sequence[str] | None = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
    ):
        super().__init__(
            task_type="regression",
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
        )

        # Set tasks
        self.tasks = set(tasks) if tasks is not None else MOLECULEACE_TASKS

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model on MoleculeACE datasets using linear probes."""
        aggregate_metrics = defaultdict(list)
        all_task_metrics = {}

        for task in tqdm(self.tasks, desc=f"{self.__class__.__name__}"):
            train_dataset = MoleculeACEDataset(task=task, train=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataset = MoleculeACEDataset(task=task, train=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            try:
                # Use base callback embedding helper with explicit SMILES modality
                train_embeddings, train_targets = self._get_embeddings(module, train_loader, modality="SMILES")
                test_embeddings, test_targets = self._get_embeddings(module, test_loader, modality="SMILES")

                probe = self._train_probe(train_embeddings, train_targets)
                self.probes[task] = probe

                metrics = self._evaluate_probe(probe, test_embeddings, test_targets)
                all_task_metrics[task] = metrics

                if trainer is not None:
                    for metric_name, value in metrics.items():
                        trainer.logger.log_metrics({f"moleculeace_linear_probe/{task}/{metric_name}": value})

                for metric_name, value in metrics.items():
                    aggregate_metrics[metric_name].append(value)

            except Exception as e:
                print(f"Error processing task {task}: {str(e)}")
                continue

        mean_metrics = {}
        for metric_name, values in aggregate_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                mean_metrics[metric_name] = avg_value
                if trainer is not None:
                    trainer.logger.log_metrics({f"moleculeace_linear_probe/mean/{metric_name}": avg_value})

        all_task_metrics["mean"] = mean_metrics
        return all_task_metrics
