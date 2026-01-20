import logging
from collections.abc import Sequence
from typing import override

import lightning as L
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from lobster.constants import MOLECULEACE_TASKS, SklearnProbeType
from lobster.datasets import MoleculeACEDataset

from ._sklearn_probe_callback import SklearnProbeCallback, SklearnProbeTaskConfig

logger = logging.getLogger(__name__)


class MoleculeACESklearnProbeCallback(SklearnProbeCallback):
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
        probe_type: SklearnProbeType = "linear",
        use_protein_sequences: bool = False,
        ignore_errors: bool = False,
        seed: int = 0,
    ):
        super().__init__(batch_size=batch_size, seed=seed)

        self.probe_type = probe_type
        self.use_protein_sequences = use_protein_sequences
        self.tasks = set(tasks) if tasks is not None else MOLECULEACE_TASKS
        self.ignore_errors = ignore_errors

    @override
    def get_embeddings(
        self,
        model,
        dataset: MoleculeACEDataset,
        *,
        modality: str = "SMILES",
        aggregate: bool = True,
    ):
        "Override to allow for protein sequences to be embedded as well."

        if not self.use_protein_sequences:
            return super().get_embeddings(model, dataset, modality=modality, aggregate=aggregate)
        else:
            protein_sequence_dataset = StringDataset(
                items=[x[1] for x, _ in dataset], labels=[y[0] for _, y in dataset]
            )
            smiles_dataset = StringDataset(items=[x[0] for x, _ in dataset], labels=[y[0] for _, y in dataset])
            protein_embeddings, _ = super().get_embeddings(
                model, protein_sequence_dataset, modality="AMINO_ACID", aggregate=aggregate
            )
            smiles_embeddings, _ = super().get_embeddings(model, smiles_dataset, modality="SMILES", aggregate=aggregate)

            return torch.cat([smiles_embeddings, protein_embeddings], dim=1)

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model on MoleculeACE datasets using linear probes."""
        all_task_metrics = {}

        for task in tqdm(self.tasks, desc=self.__class__.__name__):
            train_dataset = MoleculeACEDataset(
                task=task, train=True, include_protein_sequences=self.use_protein_sequences
            )
            test_dataset = MoleculeACEDataset(
                task=task, train=False, include_protein_sequences=self.use_protein_sequences
            )

            # Create task configuration
            config = SklearnProbeTaskConfig(
                task_name=task,
                task_type="regression",
                probe_type=self.probe_type,
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
                    metrics = {}
                    continue
                else:
                    raise e

        # Calculate mean metrics from all_task_metrics
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


class StringDataset(Dataset):
    def __init__(self, items: list[str], labels: list[str]):
        self.items = items
        self.labels = labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index], self.labels[index]
