from collections import defaultdict
from collections.abc import Sequence

import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm

from lobster.constants import MOLECULEACE_TASKS
from lobster.datasets import MoleculeACEDataset
from lobster.tokenization import UMETokenizerTransform

from ._linear_probe_callback import LinearProbeCallback


class MoleculeACELinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the Molecule Activity Cliff
    Estimation (MoleculeACE) dataset from Tilborg et al. (2022).

    Currently only supports UME embeddings and uses UMETokenizerTransform.

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
        tasks: Sequence[str] | None = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
    ):
        tokenizer_transform = UMETokenizerTransform(
            modality="SMILES",
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

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model on MoleculeACE datasets using linear probes.

        This method can be used both during training (with a trainer)
        and standalone (with just a model).

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
        """
        aggregate_metrics = defaultdict(list)
        all_task_metrics = {}

        for task in tqdm(self.tasks, desc=f"{self.__class__.__name__}"):
            # Create datasets
            train_dataset = MoleculeACEDataset(task=task, transform_fn=self.transform_fn, train=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataset = MoleculeACEDataset(task=task, transform_fn=self.transform_fn, train=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            try:
                # Get embeddings
                train_embeddings, train_targets = self._get_embeddings(module, train_loader)
                test_embeddings, test_targets = self._get_embeddings(module, test_loader)

                # Train probe
                probe = self._train_probe(train_embeddings, train_targets)
                self.probes[task] = probe

                # Evaluate
                metrics = self._evaluate_probe(probe, test_embeddings, test_targets)
                all_task_metrics[task] = metrics

                # Log metrics if trainer is provided
                if trainer is not None:
                    for metric_name, value in metrics.items():
                        trainer.logger.log_metrics(
                            {f"moleculeace_linear_probe/{task}/{metric_name}": value},
                        )

                # Store for aggregate metrics
                for metric_name, value in metrics.items():
                    aggregate_metrics[metric_name].append(value)

            except Exception as e:
                print(f"Error processing task {task}: {str(e)}")
                continue

        # Calculate and log aggregate metrics
        mean_metrics = {}
        for metric_name, values in aggregate_metrics.items():
            if values:  # Only process if we have values
                avg_value = sum(values) / len(values)
                mean_metrics[metric_name] = avg_value

                # Log if trainer is provided
                if trainer is not None:
                    trainer.logger.log_metrics(
                        {f"moleculeace_linear_probe/mean/{metric_name}": avg_value},
                    )

        # Add mean metrics to the result
        all_task_metrics["mean"] = mean_metrics

        return all_task_metrics
