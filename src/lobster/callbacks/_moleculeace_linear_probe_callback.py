from collections import defaultdict
from collections.abc import Callable, Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lobster.constants import MOLECULEACE_TASKS
from lobster.datasets import MoleculeACEDataset
from lobster.tokenization import UMETokenizerTransform

from ._linear_probe_callback import LinearProbeCallback


class MoleculeACELinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the Molecule Activity Cliff
    Estimation (MoleculeACE) dataset from Tilborg et al. (2022).

    Supports both UME-style models (requiring tokenization) and direct embedding
    models (taking raw SMILES strings).

    This callback assesses how well a molecular embedding model captures activity
    cliffs - pairs of molecules that are structurally similar but show large
    differences in biological activity (potency). It does this by training linear
    probes on the frozen embeddings to predict pEC50/pKi values for 30 different
    protein targets from ChEMBL.

    Model compatibility:
    - UME models: Use requires_tokenization=True (default)
    - Other molecular models: Use requires_tokenization=False for models that
      take raw SMILES strings directly

    Parameters
    ----------
    tasks : Sequence[str] | None, default=None
        Specific MoleculeACE tasks to evaluate. If None, all tasks are used.
    batch_size : int, default=32
        Batch size for embedding extraction and evaluation.
    run_every_n_epochs : int | None, default=None
        Run this callback every n epochs. If None, runs every validation epoch.
    requires_tokenization : bool, default=True
        Whether the model requires tokenized inputs (via UMETokenizerTransform) or
        can accept raw SMILES strings directly.
    transform_fn : Callable | None, default=None
        Custom transform function to apply to inputs. If None and requires_tokenization
        is True, UMETokenizerTransform will be used.

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
        requires_tokenization: bool = True,
        transform_fn: Callable | None = None,
    ):
        self.requires_tokenization = requires_tokenization

        # Set up transform function based on requirements
        if requires_tokenization and transform_fn is None:
            # Use default UME tokenizer for SMILES
            transform_fn = UMETokenizerTransform(
                modality="SMILES",
                max_length=512,  # Default max length for molecules
            )
        # If requires_tokenization=False and transform_fn=None, leave it as None
        # If transform_fn is provided, use it as-is

        super().__init__(
            transform_fn=transform_fn,
            task_type="regression",
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
        )

        # Set tasks
        self.tasks = set(tasks) if tasks is not None else MOLECULEACE_TASKS

    def _process_and_embed(
        self,
        pl_module: L.LightningModule,
        inputs,
        modality: str = "SMILES",
        aggregate: bool = True,
    ):
        """Process inputs and extract embeddings with support for different model types.

        Parameters
        ----------
        pl_module : L.LightningModule
            The lightning module with a model that can extract embeddings
        inputs : various
            Either tokenized inputs (dict with input_ids, attention_mask)
            or raw inputs (list of strings or single string)
        modality : str, default="SMILES"
            The modality of the inputs
        aggregate : bool, default=True
            Whether to average pool over sequence length

        Returns
        -------
        torch.Tensor
            Embeddings tensor
        """
        # Safety check: Detect potentially incompatible model/tokenization combinations
        model_name = getattr(pl_module.__class__, "__name__", str(type(pl_module)))
        is_tokenized_input = isinstance(inputs, dict)
        is_raw_input = isinstance(inputs, (list, str))

        # Check for dangerous combinations - hypothetical molecular models that might be like ESM
        if not self.requires_tokenization and is_tokenized_input:
            raise ValueError(
                f"Model {model_name} expects raw sequences, not tokenized inputs. "
                f"Set requires_tokenization=False when using models that take raw SMILES. "
                f"Current config: requires_tokenization={self.requires_tokenization}, "
                f"inputs are tokenized={is_tokenized_input}"
            )

        # Handle raw sequences directly using embed_sequences method
        if is_raw_input and not isinstance(inputs, dict):
            # Use embed_sequences method directly for raw sequences
            return pl_module.embed_sequences(inputs, modality=modality, aggregate=aggregate)

        # Handle tokenized inputs using embed method
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            # For tokenized inputs, use embed method
            return pl_module.embed(inputs, aggregate=aggregate)

        # Fallback - try to use as is
        else:
            try:
                return pl_module.embed(inputs, aggregate=aggregate)
            except Exception as e:
                raise ValueError(f"Could not process inputs of type {type(inputs)}: {e}") from e

    def _get_embeddings(self, pl_module: L.LightningModule, dataloader, modality: str = "SMILES"):
        """Extract embeddings with enhanced model compatibility."""
        embeddings = []
        targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch

                # Use the enhanced embedding extraction
                batch_embeddings = self._process_and_embed(pl_module, x, modality=modality, aggregate=True)

                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

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
            train_dataset = MoleculeACEDataset(task=task, train=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataset = MoleculeACEDataset(task=task, train=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            try:
                # Get embeddings using enhanced method with SMILES modality
                train_embeddings, train_targets = self._get_embeddings(module, train_loader, modality="SMILES")
                test_embeddings, test_targets = self._get_embeddings(module, test_loader, modality="SMILES")

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
