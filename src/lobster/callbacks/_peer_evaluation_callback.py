import logging
import tempfile
from collections import defaultdict
from collections.abc import Sequence

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.multioutput import MultiOutputClassifier
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from lobster.constants import (
    PEER_STRUCTURE_TASKS,
    PEER_TASK_CATEGORIES,
    PEER_TASK_SPLITS,
    PEER_TASKS,
    Modality,
    PEERTask,
    PEERTaskCategory,
)
from lobster.datasets import PEERDataset

from ._linear_probe_callback import LinearProbeCallback
from ._peer_utils import (
    calculate_mean_metrics,
    convert_numpy_to_python,
    get_peer_task_metric,
    peer_default_collate_fn,
    peer_structure_collate_fn,
)

logger = logging.getLogger(__name__)


class PEEREvaluationCallback(LinearProbeCallback):
    """Callback for evaluating model embeddings on PEER benchmark tasks.

    By default, evaluates 16 out of 17 PEER tasks (excludes PROTEINNET due to high-memory issues).

    The callback handles various input types:
    - Single sequence inputs (function prediction, localization)
    - Paired sequence inputs (protein-protein, protein-ligand interactions)
    - Per-residue tasks (secondary structure prediction)

    Available task categories (16 tasks by default):
    - Function prediction: fluorescence, stability, betalactamase, solubility, etc.
    - Localization: binary and multi-class subcellular localization
    - Protein-ligand interaction: binding affinity (BindingDB, PDBbind)
    - Protein-protein interaction: human/yeast PPI classification and affinity
    - Structure prediction: fold classification, secondary structure

    Excluded by default: PROTEINNET (contact map prediction, quadratic memory scaling)

    Reference: Guo et al. (2023) "PEER: A Comprehensive and Multi-Task Benchmark for
    Protein Sequence Understanding" https://arxiv.org/abs/2206.02096

    Model compatibility:
    - UME models: Use requires_tokenization=True (default)
    - ESM models: Use requires_tokenization=False

    WARNING: Using ESM models with requires_tokenization=True will cause runtime errors!
    ESM models expect raw sequences, not tokenized inputs.

    Both UME and ESM models must implement:
    - embed_sequences(sequences, modality, aggregate) method
    - embed(inputs, aggregate) method (UME for tokenized inputs, ESM for raw sequences)

    Parameters
    ----------
    tasks: Sequence[PEERTask | str] | None
        The tasks to evaluate.
    batch_size: int
        The batch size to use for evaluation.
    run_every_n_epochs: int | None
        The number of epochs to run the evaluation every.
    clear_cache_between_tasks: bool
        Whether to clear the cache between tasks.
    max_embeddings_per_chunk: int
        The maximum number of embeddings to process at once.
    """

    def __init__(
        self,
        tasks: Sequence[PEERTask | str] | None = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
        clear_cache_between_tasks: bool = True,
        max_embeddings_per_chunk: int = 10000,
    ):
        super().__init__(
            transform_fn=None,
            task_type="regression",  # Will be updated per task
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
        )

        self.clear_cache_between_tasks = clear_cache_between_tasks
        self.max_embeddings_per_chunk = max_embeddings_per_chunk

        # Set up selected tasks
        self.selected_tasks = set()
        if tasks is not None:
            for task in tasks:
                task_enum = PEERTask(task) if isinstance(task, str) else task
                self.selected_tasks.add(task_enum)

        # If no tasks specified, use all tasks except PROTEINNET (which causes OOM)
        if not self.selected_tasks:
            self.selected_tasks = set(PEER_TASKS.keys()) - {PEERTask.PROTEINNET}

        # Cache for datasets
        self.datasets = {}

        # Define memory-intensive tasks that get a reduced batch size
        self.memory_intensive_tasks = {
            PEERTask.PROTEINNET,
            PEERTask.SECONDARY_STRUCTURE,
            PEERTask.FOLD,
            PEERTask.SUBCELLULAR_LOCALIZATION,
            PEERTask.BINDINGDB,
            PEERTask.PDBBIND,
        }

    def _get_batch_size_for_task(self, task: PEERTask) -> int:
        """Get batch size for a specific task with memory management."""
        # Tasks that need batch_size=1 due to collation issues or extreme memory usage
        if task in {PEERTask.PROTEINNET, PEERTask.SECONDARY_STRUCTURE, PEERTask.FOLD}:
            return 1

        # Moderate reduction for memory-intensive tasks
        if task in self.memory_intensive_tasks:
            return max(1, self.batch_size // 2)

        return self.batch_size

    def _embed_sequences(
        self,
        pl_module: L.LightningModule,
        sequences: str | list[str],
        modality: str = "amino_acid",
        aggregate: bool = True,
    ) -> Tensor:
        """Extract embeddings using the model's embed_sequences method.

        Parameters
        ----------
        pl_module : L.LightningModule
            The lightning module with embed_sequences method
        sequences : str | list[str]
            Raw sequences to embed
        modality : str, default="amino_acid"
            The modality of the sequences
        aggregate : bool, default=True
            Whether to aggregate embeddings over sequence length

        Returns
        -------
        Tensor
            Embeddings tensor
        """
        return pl_module.embed_sequences(sequences, modality=modality, aggregate=aggregate)

    def _get_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask | None = None
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader."""
        if task is None:
            return self._get_standard_embeddings(pl_module, dataloader, task)

        category = PEER_TASK_CATEGORIES[task]

        # Protein-protein or protein-ligand interactions
        if category in {PEERTaskCategory.PROTEIN_PROTEIN_INTERACTION, PEERTaskCategory.PROTEIN_LIGAND_INTERACTION}:
            return self._get_paired_embeddings(pl_module, dataloader, task)

        # Structure prediction tasks
        elif category == PEERTaskCategory.STRUCTURE_PREDICTION:
            return self._get_structure_embeddings(pl_module, dataloader, task)

        # Standard single sequence tasks (function prediction, localization)
        else:
            return self._get_standard_embeddings(pl_module, dataloader, task)

    def _get_standard_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask | None = None
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for standard tasks with chunked processing."""
        embeddings = []
        targets = []
        current_embeddings = []
        current_targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                sequences, y = batch

                # Extract sequences if they're in a container
                if hasattr(sequences, "sequences"):
                    sequences = sequences.sequences
                elif isinstance(sequences, dict) and "sequences" in sequences:
                    sequences = sequences["sequences"]
                elif isinstance(sequences, (list, tuple)) and all(isinstance(seq, str) for seq in sequences):
                    sequences = sequences
                else:
                    raise ValueError(
                        f"Expected sequences to be a list or dict with 'sequences' key, got {type(sequences)}"
                    )

                # Get embeddings using embed_sequences
                batch_embeddings = self._embed_sequences(
                    pl_module, sequences, modality=Modality.AMINO_ACID, aggregate=True
                )

                current_embeddings.append(batch_embeddings.cpu())
                current_targets.append(y.cpu())

                # Check if we should process current chunk
                current_size = sum(emb.shape[0] for emb in current_embeddings)
                is_last_batch = batch_idx == len(dataloader) - 1

                should_chunk = (
                    task and task in self.memory_intensive_tasks and current_size >= self.max_embeddings_per_chunk
                ) or is_last_batch

                if should_chunk and current_embeddings:
                    chunk_embeddings = torch.cat(current_embeddings, dim=0)
                    chunk_targets = torch.cat(current_targets, dim=0)

                    embeddings.append(chunk_embeddings)
                    targets.append(chunk_targets)

                    current_embeddings.clear()
                    current_targets.clear()
                    del chunk_embeddings, chunk_targets

        # Final concatenation
        if embeddings:
            final_embeddings = torch.cat(embeddings, dim=0)
            final_targets = torch.cat(targets, dim=0)
            embeddings.clear()
            targets.clear()
            return final_embeddings, final_targets
        else:
            return torch.tensor([]), torch.tensor([])

    def _get_paired_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for paired inputs (protein-protein or protein-ligand)."""
        task_type = PEER_TASKS[task][0]
        embeddings = []
        targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, y = batch

                # For paired inputs, inputs will be a list of two items
                if isinstance(inputs, list) and len(inputs) == 2:
                    seq1, seq2 = inputs

                    # Handle protein-protein interactions
                    if task in {PEERTask.HUMANPPI, PEERTask.YEASTPPI, PEERTask.PPIAFFINITY}:
                        # Get embeddings for each protein separately
                        embeddings1 = self._embed_sequences(
                            pl_module, seq1, modality=Modality.AMINO_ACID, aggregate=True
                        )
                        embeddings2 = self._embed_sequences(
                            pl_module, seq2, modality=Modality.AMINO_ACID, aggregate=True
                        )

                        # Concatenate the embeddings
                        batch_embeddings = torch.cat([embeddings1, embeddings2], dim=1)

                    # Handle protein-ligand interactions
                    elif task in {PEERTask.BINDINGDB, PEERTask.PDBBIND}:
                        # Get embeddings for protein and ligand
                        protein_embeddings = self._embed_sequences(
                            pl_module, seq1, modality=Modality.AMINO_ACID, aggregate=True
                        )
                        ligand_embeddings = self._embed_sequences(
                            pl_module, seq2, modality=Modality.SMILES, aggregate=True
                        )

                        # Concatenate the embeddings
                        batch_embeddings = torch.cat([protein_embeddings, ligand_embeddings], dim=1)
                else:
                    raise ValueError(f"Expected paired inputs for task {task}, but got: {type(inputs)}")

                # Cast targets correctly
                if task_type == "regression":
                    y = y.float()
                else:
                    y = y.long()

                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _get_structure_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for structure prediction tasks."""
        embeddings = []
        targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x, y = batch

                    # Handle batch format from custom collation
                    if isinstance(x, list) and isinstance(y, list):
                        for single_x, single_y in zip(x, y):
                            self._process_single_structure_item(
                                pl_module, task, single_x, single_y, embeddings, targets
                            )
                    else:
                        self._process_single_structure_item(pl_module, task, x, y, embeddings, targets)
                else:
                    raise ValueError(f"Expected batch to be a tuple/list of (inputs, targets), got {type(batch)}")

        if not embeddings or not targets:
            return torch.tensor([]), torch.tensor([])

        try:
            concatenated_embeddings = torch.cat(embeddings, dim=0)
            concatenated_targets = torch.cat(targets, dim=0)
            return concatenated_embeddings, concatenated_targets
        except RuntimeError as e:
            logger.warning(f"Failed to concatenate embeddings/targets: {e}")
            return torch.tensor([]), torch.tensor([])

    def _process_single_structure_item(self, pl_module: L.LightningModule, task: PEERTask, x, y, embeddings, targets):
        """Process a single structure prediction item."""
        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                # Extract sequence from input
                if isinstance(x, dict) and "sequences" in x:
                    sequences = x["sequences"]
                elif hasattr(x, "sequences"):
                    sequences = x.sequences
                else:
                    sequences = x

                # Get per-residue embeddings (no aggregation)
                batch_embeddings = self._embed_sequences(
                    pl_module, sequences, modality=Modality.AMINO_ACID, aggregate=False
                )

                # Flatten embeddings and targets for per-residue prediction
                if batch_embeddings.dim() == 3:  # (batch, seq_len, hidden_dim)
                    batch_embeddings = batch_embeddings.view(-1, batch_embeddings.size(-1))
                if y.dim() == 2:  # (batch, seq_len)
                    y = y.view(-1)

                # Filter out padding tokens (assuming -100 is ignore value)
                valid_mask = y != -100
                if valid_mask.any():
                    result_embeddings = batch_embeddings[valid_mask]
                    result_targets = y[valid_mask]

                    if result_embeddings.numel() > 0 and result_targets.numel() > 0:
                        embeddings.append(result_embeddings.cpu())
                        targets.append(result_targets.cpu())

            case PEERTask.FOLD:
                # Extract sequence from input
                if isinstance(x, dict) and "sequences" in x:
                    sequences = x["sequences"]
                elif hasattr(x, "sequences"):
                    sequences = x.sequences
                else:
                    sequences = x

                # Get aggregated embeddings for fold classification
                batch_embeddings = self._embed_sequences(
                    pl_module, sequences, modality=Modality.AMINO_ACID, aggregate=True
                )

                if batch_embeddings.numel() > 0 and y.numel() > 0:
                    embeddings.append(batch_embeddings.cpu())
                    targets.append(y.cpu())

            case PEERTask.PROTEINNET:
                # Extract sequence from input
                if isinstance(x, dict) and "sequences" in x:
                    sequences = x["sequences"]
                elif hasattr(x, "sequences"):
                    sequences = x.sequences
                else:
                    sequences = x

                # Get per-residue embeddings for contact prediction
                batch_embeddings = self._embed_sequences(
                    pl_module, sequences, modality=Modality.AMINO_ACID, aggregate=False
                )

                # Process contact maps (this is simplified - real implementation would be more complex)
                if batch_embeddings.numel() > 0 and y.numel() > 0:
                    # For contact prediction, we typically need pairwise representations
                    # This is a simplified version
                    seq_len = batch_embeddings.size(1)
                    # Create pairwise features by concatenating residue embeddings
                    pairwise_embeddings = []
                    pairwise_targets = []

                    for i in range(seq_len):
                        for j in range(i + 1, seq_len):
                            pair_emb = torch.cat([batch_embeddings[0, i], batch_embeddings[0, j]], dim=0)
                            pairwise_embeddings.append(pair_emb)
                            if y.dim() == 2:  # Contact map
                                pairwise_targets.append(y[i, j])

                    if pairwise_embeddings:
                        result_embeddings = torch.stack(pairwise_embeddings)
                        result_targets = torch.stack(pairwise_targets)

                        embeddings.append(result_embeddings.cpu())
                        targets.append(result_targets.cpu())

    def _get_task_test_splits(self, task: PEERTask) -> list[str]:
        """Get the most relevant test split for each task."""
        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                return ["cb513"]
            case PEERTask.BINDINGDB:
                return ["holdout_test"]
            case PEERTask.FOLD:
                return ["test_superfamily_holdout"]
            case PEERTask.HUMANPPI | PEERTask.YEASTPPI:
                return ["test"]
            case _:
                return [split for split in PEER_TASK_SPLITS[task] if split not in ["train", "valid"]][:1]

    def _get_task_datasets(self, task: PEERTask) -> tuple[PEERDataset, dict[str, PEERDataset]]:
        """Get or create train and test datasets for a given task."""
        cache_key = task.value

        if cache_key in self.datasets:
            return self.datasets[cache_key]

        train_split = "train"
        test_splits = self._get_task_test_splits(task)

        # Create datasets with no transform (raw sequences)
        train_dataset = PEERDataset(task=task, split=train_split, transform_fn=None)

        test_datasets = {}
        for split in test_splits:
            test_datasets[split] = PEERDataset(task=task, split=split, transform_fn=None)

        # Cache datasets
        self.datasets[cache_key] = (train_dataset, test_datasets)
        return train_dataset, test_datasets

    def _train_probe(self, embeddings: Tensor, targets: Tensor, task: PEERTask = None):
        """Train a probe on the given embeddings and targets."""
        if task is None:
            return super()._train_probe(embeddings, targets)

        # Get task type and num_classes from constants
        task_type, num_classes = PEER_TASKS[task]

        # Set metrics based on the task type and num_classes
        self._set_metrics(task_type, num_classes)

        # Train probe based on task type
        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()

        match task_type:
            case "regression":
                probe = LinearRegression()
                probe.fit(embeddings_np, targets_np)
            case "multilabel":
                base_classifier = LogisticRegression(random_state=42, max_iter=1000)
                probe = MultiOutputClassifier(base_classifier)
                probe.fit(embeddings_np, targets_np)
            case "binary" | "multiclass":
                probe = LogisticRegression(random_state=42, max_iter=1000)
                probe.fit(embeddings_np, targets_np.ravel())

        return probe

    def _evaluate_probe(self, probe, embeddings: Tensor, targets: Tensor, task: PEERTask = None) -> dict[str, float]:
        """Evaluate a trained probe on the given embeddings and targets."""
        if task is None:
            return super()._evaluate_probe(probe, embeddings, targets)

        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()

        relevant_metric = get_peer_task_metric(task)
        metrics = {}

        predictions = probe.predict(embeddings_np)

        match relevant_metric:
            case "accuracy":
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    targets_np = targets_np.ravel() if targets_np.ndim > 1 else targets_np
                metrics["accuracy"] = accuracy_score(targets_np, predictions)

            case "spearman":
                if predictions.ndim > 1:
                    correlations = []
                    for i in range(predictions.shape[1]):
                        corr, _ = spearmanr(targets_np[:, i], predictions[:, i])
                        correlations.append(corr)
                    metrics["spearman"] = np.mean(correlations)
                else:
                    corr, _ = spearmanr(targets_np, predictions)
                    metrics["spearman"] = corr

            case "rmse":
                rmse = np.sqrt(mean_squared_error(targets_np, predictions))
                metrics["rmse"] = rmse

            case "l5_precision":
                if task == PEERTask.PROTEINNET and hasattr(probe, "predict_proba"):
                    probs = probe.predict_proba(embeddings_np)[:, 1]
                    L = int(np.sqrt(len(targets_np) * 2))
                    top_k = max(1, L // 5)
                    top_indices = np.argsort(probs)[-top_k:]
                    metrics["l5_precision"] = np.mean(targets_np[top_indices])
                else:
                    metrics["accuracy"] = accuracy_score(targets_np, predictions)

        return convert_numpy_to_python(metrics)

    def _get_collate_fn(self, task: PEERTask):
        """Get the appropriate collation function for the task."""
        return peer_structure_collate_fn if task in PEER_STRUCTURE_TASKS else peer_default_collate_fn

    def _evaluate_task(
        self, task: PEERTask, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> dict[str, dict[str, float]]:
        """Evaluate a single PEER task across all its test splits."""
        try:
            train_dataset, test_datasets = self._get_task_datasets(task)
            collate_fn = self._get_collate_fn(task)
            batch_size = self._get_batch_size_for_task(task)

            # Determine number of workers
            if task in {PEERTask.PROTEINNET, PEERTask.SECONDARY_STRUCTURE, PEERTask.FOLD}:
                num_workers = 1
            elif task in self.memory_intensive_tasks:
                num_workers = 2
            else:
                num_workers = 4

            logger.info(f"Starting task {task} with batch_size={batch_size}, num_workers={num_workers}")

            # Get train embeddings and train probe
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
            )
            train_embeddings, train_targets = self._get_embeddings(pl_module, train_loader, task)

            if train_embeddings.numel() == 0 or train_targets.numel() == 0:
                logger.warning(f"No valid training data for task {task}, skipping...")
                return {}

            probe = self._train_probe(train_embeddings, train_targets, task)
            self.probes[task.value] = probe

            # Clear training data to free memory
            del train_embeddings, train_targets

            # Evaluate on each test split
            split_metrics = {}
            for split_name, test_dataset in test_datasets.items():
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
                )

                test_embeddings, test_targets = self._get_embeddings(pl_module, test_loader, task)

                if test_embeddings.numel() == 0 or test_targets.numel() == 0:
                    logger.warning(f"No valid test data for task {task}, split {split_name}, skipping...")
                    continue

                metrics = self._evaluate_probe(probe, test_embeddings, test_targets, task)
                split_metrics[split_name] = metrics

                # Clear test data to free memory
                del test_embeddings, test_targets

            return split_metrics

        except Exception as e:
            logger.exception(f"Error evaluating task {task}: {str(e)}")
            return {}

    def _run_evaluation(
        self,
        pl_module: L.LightningModule,
        trainer: L.Trainer,
        step: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Core evaluation logic used by both on_validation_epoch_end and evaluate."""
        all_metrics = defaultdict(list)
        all_task_metrics = {}
        split_count = 0

        # Evaluate each selected task
        for task in tqdm(self.selected_tasks, desc=f"{self.__class__.__name__}"):
            split_metrics = self._evaluate_task(task, trainer, pl_module)

            relevant_metric = get_peer_task_metric(task)

            if split_metrics:
                # Get the single split result and store it directly under the task name
                split_name, metrics = next(iter(split_metrics.items()))
                all_task_metrics[task.value] = metrics

                # Log the relevant metric for this task
                if relevant_metric in metrics:
                    value = metrics[relevant_metric]
                    metric_key = f"peer_linear_probe/{task}/{relevant_metric}"
                    trainer.logger.log_metrics({metric_key: value}, step=step)

                    # Collect metrics for global averages
                    all_metrics[relevant_metric].append(value)

                split_count += 1

            # Clear dataset cache between tasks if enabled
            if self.clear_cache_between_tasks:
                task_cache_key = task.value
                if task_cache_key in self.datasets:
                    del self.datasets[task_cache_key]

        # Calculate and log overall averages for each metric type
        mean_metrics = calculate_mean_metrics(all_metrics)
        for metric_name, avg_value in mean_metrics.items():
            metric_key = f"peer_linear_probe/mean/{metric_name}"
            trainer.logger.log_metrics({metric_key: avg_value}, step=step)

        # Add mean metrics to result
        all_task_metrics["mean"] = mean_metrics

        # Log total number of splits evaluated
        trainer.logger.log_metrics({"peer_linear_probe/total_splits_evaluated": split_count}, step=step)

        return convert_numpy_to_python(all_task_metrics)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes on PEER tasks at specified epochs."""
        if self._skip(trainer):
            return

        self._run_evaluation(pl_module, trainer, step=trainer.current_epoch)

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate model on PEER benchmark tasks using linear probes.

        Parameters
        ----------
        module : L.LightningModule
            The model to evaluate
        trainer : L.Trainer | None, optional
            Trainer for logging metrics, by default None

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary of task results with metrics and averages
        """
        if trainer is None:
            trainer = L.Trainer(
                logger=CSVLogger(tempfile.mkdtemp()),
                accelerator="auto",
                devices=1,
            )

        return self._run_evaluation(module, trainer, step=0)
