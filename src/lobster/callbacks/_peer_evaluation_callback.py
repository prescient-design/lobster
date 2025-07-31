import logging
import tempfile
import gc
from collections import defaultdict
from collections.abc import Callable, Sequence

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
from transformers.tokenization_utils_base import BatchEncoding

from lobster.constants import (
    PEER_TASK_CATEGORIES,
    PEER_TASK_SPLITS,
    PEER_TASKS,
    PEER_STRUCTURE_TASKS,
    PEERTask,
    PEERTaskCategory,
    Modality,
)
from lobster.datasets import PEERDataset
from lobster.tokenization import UMETokenizerTransform

from ._linear_probe_callback import LinearProbeCallback
from ._peer_utils import (
    peer_structure_collate_fn,
    peer_default_collate_fn,
    flatten_and_filter_token_embeddings,
    process_secondary_structure_item,
    process_proteinnet_item,
    process_fold_item,
    calculate_mean_metrics,
    get_peer_task_metric,
    convert_numpy_to_python,
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
    """

    def __init__(
        self,
        max_length: int | None = None,
        tasks: Sequence[PEERTask | str] | None = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
        requires_tokenization: bool = True,
        transform_fn: Callable | None = None,
        clear_cache_between_tasks: bool = True,
        manage_memory: bool = True,
        max_embeddings_per_chunk: int = 10000,
    ):
        """Initialize the PEER benchmark callback.

        Parameters
        ----------
        max_length : int, default=None
            Maximum sequence length for tokenization (if needed).
        tasks : Sequence[PEERTask | str] | None, default=None
            Specific PEER tasks to evaluate. If None, all tasks are used.
        batch_size : int, default=32
            Batch size for embedding extraction and evaluation.
        run_every_n_epochs : int | None, default=None
            Run this callback every n epochs. If None, runs every validation epoch.
        requires_tokenization : bool, default=True
            Whether the model requires tokenized inputs (via UMETokenizerTransform) or
            can accept raw sequences directly.
        transform_fn : Callable | None, default=None
            Custom transform function to apply to inputs. If None and requires_tokenization
            is True, UMETokenizerTransform will be used.
        clear_cache_between_tasks : bool, default=True
            Whether to clear dataset cache between tasks to reduce memory usage.
        manage_memory : bool, default=True
            Whether to use enhanced memory management techniques. This includes chunked
            processing, memory cache clearing, and adaptive batch sizing for memory-intensive tasks.
        max_embeddings_per_chunk : int, default=10000
            Maximum number of embeddings to process at once to prevent memory issues.
        """
        self.requires_tokenization = requires_tokenization
        self.clear_cache_between_tasks = clear_cache_between_tasks
        self.manage_memory = manage_memory
        self.max_embeddings_per_chunk = max_embeddings_per_chunk

        if requires_tokenization and transform_fn is None:
            if max_length is None:
                raise ValueError("max_length must be provided if requires_tokenization is True")

            transform_fn = UMETokenizerTransform(
                modality="amino_acid",
                max_length=max_length,
            )
        # If requires_tokenization=False and transform_fn=None, leave it as None
        # If transform_fn is provided, use it as-is

        super().__init__(
            transform_fn=transform_fn,
            # Default task type will be updated per task
            task_type="regression",
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
        )

        self.selected_tasks = set()

        if tasks is not None:
            for task in tasks:
                task_enum = PEERTask(task) if isinstance(task, str) else task
                self.selected_tasks.add(task_enum)

        # If no tasks specified, use all tasks except PROTEINNET (which causes OOM)
        if not self.selected_tasks:
            self.selected_tasks = set(PEER_TASKS.keys()) - {PEERTask.PROTEINNET}

        # Store embedders for different modalities (AA, SMILES)
        self.embedders = {}

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

    def _clear_memory_cache(self):
        """Clear memory caches and force garbage collection."""
        if self.manage_memory:
            # Clear Python garbage
            gc.collect()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.debug("Cleared memory caches and forced garbage collection")

    def _requires_minimal_batch_size(self, task: PEERTask) -> bool:
        """Determine if a task requires minimal batch size (1) due to extreme memory usage or collation issues."""
        # Tasks that require most restrictive settings due to high memory usage or collation complexity
        return task in {
            PEERTask.PROTEINNET,
            PEERTask.SECONDARY_STRUCTURE,
            PEERTask.FOLD,
        }

    def _get_batch_size_for_task(self, task: PEERTask) -> int:
        """Get batch size for a specific task with enhanced memory management."""
        # Tasks that need batch_size=1 due to collation issues or extreme memory usage
        if self._requires_minimal_batch_size(task):
            return 1

        # Moderate reduction for memory-intensive tasks only
        if task in self.memory_intensive_tasks:
            if self.manage_memory:
                return max(1, self.batch_size // 2)  # Less aggressive reduction
            else:
                return max(1, self.batch_size // 2)

        # For non-memory-intensive tasks, use full batch size even with memory management
        return self.batch_size

    def _process_and_embed(
        self,
        pl_module: L.LightningModule,
        inputs: dict[str, Tensor] | list[str] | str | BatchEncoding,
        modality: str = "amino_acid",
        aggregate: bool = True,
    ) -> Tensor:
        """Process inputs and extract embeddings using UME's built-in embedding methods.

        This method is compatible with UME and also supports ESM models.

        Parameters
        ----------
        pl_module : L.LightningModule
            The lightning module with a model that can extract embeddings
        inputs : dict[str, Tensor] | list[str] | str | BatchEncoding
            Either tokenized inputs (dict with input_ids, attention_mask)
            or raw inputs (list of strings or single string)
        modality : str, default="amino_acid"
            The modality of the inputs
        aggregate : bool, default=True
            Whether to average pool over sequence length

        Returns
        -------
        Tensor
            Embeddings tensor of shape (batch_size, hidden_size) if aggregate=True
            or (batch_size, seq_len, hidden_size) if aggregate=False
        """
        # Safety check: Detect potentially incompatible model/tokenization combinations
        model_name = getattr(pl_module, '__class__', {}).get('__name__', str(type(pl_module)))
        is_tokenized_input = isinstance(inputs, (dict, BatchEncoding))
        is_raw_input = isinstance(inputs, (list, str))
        
        # Check for dangerous combinations
        if "ESM" in model_name and self.requires_tokenization and is_tokenized_input:
            raise ValueError(
                f"ESM models expect raw sequences, not tokenized inputs. "
                f"Set requires_tokenization=False when using {model_name}. "
                f"Current config: requires_tokenization={self.requires_tokenization}, "
                f"inputs are tokenized={is_tokenized_input}"
            )

        # Handle raw sequences directly using embed_sequences method
        if isinstance(inputs, (str, list)) and not isinstance(inputs, dict):
            # Use embed_sequences method directly for raw sequences (UME and ESM)
            return pl_module.embed_sequences(inputs, modality=modality, aggregate=aggregate)

        # Handle tokenized inputs using embed method
        elif isinstance(inputs, (dict, BatchEncoding)) and "input_ids" in inputs:
            # For tokenized inputs, use embed method (UME)
            try:
                return pl_module.embed(inputs, aggregate=aggregate)
            except NotImplementedError:
                # ESM doesn't support tokenized inputs, so extract sequences if available
                if hasattr(inputs, "original_sequence"):
                    sequences = inputs.original_sequence
                    return pl_module.embed_sequences(sequences, modality=modality, aggregate=aggregate)
                else:
                    raise ValueError(f"Model {type(pl_module)} doesn't support tokenized inputs "
                                   f"and no original sequences available in the data")

        # Handle mixed case - try to extract sequences if possible
        else:
            # Try to extract sequences from complex input structures
            if hasattr(inputs, "original_sequence"):
                sequences = inputs.original_sequence
                return pl_module.embed_sequences(sequences, modality=modality, aggregate=aggregate)

            # Fallback - try to use as is
            try:
                return pl_module.embed(inputs, aggregate=aggregate)
            except Exception as e:
                raise ValueError(f"Could not process inputs of type {type(inputs)}: {e}") from e

    def _get_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask | None = None
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader with task-specific handling."""
        if task is None:
            # Fall back to parent implementation if no task specified
            return super()._get_embeddings(pl_module, dataloader)

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
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for standard tasks with chunked processing to prevent memory issues."""
        embeddings = []
        targets = []
        current_embeddings = []
        current_targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch

                # Use simplified embedding extraction
                batch_embeddings = self._process_and_embed(pl_module, x, modality=Modality.AMINO_ACID, aggregate=True)

                current_embeddings.append(batch_embeddings.cpu())
                current_targets.append(y.cpu())

                # Check if we've accumulated enough embeddings or if this is the last batch
                current_size = sum(emb.shape[0] for emb in current_embeddings)
                is_last_batch = batch_idx == len(dataloader) - 1

                # Only chunk for memory-intensive tasks, or if we hit the chunk limit
                should_chunk = (
                    task in self.memory_intensive_tasks and current_size >= self.max_embeddings_per_chunk
                ) or is_last_batch

                if should_chunk:
                    # Process current chunk
                    if current_embeddings:
                        chunk_embeddings = torch.cat(current_embeddings, dim=0)
                        chunk_targets = torch.cat(current_targets, dim=0)

                        embeddings.append(chunk_embeddings)
                        targets.append(chunk_targets)

                        # Clear current chunk and force memory cleanup
                        current_embeddings.clear()
                        current_targets.clear()
                        del chunk_embeddings, chunk_targets

                        # Clear memory cache only for memory-intensive tasks
                        if self.manage_memory and task in self.memory_intensive_tasks:
                            self._clear_memory_cache()

        # Final concatenation
        if embeddings:
            final_embeddings = torch.cat(embeddings, dim=0)
            final_targets = torch.cat(targets, dim=0)

            # Clear intermediate results
            embeddings.clear()
            targets.clear()

            return final_embeddings, final_targets
        else:
            return torch.tensor([]), torch.tensor([])

    def _get_task_test_splits(self, task: PEERTask) -> list[str]:
        """Get the most relevant test split for each task.

        Returns only the single most important test split for each task
        to simplify evaluation results.
        """
        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                return ["cb513"]  # Only CB513 benchmark

            case PEERTask.BINDINGDB:
                return ["holdout_test"]  # Only holdout test split

            case PEERTask.FOLD:
                return ["test_superfamily_holdout"]  # Only superfamily holdout for remote homology

            case PEERTask.HUMANPPI | PEERTask.YEASTPPI:
                return ["test"]  # Only standard test split

            case _:
                # Default case: return single test split
                return [split for split in PEER_TASK_SPLITS[task] if split not in ["train", "valid"]][:1]

    def _get_task_datasets(self, task: PEERTask) -> tuple[PEERDataset, dict[str, PEERDataset]]:
        """Get or create train and test datasets for a given task.

        Returns:
            Tuple containing (train_dataset, test_datasets_dict)
            where test_datasets_dict maps split names to datasets
        """
        cache_key = task.value

        if cache_key in self.datasets:
            return self.datasets[cache_key]

        train_split = "train"

        # Get all test splits for this task
        test_splits = self._get_task_test_splits(task)

        # Create train dataset
        train_dataset = PEERDataset(
            task=task,
            split=train_split,
            transform_fn=self.transform_fn,
        )

        # Create test datasets for each test split
        test_datasets = {}
        for split in test_splits:
            test_datasets[split] = PEERDataset(
                task=task,
                split=split,
                transform_fn=self.transform_fn,
            )

        # Cache datasets
        self.datasets[cache_key] = (train_dataset, test_datasets)

        return train_dataset, test_datasets

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
                        embeddings1 = self._process_and_embed(
                            pl_module, seq1, modality=Modality.AMINO_ACID, aggregate=True
                        )
                        embeddings2 = self._process_and_embed(
                            pl_module, seq2, modality=Modality.AMINO_ACID, aggregate=True
                        )

                        # Concatenate the embeddings
                        batch_embeddings = torch.cat([embeddings1, embeddings2], dim=1)

                    # Handle protein-ligand interactions
                    elif task in {PEERTask.BINDINGDB, PEERTask.PDBBIND}:
                        # Get embeddings for protein and ligand
                        protein_embeddings = self._process_and_embed(
                            pl_module, seq1, modality=Modality.AMINO_ACID, aggregate=True
                        )
                        ligand_embeddings = self._process_and_embed(
                            pl_module, seq2, modality=Modality.SMILES, aggregate=True
                        )

                        # Concatenate the embeddings
                        batch_embeddings = torch.cat([protein_embeddings, ligand_embeddings], dim=1)
                else:
                    raise ValueError(f"Expected paired inputs for task {task}, but got: {type(inputs)}")

                # Make sure targets cast correctly
                if task_type == "regression":
                    y = y.float()
                else:
                    y = y.long()

                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _flatten_and_filter_token_embeddings(
        self,
        batch_embeddings: Tensor,
        targets: Tensor,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        ignore_target_value: int = -100,
    ) -> tuple[Tensor, Tensor]:
        """Helper method to flatten embeddings and filter special tokens for token-level tasks."""
        tokenizer = self.transform_fn.tokenizer if hasattr(self.transform_fn, "tokenizer") else None
        return flatten_and_filter_token_embeddings(
            batch_embeddings=batch_embeddings,
            targets=targets,
            input_ids=input_ids,
            attention_mask=attention_mask,
            ignore_target_value=ignore_target_value,
            tokenizer=tokenizer,
            requires_tokenization=self.requires_tokenization,
        )

    def _get_structure_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings for structure prediction tasks with special handling."""
        embeddings = []
        targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Handle both tuple and list batch formats
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x, y = batch

                    # If x and y are lists (from custom collation), process each item
                    if isinstance(x, list) and isinstance(y, list):
                        # Process each item in the batch individually
                        for single_x, single_y in zip(x, y):
                            self._process_single_structure_item(
                                pl_module, task, single_x, single_y, embeddings, targets
                            )
                    else:
                        # Regular batch processing for backward compatibility
                        self._process_single_structure_item(pl_module, task, x, y, embeddings, targets)
                else:
                    raise ValueError(f"Expected batch to be a tuple/list of (inputs, targets), got {type(batch)}")

        # If we have no valid embeddings, return empty tensors
        if not embeddings or not targets:
            return torch.tensor([]), torch.tensor([])

        # Handle concatenation with proper error handling
        try:
            concatenated_embeddings = torch.cat(embeddings, dim=0)
            concatenated_targets = torch.cat(targets, dim=0)
            return concatenated_embeddings, concatenated_targets
        except RuntimeError as e:
            logger.warning(f"Failed to concatenate embeddings/targets: {e}")
            logger.warning(f"Embeddings shapes: {[emb.shape for emb in embeddings]}")
            logger.warning(f"Targets shapes: {[tgt.shape for tgt in targets]}")
            return torch.tensor([]), torch.tensor([])

    def _process_single_structure_item(self, pl_module: L.LightningModule, task: PEERTask, x, y, embeddings, targets):
        """Process a single structure prediction item."""
        # Extract input_ids and attention_mask if available (for token filtering)
        input_ids = x.get("input_ids") if isinstance(x, dict) else None
        attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
        tokenizer = self.transform_fn.tokenizer if hasattr(self.transform_fn, "tokenizer") else None

        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                result_embeddings, result_targets = process_secondary_structure_item(
                    pl_module=pl_module,
                    x=x,
                    y=y,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    process_and_embed_fn=self._process_and_embed,
                    tokenizer=tokenizer,
                    requires_tokenization=self.requires_tokenization,
                )
                if result_embeddings.numel() > 0 and result_targets.numel() > 0:
                    embeddings.append(result_embeddings)
                    targets.append(result_targets)

            case PEERTask.PROTEINNET:
                result_embeddings, result_targets = process_proteinnet_item(
                    pl_module=pl_module,
                    x=x,
                    y=y,
                    process_and_embed_fn=self._process_and_embed,
                    max_length=512,  # Default max length
                    manage_memory=self.manage_memory,
                    clear_memory_cache_fn=self._clear_memory_cache,
                )
                if result_embeddings.numel() > 0 and result_targets.numel() > 0:
                    embeddings.append(result_embeddings)
                    targets.append(result_targets)

            case PEERTask.FOLD:
                result_embeddings, result_targets = process_fold_item(
                    pl_module=pl_module,
                    x=x,
                    y=y,
                    process_and_embed_fn=self._process_and_embed,
                )
                if result_embeddings.numel() > 0 and result_targets.numel() > 0:
                    embeddings.append(result_embeddings)
                    targets.append(result_targets)

    def _train_probe(self, embeddings: Tensor, targets: Tensor, task: PEERTask = None):
        """Train a probe on the given embeddings and targets with task-specific handling."""
        if task is None:
            # Fallback to parent implementation if no task specified
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
        """Evaluate a trained probe on the given embeddings and targets, returning task-specific metrics."""
        if task is None:
            # Fallback to parent implementation if no task specified
            return super()._evaluate_probe(probe, embeddings, targets)

        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()

        relevant_metric = get_peer_task_metric(task)
        metrics = {}

        # Get predictions from the probe
        predictions = probe.predict(embeddings_np)

        # Calculate the relevant metric based on the task
        match relevant_metric:
            case "accuracy":
                # For classification tasks
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    # Multi-class: reshape if needed
                    targets_np = targets_np.ravel() if targets_np.ndim > 1 else targets_np

                metrics["accuracy"] = accuracy_score(targets_np, predictions)

            case "spearman":
                # For regression tasks requiring Spearman correlation
                if predictions.ndim > 1:
                    # Multiple outputs - calculate correlation for each and average
                    correlations = []
                    for i in range(predictions.shape[1]):
                        corr, _ = spearmanr(targets_np[:, i], predictions[:, i])
                        correlations.append(corr)
                    metrics["spearman"] = np.mean(correlations)
                else:
                    # Single output
                    corr, _ = spearmanr(targets_np, predictions)
                    metrics["spearman"] = corr

            case "rmse":
                # For regression tasks requiring RMSE
                rmse = np.sqrt(mean_squared_error(targets_np, predictions))
                metrics["rmse"] = rmse

            case "l5_precision":
                # For contact prediction tasks
                if task == PEERTask.PROTEINNET:
                    # For contact prediction, we need L/5 precision
                    # We sort predictions by confidence and take top L/5
                    if hasattr(probe, "predict_proba"):
                        # Get probabilities for positive class
                        probs = probe.predict_proba(embeddings_np)[:, 1]

                        # Get sequence length L (approximated from data)
                        # This is simplified - in practice you might want to group by protein
                        L = int(np.sqrt(len(targets_np) * 2))
                        top_k = max(1, L // 5)

                        # Get top L/5 predictions
                        top_indices = np.argsort(probs)[-top_k:]
                        metrics["l5_precision"] = np.mean(targets_np[top_indices])
                    else:
                        # Fallback if predict_proba is not available
                        metrics["accuracy"] = accuracy_score(targets_np, predictions)

        # Convert NumPy scalars to Python types for clean YAML formatting
        return convert_numpy_to_python(metrics)

    def _get_collate_fn(self, task: PEERTask):
        """Get the appropriate collation function for the task."""
        # Tasks that need custom collation due to variable-length tensors
        return peer_structure_collate_fn if task in PEER_STRUCTURE_TASKS else peer_default_collate_fn

    def _evaluate_task(
        self, task: PEERTask, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> dict[str, dict[str, float]]:
        """Evaluate a single PEER task across all its test splits.

        Returns:
            Dict mapping split names to dictionaries of metrics
        """
        try:
            # Clear memory cache before starting memory-intensive tasks
            if task in self.memory_intensive_tasks:
                self._clear_memory_cache()
                logger.debug(f"Cleared memory cache before starting task: {task}")

            train_dataset, test_datasets = self._get_task_datasets(task)

            # Get appropriate collation function for this task
            collate_fn = self._get_collate_fn(task)

            # Clear GPU cache for memory-intensive tasks
            if task in self.memory_intensive_tasks:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(f"Cleared GPU cache before memory-intensive task: {task}")

            # Get batch size for this task
            batch_size = self._get_batch_size_for_task(task)

            # Use fewer workers for memory-intensive tasks, but not zero
            if self._requires_minimal_batch_size(task):
                num_workers = 1  # Single worker to minimize overhead
            elif task in self.memory_intensive_tasks:
                num_workers = 2  # Reduced workers for memory-intensive tasks
            else:
                num_workers = 4  # Full workers for regular tasks

            logger.info(f"Starting task {task} with batch_size={batch_size}, num_workers={num_workers}")

            # Get train embeddings and probe
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
            )
            train_embeddings, train_targets = self._get_embeddings(pl_module, train_loader, task)

            # Check if we have valid training data
            if train_embeddings.numel() == 0 or train_targets.numel() == 0:
                logger.warning(f"No valid training data for task {task}, skipping...")
                return {}

            probe = self._train_probe(train_embeddings, train_targets, task)
            self.probes[task.value] = probe

            # Clear training data to free memory
            del train_embeddings, train_targets
            if task in self.memory_intensive_tasks:
                self._clear_memory_cache()

            # Evaluate on each test split
            split_metrics = {}
            for split_name, test_dataset in test_datasets.items():
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
                )

                test_embeddings, test_targets = self._get_embeddings(pl_module, test_loader, task)

                # Check if we have valid test data
                if test_embeddings.numel() == 0 or test_targets.numel() == 0:
                    logger.warning(f"No valid test data for task {task}, split {split_name}, skipping...")
                    continue

                # Evaluate probe with task-specific metrics
                metrics = self._evaluate_probe(probe, test_embeddings, test_targets, task)
                split_metrics[split_name] = metrics

                # Clear test data to free memory
                del test_embeddings, test_targets
                if task in self.memory_intensive_tasks:
                    self._clear_memory_cache()

            return split_metrics

        except Exception as e:
            logger.exception(f"Error evaluating task {task}: {str(e)}")
            # Clear memory on error
            self._clear_memory_cache()
            return {}

    def _run_evaluation(
        self,
        pl_module: L.LightningModule,
        trainer: L.Trainer,
        step: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Core evaluation logic used by both on_validation_epoch_end and evaluate."""
        # Track metrics
        all_metrics = defaultdict(list)
        all_task_metrics = {}
        split_count = 0

        # Evaluate each selected task
        for task in tqdm(self.selected_tasks, desc=f"{self.__class__.__name__}"):
            split_metrics = self._evaluate_task(task, trainer, pl_module)

            # Get the relevant metric for this task
            relevant_metric = get_peer_task_metric(task)

            # Store task metrics for return value (simplified since we only have one split per task)
            if split_metrics:
                # Get the single split result and store it directly under the task name
                split_name, metrics = next(iter(split_metrics.items()))
                all_task_metrics[task.value] = metrics

                # Only log the relevant metric for this task
                if relevant_metric in metrics:
                    value = metrics[relevant_metric]
                    metric_key = f"peer_linear_probe/{task}/{relevant_metric}"
                    trainer.logger.log_metrics({metric_key: value}, step=step)

                    # Collect metrics for global averages - using the metric type as the key
                    all_metrics[relevant_metric].append(value)

                split_count += 1

            # Clear dataset cache between tasks if enabled
            if self.clear_cache_between_tasks:
                task_cache_key = task.value
                if task_cache_key in self.datasets:
                    del self.datasets[task_cache_key]
                    logger.debug(f"Cleared dataset cache for task {task}")

            # Force memory cleanup after memory-intensive tasks
            if task in self.memory_intensive_tasks:
                self._clear_memory_cache()
                logger.debug(f"Completed memory-intensive task {task} and cleared memory")

        # Calculate and log overall averages for each metric type
        mean_metrics = calculate_mean_metrics(all_metrics)
        for metric_name, avg_value in mean_metrics.items():
            metric_key = f"peer_linear_probe/mean/{metric_name}"
            trainer.logger.log_metrics({metric_key: avg_value}, step=step)

        # Add mean metrics to result
        all_task_metrics["mean"] = mean_metrics

        # Log total number of splits evaluated
        trainer.logger.log_metrics({"peer_linear_probe/total_splits_evaluated": split_count}, step=step)

        # Convert NumPy scalars to Python types for clean YAML formatting
        return convert_numpy_to_python(all_task_metrics)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes on PEER tasks at specified epochs.

        This method is automatically called by Lightning during training.
        """
        if self._skip(trainer):
            return

        self._run_evaluation(pl_module, trainer, step=trainer.current_epoch)

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate model on PEER benchmark tasks using linear probes.

        This method can be called manually at any time to evaluate a model.

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
        # Create a simple trainer with logger if none provided
        if trainer is None:
            trainer = L.Trainer(
                logger=CSVLogger(tempfile.mkdtemp()),
                accelerator="auto",
                devices=1,
            )

        return self._run_evaluation(module, trainer, step=0)
