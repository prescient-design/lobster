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

from lobster.constants import PEER_TASK_CATEGORIES, PEER_TASK_SPLITS, PEER_TASKS, PEERTask, PEERTaskCategory
from lobster.datasets import PEERDataset
from lobster.tokenization import UMETokenizerTransform

from ._linear_probe_callback import LinearProbeCallback

logger = logging.getLogger(__name__)


def peer_structure_collate_fn(batch):
    """Custom collation function for PEER structure prediction tasks.

    Handles variable-length tensors that can't be stacked by default collation.
    """
    inputs = []
    targets = []

    for item in batch:
        x, y = item
        inputs.append(x)
        targets.append(y)

    # For structure tasks, we don't stack the targets - return them as lists
    return inputs, targets


def peer_default_collate_fn(batch):
    """Default collation function for PEER tasks that can be batched normally."""
    try:
        return torch.utils.data.default_collate(batch)
    except Exception:
        # If default collation fails, fall back to list format
        return peer_structure_collate_fn(batch)


class PEEREvaluationCallback(LinearProbeCallback):
    """Callback for evaluating model embeddings on PEER benchmark tasks:

    By default, evaluates 16 out of 17 PEER tasks (excludes PROTEINNET due to high-memory issues).

    The callback handles various input types including:
    - Single sequence inputs
    - Paired sequence inputs (protein-protein, protein-ligand)
    - Per-residue tasks (secondary structure)

    Available tasks (16 by default):
    - Function prediction tasks (regression):
        - "aav": AAV variant fitness
        - "betalactamase": Beta-lactamase stability
        - "fluorescence": Protein fluorescence
        - "gb1": GB1 protein stability
        - "stability": Protein stability
        - "thermostability": Protein thermostability

    - Function prediction tasks (classification):
        - "solubility": Protein solubility (binary, 2 classes)

    - Localization prediction tasks:
        - "binarylocalization": Binary subcellular localization (binary, 2 classes)
        - "subcellularlocalization": Multi-class subcellular localization (multiclass, 10 classes)

    - Protein-ligand interaction tasks (regression):
        - "bindingdb": Protein-ligand binding affinity (BindingDB)
        - "pdbbind": Protein-ligand binding affinity (PDBbind)

    - Protein-protein interaction tasks:
        - "humanppi": Human protein-protein interactions (binary, 2 classes)
        - "ppiaffinity": Protein-protein binding affinity (regression)
        - "yeastppi": Yeast protein-protein interactions (binary, 2 classes)

    - Structure prediction tasks:
        - "fold": Protein fold classification (multiclass, 1195 classes)
        - "secondarystructure": Secondary structure prediction (multiclass, 3 classes - coil, strand, helix)

    Excluded by default:
        - "proteinnet": Contact map prediction (excluded due to quadratic memory scaling)

    Reference:
        Guo et al. (2023) "PEER: A Comprehensive and Multi-Task Benchmark for
        Protein Sequence Understanding"
        https://arxiv.org/abs/2206.02096

    Supports both tokenizer-based models (using UMETokenizerTransform) and
    sequence-based models like ESM.
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
            Whether to use enhanced memory management techniques.
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

        # Define memory-intensive tasks that need special handling
        self.memory_intensive_tasks = {
            PEERTask.PROTEINNET,
            PEERTask.SECONDARY_STRUCTURE,
            PEERTask.FOLD,
            PEERTask.SUBCELLULAR_LOCALIZATION,
            PEERTask.BINDINGDB,
            PEERTask.PDBBIND,
        }

        # Define high memory-intensive tasks that need batch_size=1
        self.high_memory_tasks = {
            PEERTask.PROTEINNET,
            PEERTask.SECONDARY_STRUCTURE,
            PEERTask.FOLD,
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

    def _get_batch_size_for_task(self, task: PEERTask) -> int:
        """Get batch size for a specific task with enhanced memory management."""
        # Tasks that need batch_size=1 due to collation issues or high memory usage
        if task in self.high_memory_tasks:
            return 1

        # Moderate reduction for memory-intensive tasks only
        if task in self.memory_intensive_tasks:
            if self.manage_memory:
                return max(1, self.batch_size // 2)  # Less aggressive reduction
            else:
                return max(1, self.batch_size // 2)

        # For non-memory-intensive tasks, use full batch size even with memory management
        return self.batch_size

    def _get_relevant_metric(self, task: PEERTask) -> str:
        """Get the relevant metric for a given PEER task.

        Parameters
        ----------
        task : PEERTask
            The PEER task to get the relevant metric for.

        Returns
        -------
        str
            The name of the relevant metric for this task.
        """
        # Function prediction tasks
        if task in {PEERTask.FLUORESCENCE, PEERTask.STABILITY, PEERTask.BETALACTAMASE}:
            return "spearman"
        # Localization and function classification tasks
        elif task in {
            PEERTask.SOLUBILITY,
            PEERTask.SUBCELLULAR_LOCALIZATION,
            PEERTask.BINARY_LOCALIZATION,
            PEERTask.FOLD,
            PEERTask.SECONDARY_STRUCTURE,
        }:
            return "accuracy"
        # Contact prediction
        elif task == PEERTask.PROTEINNET:
            return "l5_precision"
        # PPI classification
        elif task in {PEERTask.HUMANPPI, PEERTask.YEASTPPI}:
            return "accuracy"
        # Regression tasks with RMSE
        elif task in {PEERTask.PPIAFFINITY, PEERTask.PDBBIND, PEERTask.BINDINGDB}:
            return "rmse"
        # Default to accuracy for classification and rmse for regression
        else:
            task_type = PEER_TASKS[task][0]
            if task_type in {"binary", "multiclass", "multilabel"}:
                return "accuracy"
            else:  # regression
                return "rmse"

    def _process_and_embed(
        self,
        pl_module: L.LightningModule,
        inputs: dict[str, Tensor] | list[str] | str | BatchEncoding,
        modality: str = "amino_acid",
        aggregate: bool = True,
    ) -> Tensor:
        """Process inputs (tokenize if needed) and extract embeddings using the model's embed method.

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
        # Models that accept sequence inputs directly (like ESM)
        if not self.requires_tokenization:
            # Check if inputs are already tokenized (BatchEncoding or dict with input_ids)
            if isinstance(inputs, (dict, BatchEncoding)) and "input_ids" in inputs:  # noqa: UP038
                # Extract the original sequences if possible and available
                if hasattr(inputs, "original_sequence"):
                    # Use the original sequence if available
                    sequences = inputs.original_sequence
                elif hasattr(self.transform_fn, "tokenizer") and hasattr(self.transform_fn.tokenizer, "decode"):
                    # Try to decode the input_ids back to sequences
                    tokenizer = self.transform_fn.tokenizer
                    sequences = [tokenizer.decode(ids) for ids in inputs["input_ids"]]
                else:
                    # We can't handle this case - return error
                    raise ValueError(
                        "Model requires sequence inputs but received tokenized inputs "
                        "without a way to recover the original sequences"
                    )
            else:
                # Inputs are already sequences or can be used directly
                sequences = inputs

            # Call the model's embed method with the sequences
            return pl_module.embed(sequences, aggregate=aggregate)

        # Models that require tokenized inputs (traditional approach)
        else:
            # Check if inputs are already tokenized (either dict or BatchEncoding)
            if isinstance(inputs, (dict, BatchEncoding)) and "input_ids" in inputs:  # noqa: UP038
                tokenized_inputs = inputs
            else:
                # Tokenize the inputs
                if hasattr(pl_module, "tokenizer_transforms") and modality in pl_module.tokenizer_transforms:
                    tokenizer_transform = pl_module.tokenizer_transforms[modality]
                    tokenized_inputs = tokenizer_transform(inputs)
                else:
                    # Fall back to the transform_fn from this callback
                    tokenized_inputs = self.transform_fn(inputs)

            # Use the embed method from the model
            return pl_module.embed(tokenized_inputs, aggregate=aggregate)

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

                # Get embeddings using our modified method that handles both tokenized and direct inputs
                batch_embeddings = self._process_and_embed(pl_module, x, modality="amino_acid", aggregate=True)

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
        """Get all appropriate test splits for a task.

        For most tasks, this will be a single 'test' split.
        For tasks with multiple test splits, returns all of them.
        """
        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                return ["casp12", "cb513", "ts115"]

            case PEERTask.BINDINGDB:
                return ["random_test", "holdout_test"]

            case PEERTask.FOLD:
                return [split for split in PEER_TASK_SPLITS[task] if "test" in split and "holdout" in split]

            case PEERTask.HUMANPPI | PEERTask.YEASTPPI:
                return ["test", "cross_species_test"]

            case _:
                # Default case: return anything with test
                return [split for split in PEER_TASK_SPLITS[task] if split not in ["train", "valid"]]

    def _get_task_datasets(self, task: PEERTask) -> tuple[PEERDataset, dict[str, PEERDataset]]:
        """Get or create train and test datasets for a given task.

        Returns:
            Tuple containing (train_dataset, test_datasets_dict)
            where test_datasets_dict maps split names to datasets
        """
        cache_key = str(task)

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
                        embeddings1 = self._process_and_embed(pl_module, seq1, modality="amino_acid", aggregate=True)
                        embeddings2 = self._process_and_embed(pl_module, seq2, modality="amino_acid", aggregate=True)

                        # Concatenate the embeddings
                        batch_embeddings = torch.cat([embeddings1, embeddings2], dim=1)

                    # Handle protein-ligand interactions
                    elif task in {PEERTask.BINDINGDB, PEERTask.PDBBIND}:
                        # Get embeddings for protein and ligand
                        protein_embeddings = self._process_and_embed(
                            pl_module, seq1, modality="amino_acid", aggregate=True
                        )

                        # For sequence-based models that don't have ligand tokenizers
                        if not self.requires_tokenization:
                            # Assume the model can handle ligand sequences directly
                            ligand_embeddings = self._process_and_embed(
                                pl_module, seq2, modality="amino_acid", aggregate=True
                            )
                        else:
                            # Use the specific ligand tokenizer for tokenizer-based models
                            ligand_embeddings = self._process_and_embed(
                                pl_module, seq2, modality="ligand", aggregate=True
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
        """Helper method to flatten embeddings and filter special tokens for token-level tasks.

        Parameters
        ----------
        batch_embeddings : Tensor
            Embeddings with shape [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
        targets : Tensor
            Target labels with shape [batch_size, seq_len] or [batch_size*seq_len] or [orig_seq_len]
        input_ids : Tensor | None, default=None
            Token IDs from tokenizer, used to identify special tokens
        attention_mask : Tensor | None, default=None
            Attention mask from tokenizer
        ignore_target_value : int, default=-100
            Value in targets to ignore (typically padding value)

        Returns
        -------
        tuple[Tensor, Tensor]
            Filtered embeddings and targets
        """
        # Handle both batched and single sequence cases
        if batch_embeddings.dim() == 3:
            _batch_size, _seq_len, hidden_size = batch_embeddings.shape
            # Flatten embeddings to (batch_size*seq_len, hidden_size)
            batch_embeddings_flat = batch_embeddings.reshape(-1, hidden_size)
        else:
            # Single sequence case [seq_len, hidden_size]
            _seq_len, hidden_size = batch_embeddings.shape
            batch_embeddings_flat = batch_embeddings

        # Handle target length mismatch with tokenized sequence
        orig_target_len = targets.numel()
        expected_len = batch_embeddings_flat.shape[0]

        # Flatten targets if not already flattened
        if targets.dim() > 1:
            targets_flat = targets.reshape(-1)
        else:
            targets_flat = targets

        # Handle length mismatch between targets and embeddings
        if orig_target_len != expected_len:
            if orig_target_len < expected_len:
                # Pad targets with ignore_target_value for padded positions
                padded_targets = torch.full(
                    (expected_len,), ignore_target_value, dtype=targets_flat.dtype, device=targets_flat.device
                )
                padded_targets[:orig_target_len] = targets_flat
                targets_flat = padded_targets
            else:
                # Truncate targets if they're longer than embeddings
                targets_flat = targets_flat[:expected_len]

        # If we have tokenized input and we're using a tokenizer-based model, use it to filter special tokens
        if (
            input_ids is not None
            and attention_mask is not None
            and self.requires_tokenization
            and hasattr(self.transform_fn, "tokenizer")
        ):
            tokenizer = self.transform_fn.tokenizer
            special_token_ids = {
                tokenizer.cls_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.sep_token_id,
                tokenizer.mask_token_id,
                tokenizer.unk_token_id,
            }

            # Create a mask for real tokens (not special tokens and not padding)
            valid_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
            for special_id in special_token_ids:
                if special_id is not None:  # In case some IDs are None
                    valid_token_mask &= input_ids != special_id
            valid_token_mask &= attention_mask.bool()

            # Also filter based on target values to ignore
            target_mask = targets_flat != ignore_target_value

            # Flatten mask & filter embeddings based on token mask
            valid_token_mask_flat = valid_token_mask.reshape(-1)

            # Ensure mask length matches
            if len(valid_token_mask_flat) != len(targets_flat):
                min_len = min(len(valid_token_mask_flat), len(targets_flat))
                valid_token_mask_flat = valid_token_mask_flat[:min_len]
                targets_flat = targets_flat[:min_len]
                batch_embeddings_flat = batch_embeddings_flat[:min_len]

            # Combine both masks
            combined_mask = valid_token_mask_flat & target_mask

            filtered_embeddings = batch_embeddings_flat[combined_mask]
            filtered_targets = targets_flat[combined_mask]
        else:
            # If no tokenizer info provided or using sequence-based model, just filter based on ignore value
            valid_mask = targets_flat != ignore_target_value
            filtered_embeddings = batch_embeddings_flat[valid_mask]
            filtered_targets = targets_flat[valid_mask]

        return filtered_embeddings, filtered_targets

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
        # For sequence-based models that don't require tokenization
        if not self.requires_tokenization:
            if isinstance(x, dict) and "input_ids" in x:
                # Get the raw sequences if we can
                if hasattr(x, "original_sequence"):
                    x = x.original_sequence
                elif hasattr(self.transform_fn, "tokenizer") and hasattr(self.transform_fn.tokenizer, "decode"):
                    # Try to decode from input_ids
                    tokenizer = self.transform_fn.tokenizer
                    x = [tokenizer.decode(ids) for ids in x["input_ids"]]

        # Extract input_ids and attention_mask if available (for token filtering)
        input_ids = x.get("input_ids") if isinstance(x, dict) else None
        attention_mask = x.get("attention_mask") if isinstance(x, dict) else None

        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                # Get per-residue embeddings
                batch_embeddings = self._process_and_embed(pl_module, x, modality="amino_acid", aggregate=False)

                # Handle single item case (batch_size=1 due to collation)
                if batch_embeddings.dim() == 3:
                    batch_embeddings = batch_embeddings.squeeze(0)  # Remove batch dimension

                # Use helper method to flatten and filter token-level embeddings
                filtered_embeddings, filtered_targets = self._flatten_and_filter_token_embeddings(
                    batch_embeddings=batch_embeddings,
                    targets=y,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ignore_target_value=-100,  # Use -100 as the padding value for targets
                )

                # Only append if we have valid embeddings/targets
                if filtered_embeddings.numel() > 0 and filtered_targets.numel() > 0:
                    embeddings.append(filtered_embeddings.cpu())
                    targets.append(filtered_targets.cpu())

            case PEERTask.PROTEINNET:
                # Handle contact map prediction with tertiary structure coordinates
                tertiary_coords, valid_mask = y

                # Get per-residue embeddings
                batch_embeddings = self._process_and_embed(pl_module, x, modality="amino_acid", aggregate=False)

                # Handle single item case (batch_size=1 due to collation)
                if batch_embeddings.dim() == 3:
                    batch_embeddings = batch_embeddings.squeeze(0)  # Remove batch dimension

                # Ensure mask length matches embedding sequence length
                seq_len = batch_embeddings.shape[0]  # Should be 512 (padded length)
                orig_len = len(valid_mask)  # Original sequence length

                # Truncate or pad the mask to match embedding length
                if orig_len > seq_len:
                    # Truncate mask if original sequence is longer than max_length
                    residue_mask = valid_mask[:seq_len].bool()
                    tertiary_coords = tertiary_coords[:seq_len]
                elif orig_len < seq_len:
                    # Pad mask with False for padded positions
                    padded_mask = torch.zeros(seq_len, dtype=torch.bool)
                    padded_mask[:orig_len] = valid_mask.bool()
                    residue_mask = padded_mask

                    # Pad tertiary coords with zeros
                    padded_coords = torch.zeros((seq_len, tertiary_coords.shape[1]), dtype=tertiary_coords.dtype)
                    padded_coords[:orig_len] = tertiary_coords
                    tertiary_coords = padded_coords
                else:
                    residue_mask = valid_mask.bool()

                # Only use valid (non-padded) positions for contact prediction
                true_residue_mask = residue_mask[:orig_len]  # Only original sequence positions

                if true_residue_mask.sum() > 0:
                    valid_embeddings = batch_embeddings[:orig_len][true_residue_mask]
                    valid_coords = tertiary_coords[:orig_len][true_residue_mask]

                    # Limit sequence length for PROTEINNET to prevent memory explosion
                    n_residues = valid_coords.shape[0]
                    if n_residues > self.proteinnet_max_length:
                        # Truncate to prevent quadratic memory explosion
                        valid_embeddings = valid_embeddings[: self.proteinnet_max_length]
                        valid_coords = valid_coords[: self.proteinnet_max_length]
                        n_residues = self.proteinnet_max_length
                        logger.debug(f"Truncated PROTEINNET sequence to {n_residues} residues to prevent OOM")

                    # Calculate pairwise distances between 3D coordinates
                    distances = torch.cdist(valid_coords, valid_coords)

                    # Define contacts as residues closer than 8 Angstroms
                    contacts = (distances < 8.0).float()

                    # Process residue pairs in chunks to avoid memory accumulation
                    chunk_size = 5000  # Process pairs in chunks of 5000
                    contact_targets = []
                    contact_embeddings = []

                    pair_count = 0
                    current_chunk_embeddings = []
                    current_chunk_targets = []

                    # For each residue pair, concatenate their embeddings
                    for i in range(n_residues):
                        for j in range(i + 4, n_residues):  # Skip local contacts (i to i+3)
                            current_chunk_embeddings.append(torch.cat([valid_embeddings[i], valid_embeddings[j]]))
                            current_chunk_targets.append(contacts[i, j])
                            pair_count += 1

                            # Process chunk when it's full
                            if len(current_chunk_embeddings) >= chunk_size:
                                if current_chunk_embeddings:
                                    chunk_embeddings = torch.stack(current_chunk_embeddings)
                                    chunk_targets = torch.tensor(current_chunk_targets)

                                    contact_embeddings.append(chunk_embeddings.cpu())
                                    contact_targets.append(chunk_targets.cpu())

                                    # Clear current chunk and force memory cleanup
                                    current_chunk_embeddings.clear()
                                    current_chunk_targets.clear()
                                    del chunk_embeddings, chunk_targets

                                    # Clear memory cache for PROTEINNET
                                    if self.manage_memory:
                                        self._clear_memory_cache()

                    # Process remaining pairs in the last chunk
                    if current_chunk_embeddings:
                        chunk_embeddings = torch.stack(current_chunk_embeddings)
                        chunk_targets = torch.tensor(current_chunk_targets)

                        contact_embeddings.append(chunk_embeddings.cpu())
                        contact_targets.append(chunk_targets.cpu())

                        # Clear memory
                        del chunk_embeddings, chunk_targets

                    # Concatenate all chunks
                    if contact_embeddings:
                        batch_embeddings = torch.cat(contact_embeddings, dim=0)
                        batch_targets = torch.cat(contact_targets, dim=0)

                        # Only append if we have valid embeddings/targets
                        if batch_embeddings.numel() > 0 and batch_targets.numel() > 0:
                            embeddings.append(batch_embeddings.cpu())
                            targets.append(batch_targets.cpu())

                            # Clear intermediate results
                            del batch_embeddings, batch_targets
                            contact_embeddings.clear()
                            contact_targets.clear()

                    logger.debug(f"Processed {pair_count} residue pairs for PROTEINNET")

            case PEERTask.FOLD:
                # Standard fold classification - sequence-level task
                batch_embeddings = self._process_and_embed(pl_module, x, modality="amino_acid", aggregate=True)

                # Ensure target has at least one dimension for concatenation
                target = y.cpu()
                if target.dim() == 0:
                    target = target.unsqueeze(0)

                # Only append if we have valid embeddings/targets
                if batch_embeddings.numel() > 0 and target.numel() > 0:
                    embeddings.append(batch_embeddings.cpu())
                    targets.append(target)

    def _train_probe(self, embeddings: Tensor, targets: Tensor, task: PEERTask = None):
        """Train a probe on the given embeddings and targets with task-specific handling."""
        if task is None:
            # Fallback to parent implementation if no task specified
            return super()._train_probe(embeddings, targets)

        task_type, num_classes = PEER_TASKS[task]

        # Update task_type and num_classes based on the specific task
        match task:
            case PEERTask.SECONDARY_STRUCTURE:
                # Secondary structure is a multiclass classification problem
                task_type = "multiclass"
                num_classes = 3  # 3 secondary structure classes (helix, sheet, coil)

            case PEERTask.SUBCELLULAR_LOCALIZATION:
                # Subcellular localization is a multiclass problem
                task_type = "multiclass"
                num_classes = 10

            case PEERTask.FOLD:
                # Fold classification is a multiclass problem
                task_type = "multiclass"
                num_classes = 1195  # TODO - verify number of fold classes

            case PEERTask.PROTEINNET:
                # Contact prediction is a binary classification problem
                task_type = "binary"
                num_classes = None

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
                base_classifier = LogisticRegression(random_state=42)
                probe = MultiOutputClassifier(base_classifier)
                probe.fit(embeddings_np, targets_np)

            case "binary" | "multiclass":
                # Use appropriate classifier for binary vs multiclass
                if task_type == "binary":
                    probe = LogisticRegression(
                        random_state=42,
                        max_iter=1000,  # Increase for convergence
                    )
                else:  # multiclass
                    probe = LogisticRegression(
                        random_state=42,
                        max_iter=1000,  # Increase for convergence
                    )
                probe.fit(embeddings_np, targets_np.ravel())

        return probe

    def _evaluate_probe(self, probe, embeddings: Tensor, targets: Tensor, task: PEERTask = None) -> dict[str, float]:
        """Evaluate a trained probe on the given embeddings and targets, returning task-specific metrics."""
        if task is None:
            # Fallback to parent implementation if no task specified
            return super()._evaluate_probe(probe, embeddings, targets)

        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()

        relevant_metric = self._get_relevant_metric(task)
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

        return metrics

    def _get_collate_fn(self, task: PEERTask):
        """Get the appropriate collation function for the task."""
        # Tasks that need custom collation due to variable-length tensors
        if task in {PEERTask.PROTEINNET, PEERTask.SECONDARY_STRUCTURE, PEERTask.FOLD}:
            return peer_structure_collate_fn
        else:
            return peer_default_collate_fn

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
            if task in self.high_memory_tasks:
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
            self.probes[str(task)] = probe

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
        category_metrics = defaultdict(lambda: defaultdict(list))
        all_metrics = defaultdict(list)
        all_task_metrics = {}
        split_count = 0

        # Evaluate each selected task
        for task in tqdm(self.selected_tasks, desc=f"{self.__class__.__name__}"):
            task_category = PEER_TASK_CATEGORIES[task]
            split_metrics = self._evaluate_task(task, trainer, pl_module)

            # Get the relevant metric for this task
            relevant_metric = self._get_relevant_metric(task)

            # Store task metrics for return value
            all_task_metrics[str(task)] = split_metrics

            # Process metrics for each split, but only log the relevant metric
            for split_name, metrics in split_metrics.items():
                # Only log the relevant metric for this task
                if relevant_metric in metrics:
                    value = metrics[relevant_metric]
                    metric_key = f"peer_linear_probe/{task}/{split_name}/{relevant_metric}"
                    trainer.logger.log_metrics({metric_key: value}, step=step)

                    # Collect metrics for category averages
                    if len(split_metrics) > 1:
                        category_metrics[task_category][f"{relevant_metric}_by_split"].append(value)

                    # Collect metrics for global averages - using the metric type as the key
                    all_metrics[relevant_metric].append(value)

                split_count += 1

            # Calculate and log task averages across splits if there are multiple splits
            if len(split_metrics) > 1:
                task_avg_metrics = {}
                # Only average the relevant metric
                if any(relevant_metric in m for m in split_metrics.values()):
                    values = [m.get(relevant_metric) for m in split_metrics.values() if relevant_metric in m]
                    if values:
                        avg_value = sum(values) / len(values)
                        task_avg_key = f"peer_linear_probe/{task}/average/{relevant_metric}"
                        trainer.logger.log_metrics({task_avg_key: avg_value}, step=step)
                        task_avg_metrics[relevant_metric] = avg_value

                if task_avg_metrics:
                    all_task_metrics[f"{task}/average"] = task_avg_metrics

            # Clear dataset cache between tasks if enabled
            if self.clear_cache_between_tasks:
                task_cache_key = str(task)
                if task_cache_key in self.datasets:
                    del self.datasets[task_cache_key]
                    logger.debug(f"Cleared dataset cache for task {task}")

            # Force memory cleanup after memory-intensive tasks
            if task in self.memory_intensive_tasks:
                self._clear_memory_cache()
                logger.debug(f"Completed memory-intensive task {task} and cleared memory")

        # Log category averages - but only for metrics that are relevant to tasks in that category
        category_metrics_dict = {}
        for category, metrics_dict in category_metrics.items():
            category_metrics_dict[str(category)] = {}
            for metric_name, values in metrics_dict.items():
                if values:
                    avg_value = sum(values) / len(values)
                    metric_key = f"peer_linear_probe/category/{category}/{metric_name}"
                    trainer.logger.log_metrics({metric_key: avg_value}, step=step)
                    category_metrics_dict[str(category)][metric_name] = avg_value

        if category_metrics_dict:
            all_task_metrics["categories"] = category_metrics_dict

        # Calculate and log overall averages for each metric type
        mean_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                metric_key = f"peer_linear_probe/mean/{metric_name}"
                trainer.logger.log_metrics({metric_key: avg_value}, step=step)
                mean_metrics[metric_name] = avg_value

        # Add mean metrics to result
        all_task_metrics["mean"] = mean_metrics

        # Log total number of splits evaluated
        trainer.logger.log_metrics({"peer_linear_probe/total_splits_evaluated": split_count}, step=step)

        return all_task_metrics

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
