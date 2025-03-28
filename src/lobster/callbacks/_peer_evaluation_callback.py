import logging
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, Union

import lightning as L
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from lobster.constants import PEER_TASK_CATEGORIES, PEER_TASK_SPLITS, PEER_TASKS, PEERTask, PEERTaskCategory
from lobster.datasets import PEERDataset
from lobster.tokenization import UmeTokenizerTransform

from ._linear_probe_callback import LinearProbeCallback

logger = logging.getLogger(__name__)


class PEEREvaluationCallback(LinearProbeCallback):
    """Callback for evaluating model embeddings on PEER benchmark tasks:

    The callback handles various input types including:
    - Single sequence inputs
    - Paired sequence inputs (protein-protein, protein-ligand)
    - Per-residue tasks (secondary structure)
    - Contact map prediction

    Available tasks:
    - Function prediction tasks:
        - "aav": AAV variant fitness
        - "betalactamase": Beta-lactamase stability
        - "fluorescence": Protein fluorescence
        - "gb1": GB1 protein stability
        - "solubility": Protein solubility
        - "stability": Protein stability
        - "thermostability": Protein thermostability

    - Localization prediction tasks:
        - "binarylocalization": Binary subcellular localization
        - "subcellularlocalization": Multi-class subcellular localization

    - Protein-ligand interaction tasks:
        - "bindingdb": Protein-ligand binding affinity (BindingDB)
        - "pdbbind": Protein-ligand binding affinity (PDBbind)

    - Protein-protein interaction tasks:
        - "humanppi": Human protein-protein interactions
        - "ppiaffinity": Protein-protein binding affinity
        - "yeastppi": Yeast protein-protein interactions

    - Structure prediction tasks:
        - "fold": Protein fold classification
        - "proteinnet": Contact map prediction
        - "secondarystructure": Secondary structure prediction

    Reference:
        Guo et al. (2023) "PEER: A Comprehensive and Multi-Task Benchmark for
        Protein Sequence Understanding"
        https://arxiv.org/abs/2206.02096

    Currently only supports Ume embeddings and uses UmeTokenizerTransform.
    """

    def __init__(
        self,
        max_length: int,
        tasks: Optional[Sequence[Union[PEERTask, str]]] = None,
        batch_size: int = 32,
        run_every_n_epochs: Optional[int] = None,
    ):
        """Initialize the PEER benchmark callback.

        Parameters
        ----------
        max_length : int
            Maximum sequence length for tokenization.
        tasks : Optional[Sequence[Union[PEERTask, str]]], default=None
            Specific PEER tasks to evaluate. If None, all tasks are used.
        batch_size : int, default=32
            Batch size for embedding extraction and evaluation.
        run_every_n_epochs : Optional[int], default=None
            Run this callback every n epochs. If None, runs every validation epoch.
        """
        tokenizer_transform = UmeTokenizerTransform(
            modality="amino_acid",
            max_length=max_length,
        )

        super().__init__(
            transform_fn=tokenizer_transform,
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

        # If no tasks specified, use all tasks
        if not self.selected_tasks:
            self.selected_tasks = set(PEER_TASKS.keys())

        # Store embedders for different modalities (AA, SMILES)
        self.embedders = {}

        # Cache for datasets
        self.datasets = {}

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

    def _get_task_datasets(self, task: PEERTask) -> Tuple[PEERDataset, Dict[str, PEERDataset]]:
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
    ) -> Tuple[Tensor, Tensor]:
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
                        embeddings1 = pl_module.get_embeddings([seq1], modality="protein", per_residue=False)
                        embeddings2 = pl_module.get_embeddings([seq2], modality="protein", per_residue=False)

                        # Concatenate the embeddings. TODO - update later w/ cross-attention head
                        batch_embeddings = torch.cat([embeddings1, embeddings2], dim=1)

                    # Handle protein-ligand interactions
                    elif task in {PEERTask.BINDINGDB, PEERTask.PDBBIND}:
                        # Get embeddings for protein and ligand
                        protein_embeddings = pl_module.get_embeddings([seq1], modality="protein", per_residue=False)
                        ligand_embeddings = pl_module.get_embeddings([seq2], modality="ligand", per_residue=False)

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
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Helper method to flatten embeddings and filter special tokens for token-level tasks.

        Parameters
        ----------
        batch_embeddings : Tensor
            Embeddings with shape [batch_size, seq_len, hidden_size]
        targets : Tensor
            Target labels with shape [batch_size, seq_len] or [batch_size*seq_len]
        input_ids : Optional[Tensor], default=None
            Token IDs from tokenizer, used to identify special tokens
        attention_mask : Optional[Tensor], default=None
            Attention mask from tokenizer

        Returns
        -------
        Tuple[Tensor, Tensor]
            Filtered embeddings and targets
        """
        _batch_size, _seq_len, hidden_size = batch_embeddings.shape

        # Flatten embeddings to (batch_size*seq_len, hidden_size)
        batch_embeddings_flat = batch_embeddings.reshape(-1, hidden_size)

        # Flatten targets if not already flattened
        if targets.dim() > 1:
            targets_flat = targets.reshape(-1)
        else:
            targets_flat = targets

        # If we have tokenized input, use it to filter special tokens
        if input_ids is not None and attention_mask is not None:
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

            # Flatten mask & filter embeddings based on token mask
            valid_token_mask_flat = valid_token_mask.reshape(-1)
            filtered_embeddings = batch_embeddings_flat[valid_token_mask_flat]

            # Make sure the number of embeddings matches the number of targets
            min_len = min(filtered_embeddings.size(0), targets_flat.size(0))
            filtered_embeddings = filtered_embeddings[:min_len]
            filtered_targets = targets_flat[:min_len]
        else:
            # If no tokenizer info provided, use all embeddings and targets as is
            filtered_embeddings = batch_embeddings_flat
            filtered_targets = targets_flat

            # Make sure lengths match
            min_len = min(filtered_embeddings.size(0), filtered_targets.size(0))
            filtered_embeddings = filtered_embeddings[:min_len]
            filtered_targets = filtered_targets[:min_len]

        return filtered_embeddings, filtered_targets

    def _get_structure_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask
    ) -> Tuple[Tensor, Tensor]:
        """Extract embeddings for structure prediction tasks with special handling."""
        embeddings = []
        targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch - all batches will have input_ids and attention_mask
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, y = batch
                    # x should be a dictionary with tokenizer outputs
                    input_ids = x["input_ids"]
                    attention_mask = x["attention_mask"]
                else:
                    raise ValueError(f"Expected batch to be a tuple of (inputs, targets), got {type(batch)}")

                match task:
                    case PEERTask.SECONDARY_STRUCTURE:
                        # Get per-residue embeddings
                        batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=True)

                        # Use helper method to flatten and filter token-level embeddings
                        filtered_embeddings, filtered_targets = self._flatten_and_filter_token_embeddings(
                            batch_embeddings=batch_embeddings,
                            targets=y,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            ignore_target_value=-1,  # Assuming -1 is the padding value for targets
                        )

                        embeddings.append(filtered_embeddings.cpu())
                        targets.append(filtered_targets.cpu())

                    case PEERTask.PROTEINNET:
                        # Handle contact map prediction with tertiary structure coordinates
                        tertiary_coords, valid_mask = y

                        # Get per-residue embeddings
                        batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=True)
                        batch_size, _seq_len, hidden_size = batch_embeddings.shape

                        # Extract valid embeddings and coords based on mask
                        valid_embeddings = []
                        valid_coords = []

                        for i in range(batch_size):
                            # Create combined mask using both valid_mask and token masks if available
                            residue_mask = valid_mask[i].bool()

                            if residue_mask.sum() > 0:
                                valid_embeddings.append(batch_embeddings[i, residue_mask])
                                valid_coords.append(tertiary_coords[i, residue_mask])

                        if not valid_embeddings:
                            continue

                        # Create contact map representations
                        contact_targets = []
                        contact_embeddings = []

                        for emb, coords in zip(valid_embeddings, valid_coords):
                            # Calculate pairwise distances between 3D coordinates
                            n_residues = coords.shape[0]
                            distances = torch.cdist(coords, coords)

                            # Define contacts as residues closer than 8 Angstroms
                            contacts = (distances < 8.0).float()

                            # For each residue pair, concatenate their embeddings # TODO - consider updated w/ outer product
                            for i in range(n_residues):
                                for j in range(i + 4, n_residues):  # Skip local contacts (i to i+3)
                                    contact_embeddings.append(torch.cat([emb[i], emb[j]]))
                                    contact_targets.append(contacts[i, j])

                        if contact_embeddings:
                            batch_embeddings = torch.stack(contact_embeddings)
                            batch_targets = torch.tensor(contact_targets)

                            embeddings.append(batch_embeddings.cpu())
                            targets.append(batch_targets.cpu())

                    case PEERTask.FOLD:
                        # Standard fold classification - sequence-level task
                        batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=False)
                        embeddings.append(batch_embeddings.cpu())
                        targets.append(y.cpu())

        # If we have no valid embeddings, return empty tensors
        if not embeddings:
            return torch.tensor([]), torch.tensor([])

        return torch.cat(embeddings), torch.cat(targets)

    def _get_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask = None
    ) -> Tuple[Tensor, Tensor]:
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
            embeddings = []
            targets = []

            pl_module.eval()
            with torch.no_grad():
                for batch in dataloader:
                    x, y = batch

                    batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=False)

                    embeddings.append(batch_embeddings.cpu())
                    targets.append(y.cpu())

            return torch.cat(embeddings), torch.cat(targets)

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
                probe = LogisticRegression(
                    multi_class="ovr" if task_type == "binary" else "multinomial",
                    random_state=42,
                    max_iter=1000,  # Increase for convergence
                )
                probe.fit(embeddings_np, targets_np.ravel())

        return probe

    def _evaluate_task(
        self, task: PEERTask, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a single PEER task across all its test splits.

        Returns:
            Dict mapping split names to dictionaries of metrics
        """
        try:
            train_dataset, test_datasets = self._get_task_datasets(task)

            # Get train embeddings and probe
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            train_embeddings, train_targets = self._get_embeddings(pl_module, train_loader, task)
            probe = self._train_probe(train_embeddings, train_targets, task)
            self.probes[str(task)] = probe

            # Evaluate on each test split
            split_metrics = {}
            for split_name, test_dataset in test_datasets.items():
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

                test_embeddings, test_targets = self._get_embeddings(pl_module, test_loader, task)

                # Evaluate probe
                metrics = self._evaluate_probe(probe, test_embeddings, test_targets)
                split_metrics[split_name] = metrics

            return split_metrics

        except Exception as e:
            logger.exception(f"Error evaluating task {task}: {str(e)}")
            return {}

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes on PEER tasks at specified epochs."""
        if self._skip(trainer):
            return

        # Reset aggregate metrics for new epoch
        category_metrics = defaultdict(lambda: defaultdict(list))
        all_metrics = defaultdict(list)
        split_count = 0  # Track total number of splits evaluated

        # Evaluate each selected task
        for task in tqdm(self.selected_tasks, desc=f"{self.__class__.__name__}"):
            task_category = PEER_TASK_CATEGORIES[task]
            split_metrics = self._evaluate_task(task, trainer, pl_module)

            # Process metrics for each split
            for split_name, metrics in split_metrics.items():
                # Log split-specific metrics
                for metric_name, value in metrics.items():
                    metric_key = f"peer_linear_probe/{task}/{split_name}/{metric_name}"
                    trainer.logger.log_metrics({metric_key: value}, step=trainer.current_epoch)

                    # Also log a task-level average if there are multiple splits
                    if len(split_metrics) > 1:
                        # Add to task-level averages for this metric
                        category_metrics[task_category][f"{metric_name}_by_split"].append(value)

                # For computing global averages, count each split as a separate evaluation
                for metric_name, value in metrics.items():
                    all_metrics[metric_name].append(value)

                split_count += 1

            # Calculate and log task averages across splits if there are multiple splits
            if len(split_metrics) > 1:
                for metric_name in next(iter(split_metrics.values()), {}):
                    # Calculate average across all splits for this task
                    values = [metrics[metric_name] for metrics in split_metrics.values() if metric_name in metrics]
                    if values:
                        avg_value = sum(values) / len(values)
                        task_avg_key = f"peer_linear_probe/{task}/average/{metric_name}"
                        trainer.logger.log_metrics({task_avg_key: avg_value}, step=trainer.current_epoch)

        # Log category averages
        for category, metrics_dict in category_metrics.items():
            for metric_name, values in metrics_dict.items():
                if values:
                    avg_value = sum(values) / len(values)
                    trainer.logger.log_metrics(
                        {f"peer_linear_probe/category/{category}/{metric_name}": avg_value}, step=trainer.current_epoch
                    )

        # Log overall averages
        for metric_name, values in all_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                trainer.logger.log_metrics(
                    {f"peer_linear_probe/mean/{metric_name}": avg_value}, step=trainer.current_epoch
                )

        # Log total number of splits evaluated
        trainer.logger.log_metrics(
            {"peer_linear_probe/total_splits_evaluated": split_count}, step=trainer.current_epoch
        )
