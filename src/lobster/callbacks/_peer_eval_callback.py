from collections import defaultdict
from typing import Optional, Sequence, Dict, Tuple, Union, Callable
import logging

import lightning as L
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

from lobster.constants import PEER_TASKS, PEER_TASK_CATEGORIES, PEER_TASK_SPLITS, PEERTask, PEERTaskCategory
from lobster.datasets import PEERDataset
from lobster.tokenization import NucleotideTokenizerFast
from lobster.transforms import TokenizerTransform

from ._linear_probe_callback import LinearProbeCallback


logger = logging.getLogger(__name__)


class PEERLinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the PEER benchmark tasks.
    
    This callback assesses model performance across diverse protein-related prediction tasks:
    - Function prediction (e.g., stability, fluorescence)
    - Localization prediction (subcellular localization)
    - Protein-protein interactions
    - Protein-ligand interactions
    - Structure prediction (fold classification, secondary structure, contact maps)
    
    The callback handles various input types including:
    - Single sequence inputs
    - Paired sequence inputs (protein-protein, protein-ligand)
    - Per-residue tasks (secondary structure)
    - Contact map prediction
    
    Reference:
        Guo et al. (2023) "PEER: A Comprehensive and Multi-Task Benchmark for 
        Protein Sequence Understanding"
        https://arxiv.org/abs/2206.02096
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
        # Create a tokenizer transform for nucleotide sequences
        tokenizer_transform = TokenizerTransform(
            tokenizer=NucleotideTokenizerFast(), # TODO - reaplce with ProteinTokenizerFast
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        super().__init__(
            transform_fn=tokenizer_transform,
            # Default task type will be updated per task
            task_type="regression",
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
        )

        # Set up tasks based on input parameters
        self.selected_tasks = set()
        
        # If specific tasks are provided, add them
        if tasks is not None:
            for task in tasks:
                task_enum = PEERTask(task) if isinstance(task, str) else task
                self.selected_tasks.add(task_enum)
        
        # If no tasks specified, use all tasks
        if not self.selected_tasks:
            self.selected_tasks = set(PEER_TASKS.keys())
        
        # Dictionary to store embedders for different modalities
        self.embedders = {}
        
        # Cache for datasets
        self.datasets = {}

    def _get_task_test_splits(self, task: PEERTask) -> list[str]:
        """Get all appropriate test splits for a task.
        
        For most tasks, this will be a single 'test' split.
        For tasks with multiple test splits, returns all of them.
        """
        available_splits = PEER_TASK_SPLITS[task]
        test_splits = []
        # TODO - make match case here?
        # Handle secondary structure special case
        if task == PEERTask.SECONDARY_STRUCTURE:
            # All of these are considered test sets for secondary structure
            return [split for split in ["casp12", "cb513", "ts115"] if split in available_splits]
        
        # Handle bindingdb special case (both random and holdout tests)
        if task == PEERTask.BINDINGDB:
            return [split for split in ["random_test", "holdout_test"] if split in available_splits]
        
        # Handle fold special case (all holdout tests)
        if task == PEERTask.FOLD:
            holdout_splits = [
                split for split in available_splits 
                if "test" in split and "holdout" in split
            ]
            return holdout_splits if holdout_splits else ["test"]
        
        # Handle PPI special case (include cross-species tests)
        if task in {PEERTask.HUMANPPI, PEERTask.YEASTPPI}:
            return [split for split in ["test", "cross_species_test"] if split in available_splits]
        
        # Default case: just use 'test' if available
        if "test" in available_splits:
            return ["test"]
        
        # Fallback to the last split if no explicit test split
        return [available_splits[-1]]

    def _get_task_datasets(self, task: PEERTask) -> Tuple[PEERDataset, Dict[str, PEERDataset]]:
        """Get or create train and test datasets for a given task.
        
        Returns:
            Tuple containing (train_dataset, test_datasets_dict)
            where test_datasets_dict maps split names to datasets
        """
        cache_key = str(task)
        
        if cache_key in self.datasets:
            return self.datasets[cache_key]
        
        # Always use 'train' for training
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
                # Unpack batch
                inputs, y = batch
                
                # For paired inputs, inputs will be a list of two items
                if isinstance(inputs, list) and len(inputs) == 2:
                    seq1, seq2 = inputs
                    
                    # Handle protein-protein interactions
                    if task in {PEERTask.HUMANPPI, PEERTask.YEASTPPI, PEERTask.PPIAFFINITY}:
                        # Get embeddings for each protein separately
                        embeddings1 = pl_module.get_embeddings([seq1], modality="protein", per_residue=False)
                        embeddings2 = pl_module.get_embeddings([seq2], modality="protein", per_residue=False)
                        
                        # Concatenate the embeddings
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
                
                # Convert targets to appropriate format
                if task_type == "regression":
                    y = y.float()
                else:
                    y = y.long()
                
                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())
        
        return torch.cat(embeddings), torch.cat(targets)

    def _get_structure_embeddings(
        self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask
    ) -> Tuple[Tensor, Tensor]:
        """Extract embeddings for structure prediction tasks with special handling."""
        embeddings = []
        targets = []
        
        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                
                # Secondary structure prediction (per-residue)
                if task == PEERTask.SECONDARY_STRUCTURE:
                    # Get per-residue embeddings
                    batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=True)
                    # Flatten the embeddings to (n_residues, hidden_size)
                    batch_size, seq_len, hidden_size = batch_embeddings.shape
                    batch_embeddings = batch_embeddings.reshape(-1, hidden_size)
                    
                    # Flatten targets to match
                    y = y.view(-1)
                    
                    valid_indices = y != -1  # Assume -1 is padding # TODO - get padding token
                    batch_embeddings = batch_embeddings[valid_indices]
                    y = y[valid_indices]
                
                # Contact map prediction
                elif task == PEERTask.PROTEINNET:
                    # For contact map prediction, we need to handle special target format
                    # y is a tuple of (tertiary_coords, valid_mask)
                    tertiary_coords, valid_mask = y
                    
                    # Get per-residue embeddings
                    batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=True)
                    
                    # Process embeddings to create pairwise features
                    # For simplicity, we'll use outer product of embeddings
                    # In practice, you might want a more sophisticated approach
                    batch_size, seq_len, hidden_size = batch_embeddings.shape
                    
                    # Extract valid embeddings and coords based on mask
                    valid_embeddings = []
                    valid_coords = []
                    for i in range(batch_size):
                        mask = valid_mask[i].bool()
                        if mask.sum() > 0:
                            valid_embeddings.append(batch_embeddings[i, mask])
                            valid_coords.append(tertiary_coords[i, mask])
                    
                    # Skip if no valid embeddings
                    if not valid_embeddings:
                        continue
                    
                    # Create distance matrices as targets
                    contact_targets = []
                    contact_embeddings = []
                    
                    for emb, coords in zip(valid_embeddings, valid_coords):
                        # Calculate pairwise distances
                        n_residues = coords.shape[0]
                        distances = torch.cdist(coords, coords)
                        
                        # Define contacts (e.g., residues closer than 8 Angstroms)
                        contacts = (distances < 8.0).float()
                        
                        # Create embeddings for each residue pair
                        for i in range(n_residues):
                            for j in range(i + 4, n_residues):  # Skip local contacts
                                contact_embeddings.append(torch.cat([emb[i], emb[j]]))
                                contact_targets.append(contacts[i, j])
                    
                    if contact_embeddings:
                        batch_embeddings = torch.stack(contact_embeddings)
                        y = torch.tensor(contact_targets)
                    else:
                        continue
                
                # Standard fold classification
                elif task == PEERTask.FOLD:
                    batch_embeddings = pl_module.get_embeddings([x], modality="protein", per_residue=False)
                
                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())
        
        return torch.cat(embeddings), torch.cat(targets)

    def _get_embeddings(self, pl_module: L.LightningModule, dataloader: DataLoader, task: PEERTask = None) -> Tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader with task-specific handling."""
        # Use task-specific embedding extraction based on category
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
                    
                    # Get embeddings using model's get_embeddings method
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
        self._set_metrics(task_type, num_classes)
        
        # Update task_type for certain tasks
        if task == PEERTask.SECONDARY_STRUCTURE:
            # Secondary structure is a multiclass classification problem
            task_type = "multiclass"
            num_classes = 3  # 3 secondary structure classes
            self._set_metrics(task_type, num_classes)
        elif task == PEERTask.SUBCELLULAR_LOCALIZATION:
            # Subcellular localization is a multiclass problem
            task_type = "multiclass"
            num_classes = 10  # 10 localization classes
            self._set_metrics(task_type, num_classes)
        elif task == PEERTask.FOLD:
            # Fold classification is a multiclass problem
            task_type = "multiclass"
            num_classes = 1195  # 1195 fold classes in PEER
            self._set_metrics(task_type, num_classes)
        elif task == PEERTask.PROTEINNET:
            # Contact prediction is a binary classification problem
            task_type = "binary"
            self._set_metrics(task_type)
        
        # Train probe based on task type
        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()
        
        if task_type == "regression":
            probe = LinearRegression()
            probe.fit(embeddings_np, targets_np)
        elif task_type == "multilabel":
            base_classifier = LogisticRegression(random_state=42)
            probe = MultiOutputClassifier(base_classifier)
            probe.fit(embeddings_np, targets_np)
        else:  # binary or multiclass
            probe = LogisticRegression(
                multi_class="ovr" if task_type == "binary" else "multinomial",
                random_state=42,
                max_iter=1000,  # Increase for convergence
            )
            probe.fit(embeddings_np, targets_np.ravel())
        
        return probe

    def _evaluate_task(self, task: PEERTask, trainer: L.Trainer, pl_module: L.LightningModule) -> Dict[str, Dict[str, float]]:
        """Evaluate a single PEER task across all its test splits.
        
        Returns:
            Dict mapping split names to dictionaries of metrics
        """
        try:
            # Get datasets for this task
            train_dataset, test_datasets = self._get_task_datasets(task)
            
            # Create train loader
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )
            
            # Extract train embeddings
            train_embeddings, train_targets = self._get_embeddings(pl_module, train_loader, task)
            
            # Train probe
            probe = self._train_probe(train_embeddings, train_targets, task)
            self.probes[str(task)] = probe
            
            # Evaluate on each test split
            split_metrics = {}
            for split_name, test_dataset in test_datasets.items():
                test_loader = DataLoader(
                    test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
                )
                
                # Extract test embeddings
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
                
                # For computing global averages, we count each split as a separate evaluation
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
                        {f"peer_linear_probe/category/{category}/{metric_name}": avg_value},
                        step=trainer.current_epoch
                    )
        
        # Log overall averages
        for metric_name, values in all_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                trainer.logger.log_metrics(
                    {f"peer_linear_probe/mean/{metric_name}": avg_value},
                    step=trainer.current_epoch
                )
        
        # Log total number of splits evaluated
        trainer.logger.log_metrics(
            {f"peer_linear_probe/total_splits_evaluated": split_count},
            step=trainer.current_epoch
        )