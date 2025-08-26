import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence

import lightning as L
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from torch import Tensor
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

from lobster.constants import CALM_TASKS, CALM_TASK_SPECIES
from lobster.datasets import CalmPropertyDataset

from ._linear_probe_callback import LinearProbeCallback
from ._peer_utils import convert_numpy_to_python

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


class CalmLinearProbeCallback(LinearProbeCallback):
    """Callback for evaluating embedding models on the CALM dataset collection.
    This callback performs linear probing evaluation on various cDNA sequence property
    prediction tasks from the CALM dataset collection. It creates train/test splits for each task,
    extracts embeddings from the model, trains linear probes on these embeddings, and evaluates
    their performance.

    Currently only supports UME embeddings and uses UMETokenizerTransform.

    Parameters
    ----------
    max_length : int
        Maximum sequence length for tokenization.
    tasks : Optional[Sequence[str]], default=None
        Specific CALM tasks to evaluate. If None, all available tasks will be used.
        Available tasks:
        - 'meltome': Predicts protein melting temperature (regression)
        - 'solubility': Predicts protein solubility (regression)
        - 'localization': Predicts cellular localization (multilabel, 10 classes)
        - 'protein_abundance': Predicts protein abundance (regression, species-specific)
        - 'transcript_abundance': Predicts transcript abundance (regression, species-specific)
        - 'function_bp': Predicts Gene Ontology biological process terms (multilabel, 4 classes)
        - 'function_cc': Predicts Gene Ontology cellular component terms (multilabel, 4 classes)
        - 'function_mf': Predicts Gene Ontology molecular function terms (multilabel, 4 classes)
    species : Optional[Sequence[str]], default=None
        Species to include for species-specific tasks ('protein_abundance' and 'transcript_abundance').
        If None, defaults to ['hsapiens', 'ecoli', 'scerevisiae'] for comprehensive evaluation.
        Available species:
        - For protein_abundance: 'athaliana', 'dmelanogaster', 'ecoli', 'hsapiens', 'scerevisiae'
        - For transcript_abundance: All of the above plus 'hvolcanii' and 'ppastoris'
    batch_size : int, default=32
        Batch size for embedding extraction and evaluation.
    run_every_n_epochs : Optional[int], default=None
        Run this callback every n epochs. If None, runs every validation epoch.
    test_size : float, default=0.2
        Fraction of data to use for testing. Ignored if use_cross_validation=True.
    max_samples : int, default=3000
        Maximum number of samples to use from each dataset.
    use_cross_validation : bool, default=False
        Whether to use k-fold cross validation instead of single train/test split.
    n_folds : int, default=5
        Number of folds for cross validation (only used if use_cross_validation=True).
    dimensionality_reduction : bool, default=False
        Whether to apply PCA dimensionality reduction to embeddings before training probes.
    reduced_dim : int, default=320
        Number of dimensions to reduce to (only used if dimensionality_reduction=True).
    probe_type : str, default="linear"
        Type of probe to use. Options: "linear", "elastic", "svm".

    Attributes
    ----------
    dataset_splits : dict
        Cache of train/test splits for each task.
    aggregate_metrics : defaultdict
        Collection of metrics across all tasks for averaging.
    probes : dict
        Trained linear probes for each task.
    """

    def __init__(
        self,
        max_length: int = None,  # Keep for API compatibility but not used
        tasks: Sequence[str] | None = None,
        species: Sequence[str] | None = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
        test_size: float = 0.2,
        max_samples: int = 3000,
        use_cross_validation: bool = False,
        n_folds: int = 5,
        dimensionality_reduction: bool = False,
        reduced_dim: int = 320,
        probe_type: str = "linear",
    ):
        # Don't use transform_fn - we'll handle raw sequences with explicit modality
        super().__init__(
            transform_fn=None,
            task_type="regression",
            batch_size=batch_size,
            run_every_n_epochs=run_every_n_epochs,
            use_cross_validation=use_cross_validation,
            n_folds=n_folds,
            dimensionality_reduction=dimensionality_reduction,
            reduced_dim=reduced_dim,
            probe_type=probe_type,
        )

        self.tasks = set(tasks) if tasks else set(CALM_TASKS.keys())
        print(f"[CALM DEBUG] CALM tasks to evaluate: {sorted(self.tasks)}")
        logger.info(f"CALM tasks to evaluate: {sorted(self.tasks)}")
        
        # Set default species if none provided - include commonly used model organisms
        if species is None:
            # Use a subset that represents diverse biology: human, E. coli, and yeast
            default_species = ["hsapiens", "ecoli", "scerevisiae"]
            self.species = set(default_species)
            logger.info(f"No species specified, using default species: {default_species}")
        else:
            self.species = set(species)
            logger.info(f"Using specified species: {sorted(self.species)}")

        self.test_size = test_size
        self.max_samples = max_samples

        self.dataset_splits = {}
        self.aggregate_metrics = defaultdict(list)

    def _create_split_datasets(
        self, task: str, species: str | None = None
    ) -> tuple[CalmPropertyDataset, CalmPropertyDataset]:
        """Create train/test splits for a given task.

        Parameters
        ----------
        task : str
            The CALM task name to create splits for
        species : Optional[str], default=None
            The species name for species-specific tasks

        Returns
        -------
        Tuple[CalmPropertyDataset, CalmPropertyDataset]
            A tuple containing (train_dataset, test_dataset)
        """
        # Check cache for existing splits
        split_key = f"{task}_{species}" if species else task
        if split_key in self.dataset_splits:
            return self.dataset_splits[split_key]

        dataset = CalmPropertyDataset(task=task, species=species)

        indices = np.arange(len(dataset))

        # If dataset is too large, subsample it first
        if len(indices) > self.max_samples:
            indices = np.random.choice(indices, size=self.max_samples, replace=False)

        # Create train/test split from (possibly subsampled) indices
        test_size = int(len(indices) * self.test_size)
        train_size = len(indices) - test_size
        shuffled_indices = np.random.permutation(indices)
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        # Cache the splits
        self.dataset_splits[split_key] = (train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _get_embeddings(
        self, model: L.LightningModule | torch.nn.Module, dataloader: DataLoader
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader.
        
        Overrides parent method to handle CALM-specific data format where
        sequences come as raw strings that need to be processed through UME's
        embed_sequences method which properly handles padding tokens.

        Parameters
        ----------
        model : Union[L.LightningModule, torch.nn.Module]
            The model to extract embeddings from
        dataloader : DataLoader
            DataLoader for the data to extract embeddings for

        Returns
        -------
        Tuple[Tensor, Tensor]
            Tuple of (embeddings, targets)
        """
        embeddings = []
        targets = []

        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                
                # x comes as raw strings - convert to list if needed
                if isinstance(x, (list, tuple)):
                    sequences = list(x)
                else:
                    # Handle case where x might be a single string
                    sequences = [x] if isinstance(x, str) else x.tolist()

                # Use UME's embed_sequences method which properly handles tokenization,
                # padding tokens, and mean pooling
                batch_embeddings = model.embed_sequences(
                    sequences=sequences,
                    modality="nucleotide", 
                    aggregate=True  # This does mean pooling and handles padding properly
                )

                embeddings.append(batch_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _evaluate_task(
        self,
        task_key: str,
        task: str,
        train_dataset,
        test_dataset,
        module: L.LightningModule,
        trainer: L.Trainer = None,
    ) -> dict[str, float]:
        """Evaluate a single task.

        Parameters
        ----------
        task_key : str
            The task key for logging (task name or task_species)
        task : str
            The actual task name
        train_dataset : Dataset
            Training dataset
        test_dataset : Dataset
            Test dataset
        module : L.LightningModule
            Model to evaluate
        trainer : Optional[L.Trainer]
            Optional trainer for logging

        Returns
        -------
        Dict[str, float]
            Dictionary of metric_name -> value
        """
        task_type, num_classes = CALM_TASKS[task]

        # For multilabel tasks (e.g., localization, function_*), infer label count from dataset
        if task_type == "multilabel":
            # Unwrap Subset to access underlying dataset
            base_dataset = train_dataset
            while isinstance(base_dataset, Subset):
                base_dataset = base_dataset.dataset
            inferred_num = getattr(base_dataset, "num_label_columns", None)
            if isinstance(inferred_num, int) and inferred_num > 0:
                num_classes = inferred_num

        self._set_metrics(task_type, num_classes)

        try:
            if self.use_cross_validation:
                # For cross-validation, we need all data combined
                # Create a combined dataset from train and test
                combined_dataset = ConcatDataset([train_dataset, test_dataset])
                combined_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
                
                # Get all embeddings and targets
                all_embeddings, all_targets = self._get_embeddings(module, combined_loader)
                
                # Use cross-validation evaluation
                metrics = self._evaluate_with_cross_validation(all_embeddings, all_targets, task_key)
                
            else:
                # Use traditional train/test split
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                train_embeddings, train_targets = self._get_embeddings(module, train_loader)
                test_embeddings, test_targets = self._get_embeddings(module, test_loader)

                probe = self._train_probe(train_embeddings, train_targets, task_key)
                self.probes[task_key] = probe

                metrics = self._evaluate_probe(probe, test_embeddings, test_targets, task_key)

            # Log metrics if trainer is provided
            if trainer is not None:
                for metric_name, value in metrics.items():
                    metric_key = f"calm_linear_probe/{task_key}/{metric_name}"
                    trainer.logger.log_metrics({metric_key: value})

            return metrics

        except Exception as e:
            logger.debug(f"Error in _evaluate_task for {task_key}: {str(e)}")
            raise

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model on CALM datasets using linear probes.

        This method can be used both during training (with a trainer)
        and standalone (with just a model).

        Parameters
        ----------
        model : L.LightningModule
            The model to evaluate
        trainer : Optional[L.Trainer]
            Optional trainer for logging metrics

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of task_name -> metric_name -> value
        """
        # Clear metrics for this run
        aggregate_metrics = defaultdict(list)
        all_task_metrics = {}

        for task in tqdm(self.tasks, desc=f"{self.__class__.__name__}"):
            print(f"[CALM DEBUG] Processing task: {task}")
            logger.info(f"Processing task: {task}")
            # Handle species-specific tasks
            if task in ["protein_abundance", "transcript_abundance"]:
                logger.info(f"Task {task} is species-specific, processing for species: {sorted(self.species)}")
                for species in self.species:
                    task_key = f"{task}_{species}"
                    logger.info(f"Starting evaluation for {task_key}")
                    try:
                        train_dataset, test_dataset = self._create_split_datasets(task, species)
                        logger.info(f"Created datasets for {task_key}: train={len(train_dataset)}, test={len(test_dataset)}")
                        metrics = self._evaluate_task(
                            task_key,
                            task,
                            train_dataset,
                            test_dataset,
                            module,
                            trainer,
                        )
                        all_task_metrics[task_key] = metrics
                        logger.info(f"Successfully completed {task_key} with metrics: {metrics}")

                        # Store for aggregate metrics
                        for metric_name, value in metrics.items():
                            aggregate_metrics[metric_name].append(value)
                    except Exception as e:
                        logger.error(f"Error processing {task_key}: {str(e)}")
                        print(f"[CALM ERROR] {task_key} failed: {e}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
            else:
                logger.info(f"Task {task} is non-species-specific")
                try:
                    train_dataset, test_dataset = self._create_split_datasets(task)
                    logger.info(f"Created datasets for {task}: train={len(train_dataset)}, test={len(test_dataset)}")
                    metrics = self._evaluate_task(
                        task,
                        task,
                        train_dataset,
                        test_dataset,
                        module,
                        trainer,
                    )
                    all_task_metrics[task] = metrics
                    logger.info(f"Successfully completed {task} with metrics: {metrics}")

                    # Store for aggregate metrics
                    for metric_name, value in metrics.items():
                        aggregate_metrics[metric_name].append(value)
                except Exception as e:
                    logger.error(f"Error processing {task}: {str(e)}")
                    print(f"[CALM ERROR] {task} failed: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")

        # Calculate and log aggregate metrics
        mean_metrics = {}
        for metric_name, values in aggregate_metrics.items():
            if values:  # Only process if we have values
                avg_value = sum(values) / len(values)
                mean_metrics[metric_name] = avg_value

                # Log if trainer is provided
                if trainer is not None:
                    trainer.logger.log_metrics(
                        {f"calm_linear_probe/mean/{metric_name}": avg_value},
                    )

        # Add mean metrics to the result
        all_task_metrics["mean"] = mean_metrics

        # Log final summary
        successful_tasks = [k for k in all_task_metrics.keys() if k != "mean"]
        logger.info(f"Evaluation completed. Successful tasks: {successful_tasks}")
        logger.info(f"Total successful tasks: {len(successful_tasks)}")

        # Convert NumPy scalars to Python types for clean YAML formatting
        return convert_numpy_to_python(all_task_metrics)