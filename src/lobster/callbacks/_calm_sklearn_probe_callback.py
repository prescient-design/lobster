import logging
from collections.abc import Sequence

import lightning as L
import torch
import torch.utils.data
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from lobster.constants import SklearnProbeType, CALM_TASKS, CALM_DEFAULT_SPECIES, CALM_SPECIES_SPECIFIC_TASKS
from lobster.datasets import CalmPropertyDataset

from ._sklearn_probe_callback import SklearnProbeCallback, SklearnProbeTaskConfig

logger = logging.getLogger(__name__)


class CalmSklearnProbeCallback(SklearnProbeCallback):
    """Callback for evaluating embedding models on the CALM dataset collection.

    This callback performs scikit-learn probing evaluation on various cDNA sequence property
    prediction tasks from the CALM dataset collection. It creates train/test splits for each task,
    extracts embeddings from the model, trains linear probes on these embeddings, and evaluates
    their performance.

    Parameters
    ----------
    tasks : Sequence[str] | None, default=None
        Specific CALM tasks to evaluate. If None, all available tasks will be used.
        Available tasks:
        - 'meltome': Predicts protein melting temperature (regression)
        - 'solubility': Predicts protein solubility (regression)
        - 'localization': Predicts cellular localization (multilabel, 10 classes)
        - 'protein_abundance': Predicts protein abundance (regression, species-specific)
        - 'transcript_abundance': Predicts transcript abundance (regression, species-specific)
        - 'function_bp': Predicts Gene Ontology biological process terms (multilabel, 5 classes)
        - 'function_cc': Predicts Gene Ontology cellular component terms (multilabel, 5 classes)
        - 'function_mf': Predicts Gene Ontology molecular function terms (multilabel, 5 classes)
    species : Sequence[str] | None, default=None
        Species to include for species-specific tasks ('protein_abundance' and 'transcript_abundance').
        If None, defaults to ['hsapiens', 'ecoli', 'scerevisiae'] for comprehensive evaluation.
        Available species:
        - For protein_abundance: 'athaliana', 'dmelanogaster', 'ecoli', 'hsapiens', 'scerevisiae'
        - For transcript_abundance: All of the above plus 'hvolcanii' and 'ppastoris'
    batch_size : int, default=32
        Batch size for embedding extraction and evaluation.
    probe_type : SklearnProbeType, default="linear"
        Type of probe to use. Options: "linear", "elastic", "svm".
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
    classification_threshold : float, default=0.5
        Threshold for multilabel classification tasks.
    seed : int, default=0
        Random seed for reproducibility.
    """

    def __init__(
        self,
        tasks: Sequence[str] | None = None,
        species: Sequence[str] | None = None,
        batch_size: int = 32,
        probe_type: SklearnProbeType = "linear",
        test_size: float = 0.2,
        max_samples: int = 3000,
        use_cross_validation: bool = False,
        n_folds: int = 5,
        dimensionality_reduction: bool = False,
        reduced_dim: int = 320,
        classification_threshold: float = 0.5,
        ignore_errors: bool = False,
        seed: int = 0,
    ):
        super().__init__(batch_size=batch_size, seed=seed)

        self.probe_type = probe_type
        self.test_size = test_size
        self.max_samples = max_samples
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.dimensionality_reduction = dimensionality_reduction
        self.reduced_dim = reduced_dim
        self.classification_threshold = classification_threshold
        self.ignore_errors = ignore_errors

        self.tasks = set(tasks) if tasks else set(CALM_TASKS.keys())
        logger.info(f"CALM tasks to evaluate: {sorted(self.tasks)}")

        if species is None:
            default_species = CALM_DEFAULT_SPECIES
            self.species = set(default_species)
            logger.info(f"No species specified, using default species: {default_species}")
        else:
            self.species = set(species)
            logger.info(f"Using specified species: {sorted(self.species)}")

        self.dataset_splits = {}

    def _random_split_dataset(
        self,
        dataset: Dataset,
    ) -> tuple:
        """Create train/test splits for a given task.

        Parameters
        ----------
        task : str
            The CALM task name to create splits for
        species : str | None, default=None
            The species name for species-specific tasks

        Returns
        -------
        tuple
            A tuple containing (train_dataset, test_dataset)
        """
        generator = torch.Generator().manual_seed(self.seed)

        # Convert fraction to actual counts
        total_size = len(dataset)
        test_count = int(total_size * self.test_size)
        train_count = total_size - test_count

        return torch.utils.data.random_split(dataset, [train_count, test_count], generator=generator)

    def _subsample_dataset(self, dataset: Dataset) -> Dataset:
        """Create full dataset for a given task (for cross-validation).

        Parameters
        ----------
        dataset : Dataset
            The dataset to subsample

        Returns
        -------
        Dataset
            The full dataset, potentially subsampled if too large
        """
        # If dataset is too large, subsample it
        if len(dataset) > self.max_samples:
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(self.seed))[
                : self.max_samples
            ].tolist()  # Convert tensor to list of integers
            dataset = Subset(dataset, indices)

        return dataset

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
        module : L.LightningModule
            The model to evaluate
        trainer : L.Trainer | None
            Optional trainer for logging metrics

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary of task_name -> metric_name -> value
        """
        all_task_metrics = {}

        for task in tqdm(self.tasks, desc=self.__class__.__name__):
            logger.info(f"Evaluating task: {task}")
            task_type, num_classes = CALM_TASKS[task]

            # Handle species-specific tasks
            if task in CALM_SPECIES_SPECIFIC_TASKS:
                species_list = self.species
            else:
                species_list = [None]

            per_species_metrics = {}

            # Run evaluation for each species with or without cross-validation
            for species in species_list:
                task_key = f"{task}_{species}" if species is not None else task
                logger.info(f"Starting evaluation for {task_key} and species: {species}")

                dataset = CalmPropertyDataset(task=task, species=species)
                dataset = self._subsample_dataset(dataset=dataset)

                if task_type == "multilabel":
                    num_classes = getattr(dataset, "num_label_columns", None)

                    if not isinstance(num_classes, int) or num_classes <= 0:
                        if self.ignore_errors:
                            logger.warning(f"Could not infer number of label columns for task: {task}. Skipping task.")
                            continue
                        else:
                            raise ValueError(f"Could not infer number of label columns for task: {task}")

                config = SklearnProbeTaskConfig(
                    task_name=task_key,
                    task_type=task_type,
                    probe_type=self.probe_type,
                    num_classes=num_classes,
                    modality="nucleotide",
                    dimensionality_reduction=self.dimensionality_reduction,
                    reduced_dim=self.reduced_dim,
                    classification_threshold=self.classification_threshold,
                )

                try:
                    # Evaluate with cross-validation
                    if self.use_cross_validation:
                        result = self.train_and_evaluate_cv_probe_on_task(
                            model=module,
                            dataset=dataset,
                            task_config=config,
                            n_folds=self.n_folds,
                        )

                    # Without cross-validation
                    else:
                        train_dataset, test_dataset = self._random_split_dataset(dataset)
                        logger.info(
                            f"Created datasets for {task_key}: train={len(train_dataset)}, test={len(test_dataset)}"
                        )
                        result = self.train_and_evaluate_probe_on_task(
                            model=module,
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            task_config=config,
                        )

                except Exception as e:
                    if self.ignore_errors:
                        logger.error(f"Error processing {task_key}: {str(e)}. Skipping task.")
                        metrics = {}
                        continue
                    else:
                        raise e

                metrics = result.metrics

                per_species_metrics[task_key] = metrics

                self.log_metrics(
                    metrics=metrics,
                    task_name=task_key,
                    probe_type=self.probe_type,
                    trainer=trainer,
                )

            mean_task_metrics = self._compute_mean_metrics(per_species_metrics)
            all_task_metrics[task] = mean_task_metrics

            # Only log mean metrics across species if species was specified
            if species is not None:
                self.log_metrics(
                    metrics=mean_task_metrics,
                    task_name=task,
                    probe_type=self.probe_type,
                    is_mean=True,
                    trainer=trainer,
                )

        mean_metrics_across_tasks = self._compute_mean_metrics(all_task_metrics)

        # Add mean metrics to the result
        all_task_metrics["mean"] = mean_metrics_across_tasks

        self.log_metrics(
            metrics=mean_metrics_across_tasks,
            task_name="mean",
            probe_type=self.probe_type,
            is_mean=True,
            trainer=trainer,
        )

        successful_tasks = [k for k in all_task_metrics.keys() if k != "mean"]
        logger.info(
            f"Evaluation completed. Successful tasks: (n={len(successful_tasks)}/{len(self.tasks)}) {successful_tasks}"
        )
        logger.info(f"Results: {all_task_metrics}")

        return all_task_metrics
