import warnings
from collections.abc import Callable
from typing import Literal

import lightning as L
import numpy as np
import torch
import math
from lobster.transforms import Transform
from lightning.pytorch.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef

TaskType = Literal["regression", "binary", "multiclass", "multilabel"]
ProbeType = Literal["linear", "elastic", "svm"]

warnings.filterwarnings(
    "ignore",
    message="Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer",
    category=UserWarning,
)


class LinearProbeCallback(Callback):
    """Callback for evaluating embedding models using scikit-learn linear probes.

    Assumes the underlying model is UME as it accesses
    `module.model.tokens_to_latents` to extract embeddings. To use with other
    models, you may need to override `_get_embeddings`.

    Sublclasses must implement the `evaluate` method to evaluate the model on
    specific tasks.
    """

    def __init__(
        self,
        task_type: TaskType = "regression",
        transform_fn: Transform | Callable | None = None,
        num_classes: int | None = None,
        batch_size: int = 32,
        run_every_n_epochs: int | None = None,
        use_cross_validation: bool = False,
        n_folds: int = 5,
        dimensionality_reduction: bool = False,
        reduced_dim: int = 320,
        probe_type: ProbeType = "linear",
        classification_threshold: float = 0.5,
    ):
        super().__init__()
        self.transform_fn = transform_fn
        self.task_type = task_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.run_every_n_epochs = run_every_n_epochs
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.dimensionality_reduction = dimensionality_reduction
        self.reduced_dim = reduced_dim
        self.probe_type = probe_type
        self.classification_threshold = classification_threshold

        # Initialize metrics based on task type
        self._set_metrics(task_type, num_classes)

        # Dictionary to store trained probes and dimensionality reducers
        self.probes: dict[str, LinearRegression | LogisticRegression | ElasticNet | SVR | SVC] = {}
        self.dim_reducers: dict[str, PCA] = {}
        self.feature_scalers: dict[str, StandardScaler] = {}

    def _set_metrics(self, task_type: TaskType, num_classes: int | None = None) -> None:
        """Initialize metrics based on task type."""
        # Use a slightly lower default threshold for multilabel to reduce all-negative predictions
        classification_threshold = self.classification_threshold
        if task_type == "regression":
            self.mse = MeanSquaredError()
            self.r2 = R2Score()
            self.spearman = SpearmanCorrCoef()
            self.pearson = PearsonCorrCoef()
            self.accuracy = None
            self.f1 = None
            self.f1_weighted = None
            self.auroc = None

        elif task_type in {"binary", "multiclass", "multilabel"}:
            # Use correct parameter names for different task types
            metric_task = task_type
            if task_type == "multilabel":
                # For multilabel, use num_labels parameter
                self.accuracy = Accuracy(task=metric_task, num_labels=num_classes, threshold=classification_threshold)
                self.f1 = F1Score(task=metric_task, num_labels=num_classes, threshold=classification_threshold)
                self.f1_weighted = F1Score(
                    task=metric_task, num_labels=num_classes, average="weighted", threshold=classification_threshold
                )
                self.auroc = AUROC(task=metric_task, num_labels=num_classes)
            else:
                # For binary and multiclass, use num_classes parameter
                self.accuracy = Accuracy(task=metric_task, num_classes=num_classes)
                self.f1 = F1Score(task=metric_task, num_classes=num_classes)
                self.f1_weighted = F1Score(task=metric_task, num_classes=num_classes, average="weighted")
                self.auroc = AUROC(task=metric_task, num_classes=num_classes)
            self.mse = None
            self.r2 = None
            self.spearman = None
            self.pearson = None

        else:
            raise ValueError("task_type must be: regression, binary, multiclass, or multilabel")

        self.task_type = task_type
        self.num_classes = num_classes

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip validation this epoch."""
        # Don't skip if run_every_n_epochs is not set
        if self.run_every_n_epochs is None:
            return False

        # Skip if not in the main process
        if trainer.global_rank != 0:
            return True

        return trainer.current_epoch % self.run_every_n_epochs != 0

    def _get_embeddings(
        self, model: L.LightningModule | torch.nn.Module, dataloader: DataLoader
    ) -> tuple[Tensor, Tensor]:
        """Extract embeddings from the model for a given dataloader.

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
                x = {k: v.to(model.device) for k, v in x.items()}

                # Get token-level embeddings
                batch_embeddings = model.model.tokens_to_latents(**x)

                # Reshape to (batch_size, seq_len, hidden_size)
                batch_size = len(y)
                seq_len = x["input_ids"].size(-1)
                batch_embeddings = batch_embeddings.view(batch_size, seq_len, -1)

                # Simple mean pooling over sequence length dimension
                seq_embeddings = batch_embeddings.mean(dim=1)

                embeddings.append(seq_embeddings.cpu())
                targets.append(y.cpu())

        return torch.cat(embeddings), torch.cat(targets)

    def _train_probe(self, embeddings: Tensor, targets: Tensor, task_key: str = "default"):
        """Train a probe on the given embeddings and targets."""
        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()

        # Standardize features per task_key to stabilize probe training
        if task_key not in self.feature_scalers:
            scaler = StandardScaler()
            embeddings_np = scaler.fit_transform(embeddings_np)
            self.feature_scalers[task_key] = scaler
        else:
            embeddings_np = self.feature_scalers[task_key].transform(embeddings_np)

        # Apply dimensionality reduction if requested
        if self.dimensionality_reduction:
            if task_key not in self.dim_reducers:
                # Ensure we don't try to reduce to more dimensions than we have samples/features
                n_samples, n_features = embeddings_np.shape
                actual_reduced_dim = min(self.reduced_dim, n_samples - 1, n_features)
                pca = PCA(n_components=actual_reduced_dim, random_state=42)
                embeddings_np = pca.fit_transform(embeddings_np)
                self.dim_reducers[task_key] = pca
            else:
                embeddings_np = self.dim_reducers[task_key].transform(embeddings_np)

        # Train probe based on task type and probe type
        if self.task_type == "regression":
            if self.probe_type == "linear":
                probe = LinearRegression()
            elif self.probe_type == "elastic":
                probe = ElasticNet(random_state=42, max_iter=5000)
            elif self.probe_type == "svm":
                probe = SVR(kernel="linear")
            probe.fit(embeddings_np, targets_np)

            # Fallback: if model predicts a constant on training set, retry with LinearRegression
            try:
                train_pred = probe.predict(embeddings_np)
                if float(np.std(train_pred)) == 0.0:
                    probe = LinearRegression()
                    probe.fit(embeddings_np, targets_np)
            except Exception:
                pass

        elif self.task_type == "multilabel":
            # Ensure targets are integers for multilabel classification
            targets_np = targets_np.astype(int)

            if self.probe_type == "linear":
                base_classifier = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
                probe = MultiOutputClassifier(base_classifier)
            elif self.probe_type == "elastic":
                # For multilabel with ElasticNet, use LogisticRegression with elastic net penalty
                base_classifier = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    random_state=42,
                    max_iter=1000,
                    class_weight="balanced",
                )
                probe = MultiOutputClassifier(base_classifier)
            elif self.probe_type == "svm":
                base_classifier = SVC(kernel="linear", probability=True, random_state=42, class_weight="balanced")
                probe = MultiOutputClassifier(base_classifier)
            probe.fit(embeddings_np, targets_np)

        else:  # binary or multiclass
            if self.probe_type == "linear":
                probe = LogisticRegression(
                    multi_class="ovr" if self.task_type == "binary" else "multinomial",
                    random_state=42,
                    max_iter=1000,
                )
            elif self.probe_type == "elastic":
                # For classification with ElasticNet, we need to use LogisticRegression with elasticnet penalty
                probe = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    multi_class="ovr" if self.task_type == "binary" else "multinomial",
                    random_state=42,
                    max_iter=1000,
                )
            elif self.probe_type == "svm":
                probe = SVC(kernel="linear", probability=True, random_state=42)
            probe.fit(embeddings_np, targets_np.ravel())

        return probe

    def _evaluate_probe(
        self, probe, embeddings: Tensor, targets: Tensor, task_key: str = "default"
    ) -> dict[str, float]:
        """Evaluate a trained probe using task-appropriate metrics."""
        embeddings_np = embeddings.numpy()  # Convert to numpy for probe prediction

        # Apply standardization and dimensionality reduction if used during training
        if task_key in self.feature_scalers:
            embeddings_np = self.feature_scalers[task_key].transform(embeddings_np)
        if self.dimensionality_reduction and task_key in self.dim_reducers:
            embeddings_np = self.dim_reducers[task_key].transform(embeddings_np)

        metrics = {}

        if self.task_type == "regression":
            predictions_np = probe.predict(embeddings_np)
            predictions = torch.from_numpy(predictions_np).float()

            # Ensure predictions and targets have matching shapes
            if predictions.dim() != targets.dim():
                if targets.dim() == 2 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                elif predictions.dim() == 1 and targets.dim() == 2:
                    predictions = predictions.unsqueeze(1)

            # Use fresh metric instances per evaluation to avoid state leakage across folds
            mse_metric = MeanSquaredError()
            r2_metric = R2Score()
            spearman_metric = SpearmanCorrCoef()
            pearson_metric = PearsonCorrCoef()

            metrics["mse"] = mse_metric(predictions, targets).item()
            metrics["r2"] = r2_metric(predictions, targets).item()
            spearman_val = spearman_metric(predictions.squeeze(), targets.squeeze()).item()
            metrics["spearman"] = 0.0 if math.isnan(spearman_val) else spearman_val
            pearson_val = pearson_metric(predictions.squeeze(), targets.squeeze()).item()
            if math.isnan(pearson_val):
                pearson_val = 0.0
            metrics["pearson"] = pearson_val

        else:  # binary, multiclass, or multilabel
            if self.task_type == "multilabel":
                # Ensure targets are integers for multilabel classification
                targets = targets.int()

                # All supported multilabel estimators expose predict_proba
                predictions_np = np.stack(
                    [est.predict_proba(embeddings_np)[:, 1] for est in probe.estimators_], axis=1
                )

            else:  # binary or multiclass
                if hasattr(probe, "predict_proba"):
                    predictions_np = probe.predict_proba(embeddings_np)
                    if self.task_type == "binary":
                        predictions_np = predictions_np[:, 1]
                else:
                    # For models without predict_proba (like ElasticNet for classification), use decision_function
                    if hasattr(probe, "decision_function"):
                        predictions_np = probe.decision_function(embeddings_np)
                        # Apply sigmoid for binary classification or softmax for multiclass
                        if self.task_type == "binary":
                            predictions_np = 1 / (1 + np.exp(-predictions_np))  # sigmoid
                        else:
                            # For multiclass, apply softmax
                            exp_pred = np.exp(predictions_np)
                            predictions_np = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                    else:
                        # Fallback to predict (will affect metric accuracy)
                        predictions_np = probe.predict(embeddings_np).astype(float)

            predictions = torch.from_numpy(predictions_np).float()

            # Fresh metric instances per evaluation
            if self.task_type == "multilabel":
                accuracy_metric = Accuracy(task="multilabel", num_labels=self.num_classes)
                f1_metric = F1Score(task="multilabel", num_labels=self.num_classes)
                f1_weighted_metric = F1Score(task="multilabel", num_labels=self.num_classes, average="weighted")
                auroc_metric = AUROC(task="multilabel", num_labels=self.num_classes)
            else:
                accuracy_metric = Accuracy(task=self.task_type, num_classes=self.num_classes)
                f1_metric = F1Score(task=self.task_type, num_classes=self.num_classes)
                f1_weighted_metric = F1Score(task=self.task_type, num_classes=self.num_classes, average="weighted")
                auroc_metric = AUROC(task=self.task_type, num_classes=self.num_classes)

            metrics["accuracy"] = accuracy_metric(predictions, targets).item()
            metrics["f1"] = f1_metric(predictions, targets).item()
            metrics["f1_weighted"] = f1_weighted_metric(predictions, targets).item()
            metrics["auroc"] = auroc_metric(predictions, targets).item()

        return metrics

    def _evaluate_with_cross_validation(self, embeddings: Tensor, targets: Tensor, task_key: str) -> dict[str, float]:
        """Evaluate using k-fold cross validation."""
        embeddings_np = embeddings.numpy()
        targets_np = targets.numpy()

        # Choose a splitter. For multilabel tasks, prefer iterative stratification if available.
        splitter = None
        if self.task_type == "multilabel":
            try:
                from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

                splitter = MultilabelStratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            except Exception:
                splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        else:
            splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_metrics = {
            "mse": [],
            "r2": [],
            "spearman": [],
            "pearson": [],
            "accuracy": [],
            "f1": [],
            "f1_weighted": [],
            "auroc": [],
        }

        # For multilabel, pass targets to splitter for stratification; otherwise use features only
        split_iter = (
            splitter.split(embeddings_np, targets_np)
            if self.task_type == "multilabel"
            else splitter.split(embeddings_np)
        )

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            fold_task_key = f"{task_key}_fold_{fold_idx}"

            # Split embeddings and targets for this fold
            train_embeddings = torch.from_numpy(embeddings_np[train_idx]).float()
            val_embeddings = torch.from_numpy(embeddings_np[val_idx]).float()
            train_targets = torch.from_numpy(targets_np[train_idx])
            val_targets = torch.from_numpy(targets_np[val_idx])

            # Set appropriate dtype based on task type
            if self.task_type == "regression":
                train_targets = train_targets.float()
                val_targets = val_targets.float()
            else:  # classification tasks
                train_targets = train_targets.int() if self.task_type == "multilabel" else train_targets.long()
                val_targets = val_targets.int() if self.task_type == "multilabel" else val_targets.long()

            # Train probe on fold training data
            fold_probe = self._train_probe(train_embeddings, train_targets, fold_task_key)

            # Evaluate probe on fold validation data
            fold_metrics = self._evaluate_probe(fold_probe, val_embeddings, val_targets, fold_task_key)

            # Store fold metrics
            for metric_name, value in fold_metrics.items():
                if metric_name in cv_metrics:
                    cv_metrics[metric_name].append(value)

        # Average across folds (only for metrics that have values)
        avg_metrics = {}
        for metric_name, values in cv_metrics.items():
            if values:  # Only process if we have values for this metric
                avg_metrics[metric_name] = np.mean(values)
                avg_metrics[f"{metric_name}_std"] = np.std(values)

        return avg_metrics

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model using linear probes.

        This method can be used both during training (with a trainer) and
        standalone (with just a model).

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
        raise NotImplementedError("Subclasses must implement evaluate")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Train and evaluate linear probes, optionally at specified epochs."""
        if self._skip(trainer):
            return

        self.device = pl_module.device
        self.evaluate(pl_module, trainer)
