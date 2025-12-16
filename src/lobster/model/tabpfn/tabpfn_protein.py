from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from .._utils_metrics import create_all_metrics
from ._embedding_utils import (
    extract_embeddings,
    get_embedding_dim,
    initialize_embedding_model,
)
from ._ensembling_utils import (
    create_feature_subsets,
    create_sample_chunks,
    fit_ensemble_models,
    predict_with_ensemble,
    validate_limits,
)


class TabPFNProteinModel(pl.LightningModule):
    """Hybrid model combining protein embeddings with TabPFN v2.

    This model uses a protein language model to generate embeddings from
    protein sequences, then uses TabPFN v2 (commercially open) to make
    predictions on these embeddings as tabular features.

    Large Dataset Handling
    ----------------------
    TabPFN v2 has pretraining limits of 500 features and 50K samples. When these
    limits are exceeded, this model automatically uses an ensembling strategy:

    - **Feature Ensembling**: If embeddings have >500 features, multiple TabPFN models
      are created, each trained on a random subset of 500 features. Predictions are
      averaged across all feature subsets. Number of models scales dynamically:
      ceil(n_features / 500).

    - **Sample Ensembling**: If training data has >50K samples, data is split into
      sequential chunks of 50K. Each chunk gets its own TabPFN model. Predictions
      are averaged across all models.

    - **Combined Ensembling**: When both limits are exceeded, we create models for
      each (sample_chunk, feature_subset) combination, resulting in
      n_sample_chunks × n_feature_models total models.

    This ensembling can be disabled by setting `enable_large_data_ensembling=False`,
    in which case an error is raised when limits are exceeded (unless
    `tabpfn_ignore_pretraining_limits=True`).

    References
    ----------
    ```bibtex
    @article{hollmann2025tabpfn,
    title={Accurate predictions on small data with a tabular foundation model},
    author={Hollmann, Noah and M{\"u}ller, Samuel and others},
    journal={Nature},
    year={2025},
    doi={10.1038/s41586-024-08328-6}
    }
    ```

    Parameters
    ----------
    task : Literal["regression", "classification"], default="regression"
        Type of prediction task for TabPFN
    num_labels : int, default=1
        Number of output labels (1 for regression, 2 for binary, 2+ for multiclass)
    num_chains : int, default=1
        Number of protein chains in input data
    embedding_model_type : Literal["LobsterPMLM", "LobsterPCLM", "UME"], default="LobsterPMLM"
        Type of protein embedding model to use
    embedding_model_name : str | None, default=None
        Specific pretrained model name for embeddings (e.g., 'esm2_t33_650M_UR50D')
    embedding_checkpoint : str | None, default=None
        Path to checkpoint for embedding model
    pooling : Literal["mean", "max", "cls"], default="mean"
        How to pool per-residue embeddings to sequence-level
    max_length : int, default=512
        Maximum sequence length for embedding model
    tabpfn_n_ensemble : int, default=4
        Number of models in TabPFN ensemble
    tabpfn_ignore_pretraining_limits : bool, default=True
        Whether to ignore TabPFN's pretraining limits (needed for >500 features)
    enable_large_data_ensembling : bool, default=True
        Whether to enable automatic ensembling for large datasets.
        If True and n_features > 500: creates multiple models with random 500-feature subsets
        If True and n_samples > 50K: splits data into sequential 50K chunks
        If False: raises error when limits exceeded (unless ignore_pretraining_limits=True)
    metric_average : str, default="weighted"
        Averaging strategy for classification metrics
    additional_features : list[str] | None, default=None
        Additional feature names to concatenate with embeddings

    Attributes
    ----------
    embedding_model : pl.LightningModule
        The protein embedding model
    tabpfn_models : list[TabPFNRegressor | TabPFNClassifier]
        List of TabPFN models for predictions (ensemble for large datasets)
    is_fitted : bool
        Whether TabPFN has been fitted

    Examples
    --------
    >>> model = TabPFNProteinModel(
    ...     task="regression",
    ...     embedding_model_name="esm2_t33_650M_UR50D"
    ... )
    >>> trainer = pl.Trainer(max_epochs=10)
    >>> trainer.fit(model, train_dataloader)
    >>> predictions = trainer.predict(model, test_dataloader)
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        num_labels: int = 1,
        num_chains: int = 1,
        embedding_model_type: Literal[
            "LobsterPMLM",
            "LobsterPCLM",
            "LobsterConditionalPMLM",
            "LobsterConditionalClassifierPMLM",
            "LobsterCBMPMLM",
            "UME",
        ] = "LobsterPMLM",
        embedding_model_name: str | None = None,
        embedding_checkpoint: str | None = None,
        pooling: Literal["mean", "max", "cls"] = "mean",
        max_length: int = 512,
        tabpfn_n_ensemble: int = 4,
        tabpfn_ignore_pretraining_limits: bool = True,
        enable_large_data_ensembling: bool = True,
        metric_average: str = "weighted",
        additional_features: list[str] | None = None,
        ckpt_path: str | None = None,  # unused
    ):
        super().__init__()
        self.save_hyperparameters()

        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:
            raise ImportError("TabPFN is not installed. Install with: uv sync --extra tabpfn") from e

        self._task = task
        self._num_labels = num_labels
        self._num_chains = num_chains
        self._pooling = pooling
        self._tabpfn_n_ensemble = tabpfn_n_ensemble
        self._tabpfn_ignore_pretraining_limits = tabpfn_ignore_pretraining_limits
        self._enable_large_data_ensembling = enable_large_data_ensembling
        self._additional_features = additional_features or []

        self.embedding_model = initialize_embedding_model(
            embedding_model_type,
            embedding_model_name,
            embedding_checkpoint,
            max_length,
        )

        for param in self.embedding_model.parameters():
            param.requires_grad = False
        self.embedding_model.eval()

        self._embedding_dim = get_embedding_dim(self.embedding_model, self._num_chains)

        self._tabpfn_class = TabPFNRegressor if task == "regression" else TabPFNClassifier

        self.tabpfn_models = []
        self._feature_subsets = []
        self._sample_chunks = []
        self._needs_ensembling = False

        self.is_fitted = False
        self._training_embeddings = []
        self._training_labels = []

        self._setup_metrics(task, num_labels, metric_average)

    def _setup_metrics(
        self,
        task: Literal["regression", "classification"],
        num_labels: int,
        metric_average: str,
    ):
        """Setup metrics for all stages (train/val/test)."""
        all_metrics = create_all_metrics(task, num_labels, metric_average)

        for stage, metrics in all_metrics.items():
            for metric_name, metric_obj in metrics.items():
                setattr(self, f"{stage}_{metric_name}", metric_obj)

    def training_step(self, batch, batch_idx):
        """Collect embeddings during training to fit TabPFN at epoch end.

        Parameters
        ----------
        batch : dict
            Training batch
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Dummy loss (actual training happens at epoch end)
        """
        embeddings = extract_embeddings(
            self.embedding_model,
            batch,
            self._pooling,
            freeze_embeddings=True,
            additional_features=self._additional_features,
        )
        labels = batch["labels"]

        self._training_embeddings.append(embeddings.detach().cpu().numpy())
        self._training_labels.append(labels.detach().cpu().numpy())

        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def on_train_epoch_end(self):
        """Fit TabPFN on collected embeddings at the end of each epoch."""
        if len(self._training_embeddings) == 0:
            return

        X_train = np.vstack(self._training_embeddings)
        y_train = np.concatenate(self._training_labels)
        n_samples, n_features = X_train.shape

        print(f"Fitting TabPFN on {n_samples} samples with {n_features} features...")

        self._needs_ensembling = validate_limits(
            n_samples,
            n_features,
            self._enable_large_data_ensembling,
            self._tabpfn_ignore_pretraining_limits,
        )

        self.tabpfn_models.clear()
        self._feature_subsets.clear()
        self._sample_chunks.clear()

        if not self._needs_ensembling:
            self._fit_single_model(X_train, y_train, n_samples, n_features)
        else:
            self._fit_ensemble(X_train, y_train, n_samples, n_features)

        self.is_fitted = True
        self._training_embeddings.clear()
        self._training_labels.clear()
        print(f"TabPFN fitting complete. Using {len(self.tabpfn_models)} model(s).")

    def _fit_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        n_features: int,
    ):
        """Fit single TabPFN model when data is within limits."""
        from tabpfn.constants import ModelVersion

        print("Creating single TabPFN model...")
        model = self._tabpfn_class.create_default_for_version(
            ModelVersion.V2,
            n_estimators=self._tabpfn_n_ensemble,
            ignore_pretraining_limits=self._tabpfn_ignore_pretraining_limits,
        )
        model.fit(X_train, y_train)

        self.tabpfn_models.append(model)
        self._feature_subsets.append(np.arange(n_features))
        self._sample_chunks.append((0, n_samples))

    def _fit_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        n_features: int,
    ):
        """Fit ensemble of TabPFN models for large datasets."""
        feature_subsets = create_feature_subsets(n_features)
        sample_chunks = create_sample_chunks(n_samples)

        models = fit_ensemble_models(
            X_train,
            y_train,
            self._tabpfn_class,
            self._tabpfn_n_ensemble,
            self._tabpfn_ignore_pretraining_limits,
            feature_subsets,
            sample_chunks,
        )

        self.tabpfn_models = models

        for _ in sample_chunks:
            for feature_indices in feature_subsets:
                self._feature_subsets.append(feature_indices)
                self._sample_chunks.append((0, n_samples))

    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Make predictions using TabPFN model(s)."""
        return predict_with_ensemble(
            X,
            self.tabpfn_models,
            self._feature_subsets,
            self._task,
        )

    def validation_step(self, batch, batch_idx):
        """Validation step using fitted TabPFN ensemble.

        Parameters
        ----------
        batch : dict
            Validation batch
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Validation loss
        """
        if not self.is_fitted:
            return torch.tensor(0.0, device=self.device)

        embeddings = extract_embeddings(
            self.embedding_model,
            batch,
            self._pooling,
            freeze_embeddings=True,
            additional_features=self._additional_features,
        )
        labels = batch["labels"]
        X_val = embeddings.detach().cpu().numpy()

        if self._task == "regression":
            loss = self._validation_step_regression(X_val, labels)
        else:
            loss = self._validation_step_classification(X_val, labels)

        return loss

    def _validation_step_regression(self, X_val: np.ndarray, labels: torch.Tensor) -> torch.Tensor:
        """Handle regression validation step."""
        preds, _ = self._predict(X_val)
        preds_tensor = torch.from_numpy(preds).to(self.device)
        loss = nn.functional.mse_loss(preds_tensor, labels.float())

        self.val_r2.update(preds_tensor, labels)
        self.val_mae.update(preds_tensor, labels)
        self.val_spearman.update(preds_tensor, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_r2", self.val_r2, prog_bar=True)
        self.log("val_mae", self.val_mae)
        self.log("val_spearman", self.val_spearman)

        return loss

    def _validation_step_classification(self, X_val: np.ndarray, labels: torch.Tensor) -> torch.Tensor:
        """Handle classification validation step."""
        preds, preds_proba = self._predict(X_val)
        preds_tensor = torch.from_numpy(preds_proba).to(self.device)

        if self._num_labels == 2:
            loss = nn.functional.binary_cross_entropy(preds_tensor[:, 1], labels.float())
            self.val_auroc.update(preds_tensor[:, 1], labels.long())
        else:
            loss = nn.functional.cross_entropy(preds_tensor, labels.long())
            self.val_auroc.update(preds_tensor, labels.long())

        preds_class = torch.from_numpy(preds).to(self.device)
        self.val_f1.update(preds_class, labels.long())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_auroc", self.val_auroc)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step using fitted TabPFN ensemble.

        Parameters
        ----------
        batch : dict
            Test batch
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Test loss
        """
        if not self.is_fitted:
            return torch.tensor(0.0, device=self.device)

        embeddings = extract_embeddings(
            self.embedding_model,
            batch,
            self._pooling,
            freeze_embeddings=True,
            additional_features=self._additional_features,
        )
        labels = batch["labels"]
        X_test = embeddings.detach().cpu().numpy()

        if self._task == "regression":
            loss = self._test_step_regression(X_test, labels)
        else:
            loss = self._test_step_classification(X_test, labels)

        return loss

    def _test_step_regression(self, X_test: np.ndarray, labels: torch.Tensor) -> torch.Tensor:
        """Handle regression test step."""
        preds, _ = self._predict(X_test)
        preds_tensor = torch.from_numpy(preds).to(self.device)
        loss = nn.functional.mse_loss(preds_tensor, labels.float())

        self.test_r2.update(preds_tensor, labels)
        self.test_mae.update(preds_tensor, labels)
        self.test_spearman.update(preds_tensor, labels)

        self.log("test_loss", loss)
        self.log("test_r2", self.test_r2)
        self.log("test_mae", self.test_mae)
        self.log("test_spearman", self.test_spearman)

        return loss

    def _test_step_classification(self, X_test: np.ndarray, labels: torch.Tensor) -> torch.Tensor:
        """Handle classification test step."""
        preds, preds_proba = self._predict(X_test)
        preds_tensor = torch.from_numpy(preds_proba).to(self.device)

        if self._num_labels == 2:
            loss = nn.functional.binary_cross_entropy(preds_tensor[:, 1], labels.float())
            self.test_auroc.update(preds_tensor[:, 1], labels.long())
        else:
            loss = nn.functional.cross_entropy(preds_tensor, labels.long())
            self.test_auroc.update(preds_tensor, labels.long())

        preds_class = torch.from_numpy(preds).to(self.device)
        self.test_f1.update(preds_class, labels.long())

        self.log("test_loss", loss)
        self.log("test_f1", self.test_f1)
        self.log("test_auroc", self.test_auroc)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step using fitted TabPFN ensemble.

        Parameters
        ----------
        batch : dict
            Prediction batch
        batch_idx : int
            Batch index

        Returns
        -------
        dict
            Dictionary containing predictions and optionally probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        embeddings = extract_embeddings(
            self.embedding_model,
            batch,
            self._pooling,
            freeze_embeddings=True,
            additional_features=self._additional_features,
        )
        X = embeddings.detach().cpu().numpy()

        preds, preds_proba = self._predict(X)

        if self._task == "regression":
            return {"predictions": preds}
        else:
            return {
                "predictions": preds,
                "probabilities": preds_proba,
            }

    def configure_optimizers(self):
        """Configure optimizers.

        Returns
        -------
        None
            No optimizer needed as embeddings are frozen and TabPFN is fitted via sklearn API
        """
        return None

    def forward(self, batch):
        """Forward pass through embedding model and TabPFN ensemble.

        Parameters
        ----------
        batch : dict
            Input batch

        Returns
        -------
        torch.Tensor or dict
            Predictions from TabPFN ensemble
        """
        embeddings = extract_embeddings(
            self.embedding_model,
            batch,
            self._pooling,
            freeze_embeddings=True,
            additional_features=self._additional_features,
        )

        if not self.is_fitted:
            return embeddings

        X = embeddings.detach().cpu().numpy()
        preds, preds_proba = self._predict(X)

        if self._task == "regression":
            return torch.from_numpy(preds).to(self.device)
        else:
            return torch.from_numpy(preds_proba).to(self.device)
