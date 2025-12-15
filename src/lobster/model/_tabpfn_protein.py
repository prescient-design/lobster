from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import (
    AUROC,
    F1Score,
    MeanAbsoluteError,
    R2Score,
    SpearmanCorrCoef,
)

from ._utils import model_typer


class TabPFNProteinModel(pl.LightningModule):
    """Hybrid model combining protein embeddings with TabPFN v2.

    This model uses a protein language model to generate embeddings from
    protein sequences, then uses TabPFN v2 (commercially open) to make
    predictions on these embeddings as tabular features.

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
    freeze_embeddings : bool, default=True
        Whether to freeze the embedding model parameters
    pooling : Literal["mean", "max", "cls"], default="mean"
        How to pool per-residue embeddings to sequence-level
    max_length : int, default=512
        Maximum sequence length for embedding model
    tabpfn_n_ensemble : int, default=4
        Number of models in TabPFN ensemble
    lr : float, default=1e-3
        Learning rate (used if fine-tuning embeddings)
    metric_average : str, default="weighted"
        Averaging strategy for classification metrics
    additional_features : list[str] | None, default=None
        Additional feature names to concatenate with embeddings

    Attributes
    ----------
    embedding_model : pl.LightningModule
        The protein embedding model
    tabpfn_model : TabPFNRegressor or TabPFNClassifier
        The TabPFN model for predictions
    is_fitted : bool
        Whether TabPFN has been fitted

    Examples
    --------
    >>> model = TabPFNProteinModel(
    ...     task="regression",
    ...     embedding_model_name="esm2_t33_650M_UR50D",
    ...     freeze_embeddings=True
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
        freeze_embeddings: bool = True,
        pooling: Literal["mean", "max", "cls"] = "mean",
        max_length: int = 512,
        tabpfn_n_ensemble: int = 4,
        lr: float = 1e-3,
        metric_average: str = "weighted",
        additional_features: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as e:
            raise ImportError("TabPFN is not installed. Install with: uv sync --extra tabpfn") from e

        self._task = task
        self._num_labels = num_labels
        self._num_chains = num_chains
        self._embedding_model_type = embedding_model_type
        self._embedding_model_name = embedding_model_name
        self._freeze_embeddings = freeze_embeddings
        self._pooling = pooling
        self._max_length = max_length
        self._tabpfn_n_ensemble = tabpfn_n_ensemble
        self._lr = lr
        self._metric_average = metric_average
        self._additional_features = additional_features or []

        model_cls = model_typer[embedding_model_type]
        print(f"Loading embedding model: {model_cls}")

        if embedding_model_name is not None:
            if embedding_model_type == "UME":
                from ._ume import UME

                self.embedding_model = UME.from_pretrained(embedding_model_name)
            else:
                self.embedding_model = model_cls(model_name=embedding_model_name, max_length=max_length)
        elif embedding_checkpoint is not None:
            self.embedding_model = model_cls.load_from_checkpoint(
                embedding_checkpoint,
                max_length=max_length,
            )
        else:
            raise ValueError("Must provide either embedding_model_name or embedding_checkpoint")

        if self._freeze_embeddings:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
            self.embedding_model.eval()

        if hasattr(self.embedding_model, "config"):
            self._hidden_size = self.embedding_model.config.hidden_size
        elif hasattr(self.embedding_model, "embedding_dim"):
            self._hidden_size = self.embedding_model.embedding_dim
        else:
            raise ValueError("Cannot determine hidden size from embedding model")

        self._embedding_dim = self._hidden_size * self._num_chains

        print(f"Initializing TabPFN v2 for {task}")
        if task == "regression":
            self.tabpfn_model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2,
                n_estimators=tabpfn_n_ensemble,
            )
        else:
            self.tabpfn_model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                n_estimators=tabpfn_n_ensemble,
            )

        self.is_fitted = False
        self._training_embeddings = []
        self._training_labels = []

        if self._task == "regression":
            self.train_r2 = R2Score()
            self.val_r2 = R2Score()
            self.test_r2 = R2Score()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            self.test_mae = MeanAbsoluteError()
            self.train_spearman = SpearmanCorrCoef()
            self.val_spearman = SpearmanCorrCoef()
            self.test_spearman = SpearmanCorrCoef()
        else:
            self.train_f1 = F1Score(
                task="binary" if num_labels == 2 else "multiclass",
                num_classes=num_labels if num_labels > 2 else 2,
                average=metric_average,
            )
            self.val_f1 = F1Score(
                task="binary" if num_labels == 2 else "multiclass",
                num_classes=num_labels if num_labels > 2 else 2,
                average=metric_average,
            )
            self.test_f1 = F1Score(
                task="binary" if num_labels == 2 else "multiclass",
                num_classes=num_labels if num_labels > 2 else 2,
                average=metric_average,
            )
            self.train_auroc = AUROC(
                task="binary" if num_labels == 2 else "multiclass",
                num_classes=num_labels if num_labels > 2 else 2,
                average=metric_average,
            )
            self.val_auroc = AUROC(
                task="binary" if num_labels == 2 else "multiclass",
                num_classes=num_labels if num_labels > 2 else 2,
                average=metric_average,
            )
            self.test_auroc = AUROC(
                task="binary" if num_labels == 2 else "multiclass",
                num_classes=num_labels if num_labels > 2 else 2,
                average=metric_average,
            )

    def _extract_embeddings(self, batch: dict) -> torch.Tensor:
        """Extract embeddings from batch using the embedding model.

        Parameters
        ----------
        batch : dict
            Batch dictionary containing input_ids, attention_mask, etc.

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape (batch_size, embedding_dim)
        """
        if self._freeze_embeddings:
            with torch.no_grad():
                if hasattr(self.embedding_model, "model"):
                    outputs = self.embedding_model.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    outputs = self.embedding_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
        else:
            if hasattr(self.embedding_model, "model"):
                outputs = self.embedding_model.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
            else:
                outputs = self.embedding_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )

        hidden_states = outputs["hidden_states"][-1]

        if self._pooling == "mean":
            attention_mask = batch["attention_mask"].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self._pooling == "max":
            embeddings = hidden_states.max(dim=1)[0]
        elif self._pooling == "cls":
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self._pooling}")

        if len(self._additional_features) > 0:
            additional_feats = []
            for feat_name in self._additional_features:
                if feat_name in batch:
                    feat = batch[feat_name]
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(-1)
                    additional_feats.append(feat)
            if additional_feats:
                embeddings = torch.cat([embeddings] + additional_feats, dim=-1)

        return embeddings

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
        embeddings = self._extract_embeddings(batch)
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

        print(f"Fitting TabPFN on {X_train.shape[0]} training samples...")
        self.tabpfn_model.fit(X_train, y_train)
        self.is_fitted = True

        self._training_embeddings.clear()
        self._training_labels.clear()

        print("TabPFN fitting complete")

    def validation_step(self, batch, batch_idx):
        """Validation step using fitted TabPFN.

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

        embeddings = self._extract_embeddings(batch)
        labels = batch["labels"]

        X_val = embeddings.detach().cpu().numpy()

        if self._task == "regression":
            preds = self.tabpfn_model.predict(X_val)
            preds_tensor = torch.from_numpy(preds).to(self.device)
            loss = nn.functional.mse_loss(preds_tensor, labels.float())

            self.val_r2.update(preds_tensor, labels)
            self.val_mae.update(preds_tensor, labels)
            self.val_spearman.update(preds_tensor, labels)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_r2", self.val_r2, prog_bar=True)
            self.log("val_mae", self.val_mae)
            self.log("val_spearman", self.val_spearman)
        else:
            preds_proba = self.tabpfn_model.predict_proba(X_val)
            preds_tensor = torch.from_numpy(preds_proba).to(self.device)

            if self._num_labels == 2:
                loss = nn.functional.binary_cross_entropy(preds_tensor[:, 1], labels.float())
                self.val_auroc.update(preds_tensor[:, 1], labels.long())
            else:
                loss = nn.functional.cross_entropy(preds_tensor, labels.long())
                self.val_auroc.update(preds_tensor, labels.long())

            preds_class = torch.from_numpy(self.tabpfn_model.predict(X_val)).to(self.device)
            self.val_f1.update(preds_class, labels.long())

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_f1", self.val_f1, prog_bar=True)
            self.log("val_auroc", self.val_auroc)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step using fitted TabPFN.

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

        embeddings = self._extract_embeddings(batch)
        labels = batch["labels"]

        X_test = embeddings.detach().cpu().numpy()

        if self._task == "regression":
            preds = self.tabpfn_model.predict(X_test)
            preds_tensor = torch.from_numpy(preds).to(self.device)
            loss = nn.functional.mse_loss(preds_tensor, labels.float())

            self.test_r2.update(preds_tensor, labels)
            self.test_mae.update(preds_tensor, labels)
            self.test_spearman.update(preds_tensor, labels)

            self.log("test_loss", loss)
            self.log("test_r2", self.test_r2)
            self.log("test_mae", self.test_mae)
            self.log("test_spearman", self.test_spearman)
        else:
            preds_proba = self.tabpfn_model.predict_proba(X_test)
            preds_tensor = torch.from_numpy(preds_proba).to(self.device)

            if self._num_labels == 2:
                loss = nn.functional.binary_cross_entropy(preds_tensor[:, 1], labels.float())
                self.test_auroc.update(preds_tensor[:, 1], labels.long())
            else:
                loss = nn.functional.cross_entropy(preds_tensor, labels.long())
                self.test_auroc.update(preds_tensor, labels.long())

            preds_class = torch.from_numpy(self.tabpfn_model.predict(X_test)).to(self.device)
            self.test_f1.update(preds_class, labels.long())

            self.log("test_loss", loss)
            self.log("test_f1", self.test_f1)
            self.log("test_auroc", self.test_auroc)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step using fitted TabPFN.

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

        embeddings = self._extract_embeddings(batch)
        X = embeddings.detach().cpu().numpy()

        if self._task == "regression":
            preds = self.tabpfn_model.predict(X)
            return {"predictions": preds}
        else:
            preds = self.tabpfn_model.predict(X)
            preds_proba = self.tabpfn_model.predict_proba(X)
            return {
                "predictions": preds,
                "probabilities": preds_proba,
            }

    def configure_optimizers(self):
        """Configure optimizers (only needed if fine-tuning embeddings).

        Returns
        -------
        torch.optim.Optimizer
            Optimizer for trainable parameters
        """
        if self._freeze_embeddings:
            return None

        optimizer = torch.optim.AdamW(
            self.embedding_model.parameters(),
            lr=self._lr,
        )
        return optimizer

    def forward(self, batch):
        """Forward pass through embedding model and TabPFN.

        Parameters
        ----------
        batch : dict
            Input batch

        Returns
        -------
        torch.Tensor or dict
            Predictions from TabPFN
        """
        embeddings = self._extract_embeddings(batch)

        if not self.is_fitted:
            return embeddings

        X = embeddings.detach().cpu().numpy()

        if self._task == "regression":
            preds = self.tabpfn_model.predict(X)
            return torch.from_numpy(preds).to(self.device)
        else:
            preds_proba = self.tabpfn_model.predict_proba(X)
            return torch.from_numpy(preds_proba).to(self.device)
