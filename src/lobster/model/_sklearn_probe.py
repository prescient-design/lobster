from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from lobster.constants import SklearnProbeTaskType


def _create_and_fit_preprocessors(
    embeddings: np.ndarray, dimensionality_reduction: bool = False, reduced_dim: int = 320, seed: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create preprocessors and apply them to embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings to preprocess
    dimensionality_reduction : bool
        Whether to apply PCA dimensionality reduction
    reduced_dim : int
        Target dimensionality for PCA reduction
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Processed embeddings and dictionary of fitted preprocessors
    """
    preprocessors = {}
    processed_embeddings = embeddings.copy()

    # Create and apply scaler
    scaler = StandardScaler()
    processed_embeddings = scaler.fit_transform(processed_embeddings)
    preprocessors["scaler"] = scaler

    # Apply dimensionality reduction if requested
    if dimensionality_reduction:
        n_samples, n_features = processed_embeddings.shape
        actual_reduced_dim = min(reduced_dim, n_samples - 1, n_features)

        pca = PCA(n_components=actual_reduced_dim, random_state=seed)
        processed_embeddings = pca.fit_transform(processed_embeddings)
        preprocessors["pca"] = pca

    return processed_embeddings, preprocessors


def _apply_preprocessors(embeddings: np.ndarray, preprocessors: dict[str, Any]) -> np.ndarray:
    """Apply existing preprocessors to embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings to preprocess
    preprocessors : dict[str, Any]
        Dictionary of fitted preprocessors

    Returns
    -------
    np.ndarray
        Processed embeddings
    """
    processed_embeddings = embeddings.copy()

    if "scaler" in preprocessors:
        processed_embeddings = preprocessors["scaler"].transform(processed_embeddings)

    if "pca" in preprocessors:
        processed_embeddings = preprocessors["pca"].transform(processed_embeddings)

    return processed_embeddings


def _create_probe(task_type: SklearnProbeTaskType, probe_type: str, seed: int = 0):
    """Create a probe based on task configuration.

    Parameters
    ----------
    task_type : SklearnProbeTaskType
        Type of task (regression, binary, multiclass, multilabel)
    probe_type : str
        Type of probe (linear, elastic, svm, gradient_boosting)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    sklearn estimator
        Configured probe model
    """
    match task_type:
        case "regression":
            match probe_type:
                case "linear":
                    return LinearRegression()
                case "elastic":
                    return ElasticNet(random_state=seed, max_iter=5000)
                case "svm":
                    return SVR(kernel="linear", random_state=seed)
                case "gradient_boosting":
                    return GradientBoostingRegressor(random_state=seed)
                case _:
                    raise ValueError(f"Unknown probe_type for regression: {probe_type}")

        case "multilabel":
            match probe_type:
                case "linear":
                    base_classifier = LogisticRegression(random_state=seed, class_weight="balanced", max_iter=1000)
                case "elastic":
                    base_classifier = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        random_state=seed,
                        max_iter=1000,
                        class_weight="balanced",
                    )
                case "svm":
                    base_classifier = SVC(kernel="linear", probability=True, random_state=seed, class_weight="balanced")
                case "gradient_boosting":
                    base_classifier = GradientBoostingClassifier(random_state=seed)
                case _:
                    raise ValueError(f"Unknown probe_type for multilabel: {probe_type}")
            return MultiOutputClassifier(base_classifier)

        case "binary" | "multiclass":
            multi_class = "ovr" if task_type == "binary" else "multinomial"
            match probe_type:
                case "linear":
                    return LogisticRegression(
                        multi_class=multi_class,
                        random_state=seed,
                        max_iter=1000,
                    )
                case "elastic":
                    return LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        multi_class=multi_class,
                        random_state=seed,
                        max_iter=1000,
                    )
                case "svm":
                    return SVC(kernel="linear", probability=True, random_state=seed)
                case "gradient_boosting":
                    return GradientBoostingClassifier(random_state=seed)
                case _:
                    raise ValueError(f"Unknown probe_type for {task_type}: {probe_type}")

        case _:
            raise ValueError(f"Unknown task_type: {task_type}")


def _get_classification_predictions(probe, embeddings_np: np.ndarray, task_type: SklearnProbeTaskType) -> np.ndarray:
    """Get classification predictions from probe.

    Parameters
    ----------
    probe
        Trained probe model
    embeddings_np : np.ndarray
        Input embeddings
    task_type : SklearnProbeTaskType
        Type of classification task

    Returns
    -------
    np.ndarray
        Prediction probabilities
    """
    if task_type == "multilabel":
        # Ensure targets are integers for multilabel classification
        # All supported multilabel estimators expose predict_proba
        predictions_np = np.stack([est.predict_proba(embeddings_np)[:, 1] for est in probe.estimators_], axis=1)

    else:  # binary or multiclass
        if hasattr(probe, "predict_proba"):
            predictions_np = probe.predict_proba(embeddings_np)
            if task_type == "binary":
                predictions_np = predictions_np[:, 1]
        else:
            # For models without predict_proba (like ElasticNet for classification), use decision_function
            if hasattr(probe, "decision_function"):
                predictions_np = probe.decision_function(embeddings_np)
                # Apply sigmoid for binary classification or softmax for multiclass
                if task_type == "binary":
                    predictions_np = 1 / (1 + np.exp(-predictions_np))  # sigmoid
                else:
                    # For multiclass, apply softmax
                    exp_pred = np.exp(predictions_np)
                    predictions_np = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            else:
                # Fallback to predict (will affect metric accuracy)
                predictions_np = probe.predict(embeddings_np).astype(float)

    return predictions_np


def train_sklearn_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    task_type: SklearnProbeTaskType,
    probe_type: str,
    dimensionality_reduction: bool = False,
    reduced_dim: int = 320,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Train sklearn probe with preprocessing.

    Parameters
    ----------
    x : torch.Tensor
        Input embeddings of shape (n_samples, n_features)
    y : torch.Tensor
        Target values
    task_type : SklearnProbeTaskType
        Type of task (regression, binary, multiclass, multilabel)
    probe_type : str
        Type of probe (linear, elastic, svm, gradient_boosting)
    dimensionality_reduction : bool
        Whether to apply PCA dimensionality reduction
    reduced_dim : int
        Target dimensionality for PCA reduction
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[Any, dict[str, Any]]
        Tuple of (trained_probe, preprocessors_dict)
    """
    # Convert to numpy
    embeddings_np = x.detach().numpy()
    targets_np = y.detach().numpy()

    # Preprocess embeddings (scaling + dimensionality reduction)
    processed_embeddings, preprocessors = _create_and_fit_preprocessors(
        embeddings_np, dimensionality_reduction, reduced_dim, seed
    )

    # Create probe based on task configuration
    probe = _create_probe(task_type, probe_type, seed)

    # Train probe based on task type
    if task_type == "regression":
        probe.fit(processed_embeddings, targets_np)
    elif task_type == "multilabel":
        targets_np = targets_np.astype(int)  # Ensure integer targets
        probe.fit(processed_embeddings, targets_np)
    else:  # binary or multiclass
        probe.fit(processed_embeddings, targets_np.ravel())

    return probe, preprocessors


def predict_with_sklearn_probe(
    x: torch.Tensor, probe: Any, preprocessors: dict[str, Any], task_type: SklearnProbeTaskType
) -> torch.Tensor:
    """Make predictions with trained probe and preprocessors.

    Parameters
    ----------
    x : torch.Tensor
        Input embeddings of shape (n_samples, n_features)
    probe : Any
        Trained sklearn probe
    preprocessors : dict[str, Any]
        Dictionary of fitted preprocessors from training
    task_type : SklearnProbeTaskType
        Type of task (regression, binary, multiclass, multilabel)

    Returns
    -------
    torch.Tensor
        Predictions (probabilities for classification, values for regression)
    """
    # Convert to numpy and apply preprocessing
    embeddings_np = x.detach().numpy()
    processed_embeddings = _apply_preprocessors(embeddings_np, preprocessors)

    # Make predictions based on task type
    if task_type == "regression":
        predictions_np = probe.predict(processed_embeddings)
    else:  # classification tasks
        predictions_np = _get_classification_predictions(probe, processed_embeddings, task_type)

    # Convert back to torch tensor
    return torch.from_numpy(predictions_np).float()
