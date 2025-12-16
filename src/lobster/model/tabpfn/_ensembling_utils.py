from typing import Literal

import numpy as np

from lobster.constants import MAX_FEATURES_PER_MODEL, MAX_SAMPLES_PER_MODEL


def check_pretraining_limits(
    n_samples: int,
    n_features: int,
) -> tuple[bool, bool]:
    """Check if data exceeds TabPFN pretraining limits.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features

    Returns
    -------
    tuple[bool, bool]
        (exceeds_feature_limit, exceeds_sample_limit)
    """
    exceeds_feature_limit = n_features > MAX_FEATURES_PER_MODEL
    exceeds_sample_limit = n_samples > MAX_SAMPLES_PER_MODEL
    return exceeds_feature_limit, exceeds_sample_limit


def validate_limits(
    n_samples: int,
    n_features: int,
    enable_large_data_ensembling: bool,
    ignore_pretraining_limits: bool,
) -> bool:
    """Validate data size against TabPFN limits and return whether ensembling is needed.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    enable_large_data_ensembling : bool
        Whether ensembling is enabled
    ignore_pretraining_limits : bool
        Whether to ignore limits

    Returns
    -------
    bool
        True if ensembling is needed, False otherwise

    Raises
    ------
    ValueError
        If data exceeds limits and ensembling is disabled
    """
    exceeds_feature_limit, exceeds_sample_limit = check_pretraining_limits(n_samples, n_features)

    if not (exceeds_feature_limit or exceeds_sample_limit):
        return False

    if not enable_large_data_ensembling:
        if not ignore_pretraining_limits:
            error_msg = (
                f"Dataset exceeds TabPFN pretraining limits:\n"
                f"  - Samples: {n_samples} (max: {MAX_SAMPLES_PER_MODEL})\n"
                f"  - Features: {n_features} (max: {MAX_FEATURES_PER_MODEL})\n"
                f"Set enable_large_data_ensembling=True to use automatic ensembling, "
                f"or set tabpfn_ignore_pretraining_limits=True to bypass this check."
            )
            raise ValueError(error_msg)
        else:
            print("Warning: Dataset exceeds limits but proceeding due to ignore_pretraining_limits=True")
            return False

    print("Dataset exceeds limits. Using ensembling strategy:")
    if exceeds_feature_limit:
        print(f"  - Feature ensembling: {n_features} features > {MAX_FEATURES_PER_MODEL}")
    if exceeds_sample_limit:
        print(f"  - Sample ensembling: {n_samples} samples > {MAX_SAMPLES_PER_MODEL}")

    return True


def create_feature_subsets(n_features: int, seed: int = 42) -> list[np.ndarray]:
    """Create random feature subsets for feature ensembling.

    Parameters
    ----------
    n_features : int
        Total number of features
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    list[np.ndarray]
        List of feature index arrays, one per model
    """
    n_feature_models = int(np.ceil(n_features / MAX_FEATURES_PER_MODEL))
    rng = np.random.RandomState(seed)

    feature_subsets = []
    for i in range(n_feature_models):
        n_features_for_model = min(MAX_FEATURES_PER_MODEL, n_features)
        feature_indices = rng.choice(n_features, size=n_features_for_model, replace=False)
        feature_subsets.append(feature_indices)
        print(f"  Feature subset {i + 1}/{n_feature_models}: {len(feature_indices)} features")

    return feature_subsets


def create_sample_chunks(n_samples: int) -> list[tuple[int, int]]:
    """Create sequential sample chunks for sample ensembling.

    Parameters
    ----------
    n_samples : int
        Total number of samples

    Returns
    -------
    list[tuple[int, int]]
        List of (start_idx, end_idx) tuples for each chunk
    """
    n_sample_chunks = int(np.ceil(n_samples / MAX_SAMPLES_PER_MODEL))

    sample_chunks = []
    for chunk_idx in range(n_sample_chunks):
        start_idx = chunk_idx * MAX_SAMPLES_PER_MODEL
        end_idx = min((chunk_idx + 1) * MAX_SAMPLES_PER_MODEL, n_samples)
        sample_chunks.append((start_idx, end_idx))
        print(
            f"  Sample chunk {chunk_idx + 1}/{n_sample_chunks}: samples {start_idx} to {end_idx} ({end_idx - start_idx} samples)"
        )

    return sample_chunks


def fit_ensemble_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tabpfn_class: type,
    n_ensemble: int,
    ignore_pretraining_limits: bool,
    feature_subsets: list[np.ndarray],
    sample_chunks: list[tuple[int, int]],
) -> list:
    """Fit ensemble of TabPFN models on feature/sample subsets.

    Parameters
    ----------
    X_train : np.ndarray
        Training embeddings of shape (n_samples, n_features)
    y_train : np.ndarray
        Training labels
    tabpfn_class : type
        TabPFN model class (Regressor or Classifier)
    n_ensemble : int
        Number of models in TabPFN ensemble
    ignore_pretraining_limits : bool
        Whether to ignore pretraining limits
    feature_subsets : list[np.ndarray]
        List of feature index arrays
    sample_chunks : list[tuple[int, int]]
        List of (start_idx, end_idx) tuples

    Returns
    -------
    list
        List of fitted TabPFN models
    """
    from tabpfn.constants import ModelVersion

    n_feature_models = len(feature_subsets)
    n_sample_chunks = len(sample_chunks)
    total_models = n_feature_models * n_sample_chunks

    print(
        f"Creating ensemble with {n_feature_models} feature subset(s) × {n_sample_chunks} sample chunk(s) = {total_models} model(s)"
    )

    models = []
    model_idx = 0

    for chunk_idx, (start_idx, end_idx) in enumerate(sample_chunks):
        X_chunk = X_train[start_idx:end_idx]
        y_chunk = y_train[start_idx:end_idx]

        for feat_idx, feature_indices in enumerate(feature_subsets):
            model_idx += 1
            print(f"Fitting model {model_idx}/{total_models} (chunk {chunk_idx + 1}, features {feat_idx + 1})...")

            X_subset = X_chunk[:, feature_indices]

            model = tabpfn_class.create_default_for_version(
                ModelVersion.V2,
                n_estimators=n_ensemble,
                ignore_pretraining_limits=ignore_pretraining_limits,
            )
            model.fit(X_subset, y_chunk)
            models.append(model)

    return models


def predict_with_ensemble(
    X: np.ndarray,
    models: list,
    feature_subsets: list[np.ndarray],
    task: Literal["regression", "classification"],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Make predictions using ensemble of TabPFN models.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    models : list
        List of fitted TabPFN models
    feature_subsets : list[np.ndarray]
        List of feature index arrays for each model
    task : Literal["regression", "classification"]
        Type of prediction task

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        - predictions: Mean predictions across ensemble
        - probabilities: Mean probabilities for classification (None for regression)
    """
    all_preds = []
    all_probs = []

    for model, feature_indices in zip(models, feature_subsets):
        X_subset = X[:, feature_indices]
        preds = model.predict(X_subset)
        all_preds.append(preds)

        if task == "classification":
            probs = model.predict_proba(X_subset)
            all_probs.append(probs)

    mean_preds = np.mean(all_preds, axis=0)
    mean_probs = np.mean(all_probs, axis=0) if len(all_probs) > 0 else None

    return mean_preds, mean_probs
