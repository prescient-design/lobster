from typing import Literal

from torchmetrics import (
    AUROC,
    F1Score,
    MeanAbsoluteError,
    R2Score,
    SpearmanCorrCoef,
)


def create_regression_metrics(stage: Literal["train", "val", "test"]) -> dict[str, any]:
    """Create metrics for regression tasks.

    Parameters
    ----------
    stage : Literal["train", "val", "test"]
        Training stage for which to create metrics

    Returns
    -------
    dict[str, any]
        Dictionary of metric objects with keys: r2, mae, spearman
    """
    return {
        "r2": R2Score(),
        "mae": MeanAbsoluteError(),
        "spearman": SpearmanCorrCoef(),
    }


def create_classification_metrics(
    stage: Literal["train", "val", "test"],
    num_labels: int,
    metric_average: str = "weighted",
) -> dict[str, any]:
    """Create metrics for classification tasks.

    Parameters
    ----------
    stage : Literal["train", "val", "test"]
        Training stage for which to create metrics
    num_labels : int
        Number of output labels (2 for binary, 2+ for multiclass)
    metric_average : str, default="weighted"
        Averaging strategy for metrics

    Returns
    -------
    dict[str, any]
        Dictionary of metric objects with keys: f1, auroc
    """
    task_type = "binary" if num_labels == 2 else "multiclass"
    num_classes = num_labels if num_labels > 2 else 2

    return {
        "f1": F1Score(
            task=task_type,
            num_classes=num_classes,
            average=metric_average,
        ),
        "auroc": AUROC(
            task=task_type,
            num_classes=num_classes,
            average=metric_average,
        ),
    }


def create_all_metrics(
    task: Literal["regression", "classification"],
    num_labels: int,
    metric_average: str = "weighted",
) -> dict[str, dict[str, any]]:
    """Create all metrics for train/val/test stages.

    Parameters
    ----------
    task : Literal["regression", "classification"]
        Type of prediction task
    num_labels : int
        Number of output labels
    metric_average : str, default="weighted"
        Averaging strategy for classification metrics

    Returns
    -------
    dict[str, dict[str, any]]
        Nested dictionary with structure: {stage: {metric_name: metric_object}}
    """
    metrics = {}

    for stage in ["train", "val", "test"]:
        if task == "regression":
            metrics[stage] = create_regression_metrics(stage)
        else:
            metrics[stage] = create_classification_metrics(stage, num_labels, metric_average)

    return metrics
