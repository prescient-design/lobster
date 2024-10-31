from torchmetrics import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
)


def summarize_binary_classification_metrics(preds, labels):
    """
    Summarize binary classification metrics using torchmetrics.

    Args:
    ----
        preds (tensor): Predicted values (logits or probabilities).
        labels (tensor): Ground truth labels (0 or 1).

    Returns:
    -------
        dict: A dictionary containing the summarized metrics.

    """
    # Initialize metric objects
    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary", num_classes=2, average="micro")  # binary classification
    recall = Recall(task="binary", num_classes=2, average="micro")
    f1_score = F1Score(task="binary", num_classes=2, average="micro")
    auroc = AUROC(task="binary", num_classes=1)  # binary classification
    confusion_matrix = ConfusionMatrix(task="binary", num_classes=2)
    matthews_corrcoef = MatthewsCorrCoef(task="binary")

    # Compute metrics
    accuracy.update(preds, labels)
    precision.update(preds, labels)
    recall.update(preds, labels)
    f1_score.update(preds, labels)
    auroc.update(preds, labels)
    confusion_matrix.update(preds, labels)
    matthews_corrcoef.update(preds, labels)

    # Gather metric values into a dictionary
    metrics_summary = {
        "Accuracy": accuracy.compute().item(),
        "Precision": precision.compute().item(),
        "Recall": recall.compute().item(),
        "F1 Score": f1_score.compute().item(),
        "AUROC": auroc.compute().item(),
        "Confusion Matrix": confusion_matrix.compute(),
        "Matthews Correlation Coefficient": matthews_corrcoef.compute().item(),
    }

    # Round all values to 3 decimal places
    for metric, value in metrics_summary.items():
        if metric != "Confusion Matrix":
            metrics_summary[metric] = round(value, 3)

    return metrics_summary
