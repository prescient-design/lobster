import pandas as pd
import scipy
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def spearman_rho(scores, targets):
    return scipy.stats.spearmanr(scores, targets)[0]


def evaluate_binary_classifier(
    data: pd.DataFrame,
    predicted_col: str,  # column you are predicting
):

    neg_score_col = f"{predicted_col}_pred_neg_score"
    pos_score_col = f"{predicted_col}_pred_pos_score"
    predicted_col_processed = f"{predicted_col}_processed"
    class_scores = data[[neg_score_col, pos_score_col]].values
    labels = data[predicted_col_processed].values

    top_1 = class_scores.argmax(-1)
    pos_scores = class_scores[:, 1]

    metrics = {
        "accuracy": accuracy_score(labels, top_1),
        "precision": precision_score(labels, top_1),
        "recall": recall_score(labels, top_1),
        "f1": f1_score(labels, top_1),
        "aupr": average_precision_score(labels, pos_scores),
    }
    return metrics


def _evaluate_binary_classifier(
    class_scores,
    labels,
    top_1,
    pos_scores,
):

    metrics = {
        "accuracy": accuracy_score(labels, top_1),
        "precision": precision_score(labels, top_1),
        "recall": recall_score(labels, top_1),
        "f1": f1_score(labels, top_1),
        "aupr": average_precision_score(labels, pos_scores),
    }
    return metrics


def evaluate_regression(
    data: pd.DataFrame,
    predicted_col: str,  # column you are predicting
):
    point_pred_col = f"{predicted_col}_pred"
    point_pred = data[point_pred_col].values
    labels = data[predicted_col].values

    metrics = {
        "mean_absolute_error": mean_absolute_error(labels, point_pred),
        "mean_squared_error": mean_squared_error(labels, point_pred),
        "r2_score": r2_score(labels, point_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(
            labels, data[point_pred_col]
        ),
        "spearman_rho": spearman_rho(scores=point_pred, targets=labels),
    }
    return metrics
