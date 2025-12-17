from torchmetrics import MeanAbsoluteError, R2Score, SpearmanCorrCoef
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, MulticlassAUROC, MulticlassF1Score

from lobster.model._utils_metrics import (
    create_regression_metrics,
    create_classification_metrics,
    create_all_metrics,
)


def test_create_regression_metrics():
    metrics = create_regression_metrics("train")

    assert "r2" in metrics
    assert "mae" in metrics
    assert "spearman" in metrics

    assert isinstance(metrics["r2"], R2Score)
    assert isinstance(metrics["mae"], MeanAbsoluteError)
    assert isinstance(metrics["spearman"], SpearmanCorrCoef)


def test_create_regression_metrics_all_stages():
    for stage in ["train", "val", "test"]:
        metrics = create_regression_metrics(stage)
        assert len(metrics) == 3
        assert all(key in metrics for key in ["r2", "mae", "spearman"])


def test_create_classification_metrics_binary():
    metrics = create_classification_metrics("val", num_labels=2)

    assert "f1" in metrics
    assert "auroc" in metrics

    assert isinstance(metrics["f1"], BinaryF1Score)
    assert isinstance(metrics["auroc"], BinaryAUROC)


def test_create_classification_metrics_multiclass():
    metrics = create_classification_metrics("test", num_labels=5)

    assert "f1" in metrics
    assert "auroc" in metrics

    assert isinstance(metrics["f1"], MulticlassF1Score)
    assert isinstance(metrics["auroc"], MulticlassAUROC)


def test_create_classification_metrics_with_averaging():
    metrics = create_classification_metrics("train", num_labels=3, metric_average="macro")

    assert "f1" in metrics
    assert "auroc" in metrics


def test_create_all_metrics_regression():
    all_metrics = create_all_metrics("regression", num_labels=1)

    assert "train" in all_metrics
    assert "val" in all_metrics
    assert "test" in all_metrics

    for stage in ["train", "val", "test"]:
        assert "r2" in all_metrics[stage]
        assert "mae" in all_metrics[stage]
        assert "spearman" in all_metrics[stage]


def test_create_all_metrics_classification():
    all_metrics = create_all_metrics("classification", num_labels=3)

    assert "train" in all_metrics
    assert "val" in all_metrics
    assert "test" in all_metrics

    for stage in ["train", "val", "test"]:
        assert "f1" in all_metrics[stage]
        assert "auroc" in all_metrics[stage]


def test_create_all_metrics_classification_with_averaging():
    all_metrics = create_all_metrics("classification", num_labels=4, metric_average="macro")

    assert len(all_metrics) == 3
    for stage in ["train", "val", "test"]:
        assert len(all_metrics[stage]) == 2
