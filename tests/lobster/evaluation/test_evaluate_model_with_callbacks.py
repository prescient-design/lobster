from lobster.callbacks import LinearProbeCallback
from lobster.evaluation import evaluate_model_with_callbacks


class SimpleLinearProbeCallback(LinearProbeCallback):
    """Simple implementation of a linear probe callback for testing.

    Parameters
    ----------
    task_type : str
        Type of task the probe is for
    num_classes : int | None
        Number of classes for classification tasks
    """

    def __init__(self, task_type):
        super().__init__()
        self.task_type = task_type

    def evaluate(self, model, dataloader):  # noqa: ANN001
        """Evaluate the model using a simple probe.

        Parameters
        ----------
        model : L.LightningModule
            The model to evaluate

        Returns
        -------
        dict
            Dictionary of metrics
        """
        # Return dummy metrics
        if self.task_type == "regression":
            return {
                "test_metric_1": 0.75,
                "test_metric_2": 0.85,
            }
        else:
            return {
                "accuracy": 0.82,
                "f1": 0.79,
            }


def test_evaluate_model_with_callbacks(dummy_model, dummy_datamodule, tmp_path):
    """Test evaluating a model with various callbacks."""
    # Create test callbacks
    callbacks = [
        SimpleLinearProbeCallback(task_type="regression"),
        SimpleLinearProbeCallback(task_type="multiclass"),
    ]
    dummy_datamodule.setup("fit")
    dataloader = dummy_datamodule.train_dataloader()

    _, report_path = evaluate_model_with_callbacks(
        callbacks=callbacks,
        model=dummy_model,
        dataloader=dataloader,
        output_dir=tmp_path,
    )

    # Check that report was generated
    assert report_path.exists()

    # Check report content
    with open(report_path) as f:
        report_content = f.read()

    # Verify that callback results are in the report
    assert "# Model Evaluation Report" in report_content
    assert "SimpleLinearProbeCallback" in report_content
