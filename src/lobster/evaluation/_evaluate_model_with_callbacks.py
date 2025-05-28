#!/usr/bin/env python
import inspect
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import DataLoader
from upath import UPath

logger = logging.getLogger(__name__)


def _generate_evaluation_report(
    results: dict[str, Any],
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Generate a markdown report from evaluation results.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary of callback name -> evaluation results, which can be metrics dictionary
        or paths to generated files
    output_dir : str | Path
        Directory to save the report
    metadata : dict[str, Any] | None
        Arbitrary metadata to include in the report (e.g., config information)

    Returns
    -------
    Path
        Path to the generated report
    """
    markdown_report = "# Model Evaluation Report\n\n"
    markdown_report += f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add metadata section if provided
    if metadata:
        markdown_report += "## Metadata\n\n"
        # Format metadata as a YAML code block
        markdown_report += "```yaml\n"
        for key, value in metadata.items():
            markdown_report += f"{key}: {value}\n"
        markdown_report += "```\n\n"

    # Add section for each callback's results
    markdown_report += "## Evaluation Results\n\n"

    for callback_name, callback_results in results.items():
        markdown_report += f"### {callback_name}\n\n"

        # If result is a path, assume it's an image and include it in the markdown
        if isinstance(callback_results, str | Path | UPath) and Path(callback_results).exists():
            markdown_report += f"![{callback_name} Visualization]({callback_results})\n\n"
        else:
            # For all other types, just dump the raw results
            markdown_report += f"```\n{callback_results}\n```\n\n"

    report_path = output_dir / "evaluation_report.md"

    logger.info(f"Writing evaluation report to {report_path}")

    with open(report_path, "w") as f:
        f.write(markdown_report)

    return report_path


def evaluate_model_with_callbacks(
    callbacks: Sequence[L.Callback],
    model: L.LightningModule,
    dataloader: DataLoader | None = None,
    output_dir: str | Path | UPath = "evaluation_results",
    metadata: dict[str, Any] | None = None,
):
    """Evaluate a model with various callbacks and generate a markdown report.

    Callbacks are expected to have an `evaluate` method that takes a model and optionally a dataloader.
    If the callback evaluation method requires a dataloader, it must be provided.

    Parameters
    ----------
    model : L.LightningModule
        The model to evaluate
    dataloader : DataLoader | None
        The dataloader to use for evaluation, required for some callbacks
    output_dir : str | Path
        Directory to save evaluation results
    metadata : dict[str, Any] | None
        Arbitrary metadata to include in the report (e.g., config information)
    """
    logger.info("Starting model evaluation with callbacks")
    if str(output_dir).startswith("s3://"):
        raise NotImplementedError("S3 output is not supported yet")

    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    results = {}

    for callback in callbacks:
        try:
            callback_name = callback.__class__.__name__
            logger.info(f"Evaluating with callback: {callback_name}")

            if not hasattr(callback, "evaluate") or not callable(callback.evaluate):
                error_msg = f"Callback {callback_name} does not have an evaluate method"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Inspect signature if the callback evaluation method requires a dataloader
            callback_signature = inspect.signature(callback.evaluate)

            if "dataloader" in callback_signature.parameters:
                if dataloader is None:
                    error_msg = f"Callback {callback_name} requires a dataloader but none was provided"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                callback_results = callback.evaluate(model, dataloader=dataloader)
            else:
                callback_results = callback.evaluate(model)

            logger.info(f"Successfully evaluated with {callback_name}")
            logger.info(f"Callback results: {callback_results}")

            results[callback_name] = callback_results

        except Exception as e:
            logger.exception(f"Error in {callback_name}: {e}")

    # Generate markdown report
    logger.info("Generating evaluation report")
    report_path = _generate_evaluation_report(results, output_dir, metadata)

    logger.info(f"Evaluation complete, report saved to {report_path}")

    return report_path
