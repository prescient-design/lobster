#!/usr/bin/env python
import inspect
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import DataLoader
from upath import UPath


def _format_metrics_for_markdown(metrics: dict[str, dict[str, float]]) -> str:
    """Format metrics dictionary into a markdown table.

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]
        Dictionary of task -> metric -> value

    Returns
    -------
    str
        Formatted markdown table
    """
    markdown = ""

    for task, task_metrics in metrics.items():
        markdown += f"### {task}\n\n"
        markdown += "| Metric | Value |\n"
        markdown += "|--------|-------|\n"

        for metric_name, value in task_metrics.items():
            markdown += f"| {metric_name} | {value:.4f} |\n"

        markdown += "\n"

    return markdown


def _generate_evaluation_report(
    results: dict[str, Any],
    issues: list[str],
    output_dir: str | Path,
) -> Path:
    """Generate a markdown report from evaluation results.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary of callback name -> evaluation results, which can be metrics dictionary
        or paths to generated files
    issues : list[str]
        List of issues encountered during evaluation
    output_dir : str | Path
        Directory to save the report

    Returns
    -------
    Path
        Path to the generated report
    """
    markdown_report = "# Model Evaluation Report\n\n"
    markdown_report += f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add section for each callback's results
    markdown_report += "## Evaluation Results\n\n"

    for callback_name, callback_results in results.items():
        markdown_report += f"### {callback_name}\n\n"

        # If result is a path, assume it's an image and include it in the markdown
        if isinstance(callback_results, str | Path | UPath) and Path(callback_results).exists():
            markdown_report += f"![{callback_name} Visualization]({callback_results})\n\n"

        # If result is a dictionary, format as table
        elif isinstance(callback_results, dict):
            # Nested dictionary
            if all(isinstance(v, dict) for v in callback_results.values()):
                markdown_report += _format_metrics_for_markdown({callback_name: callback_results})
            else:
                # Simple dictionary, format as table
                markdown_report += "| Metric | Value |\n"
                markdown_report += "|--------|-------|\n"
                for metric_name, value in callback_results.items():
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    markdown_report += f"| {metric_name} | {value_str} |\n"
                markdown_report += "\n"
        else:
            # Other types of results
            markdown_report += f"{callback_results}\n\n"

    # Add section for issues if any
    if issues:
        markdown_report += "## Issues Encountered\n\n"
        for issue in issues:
            markdown_report += f"- {issue}\n"
        markdown_report += "\n"

    report_path = output_dir / "evaluation_report.md"

    with open(report_path, "w") as f:
        f.write(markdown_report)

    return report_path


def evaluate_model_with_callbacks(
    callbacks: Sequence[L.Callback],
    model: L.LightningModule,
    dataloader: DataLoader | None = None,
    output_dir: str | Path | UPath = "evaluation_results",
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
    """
    if str(output_dir).startswith("s3://"):
        raise NotImplementedError("S3 output is not supported yet")

    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    issues = []

    for callback in callbacks:
        try:
            callback_name = callback.__class__.__name__

            if not hasattr(callback, "evaluate") or not callable(callback.evaluate):
                raise ValueError(f"Callback {callback_name} does not have an evaluate method")

            # Inspect signature if the callback evaluation method requires a dataloader
            callback_signature = inspect.signature(callback.evaluate)

            if "dataloader" in callback_signature.parameters:
                if dataloader is None:
                    raise ValueError(f"Callback {callback_name} requires a dataloader but none was provided")

                callback_results = callback.evaluate(model, dataloader=dataloader)
            else:
                callback_results = callback.evaluate(model)

            results[callback_name] = callback_results

        except Exception as e:
            print(f"Error in {callback_name}: {e}")
            issues.append(f"{callback_name}: {str(e)}")

    # Generate markdown report
    return _generate_evaluation_report(results, issues, output_dir)
