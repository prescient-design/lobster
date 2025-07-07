"""DGEB evaluation runner for UME models."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import dgeb
from dgeb.modality import Modality as DGEBModality

from .dgeb_adapter import UMEAdapterDGEB

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dgeb_evaluation.log"),
        ],
    )


def get_tasks_for_modality(modality: Literal["protein", "dna"]) -> list[type]:
    """Get all tasks for a specific modality."""
    if modality == "protein":
        dgeb_modality = DGEBModality.PROTEIN
    elif modality == "dna":
        dgeb_modality = DGEBModality.DNA
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return dgeb.get_tasks_by_modality(dgeb_modality)


def run_evaluation(
    model_name: str,
    modality: Literal["protein", "dna"],
    tasks: list[str] | None = None,
    output_dir: str = "dgeb_results",
    batch_size: int = 32,
    max_seq_length: int = 1024,
    use_flash_attn: bool | None = None,
    l2_norm: bool = False,
    pool_type: str = "mean",
    devices: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run DGEB evaluation on a UME model.

    Parameters
    ----------
    model_name : str
        Name or path to the UME model.
    modality : Literal["protein", "dna"]
        Biological modality for evaluation.
    tasks : list[str] | None, default=None
        List of specific tasks to run. If None, runs all tasks for the modality.
    output_dir : str, default="dgeb_results"
        Directory to save evaluation results.
    batch_size : int, default=32
        Batch size for encoding.
    max_seq_length : int, default=1024
        Maximum sequence length.
    use_flash_attn : bool | None, default=None
        Whether to use flash attention.
    l2_norm : bool, default=False
        Whether to L2-normalize embeddings.
    pool_type : str, default="mean"
        Pooling strategy.
    devices : list[int] | None, default=None
        Device IDs for inference. If None, uses [0].
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation results and metadata.
    """
    logger.info(f"Starting DGEB evaluation for {model_name} on {modality} sequences")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get tasks to run
    if tasks is None:
        task_classes = get_tasks_for_modality(modality)
        task_names = [task.metadata.display_name for task in task_classes]
        logger.info(f"Running all {len(task_classes)} tasks for {modality}: {task_names}")
    else:
        task_classes = dgeb.get_tasks_by_name(tasks)
        task_names = tasks
        logger.info(f"Running {len(task_classes)} specified tasks: {task_names}")

    # Initialize the UME adapter
    logger.info("Initializing UME adapter...")
    if devices is None:
        devices = [0]

    try:
        logger.info("Creating UMEAdapterDGEB instance...")
        model = UMEAdapterDGEB(
            model_name=model_name,
            modality=modality,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            use_flash_attn=use_flash_attn,
            l2_norm=l2_norm,
            pool_type=pool_type,
            devices=devices,
        )
        logger.info("UMEAdapterDGEB instance created successfully")
        logger.info(f"Successfully initialized UME adapter with embedding dim: {model.embed_dim}")
    except Exception as e:
        logger.error(f"Failed to initialize UME adapter: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    # Initialize DGEB evaluation
    logger.info("Initializing DGEB evaluation...")
    evaluation = dgeb.DGEB(tasks=task_classes, seed=seed)

    # Run evaluation
    logger.info("Starting evaluation...")
    start_time = datetime.now()

    try:
        results = evaluation.run(model, output_folder=str(output_path))
        end_time = datetime.now()

        logger.info(f"Evaluation completed in {end_time - start_time}")
        logger.info(f"Results saved to {output_path}")

        # Compile results summary
        results_summary = {
            "model_name": model_name,
            "modality": modality,
            "tasks_run": task_names,
            "total_tasks": len(task_classes),
            "evaluation_time": str(end_time - start_time),
            "timestamp": datetime.now().isoformat(),
            "model_metadata": model.metadata,
            "results": [],
        }

        # Extract key metrics from results
        for result in results:
            task_summary = {
                "task_name": getattr(result.task, "display_name", "Unknown Task"),
                "task_type": getattr(result.task, "type", "Unknown Type"),
                "scores": {},
            }

            # Extract scores from each layer result
            for layer_result in result.results:
                layer_name = f"layer_{layer_result.layer_number}"
                # Convert list of TaskMetric objects to dictionary
                metrics_dict = {}
                for metric in layer_result.metrics:
                    metrics_dict[metric.id] = metric.value
                task_summary["scores"][layer_name] = metrics_dict

            results_summary["results"].append(task_summary)

        return results_summary

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def generate_report(results_summary: dict[str, Any], report_dir: Path) -> None:
    """Generate a human-readable evaluation report.

    Parameters
    ----------
    results_summary : dict[str, Any]
        Results summary from run_evaluation.
    report_dir : Path
        Directory to save the report.
    """
    report_path = report_dir / "evaluation_report.md"

    with open(report_path, "w") as f:
        f.write("# DGEB Evaluation Report\n\n")
        f.write(f"**Model:** {results_summary['model_name']}\n")
        f.write(f"**Modality:** {results_summary['modality']}\n")
        f.write(f"**Evaluation Date:** {results_summary['timestamp']}\n")
        f.write(f"**Total Tasks:** {results_summary['total_tasks']}\n")
        f.write(f"**Evaluation Time:** {results_summary['evaluation_time']}\n\n")

        f.write("## Model Configuration\n\n")
        for key, value in results_summary["model_metadata"].items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")

        f.write("## Results Summary\n\n")
        f.write("| Task | Type | Primary Metric | Score |\n")
        f.write("|------|------|----------------|-------|\n")

        for task_result in results_summary["results"]:
            task_name = task_result["task_name"]
            task_type = task_result["task_type"]

            # Get the primary metric (usually from layer_0)
            if "layer_0" in task_result["scores"]:
                scores = task_result["scores"]["layer_0"]
                # Try to get the main metric (this varies by task type)
                if "main_score" in scores:
                    primary_score = scores["main_score"]
                elif "accuracy" in scores:
                    primary_score = scores["accuracy"]
                elif "f1" in scores:
                    primary_score = scores["f1"]
                elif "pearson" in scores:
                    primary_score = scores["pearson"]
                elif "spearman" in scores:
                    primary_score = scores["spearman"]
                else:
                    # Use the first available score
                    primary_score = next(iter(scores.values())) if scores else "N/A"
            else:
                primary_score = "N/A"

            # Format score
            if isinstance(primary_score, (int, float)):
                primary_score = f"{primary_score:.4f}"

            f.write(f"| {task_name} | {task_type} | Primary | {primary_score} |\n")

        f.write("\n## Detailed Results\n\n")
        for task_result in results_summary["results"]:
            f.write(f"### {task_result['task_name']}\n\n")
            f.write(f"**Type:** {task_result['task_type']}\n\n")

            for layer_name, scores in task_result["scores"].items():
                f.write(f"**{layer_name}:**\n")
                for metric, value in scores.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.4f}\n")
                    else:
                        f.write(f"- {metric}: {value}\n")
                f.write("\n")

    logger.info(f"Report saved to {report_path}")


def make_json_serializable(obj):
    """Recursively convert TaskMetric and other non-serializable objects to serializable types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "model_dump"):  # Pydantic v2
        return make_json_serializable(obj.model_dump())
    elif hasattr(obj, "dict"):  # Pydantic v1
        return make_json_serializable(obj.dict())
    elif hasattr(obj, "value"):
        return make_json_serializable(obj.value)
    else:
        return obj


def main():
    """Main entry point for DGEB evaluation."""
    parser = argparse.ArgumentParser(description="Run DGEB evaluation on UME models")

    parser.add_argument(
        "model_name",
        type=str,
        help="Name or path to the UME model (e.g., 'ume-mini-base-12M' or '/path/to/checkpoint.ckpt')",
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["protein", "dna"],
        default="protein",
        help="Biological modality for evaluation",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Specific tasks to run (if not specified, runs all tasks for the modality)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dgeb_results",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use flash attention (if available)",
    )

    parser.add_argument(
        "--l2-norm",
        action="store_true",
        help="L2-normalize embeddings",
    )

    parser.add_argument(
        "--pool-type",
        type=str,
        default="mean",
        choices=["mean", "max", "cls", "last"],
        help="Pooling strategy",
    )

    parser.add_argument(
        "--devices",
        type=int,
        nargs="*",
        default=None,
        help="Device IDs for inference",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Handle flash attention flag
    use_flash_attn = args.use_flash_attn if args.use_flash_attn else None

    # Run evaluation
    try:
        results_summary = run_evaluation(
            model_name=args.model_name,
            modality=args.modality,
            tasks=args.tasks,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            use_flash_attn=use_flash_attn,
            l2_norm=args.l2_norm,
            pool_type=args.pool_type,
            devices=args.devices,
            seed=args.seed,
        )

        # Create output directory and report subdirectory
        output_path = Path(args.output_dir)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = output_path / f"report_{timestamp_str}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save results summary as JSON
        with open(report_dir / "results_summary.json", "w") as f:
            json.dump(make_json_serializable(results_summary), f, indent=2)

        # Generate report
        generate_report(results_summary, report_dir)

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Summary: {report_dir / 'results_summary.json'}")
        logger.info(f"Report: {report_dir / 'evaluation_report.md'}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
