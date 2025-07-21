"""DGEB evaluation runner using mock UME adapter for random baseline evaluation."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import dgeb
from dgeb.modality import Modality as DGEBModality

from .dgeb_mock_adapter import MockUMEAdapterDGEB

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dgeb_mock_evaluation.log"),
        ],
    )


def get_tasks_for_modality(modality: Literal["protein", "dna"]) -> list:
    """Get all tasks for a given modality."""
    if modality == "protein":
        return dgeb.get_tasks_by_modality(DGEBModality.PROTEIN)
    elif modality == "dna":
        return dgeb.get_tasks_by_modality(DGEBModality.DNA)
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format."""
    if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


def run_mock_evaluation(
    model_name: str = "mock-ume-baseline",
    modality: Literal["protein", "dna"] = "protein",
    tasks: list[str] | None = None,
    output_dir: str = "dgeb_mock_results",
    batch_size: int = 32,
    max_seq_length: int = 1024,
    use_flash_attn: bool | None = None,
    l2_norm: bool = False,
    pool_type: str = "mean",
    devices: list[int] | None = None,
    seed: int = 42,
    embed_dim: int = 768,
    num_layers: int = 12,
) -> dict[str, Any]:
    """Run DGEB evaluation using mock UME adapter with random embeddings.

    Parameters
    ----------
    model_name : str, default="mock-ume-baseline"
        Name of the mock model.
    modality : Literal["protein", "dna"], default="protein"
        Biological modality for evaluation.
    tasks : list[str] | None, default=None
        List of specific tasks to run. If None, runs all tasks for the modality.
    output_dir : str, default="dgeb_mock_results"
        Directory to save evaluation results.
    batch_size : int, default=32
        Batch size for encoding (unused in mock implementation).
    max_seq_length : int, default=1024
        Maximum sequence length.
    use_flash_attn : bool | None, default=None
        Whether to use flash attention (unused in mock implementation).
    l2_norm : bool, default=False
        Whether to L2-normalize embeddings.
    pool_type : str, default="mean"
        Pooling strategy.
    devices : list[int] | None, default=None
        Device IDs for inference (unused in mock implementation).
    seed : int, default=42
        Random seed for reproducible random embeddings.
    embed_dim : int, default=768
        Embedding dimension for the random embeddings.
    num_layers : int, default=12
        Number of layers to simulate.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation results and metadata.
    """
    logger.info(f"Starting DGEB mock evaluation for {model_name} on {modality} sequences")

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

    # Initialize the mock UME adapter
    logger.info("Initializing MockUMEAdapterDGEB...")
    try:
        model = MockUMEAdapterDGEB(
            model_name=model_name,
            modality=modality,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            use_flash_attn=use_flash_attn,
            l2_norm=l2_norm,
            pool_type=pool_type,
            devices=devices,
            embed_dim=embed_dim,
            num_layers=num_layers,
            seed=seed,
        )
        logger.info("MockUMEAdapterDGEB instance created successfully")
        logger.info(f"Successfully initialized mock adapter with embedding dim: {model.embed_dim}")
    except Exception as e:
        logger.error(f"Failed to initialize mock UME adapter: {e}")
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


def generate_mock_report(results_summary: dict[str, Any], report_dir: Path) -> None:
    """Generate a human-readable evaluation report for mock results.

    Parameters
    ----------
    results_summary : dict[str, Any]
        Results summary from run_mock_evaluation.
    report_dir : Path
        Directory to save the report.
    """
    report_path = report_dir / "mock_evaluation_report.md"

    with open(report_path, "w") as f:
        f.write("# DGEB Mock Evaluation Report (Random Baseline)\n\n")
        f.write(f"**Model:** {results_summary['model_name']}\n")
        f.write(f"**Modality:** {results_summary['modality']}\n")
        f.write(f"**Evaluation Date:** {results_summary['timestamp']}\n")
        f.write(f"**Total Tasks:** {results_summary['total_tasks']}\n")
        f.write(f"**Evaluation Time:** {results_summary['evaluation_time']}\n")
        f.write("**Note:** This is a **RANDOM BASELINE** evaluation using completely random embeddings.\n\n")

        f.write("## Model Configuration\n\n")
        for key, value in results_summary["model_metadata"].items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")

        f.write("## Results Summary (Random Baseline)\n\n")
        f.write("| Task | Type | Primary Metric | Score |\n")
        f.write("|------|------|----------------|-------|\n")

        # First pass: analyze all layers to find the best performing layer across all tasks
        layer_performance = {}
        for task_result in results_summary["results"]:
            task_name = task_result["task_name"]
            task_type = task_result["task_type"]

            for layer_name, metrics in task_result["scores"].items():
                if layer_name not in layer_performance:
                    layer_performance[layer_name] = {"wins": 0, "total_tasks": 0}

                layer_performance[layer_name]["total_tasks"] += 1

                # Determine primary metric based on task type
                primary_metric = None
                if task_type == "eds":
                    primary_metric = "top_corr"
                elif task_type == "pair_classification":
                    primary_metric = "top_ap"
                elif task_type == "classification":
                    primary_metric = "f1"
                elif task_type == "clustering":
                    primary_metric = "nmi"
                elif task_type == "bigene":
                    primary_metric = "f1"
                else:
                    # Fallback: use the first available metric
                    primary_metric = list(metrics.keys())[0] if metrics else "unknown"

                if primary_metric in metrics:
                    score = metrics[primary_metric]
                    f.write(f"| {task_name} | {task_type} | {primary_metric} | {score:.4f} |\n")

        f.write("\n")
        f.write("## Detailed Results by Task\n\n")

        for task_result in results_summary["results"]:
            task_name = task_result["task_name"]
            task_type = task_result["task_type"]

            f.write(f"### {task_name} ({task_type})\n\n")

            for layer_name, metrics in task_result["scores"].items():
                f.write(f"**{layer_name}:**\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric_name, metric_value in metrics.items():
                    f.write(f"| {metric_name} | {metric_value:.6f} |\n")

                f.write("\n")

        f.write("## Important Notes\n\n")
        f.write("- **This is a random baseline evaluation** - all embeddings are completely random\n")
        f.write("- Results represent the performance expected from random chance\n")
        f.write("- Use these results as a lower bound for model performance comparison\n")
        f.write("- Random seed was set to ensure reproducible results\n")
        f.write("- All tasks use the same random embedding generator\n")


def main():
    """Main entry point for DGEB mock evaluation."""
    parser = argparse.ArgumentParser(description="Run DGEB evaluation using mock UME adapter with random embeddings")

    parser.add_argument(
        "--model-name",
        type=str,
        default="mock-ume-baseline",
        help="Name of the mock model (default: mock-ume-baseline)",
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
        default="dgeb_mock_results",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (unused in mock implementation)",
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
        help="Use flash attention (unused in mock implementation)",
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
        help="Device IDs for inference (unused in mock implementation)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random embeddings",
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=768,
        help="Embedding dimension for the random embeddings",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of layers to simulate",
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
        results_summary = run_mock_evaluation(
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
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
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
        generate_mock_report(results_summary, report_dir)

        logger.info("Mock evaluation completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Summary: {report_dir / 'results_summary.json'}")
        logger.info(f"Report: {report_dir / 'mock_evaluation_report.md'}")

    except Exception as e:
        logger.error(f"Mock evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
