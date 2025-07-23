"""DGEB evaluation callback for UME models."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

import lightning as L
from lightning.pytorch.callbacks import Callback

from lobster.evaluation import run_evaluation, generate_report

logger = logging.getLogger(__name__)


class DGEBEvaluationCallback(Callback):
    """Callback for evaluating UME models on DGEB benchmark tasks.

    Parameters
    ----------
    model_name : str, default="UME"
        Name to use for identification and logging.
    modality : Literal["protein", "dna"], default="protein"
        Biological modality for evaluation.
    tasks : list[str] | None, default=None
        List of specific tasks to run. If None, runs all tasks for the modality.
    output_dir : str | None, default=None
        Directory to save evaluation results. If None, uses "dgeb_results".
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
    layers : list[int] | Literal["mid"] | Literal["last"] | None, default=None
        Which layers to extract embeddings from. Options:
        - None: Use last layer (default)
        - "last": Use last layer explicitly  
        - "mid": Use middle layer
        - list[int]: Use specific layer numbers (e.g., [6, 12])
    """

    def __init__(
        self,
        model_name: str = "UME",
        modality: Literal["protein", "dna"] = "protein",
        tasks: list[str] | None = None,
        output_dir: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 1024,
        use_flash_attn: bool | None = None,
        l2_norm: bool = False,
        pool_type: str = "mean",
        devices: list[int] | None = None,
        seed: int = 42,
        layers: list[int] | Literal["mid"] | Literal["last"] | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.modality = modality
        self.tasks = tasks
        self.output_dir = output_dir or "dgeb_results"
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.use_flash_attn = use_flash_attn
        self.l2_norm = l2_norm
        self.pool_type = pool_type
        self.devices = devices
        self.seed = seed
        self.layers = layers

    def _convert_results_to_callback_format(self, results_summary: dict[str, Any]) -> dict[str, Any]:
        """Convert DGEB results to callback-friendly format."""
        callback_results = {
            "_metadata": {
                "model_name": self.model_name,
                "modality": self.modality,
                "total_tasks": len(results_summary.get("results", [])),
                "evaluation_time": results_summary.get("evaluation_time", "Unknown"),
                "output_dir": self.output_dir,
                "layers": self.layers,
            }
        }

        # Extract task results (use last layer scores as main results)
        task_metrics = {"accuracy": [], "f1": [], "top_corr": []}
        
        for result in results_summary.get("results", []):
            task_name = result.get("task_name", "unknown")
            scores = result.get("scores", {})
            
            if scores:
                last_layer_scores = list(scores.values())[-1] if scores else {}
                callback_results[task_name] = last_layer_scores
                
                # Collect metrics for summary
                for metric in task_metrics:
                    if metric in last_layer_scores:
                        task_metrics[metric].append(last_layer_scores[metric])

        # Add summary statistics
        summary = {}
        for metric, values in task_metrics.items():
            if values:
                summary[f"mean_{metric}"] = sum(values) / len(values)
        
        if summary:
            callback_results["summary"] = summary

        return callback_results

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on DGEB benchmark tasks."""
        # Save module to temporary checkpoint for DGEB evaluation
        temp_checkpoint = Path(tempfile.mkdtemp()) / "dgeb_checkpoint.ckpt"
        
        try:
            logger.info(f"Running DGEB evaluation for {self.model_name} on {self.modality} tasks")
            
            # Save temporary checkpoint (DGEB adapter expects a path)
            if trainer is not None:
                # Use the existing trainer to save checkpoint
                trainer.save_checkpoint(temp_checkpoint)
            else:
                # Use Lightning's built-in checkpoint saving
                import torch
                checkpoint = {
                    'state_dict': module.state_dict(),
                    'lr_schedulers': [],
                    'epoch': 0,
                    'global_step': 0,
                    'pytorch-lightning_version': L.__version__,
                    'hyper_parameters': getattr(module, 'hparams', {}),
                }
                torch.save(checkpoint, temp_checkpoint)
            
            # Run DGEB evaluation
            results_summary = run_evaluation(
                model_name=str(temp_checkpoint),
                modality=self.modality,
                tasks=self.tasks,
                output_dir=self.output_dir,
                batch_size=self.batch_size,
                max_seq_length=self.max_seq_length,
                use_flash_attn=self.use_flash_attn,
                l2_norm=self.l2_norm,
                pool_type=self.pool_type,
                devices=self.devices,
                seed=self.seed,
                layers=self.layers,
            )

            # Update results with our model name for consistency
            results_summary["model_name"] = self.model_name

            # Generate report
            generate_report(results_summary, Path(self.output_dir))

            # Log metrics to trainer if provided
            if trainer is not None:
                for result in results_summary.get("results", []):
                    task_name = result.get("task_name", "unknown")
                    scores = result.get("scores", {})
                    if scores:
                        last_layer_scores = list(scores.values())[-1]
                        for metric_name, metric_value in last_layer_scores.items():
                            metric_key = f"dgeb/{self.modality}/{task_name}/{metric_name}"
                            trainer.logger.log_metrics({metric_key: metric_value}, step=0)

            # Convert to callback format and return
            callback_results = self._convert_results_to_callback_format(results_summary)
            logger.info(f"DGEB evaluation for {self.model_name} completed. Results saved to: {self.output_dir}")
            return callback_results

        finally:
            # Clean up temporary checkpoint
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()
                try:
                    temp_checkpoint.parent.rmdir()
                except OSError:
                    pass  # Directory not empty 