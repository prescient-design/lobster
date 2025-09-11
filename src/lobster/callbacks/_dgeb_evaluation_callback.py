import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from transformers.tokenization_utils_base import BatchEncoding

from lobster.evaluation import generate_report, run_evaluation

logger = logging.getLogger(__name__)


class DGEBEvaluationCallback(Callback):
    """Callback for evaluating UME and ESM models on DGEB benchmark tasks.

    Supports two model types:
    - UME models: requires_tokenization=True (default) - Uses checkpoint-based evaluation
    - ESM models: requires_tokenization=False - Uses direct evaluation with ESMAdapterDGEB

    Parameters
    ----------
    model_name : str, default="UME"
        Name for identification and logging.
    modality : Literal["protein", "dna"], default="protein"
        Biological modality for evaluation.
    output_dir : str | None, default=None
        Directory to save results. If None, uses "dgeb_results".
    batch_size : int, default=32
        Batch size for encoding.
    max_seq_length : int, default=1024
        Maximum sequence length.
    requires_tokenization : bool, default=True
        Whether model requires tokenized inputs (UME) or raw sequences (ESM).
    tasks : list[str] | None, default=None
        Specific tasks to run. If None, runs all tasks for the modality.
    use_flash_attn : bool | None, default=None
        Whether to use flash attention (UME only).
    l2_norm : bool, default=False
        Whether to L2-normalize embeddings.
    pool_type : str, default="mean"
        Pooling strategy.
    devices : list[int] | None, default=None
        Device IDs for inference. If None, uses [0].
    seed : int, default=42
        Random seed for reproducibility.
    layers : list[int] | Literal["mid"] | Literal["last"] | None, default="last"
        Layers to extract embeddings from. If None, uses "last" layer.
    """

    def __init__(
        self,
        model_name: str = "UME",
        modality: Literal["protein", "dna"] = "protein",
        output_dir: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 1024,
        requires_tokenization: bool = True,
        tasks: list[str] | None = None,
        use_flash_attn: bool | None = None,
        l2_norm: bool = False,
        pool_type: str = "mean",
        devices: list[int] | None = None,
        seed: int = 42,
        layers: list[int] | Literal["mid"] | Literal["last"] | None = "last",
    ):
        super().__init__()

        # Validate critical parameters
        if modality not in ["protein", "dna"]:
            raise ValueError(f"modality must be 'protein' or 'dna', got {modality}")

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
        self.requires_tokenization = requires_tokenization

    def _convert_results_to_callback_format(self, results_summary: dict[str, Any]) -> dict[str, Any]:
        """Convert DGEB results to callback-friendly format."""
        callback_results = {
            "_metadata": {
                "model_name": self.model_name,
                "modality": self.modality,
                "total_tasks": len(results_summary.get("results", [])),
                "successful_tasks": results_summary.get("successful_tasks", 0),
                "failed_tasks": results_summary.get("failed_tasks", []),
                "total_attempted_tasks": results_summary.get("total_attempted_tasks", 0),
                "evaluation_time": results_summary.get("evaluation_time", "Unknown"),
                "timestamp": results_summary.get("timestamp", "Unknown"),
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
                last_layer_scores = list(scores.values())[-1]
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

    def _process_and_embed(
        self,
        pl_module: L.LightningModule,
        inputs: dict[str, torch.Tensor] | list[str] | str | BatchEncoding,
        modality: str | None = None,
        aggregate: bool = True,
    ) -> torch.Tensor:
        """Process inputs and extract embeddings using model's built-in embedding methods.

        This method is compatible with both UME and ESM models.

        Parameters
        ----------
        pl_module : L.LightningModule
            The lightning module with a model that can extract embeddings
        inputs : dict[str, Tensor] | list[str] | str | BatchEncoding
            Either tokenized inputs (dict with input_ids, attention_mask)
            or raw inputs (list of strings or single string)
        modality : str | None, default=None
            The modality of the inputs. If None, uses self.modality converted to model format.
        aggregate : bool, default=True
            Whether to average pool over sequence length

        Returns
        -------
        torch.Tensor
            Embeddings tensor of shape (batch_size, hidden_size) if aggregate=True
            or (batch_size, seq_len, hidden_size) if aggregate=False
        """
        # Convert callback modality to model modality format if needed
        if modality is None:
            # Map callback modality terms to what models expect
            modality_map = {"protein": "amino_acid", "dna": "nucleotide"}
            modality = modality_map.get(self.modality, self.modality)

        # Handle raw sequences (preferred for both UME and ESM)
        if isinstance(inputs, (str, list)) and not isinstance(inputs, dict):
            return pl_module.embed_sequences(inputs, modality=modality, aggregate=aggregate)

        # Handle tokenized inputs (UME only)
        elif isinstance(inputs, (dict, BatchEncoding)) and "input_ids" in inputs:
            try:
                return pl_module.embed(inputs, aggregate=aggregate)
            except NotImplementedError as e:
                # Try to extract original sequences if available
                if hasattr(inputs, "original_sequence"):
                    return pl_module.embed_sequences(inputs.original_sequence, modality=modality, aggregate=aggregate)
                raise ValueError("Model doesn't support tokenized inputs and no original sequences available") from e

        # Fallback cases
        else:
            if hasattr(inputs, "original_sequence"):
                return pl_module.embed_sequences(inputs.original_sequence, modality=modality, aggregate=aggregate)

            try:
                return pl_module.embed(inputs, aggregate=aggregate)
            except Exception as e:
                raise ValueError(f"Could not process inputs of type {type(inputs)}: {e}") from e

    def evaluate(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on DGEB benchmark tasks."""
        # Route to appropriate evaluation method based on model type
        if not self.requires_tokenization:
            logger.info(f"Running direct DGEB evaluation for {self.model_name}")
            return self._evaluate_esm_direct(module, trainer)

        # Use checkpoint-based approach for UME models
        temp_checkpoint = Path(tempfile.mkdtemp()) / "dgeb_checkpoint.ckpt"

        try:
            logger.info(f"Running DGEB evaluation for {self.model_name} on {self.modality} tasks")

            # Save temporary checkpoint (DGEB adapter expects a path)
            if trainer is not None:
                # Use the existing trainer to save checkpoint
                trainer.save_checkpoint(temp_checkpoint)
            else:
                # Use Lightning's built-in checkpoint saving
                checkpoint = {
                    "state_dict": module.state_dict(),
                    "lr_schedulers": [],
                    "epoch": 0,
                    "global_step": 0,
                    "pytorch-lightning_version": L.__version__,
                    "hyper_parameters": getattr(module, "hparams", {}),
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

            # Generate report (don't let report generation failure kill the evaluation)
            try:
                generate_report(results_summary, Path(self.output_dir))
                logger.info(f"Successfully generated evaluation report at {self.output_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate evaluation report: {e}")
                logger.warning("Continuing with result processing despite report generation failure")

            # Log metrics to trainer if provided
            if trainer is not None and trainer.logger is not None:
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

        except Exception as e:
            logger.error(f"DGEB evaluation failed: {e}")
            return {"error": f"DGEB evaluation failed: {str(e)}", "_metadata": {"model_name": self.model_name}}

        finally:
            # Clean up temporary checkpoint
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()
                try:
                    temp_checkpoint.parent.rmdir()
                except OSError:
                    pass  # Directory not empty

    def _evaluate_esm_direct(
        self,
        module: L.LightningModule,
        trainer: L.Trainer | None = None,
    ) -> dict[str, Any]:
        """Direct evaluation approach for ESM models."""
        try:
            from datetime import datetime

            import dgeb
            from dgeb.modality import Modality as DGEBModality

            from lobster.evaluation import ESMAdapterDGEB

            # Create output directory
            output_path = Path(self.output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

            # Get tasks to run
            if self.tasks is None:
                modality_map = {"protein": DGEBModality.PROTEIN, "dna": DGEBModality.DNA}
                dgeb_modality = modality_map.get(self.modality)
                if dgeb_modality is None:
                    raise ValueError(f"Unsupported modality: {self.modality}")
                task_classes = dgeb.get_tasks_by_modality(dgeb_modality)
                logger.info(f"Running all {len(task_classes)} {self.modality} tasks")
            else:
                task_classes = dgeb.get_tasks_by_name(self.tasks)
                logger.info(f"Running {len(task_classes)} specified tasks: {self.tasks}")

            # Create ESM adapter
            esm_adapter = ESMAdapterDGEB(
                module=module,
                modality=self.modality,
                batch_size=self.batch_size,
                max_seq_length=self.max_seq_length,
                l2_norm=self.l2_norm,
                pool_type=self.pool_type,
                devices=self.devices or [0],
                layers=self.layers,
                process_and_embed_fn=self._process_and_embed,
            )

            # Run evaluation
            evaluation = dgeb.DGEB(tasks=task_classes, seed=self.seed)
            start_time = datetime.now()
            results = evaluation.run(esm_adapter, output_folder=str(output_path))
            end_time = datetime.now()
            evaluation_time = str(end_time - start_time)

            # Process results
            results_summary = {
                "model_name": self.model_name,
                "modality": self.modality,
                "evaluation_time": evaluation_time,
                "timestamp": datetime.now().isoformat(),
                "embedding_dim": getattr(esm_adapter, "embed_dim", "unknown"),
                "total_tasks": len(task_classes),
                "tasks_run": [getattr(task.metadata, "display_name", "Unknown") for task in task_classes],
                "model_metadata": esm_adapter.metadata,
                "results": [],
            }

            # Convert results to our format with error handling for individual tasks
            successful_tasks = 0
            failed_tasks = []

            for result in results:
                try:
                    task_name = getattr(result.task, "display_name", "Unknown Task")
                    task_summary = {
                        "task_name": task_name,
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
                    successful_tasks += 1
                    logger.info(f"Successfully processed results for task: {task_name}")

                except Exception as e:
                    task_name = getattr(getattr(result, "task", None), "display_name", "Unknown Task")
                    logger.warning(f"Failed to process results for task '{task_name}': {e}")
                    failed_tasks.append(task_name)
                    continue  # Continue with next task

            # Add summary of task processing
            results_summary["successful_tasks"] = successful_tasks
            results_summary["failed_tasks"] = failed_tasks
            results_summary["total_attempted_tasks"] = len(results)

            if failed_tasks:
                logger.warning(f"Failed to process {len(failed_tasks)} tasks: {failed_tasks}")
            logger.info(f"Successfully processed {successful_tasks}/{len(results)} tasks")

            # Generate report (don't let report generation failure kill the evaluation)
            try:
                generate_report(results_summary, output_path)
                logger.info(f"Successfully generated evaluation report at {output_path}")
            except Exception as e:
                logger.warning(f"Failed to generate evaluation report: {e}")
                logger.warning("Continuing with result processing despite report generation failure")

            # Log metrics to trainer if provided
            if trainer is not None and trainer.logger is not None:
                for result in results_summary.get("results", []):
                    task_name = result.get("task_name", "unknown")
                    scores = result.get("scores", {})
                    if scores:
                        # Handle nested scores (use last layer) or flat scores
                        if isinstance(scores, dict) and any(isinstance(v, dict) for v in scores.values()):
                            last_layer_scores = list(scores.values())[-1]
                        else:
                            last_layer_scores = scores

                        for metric_name, metric_value in last_layer_scores.items():
                            metric_key = f"dgeb/{self.modality}/{task_name}/{metric_name}"
                            trainer.logger.log_metrics({metric_key: metric_value}, step=0)

            # Convert to callback format and return
            callback_results = self._convert_results_to_callback_format(results_summary)

            # Log completion summary
            if failed_tasks:
                logger.warning(
                    f"DGEB evaluation completed with {len(failed_tasks)} failed tasks. Results saved to: {self.output_dir}"
                )
                logger.warning(f"Failed tasks: {failed_tasks}")
            else:
                logger.info(f"DGEB evaluation completed successfully. Results saved to: {self.output_dir}")

            # Always return results, even if some tasks failed
            return callback_results

        except ImportError:
            logger.error("DGEB library not available")
            return {"error": "DGEB library not available", "_metadata": {"model_name": self.model_name}}

        except Exception as e:
            logger.error(f"DGEB evaluation failed completely: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"DGEB evaluation failed: {str(e)}", "_metadata": {"model_name": self.model_name}}
