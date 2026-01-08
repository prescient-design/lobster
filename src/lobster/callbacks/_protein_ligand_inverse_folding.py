"""Callback for evaluating ligand-conditioned inverse folding on protein-ligand complexes.

This callback compares inverse folding with and without ligand context to
determine if ligand information improves sequence recovery, particularly
for binding pocket residues.
"""

import os

import lightning
import torch
from loguru import logger

from lobster.metrics.protein_ligand_inverse_folding import ProteinLigandInverseFoldingEvaluator


class ProteinLigandInverseFoldingCallback(lightning.Callback):
    """Callback for evaluating ligand-conditioned inverse folding on protein-ligand complexes.

    Compares:
    - Inverse folding with protein structure only
    - Inverse folding with protein structure + ligand context

    Hypothesis: Ligand context should improve pocket sequence recovery.

    Parameters
    ----------
    data_dir : str
        Path to PDBBind test directory with *_protein.pt and *_ligand.pt pairs
    structure_path : str, optional
        Output directory for results and structures
    save_every_n : int
        Run evaluation every N training steps
    num_samples : int
        Number of structures to evaluate (subset for speed during training)
    pocket_distance_threshold : float
        Distance threshold (Å) for defining binding pocket residues
    nsteps : int
        Number of diffusion steps for generation
    metric_prefix : str
        Prefix for logged metrics

    Example
    -------
    Add to callbacks config:

    ```yaml
    protein_ligand_inverse_folding:
      _target_: lobster.callbacks.ProteinLigandInverseFoldingCallback
      data_dir: /data2/lisanzas/pdb_bind_12_15_25/test/
      structure_path: ${paths.output_dir}/protein_ligand_eval/
      save_every_n: 5000
      num_samples: 100
      pocket_distance_threshold: 5.0
    ```
    """

    def __init__(
        self,
        data_dir: str = "/data2/lisanzas/pdb_bind_12_15_25/test/",
        structure_path: str | None = None,
        save_every_n: int = 5000,
        num_samples: int = 100,
        pocket_distance_threshold: float = 5.0,
        nsteps: int = 100,
        metric_prefix: str = "protein_ligand_inverse_folding",
    ):
        self.data_dir = data_dir
        self.structure_path = structure_path
        self.save_every_n = save_every_n
        self.num_samples = num_samples
        self.pocket_distance_threshold = pocket_distance_threshold
        self.nsteps = nsteps
        self.metric_prefix = metric_prefix

        self._evaluator = None
        self._samples = None
        self._evaluated_steps = set()  # Track steps we've already evaluated

        # Create output directory
        if structure_path:
            os.makedirs(structure_path, exist_ok=True)

    def setup(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, stage: str):
        """Load test samples once at setup.

        Only loads on rank 0 in distributed settings.
        """
        if trainer.global_rank != 0:
            return

        if self._samples is not None:
            return  # Already loaded

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get max_length from model if available
        max_length = 512  # default
        if hasattr(pl_module, "encoder") and hasattr(pl_module.encoder, "neobert"):
            if hasattr(pl_module.encoder.neobert, "config") and hasattr(pl_module.encoder.neobert.config, "max_length"):
                max_length = pl_module.encoder.neobert.config.max_length
                logger.info(f"Using model's max_length: {max_length}")

        # Create evaluator
        self._evaluator = ProteinLigandInverseFoldingEvaluator(
            data_dir=self.data_dir,
            pocket_distance_threshold=self.pocket_distance_threshold,
            num_samples=self.num_samples,
            nsteps=self.nsteps,
            device=device,
            max_length=max_length,
        )

        # Load samples
        try:
            self._samples = self._evaluator.load_test_set()
            logger.info(f"Loaded {len(self._samples)} protein-ligand test samples for inverse folding evaluation")
        except Exception as e:
            logger.error(f"Failed to load protein-ligand test samples: {e}")
            self._samples = []

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Run evaluation at specified intervals."""
        # Only run on rank 0
        if trainer.global_rank != 0:
            return

        current_step = trainer.global_step

        # Check if we should run evaluation
        if current_step % self.save_every_n != 0:
            return

        # Skip if we've already evaluated this step (avoid duplicates from multiple callback invocations)
        if current_step in self._evaluated_steps:
            return

        if not self._samples:
            logger.warning("No protein-ligand samples available for evaluation")
            return

        # Mark this step as evaluated
        self._evaluated_steps.add(current_step)
        logger.info(f"Running protein-ligand inverse folding evaluation at step {current_step}")

        # Run evaluation
        try:
            results = self._evaluator.evaluate(pl_module, self._samples)

            # Log metrics to trainer
            summary = results["summary"]
            metrics_to_log = {}

            for key, value in summary.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics_to_log[f"{self.metric_prefix}/{key}"] = value

            pl_module.log_dict(metrics_to_log, batch_size=1)

            # Log key insight
            pocket_delta = summary.get("mean_aar_pocket_delta", 0)
            nonpocket_delta = summary.get("mean_aar_nonpocket_delta", 0)

            logger.info(
                f"Protein-Ligand Inverse Folding (step {current_step}):\n"
                f"  Pocket AAR: {summary['mean_aar_pocket_no_ligand']:.2%} → "
                f"{summary['mean_aar_pocket_with_ligand']:.2%} "
                f"(Δ={pocket_delta:+.2%})\n"
                f"  Non-pocket AAR: {summary['mean_aar_nonpocket_no_ligand']:.2%} → "
                f"{summary['mean_aar_nonpocket_with_ligand']:.2%} "
                f"(Δ={nonpocket_delta:+.2%})\n"
                f"  Overall AAR: {summary['mean_aar_overall_no_ligand']:.2%} → "
                f"{summary['mean_aar_overall_with_ligand']:.2%}"
            )

            # Save CSV results
            if self.structure_path:
                output_file = os.path.join(
                    self.structure_path,
                    f"protein_ligand_inverse_folding_step{current_step}.csv",
                )
                results["results_df"].to_csv(output_file, index=False)
                logger.info(f"Saved results to {output_file}")

        except Exception as e:
            logger.error(f"Protein-ligand inverse folding evaluation failed: {e}")
            import traceback

            traceback.print_exc()

        # Clear GPU cache
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Optionally run evaluation at validation epoch end.

        This method can be used for periodic evaluation during validation
        if preferred over batch-based evaluation.
        """
        # Currently using batch-end evaluation; can enable this if needed
        pass
