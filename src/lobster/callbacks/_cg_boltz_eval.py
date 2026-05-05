"""CG Boltz2 evaluation callback.

On each validation end: submits a SLURM eval job using the latest checkpoint,
and reads results from any previously completed eval jobs to log to wandb.
No GPU work in the callback itself.
"""

import json
import os
import subprocess
from pathlib import Path

from lightning import Callback
from loguru import logger


class CGBoltzEvalCallback(Callback):
    """Submit CG Boltz2 eval at each validation, log completed results to wandb.

    Parameters
    ----------
    eval_dir : str
        Directory where eval results are stored.
    num_designs : int
        Designs per ligand for CG eval.
    data_dir : str
        Directory with test ligand .pt files.
    eval_every_n_steps : int
        Only submit eval every N global steps (to avoid flooding SLURM).
    """

    def __init__(
        self,
        eval_dir: str = "/cv/scratch/u/lisanzas/rest_finetune_i2/cg_boltz_evals",
        num_designs: int = 10,
        data_dir: str = "/cv/home/lisanzas/lobster/data/proteina_ligand_targets/processed",
        eval_every_n_steps: int = 20,
    ):
        super().__init__()
        self.eval_dir = eval_dir
        self.num_designs = num_designs
        self.data_dir = data_dir
        self.eval_every_n_steps = eval_every_n_steps
        self._submitted_steps: set[int] = set()
        self._logged_steps: set[int] = set()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        step = trainer.global_step

        # 1. Read and log any completed eval results
        if os.path.isdir(self.eval_dir):
            for subdir in sorted(Path(self.eval_dir).glob("step_*")):
                step_str = subdir.name.replace("step_", "")
                try:
                    eval_step = int(step_str)
                except ValueError:
                    continue

                if eval_step in self._logged_steps:
                    continue

                results_json = subdir / "cg_boltz_results.json"
                if not results_json.exists():
                    continue

                try:
                    with open(results_json) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                metrics = data.get("metrics", {})
                if not metrics:
                    continue

                log_dict = {
                    "cg_boltz/mean_iptm": metrics.get("mean_iptm", 0),
                    "cg_boltz/mean_ipde": metrics.get("mean_ipde", 0),
                    "cg_boltz/pass_rate": metrics.get("pass_rate", 0),
                    "cg_boltz/pass_both": metrics.get("pass_both", 0),
                    "cg_boltz/total_designs": metrics.get("total_designs", 0),
                }

                for key, value in log_dict.items():
                    trainer.logger.log_metrics({key: value}, step=eval_step)

                self._logged_steps.add(eval_step)
                logger.info(
                    f"CGBoltzEval: logged step {eval_step} — "
                    f"ipTM={metrics.get('mean_iptm', 0):.3f}, "
                    f"pass={metrics.get('pass_both', 0)}/{metrics.get('total_designs', 0)}"
                )

        # 2. Submit new eval if due
        if step in self._submitted_steps:
            return
        # Check if enough steps have passed since last submission
        if self._submitted_steps:
            last_submitted = max(self._submitted_steps)
            if step - last_submitted < self.eval_every_n_steps:
                return
        elif step == 0:
            return

        # Find the latest checkpoint
        ckpt_path = None
        for cb in trainer.callbacks:
            if hasattr(cb, "last_model_path") and cb.last_model_path:
                ckpt_path = cb.last_model_path
            if hasattr(cb, "best_model_path") and cb.best_model_path:
                if ckpt_path is None:
                    ckpt_path = cb.best_model_path

        if ckpt_path is None or not os.path.exists(ckpt_path):
            logger.warning(f"CGBoltzEval: no checkpoint found at step {step}, skipping")
            return

        eval_subdir = os.path.join(self.eval_dir, f"step_{step}")
        os.makedirs(eval_subdir, exist_ok=True)
        log_dir = os.path.join(self.eval_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "--partition=ai4dd-b200",
                "--account=llm",
                "--qos=llm",
                "--nodes=1",
                "--ntasks-per-node=1",
                "--gres=gpu:b200:1",
                "--cpus-per-task=16",
                "--mem=128G",
                "-t",
                "02:00:00",
                f"--job-name=cg-eval-{step}",
                "-o",
                f"{log_dir}/cg_eval_{step}_%j.out",
                "-e",
                f"{log_dir}/cg_eval_{step}_%j.err",
                f"--wrap=cd /cv/home/lisanzas/lobster && "
                f"uv run python scripts/eval_cg_boltz_checkpoint.py "
                f"--checkpoint '{ckpt_path}' "
                f"--output_dir '{eval_subdir}' "
                f"--num_designs {self.num_designs} "
                f"--data_dir '{self.data_dir}'",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            job_id = result.stdout.strip()
            self._submitted_steps.add(step)
            logger.info(
                f"CGBoltzEval: submitted eval for step {step} (ckpt={os.path.basename(ckpt_path)}) -> job {job_id}"
            )
        else:
            logger.warning(f"CGBoltzEval: sbatch failed at step {step}: {result.stderr[:200]}")
