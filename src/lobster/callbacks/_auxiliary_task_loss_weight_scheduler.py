import logging
import math
from typing import Literal

from lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class AuxiliaryTaskWeightScheduler(Callback):
    """
    Lightning callback to dynamically modulate auxiliary task loss weights during training.

    Supports multiple scheduling strategies to gradually introduce auxiliary tasks
    without disrupting pre-trained model performance.
    """

    def __init__(
        self,
        schedule_type: Literal["linear", "cosine", "exponential", "step"] = "linear",
        start_step: int = 0,
        ramp_steps: int = 1000,
        max_weight: float = 0.01,
        task_name: str | None = None,
    ):
        """
        Parameters
        ----------
        schedule_type : str
            Type of weight schedule ("linear", "cosine", "exponential", "step")
        start_step : int
            Global step to start increasing weights from 0
        ramp_steps : int
            Number of steps to ramp from 0 to max_weight (weight plateaus afterwards).
            If 0, weight jumps immediately to max_weight at start_step
        max_weight : float
            Maximum weight value to reach
        task_name : str, optional
            Specific auxiliary task name. If None, applies to all auxiliary tasks
        """
        self.schedule_type = schedule_type
        self.start_step = start_step
        self.ramp_steps = ramp_steps
        self.max_weight = max_weight
        self.task_name = task_name

        if ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")

    def _compute_weight(self, current_step: int) -> float:
        """Compute the current weight based on the schedule."""
        if current_step < self.start_step:
            return 0.0

        steps_since_start = current_step - self.start_step

        # If ramp_steps is 0, immediately return max_weight
        if self.ramp_steps == 0:
            return self.max_weight

        if steps_since_start >= self.ramp_steps:
            return self.max_weight

        # Progress through the ramp period
        progress = steps_since_start / self.ramp_steps

        match self.schedule_type:
            case "linear":
                weight = progress * self.max_weight
            case "cosine":
                weight = 0.5 * (1 - math.cos(math.pi * progress)) * self.max_weight
            case "exponential":
                weight = (math.exp(progress) - 1) / (math.e - 1) * self.max_weight
            case "step":
                # Step function at 50% progress
                weight = self.max_weight if progress >= 0.5 else 0.0
            case _:
                raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        return weight

    def _update_auxiliary_task_weights(self, pl_module: LightningModule, weight: float):
        """Update the auxiliary task weights in the model."""
        if not hasattr(pl_module, "auxiliary_tasks") or pl_module.auxiliary_tasks is None:
            return

        for task in pl_module.auxiliary_tasks:
            if self.task_name is None or task.name == self.task_name:
                task.loss_weight = weight

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int):
        """Update weights at the start of each training batch."""
        current_step = trainer.global_step
        weight = self._compute_weight(current_step)
        self._update_auxiliary_task_weights(pl_module, weight)

        # Log the current weight
        if hasattr(pl_module, "log"):
            if self.task_name:
                pl_module.log(f"{self.task_name}_loss_weight", weight, prog_bar=True)
            else:
                pl_module.log("aux_loss_weight", weight, prog_bar=True)


class MultiTaskWeightScheduler(Callback):
    """
    Scheduler for multiple auxiliary tasks with different schedules.
    """

    def __init__(self, task_schedules: dict[str, dict]):
        """
        Parameters
        ----------
        task_schedules : dict
            Dictionary mapping task names to their schedule configs.
            Each config should contain the same parameters as AuxiliaryTaskWeightScheduler.

        Example
        -------
        task_schedules = {
            "biopython": {
                "schedule_type": "linear",
                "start_step": 1000,
                "ramp_steps": 4000,
                "max_weight": 0.01
            },
            "secondary_structure": {
                "schedule_type": "cosine",
                "start_step": 2000,
                "ramp_steps": 6000,
                "max_weight": 0.005
            }
        }
        """
        self.schedulers = {}
        for task_name, config in task_schedules.items():
            self.schedulers[task_name] = AuxiliaryTaskWeightScheduler(task_name=task_name, **config)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int):
        """Update all task weights."""
        for scheduler in self.schedulers.values():
            scheduler.on_train_batch_start(trainer, pl_module, batch, batch_idx)
