"""Force a new learning rate on a resumed training run.

When `Trainer.fit(ckpt_path=...)` resumes a run, Lightning restores the
optimizer state via `optimizer.load_state_dict(...)` -- which overwrites
`param_groups[i]['lr']` with whatever LR was baked into the ckpt. The
LR scheduler's `base_lrs` and `_last_lr` are similarly restored. So
passing `model.optim.lr=<new>` on the Hydra command line gets clobbered
on resume, and the run continues at the original LR.

This callback runs once at `on_train_start` (after Lightning's checkpoint
restore is complete) and overwrites the LR in three places:

1. ``param_groups[i]['lr']``
2. ``param_groups[i]['initial_lr']`` (some schedulers re-derive from this)
3. ``scheduler.base_lrs`` (LambdaLR-family schedulers compute
   ``output_lr = base_lr * lr_lambda(step)``; without updating
   ``base_lrs``, the scheduler's next ``.step()`` undoes our override)

It also resets the warmup-style scheduler's step counter so the resumed
run gets a soft re-warmup at the new LR rather than jumping straight to
the post-warmup constant -- mirrors what we'd get from a fresh launch.

Wire-in (Hydra):

.. code-block:: yaml

    callbacks:
      lr_override_on_resume:
        _target_: lobster.callbacks.LrOverrideOnResume
        lr: 8e-5
        reset_scheduler_step: true
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class LrOverrideOnResume(pl.Callback):
    """Override LR (and optionally scheduler step) once on a resumed run.

    Parameters
    ----------
    lr : float
        New learning rate to write into every optimizer ``param_group`` and
        scheduler ``base_lrs``.
    reset_scheduler_step : bool, default True
        If True, also reset each scheduler's ``last_epoch`` (a.k.a.
        ``_step_count``) to 0 so any warmup phase re-runs at the new LR.
    only_on_resume : bool, default True
        If True, the override only fires when ``trainer.global_step > 0`` at
        ``on_train_start`` (i.e. a checkpoint was actually loaded). If False,
        fires unconditionally -- useful only if you're using this as a "set
        LR from cfg" hack on a fresh run.
    """

    def __init__(
        self,
        lr: float,
        reset_scheduler_step: bool = True,
        only_on_resume: bool = True,
    ) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr!r}")
        self.lr = float(lr)
        self.reset_scheduler_step = bool(reset_scheduler_step)
        self.only_on_resume = bool(only_on_resume)
        self._applied = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._applied:
            return
        if self.only_on_resume and trainer.global_step == 0:
            rank_zero_info("[LrOverrideOnResume] global_step=0 (fresh run) -- skipping override")
            self._applied = True
            return

        for opt_idx, opt in enumerate(trainer.optimizers):
            for pg_idx, pg in enumerate(opt.param_groups):
                old = pg.get("lr")
                pg["lr"] = self.lr
                if "initial_lr" in pg:
                    pg["initial_lr"] = self.lr
                rank_zero_info(f"[LrOverrideOnResume] optim[{opt_idx}].param_group[{pg_idx}]: lr {old} -> {self.lr}")

        for sch_idx, sch_cfg in enumerate(trainer.lr_scheduler_configs):
            scheduler = sch_cfg.scheduler
            if hasattr(scheduler, "base_lrs"):
                old = list(scheduler.base_lrs)
                scheduler.base_lrs = [self.lr] * len(scheduler.base_lrs)
                rank_zero_info(f"[LrOverrideOnResume] scheduler[{sch_idx}].base_lrs: {old} -> {scheduler.base_lrs}")
            if self.reset_scheduler_step and hasattr(scheduler, "last_epoch"):
                old = scheduler.last_epoch
                scheduler.last_epoch = 0
                if hasattr(scheduler, "_step_count"):
                    scheduler._step_count = 0
                rank_zero_info(
                    f"[LrOverrideOnResume] scheduler[{sch_idx}].last_epoch: {old} -> 0 (warmup will re-run from new LR)"
                )

        self._applied = True
