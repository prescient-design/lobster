"""Flow-matching sampling callback for :class:`Tokenizer3diInputFlow`.

The deterministic :class:`BackboneReconstruction` callback already dumps
predictions vs. GT from the model's training output (``outputs['x_recon']``).
For the flow variant those are still useful (they're predictions of ``x_1``
conditioned on ``x_t`` -- effectively "1-step samples" from random ``t``).

What this callback adds: a periodic *true* sampling pass that integrates
the ODE / SDE from ``x_0 ~ prior`` to ``x_1`` for a configurable list of
step counts, logs RMSD vs. GT, and writes the sampled coordinates next to
the regression PDBs in ``${structure_path}/sample/``.
"""

import logging
import os
from collections.abc import Sequence

import lightning
import torch

from lobster.model.latent_generator.io import writepdb

logger = logging.getLogger(__name__)


def _rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample masked RMSD over backbone atoms.

    Parameters
    ----------
    pred, target : Tensor
        Shape ``(B, L, A, 3)``.
    mask : Tensor
        Shape ``(B, L)``; ``1.0`` for valid residues, ``0.0`` for pad.
    """
    diff = (pred - target) ** 2
    m = mask[..., None, None].to(diff.dtype)
    diff = diff.sum(dim=-1) * m.squeeze(-1)
    n_atoms = pred.shape[-2]
    denom = mask.sum(dim=-1).clamp_min(1.0) * n_atoms
    return torch.sqrt(diff.sum(dim=(1, 2)) / denom)


def _center(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract per-sample geometric CoM over the masked backbone atoms.

    Mirrors the centering used at training time in
    :func:`Tokenizer3diInputFlow._center_geometric`, so generated and
    ground-truth structures live in the same frame before any
    distance metric.

    Parameters
    ----------
    coords : Tensor
        Shape ``(B, L, A, 3)``.
    mask : Tensor
        Shape ``(B, L)``.

    Returns
    -------
    Tensor
        Centered coords, pad positions zeroed.
    """
    m = mask[..., None, None].to(coords.dtype)
    n_atoms = coords.shape[-2]
    total = (coords * m).sum(dim=(1, 2))
    n = (mask.sum(dim=-1).clamp_min(1.0) * n_atoms).unsqueeze(-1)
    com = total / n
    return (coords - com[:, None, None, :]) * m


def _kabsch_align(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rotate per-sample ``pred`` onto ``target`` via Kabsch (no reflection).

    Both inputs are assumed CoM-centered (we re-center defensively).
    Returns the aligned ``pred`` in ``target``'s rotation frame, padded
    positions zeroed. Computed in ``float32`` because CUDA's batched SVD
    has no ``bf16`` kernel.
    """
    B, L, A, D = pred.shape
    flat_mask = mask.unsqueeze(-1).expand(-1, -1, A).reshape(B, L * A).to(pred.dtype)

    orig_dtype = pred.dtype
    p = pred.reshape(B, L * A, D).float()
    t = target.reshape(B, L * A, D).float()
    m = flat_mask.float().unsqueeze(-1)

    w = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    p = p - (p * m).sum(dim=1, keepdim=True) / w
    t = t - (t * m).sum(dim=1, keepdim=True) / w

    # Cross-covariance H = P^T Q with rows p_k, q_k (q_k=target_k). The
    # Kabsch rotation that minimizes ||R p_k - q_k||^2 is R = V D U^T (NOT
    # U D V^T -- that inverse rotation rotates q into p's frame instead).
    H = torch.einsum("bni,bnj->bij", p * m, t)

    device_type = "cuda" if pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        U, _, Vt = torch.linalg.svd(H)
    # Reflection fix: ensure det(R) = +1 so R is a proper rotation.
    d = torch.sign(torch.det(torch.bmm(Vt.transpose(-1, -2), U.transpose(-1, -2))))
    eye = torch.eye(D, device=pred.device, dtype=p.dtype).expand(B, D, D).clone()
    eye[:, -1, -1] = d
    R = torch.bmm(Vt.transpose(-1, -2), torch.bmm(eye, U.transpose(-1, -2)))
    p_rot = torch.bmm(p, R.transpose(-1, -2)) * m

    return p_rot.reshape(B, L, A, D).to(orig_dtype)


class FlowBackboneSampling(lightning.Callback):
    """Periodically sample from the flow model and log RMSD vs. GT.

    Parameters
    ----------
    structure_path : str
        Output directory. Sampled PDBs are written under
        ``{structure_path}/sample/`` (separate from the regression
        callback's ``{structure_path}/recon/``).
    sampling_step_counts : Sequence[int]
        ODE / SDE step counts to evaluate per call. Default
        ``(1, 5, 50)`` -- the 1-step value is the no-integration baseline
        (single decoder call from x_0), 50 the production setting.
    save_every_n : int
        Sample every ``N``-th validation batch (rank-0 only).
    max_samples_per_call : int
        Cap the number of PDBs saved per call to keep disk bounded.
    """

    def __init__(
        self,
        structure_path: str,
        sampling_step_counts: Sequence[int] = (1, 5, 50),
        save_every_n: int = 5_000,
        max_samples_per_call: int = 4,
    ) -> None:
        super().__init__()
        self.structure_path = structure_path
        self.sampling_step_counts = tuple(int(n) for n in sampling_step_counts)
        self.save_every_n = save_every_n
        self.max_samples_per_call = max_samples_per_call
        os.makedirs(f"{structure_path}/sample", exist_ok=True)

    def on_validation_batch_end(
        self,
        trainer,
        tokenizer,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not trainer.is_global_zero:
            return
        if batch_idx % self.save_every_n != 0:
            return
        if not hasattr(tokenizer, "sample"):
            return

        try:
            self._do_sampling(trainer, tokenizer, batch, batch_idx)
        except Exception as e:
            logger.warning("FlowBackboneSampling failed at batch %d: %s", batch_idx, e)

    @torch.no_grad()
    def _do_sampling(self, trainer, tokenizer, batch, batch_idx):
        device = next(tokenizer.parameters()).device
        states, seq_mask, residue_index = tokenizer.featurize(
            {
                "3di_states": batch["3di_states"].to(device),
                "mask": batch["mask"].to(device),
                "indices": batch["indices"].to(device),
            }
        )
        gt_raw = batch["coords_res"].to(device).to(torch.get_default_dtype())
        gt_c = _center(gt_raw, seq_mask)
        step = trainer.global_step
        n_save = min(self.max_samples_per_call, gt_raw.shape[0])

        for n_steps in self.sampling_step_counts:
            samples_raw = tokenizer.sample(
                states=states,
                seq_mask=seq_mask,
                residue_index=residue_index,
                n_steps=n_steps,
            )
            # `tokenizer.sample` centers `x_t` every step when configured, but
            # re-center defensively so this metric path is independent of model
            # internals.
            samples_c = _center(samples_raw, seq_mask)
            samples_kab = _kabsch_align(samples_c, gt_c, seq_mask)

            raw_rmsd = float(_rmsd(samples_raw, gt_raw, seq_mask).mean().item())
            com_rmsd = float(_rmsd(samples_c, gt_c, seq_mask).mean().item())
            kab_rmsd = float(_rmsd(samples_kab, gt_c, seq_mask).mean().item())

            for name, value in (
                # Kabsch (rotation + translation) is the canonical structural
                # error; we map it to the historic `val/rmsd_n_steps_{n}` so
                # WandB plots become meaningful (the legacy curve was
                # frame-misalignment-dominated and stuck at ~|CoM_gt|).
                (f"val/rmsd_n_steps_{n_steps}", kab_rmsd),
                (f"val/rmsd_kabsch_n_steps_{n_steps}", kab_rmsd),
                (f"val/rmsd_com_n_steps_{n_steps}", com_rmsd),
                (f"val/rmsd_raw_n_steps_{n_steps}", raw_rmsd),
            ):
                tokenizer.log(
                    name,
                    value,
                    batch_size=gt_raw.shape[0],
                    sync_dist=False,
                    rank_zero_only=True,
                )
            logger.info(
                "FlowBackboneSampling: step=%d n_steps=%d kabsch=%.4f com=%.4f raw=%.4f",
                step,
                n_steps,
                kab_rmsd,
                com_rmsd,
                raw_rmsd,
            )

            for i in range(n_save):
                m = seq_mask[i].bool().cpu()
                # Save the Kabsch-aligned generated structure (so it overlays
                # the GT directly in PyMOL/ChimeraX with no manual alignment)
                # and the centered GT (same frame). The model's native rotation
                # frame is not preserved on disk -- if you ever need it, the
                # transform is recoverable by re-running Kabsch.
                base = f"{self.structure_path}/sample/struc_{batch_idx}_{step}_n{n_steps}_"

                xyz_aln = samples_kab[i][m].cpu()
                seq = torch.zeros(xyz_aln.shape[0], dtype=torch.long)
                writepdb(base + f"gen_item{i}.pdb", xyz_aln, seq)

                gt_xyz = gt_c[i][m].cpu()
                gt_seq = torch.zeros(gt_xyz.shape[0], dtype=torch.long)
                writepdb(base + f"gt_item{i}.pdb", gt_xyz, gt_seq)
