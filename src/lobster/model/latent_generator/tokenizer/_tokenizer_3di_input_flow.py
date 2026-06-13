"""Latent Generator variant -- 3Di tokens in, backbone coordinates out via flow matching.

Sibling of :class:`Tokenizer3diInput`. The deterministic L2-regression
variant is left UNTOUCHED so the live SLURM job can preempt/resume
mid-build without state_dict drift; this module is only ever wired in
via ``model/latent_generator_3di_input_flow.yaml``.

Pipeline
--------

1. Featurize 3Di tokens (same as :class:`Tokenizer3diInput`).
2. CoM-center ``x_1`` (geometric centering over all backbone atoms), then
   scale by ``coord_scale`` (Proteina convention: Angstrom -> nanometer,
   factor ``0.1``) so the data variance matches the unit-variance Gaussian
   prior. The scaling is undone at the end of :meth:`sample`.
3. **Train-only:** if ``random_so3_aug`` is enabled, apply a per-sample
   uniform random SO(3) rotation to ``x_1`` (centered, so the CoM stays
   at the origin). Proteina-style data augmentation. All downstream
   tensors (``x_t`` and the loss target) consume the **rotated** ``x_1``
   by construction, so the loss is computed in the augmented frame
   (no separate alignment of ``x_1_hat`` to ``x_1`` is needed). At
   val/test time this step is a no-op so ``val/loss`` stays directly
   comparable across runs.
4. Sample noise ``x_0`` from :class:`GaussianPrior` and CoM-center it
   (Proteina-style zero-CoM noise prior).
5. Optionally apply :class:`KabschAugmentation` (Kabsch SE(3) alignment
   of ``x_0`` to ``x_1`` per sample) to keep the per-sample
   interpolation path SE(3)-equivariant under data augmentation.
   When step 3 is active, the Kabsch step aligns ``x_0`` to the
   **rotated** ``x_1`` -- this keeps the OT pairing consistent.
6. Sample ``t ~ U(0, 1)`` and ``x_t = t * x_1 + (1 - t) * x_0``.
7. CFG dropout: with prob ``p_uncond``, replace the 3Di sequence with
   the pad/null class (``num_3di_classes``).
8. Decoder predicts ``x_1_hat = f(states, x_t, t)``.
9. Loss = ``flow_matcher.loss(model_pred, target, t, x_t, target_type=...)``
   where ``target_type`` is taken from the moco interpolant's
   ``prediction_type`` (Step S):

   - ``DATA``: target = ``x_1``, with Proteina's ``1/(1-t)^2`` reweight.
   - ``VELOCITY``: target = ``x_1 - x_0``, no reweight (Proteina's
     ``target_pred=v`` mode).

   Plus ``aux_pairwise_l2_weight`` * pairwise L2 on the data-space
   prediction (``process_data_prediction(model_pred, x_t, t)``).
   When ``random_so3_aug`` is active, ``x_1`` here is the rotated GT --
   marginalising the gradient over rotations is the augmentation
   equivalent of a Kabsch-invariant loss, but without the in-loss
   alignment that would break flow-trajectory frame consistency.

Sampling
--------

ODE (Euler) by default; SDE (score-stochastic) when ``sampling_mode='sde'``.
``sample()`` accepts ``autoguidance_model`` + ``autoguidance_ratio`` stub
kwargs so an autoguidance checkpoint can be dropped in later without
code changes.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Literal

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn.functional as F
from torch import Tensor

from bionemo.moco.interpolants.continuous_time.continuous.continuous_flow_matching import (
    ContinuousFlowMatcher,
)
from bionemo.moco.schedules.inference_time_schedules import (
    LinearInferenceSchedule,
    LogInferenceSchedule,
    PowerInferenceSchedule,
)

from lobster.model.latent_generator.structure_decoder import DecoderFactory
from lobster.model.latent_generator.utils.mini3di._torch_encoder import MiniThreeDiEncoderTorch

logger = logging.getLogger(__name__)

# Standard Foldseek 3Di alphabet (20 states). Pad index follows the same
# convention as Tokenizer3diInput: one past the legal range. The CFG-null
# class reuses this same index (so the wrapped decoder's embedding table
# stays at num_3di_classes + 1 rows).
NUM_3DI_CLASSES = 20


def _center_geometric(coords: Tensor, seq_mask: Tensor, n_atoms: int) -> Tensor:
    """Subtract the per-sample geometric CoM (mean over all backbone atoms).

    Parameters
    ----------
    coords : Tensor
        Shape ``(B, L, A, 3)``.
    seq_mask : Tensor
        Shape ``(B, L)``; ``1.0`` = valid residue, ``0.0`` = pad.
    n_atoms : int
        Number of atoms per residue (``A``).

    Returns
    -------
    Tensor
        Centered coords, masked to zero at padding positions.
    """
    m = seq_mask[..., None, None].to(coords.dtype)
    total = (coords * m).sum(dim=(1, 2))
    n = (seq_mask.sum(dim=-1).clamp_min(1.0) * n_atoms).unsqueeze(-1)
    com = total / n
    return (coords - com[:, None, None, :]) * m


def _pairwise_l2(pred: Tensor, target: Tensor, seq_mask: Tensor) -> Tensor:
    """Mean pairwise residue-residue distance L2 over valid positions.

    Cheap auxiliary geometry term to keep intermediate metrics honest at
    high t. Mirrors ``lobster.model.latent_generator.tokenizer.PairWiseL2Loss``
    but operates on the 4D ``(B, L, A, 3)`` directly (no batch dict).
    """
    pred_ca = pred[:, :, 1, :]
    target_ca = target[:, :, 1, :]
    pred_dists = torch.cdist(pred_ca, pred_ca, p=2)
    target_dists = torch.cdist(target_ca, target_ca, p=2)
    diff = (pred_dists - target_dists) ** 2
    pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
    diff = diff * pair_mask
    denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (diff.sum(dim=(1, 2)) / denom).mean()


def _split_decoder_output(out) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Pull `protein_coords` and optional aux-head logits out of a decoder
    output. Backwards-compat: bare-Tensor decoders return ``(coords, None, None)``.
    Returns ``(coords, distogram_logits, three_di_logits)``.
    """
    if isinstance(out, dict):
        return (
            out["protein_coords"],
            out.get("distogram_logits", None),
            out.get("three_di_logits", None),
        )
    return out, None, None


def _distogram_ce(
    distogram_logits: Tensor,
    target_ca_dists: Tensor,
    seq_mask: Tensor,
    *,
    num_dist_buckets: int,
    max_dist_scaled: float,
) -> Tensor:
    """Per-sample distogram cross-entropy on bucketised Cα-Cα distances.

    Mirrors Proteina's `compute_auxiliary_loss` distogram branch
    (proteinfoundation/proteinflow/proteina.py:~261). Differences:

    * Pair logits come from an outer-product MLP head over single-track
      features (we have no pair-track transformer); Proteina has a
      dedicated AF-style pair head.
    * 4D spatial cross-entropy instead of flatten-to-2D + 1D CE; values
      are bit-identical, this version avoids the temporary
      ``view(B*L*L, K)`` allocation.

    Parameters
    ----------
    distogram_logits : Tensor
        Shape ``(B, L, L, K)``.
    target_ca_dists : Tensor
        Shape ``(B, L, L)`` -- pairwise Cα-Cα distances of the GROUND TRUTH,
        in the same scaled coord space the model produces (so that
        ``max_dist_scaled`` is also in scaled units).
    seq_mask : Tensor
        ``(B, L)``; 1 = valid residue.
    num_dist_buckets : int
    max_dist_scaled : float
        Upper bin edge in scaled units (``max_dist_a * coord_scale``).

    Returns
    -------
    Tensor
        Shape ``(B,)`` -- masked-mean cross-entropy per sample.
    """
    B, L, _, K = distogram_logits.shape
    pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
    # Proteina zeroes `gt_pair_dists` by `pair_mask` BEFORE bucketize so
    # that padding pairs land in bucket 0 deterministically -- safer than
    # relying on the upstream centering invariant for zero-padding.
    target_ca_dists = target_ca_dists * pair_mask
    boundaries = torch.linspace(
        0.0,
        max_dist_scaled,
        num_dist_buckets - 1,
        device=distogram_logits.device,
        dtype=target_ca_dists.dtype,
    )
    gt_bucket = torch.bucketize(target_ca_dists, boundaries)  # (B, L, L), long
    # F.cross_entropy expects (B, K, ...) -- channels-first -- for spatial CE.
    per_pair_ce = F.cross_entropy(
        distogram_logits.permute(0, 3, 1, 2),  # (B, K, L, L)
        gt_bucket,
        reduction="none",
    )  # (B, L, L)
    masked = per_pair_ce * pair_mask
    denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return masked.sum(dim=(1, 2)) / denom


class Tokenizer3diInputFlow(pl.LightningModule):
    """3Di-conditioned flow-matching backbone-coord generator."""

    def __init__(
        self,
        decoder_factory: Callable[..., DecoderFactory],
        optim: Callable[..., torch.optim.Optimizer],
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        interpolant: Callable | None = None,
        time_distribution: Callable | None = None,
        prior_distribution: Callable | None = None,
        inference_schedule: Callable | None = None,
        n_atoms: int = 3,
        num_3di_classes: int = NUM_3DI_CLASSES,
        p_uncond: float = 0.15,
        guidance_scale: float = 1.0,
        n_sampling_steps: int = 50,
        aux_pairwise_l2_weight: float = 0.01,
        aux_distogram_weight: float = 0.0,
        aux_distogram_t_lim: float = 0.5,
        num_dist_buckets: int = 64,
        max_dist_a: float = 22.0,
        aux_3di_ce_weight: float = 0.0,
        aux_3di_t_lim: float = -1.0,
        aux_3di_coord_ce_weight: float = 0.0,
        aux_3di_coord_ce_t_lim: float = -1.0,
        aux_3di_coord_ce_temperature: float = 1.0,
        sampling_mode: Literal["ode", "sde"] = "ode",
        sc_scale_noise: float = 0.0,
        sc_scale_score: float = 0.0,
        center_x1: bool = True,
        center_every_step: bool = True,
        coord_scale: float = 1.0,
        random_so3_aug: bool = False,
        autoguidance_model: torch.nn.Module | None = None,
        autoguidance_ratio: float = 0.0,
        use_self_conditioning: bool = False,
        selfcond_train_prob: float = 0.5,
        mask_3di_per_residue: bool = False,
        num_warmup_steps: int = 5_000,
        num_training_steps: int = 500_000,
        automatic_optimization: bool = True,
        ckpt_path: str | None = None,
    ):
        super().__init__()

        if isinstance(decoder_factory, omegaconf.DictConfig):
            decoder_factory = hydra.utils.instantiate(decoder_factory)
        if isinstance(optim, omegaconf.DictConfig):
            optim = hydra.utils.instantiate(optim)
        if isinstance(lr_scheduler, omegaconf.DictConfig):
            lr_scheduler = hydra.utils.instantiate(lr_scheduler)
        if isinstance(time_distribution, omegaconf.DictConfig):
            time_distribution = hydra.utils.instantiate(time_distribution)
        if isinstance(prior_distribution, omegaconf.DictConfig):
            prior_distribution = hydra.utils.instantiate(prior_distribution)
        if isinstance(inference_schedule, omegaconf.DictConfig):
            inference_schedule = hydra.utils.instantiate(inference_schedule)

        if interpolant is None:
            raise ValueError("`interpolant` is required for Tokenizer3diInputFlow")
        if time_distribution is None or prior_distribution is None:
            raise ValueError("`time_distribution` and `prior_distribution` are required")

        # Compose moco's ContinuousFlowMatcher. Three paths:
        # 1. DictConfig (the production path: train.py calls
        #    `instantiate(cfg.model, _recursive_=False)` so child keys arrive
        #    as raw configs). The yaml uses `_partial_: True`, so we MUST
        #    pass `_partial_=False` here to force full instantiation -- otherwise
        #    `instantiate` returns a `functools.partial` even with our kwargs.
        # 2. Already-built ContinuousFlowMatcher (constructed in tests).
        # 3. `functools.partial` (CPU dry-run path: when the outer
        #    `instantiate` is recursive, the partial arrives pre-resolved).
        if isinstance(interpolant, omegaconf.DictConfig):
            interpolant = hydra.utils.instantiate(
                interpolant,
                time_distribution=time_distribution,
                prior_distribution=prior_distribution,
                _partial_=False,
            )
        elif not isinstance(interpolant, ContinuousFlowMatcher):
            interpolant = interpolant(
                time_distribution=time_distribution,
                prior_distribution=prior_distribution,
            )

        if not isinstance(interpolant, ContinuousFlowMatcher):
            raise TypeError(
                "After instantiation, `interpolant` is not a "
                "ContinuousFlowMatcher (got "
                f"{type(interpolant).__name__}). The most common cause is "
                "forgetting `_partial_=False` when overriding a `_partial_: true` "
                "yaml. Please file a bug if you see this."
            )

        if inference_schedule is None:
            inference_schedule = LinearInferenceSchedule(nsteps=n_sampling_steps)

        self.decoder_factory = decoder_factory
        self.optim_factory = optim
        self.lr_scheduler = lr_scheduler
        self.flow_matcher = interpolant
        self.time_distribution = time_distribution
        self.prior_distribution = prior_distribution
        self.inference_schedule = inference_schedule

        self.n_atoms = n_atoms
        self.num_3di_classes = num_3di_classes
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.n_sampling_steps = n_sampling_steps
        self.aux_pairwise_l2_weight = aux_pairwise_l2_weight
        self.aux_distogram_weight = float(aux_distogram_weight)
        self.aux_distogram_t_lim = float(aux_distogram_t_lim)
        self.num_dist_buckets = int(num_dist_buckets)
        self.max_dist_a = float(max_dist_a)
        self.aux_3di_ce_weight = float(aux_3di_ce_weight)
        # `aux_3di_t_lim < 0` -> no time gate; otherwise (t > t_lim) per-sample.
        self.aux_3di_t_lim = float(aux_3di_t_lim)
        self.aux_3di_coord_ce_weight = float(aux_3di_coord_ce_weight)
        self.aux_3di_coord_ce_t_lim = float(aux_3di_coord_ce_t_lim)
        self.aux_3di_coord_ce_temperature = float(aux_3di_coord_ce_temperature)
        if self.aux_3di_coord_ce_weight > 0.0:
            # Frozen Foldseek VAE Dense weights + 3Di centroids registered
            # as buffers, so the module follows the parent's `.to(device)`
            # without adding trainable parameters. Built eagerly so the
            # buffers move with the module on `.cuda()`/Lightning device
            # transfer; lazy build inside `_single_step` would miss that.
            self.mini3di_torch = MiniThreeDiEncoderTorch()
        else:
            self.mini3di_torch = None
        self.sampling_mode = sampling_mode
        self.sc_scale_noise = sc_scale_noise
        self.sc_scale_score = sc_scale_score
        self.center_x1 = center_x1
        # Retained for config/back-compat. NOTE: sampling now *always* re-centers
        # intermediate states (this is a CoM-free flow, so it is never correct to
        # skip it), so this flag no longer gates the `sample()` loop.
        self.center_every_step = center_every_step
        # Coordinate scale factor applied to x_1 before flow matching (and undone
        # at the end of `sample()`). Proteina works in **nanometers** (Angstrom / 10,
        # `nm_to_ang_scale=10.0` in their `coors_utils`), so that backbone-coordinate
        # variance (~tens of Angstrom) matches the unit-variance Gaussian prior.
        # Without this the prior noise is negligible vs. the signal and the flow
        # degenerates into a 1-shot regressor. Set `coord_scale=0.1` in the flow
        # YAMLs (Angstrom -> nm); default 1.0 keeps old checkpoints byte-identical.
        self.coord_scale = coord_scale
        self.random_so3_aug = random_so3_aug
        self.autoguidance_model = autoguidance_model
        self.autoguidance_ratio = autoguidance_ratio
        self.use_self_conditioning = use_self_conditioning
        self.selfcond_train_prob = selfcond_train_prob
        # Step V: per-residue 3Di masking. When True, conditional samples (those
        # not fully nulled by `p_uncond`) get a per-sample mask rate `p ~ U(0, 1)`
        # and each residue's 3Di token is replaced by the null/pad index w.p. `p`.
        # Trains the decoder to fill in coordinates from PARTIAL 3Di conditioning,
        # so at inference one can supply tokens for some residues and let the
        # model fill in the rest. Borrows the moco discrete-flow / leflur masking
        # pattern (`rand_per_token < rand_per_sample` ~ Bernoulli with rate ~ U(0,1)).
        self.mask_3di_per_residue = mask_3di_per_residue

        # Defensive consistency check: the decoder must have been built with
        # the matching `use_self_conditioning` flag, otherwise the self-cond
        # forward pass will silently no-op (decoder=False) or trip the guard
        # in the decoder (decoder=True but we never pass x_selfcond).
        _, _decoder = self._decoder
        decoder_uses_sc = getattr(_decoder, "use_self_conditioning", False)
        if self.use_self_conditioning != decoder_uses_sc:
            raise ValueError(
                "Tokenizer3diInputFlow.use_self_conditioning="
                f"{self.use_self_conditioning} does not match decoder."
                f"use_self_conditioning={decoder_uses_sc}. Set both flags "
                "consistently in the model yaml."
            )

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.automatic_optimization = automatic_optimization
        self.ckpt_path = ckpt_path

        self.encoder = None
        self.quantizer = None
        self.freeze_decoder = False
        self.freeze_encoder = False
        self.freeze_quantizer = False

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Tokenizer3diInputFlow: total params=%d, trainable=%d", total, trainable)

    def featurize(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        states = batch["3di_states"].long()
        seq_mask = batch["mask"].float()
        residue_index = batch["indices"].long()
        states = torch.where(seq_mask.bool(), states, torch.full_like(states, self.num_3di_classes))
        return states, seq_mask, residue_index

    @property
    def _decoder(self):
        decoder_name = next(iter(self.decoder_factory.list_decoders()))
        return decoder_name, self.decoder_factory.decoders[decoder_name]

    def _sample_prior_centered(self, shape: tuple[int, ...], seq_mask: Tensor, device: torch.device) -> Tensor:
        """Sample noise from the prior and CoM-center it (zero-CoM noise)."""
        noise = self.prior_distribution.sample(shape, device=device)
        noise = _center_geometric(noise, seq_mask, self.n_atoms)
        return noise

    def _maybe_cfg_dropout(self, states: Tensor, training: bool) -> Tensor:
        """Train-time conditioning corruption.

        Two mutually-exclusive policies, both train-only:

        1. **Full-sample CFG dropout** (existing): with prob ``p_uncond``,
           replace the entire 3Di sequence with the null index
           (``num_3di_classes``). Trains the unconditional path used by
           classifier-free guidance.

        2. **Per-residue masking** (Step V, when ``mask_3di_per_residue``
           is True): for samples NOT hit by (1), sample a per-sample mask
           rate ``p ~ U(0, 1)`` and replace each residue's 3Di token by
           the null index independently w.p. ``p`` (Bernoulli per token).
           Trains the decoder to handle PARTIAL 3Di conditioning at any
           coverage level. Borrows the moco discrete-flow-matching mask
           pattern (`rand_per_token < rand_per_sample`, see leflur's
           ``_leflur_sequence_structure_encoder_lightning_module.py``).

        At inference both policies are no-ops; per-residue masking is
        applied externally by the caller (the decoder treats the null
        index identically whether it came from training-time masking,
        CFG full-drop, or padding).
        """
        if not training:
            return states
        B, L = states.shape
        device = states.device
        null = torch.full_like(states, self.num_3di_classes)

        # (1) Full-sample CFG dropout.
        if self.p_uncond > 0.0:
            full_drop = torch.rand(B, device=device) < self.p_uncond
            states = torch.where(full_drop[:, None], null, states)
        else:
            full_drop = torch.zeros(B, device=device, dtype=torch.bool)

        # (2) Per-residue masking on conditional samples (those not fully dropped).
        if self.mask_3di_per_residue:
            p_mask = torch.rand(B, 1, device=device)  # per-sample mask rate ~ U(0,1)
            per_token = torch.rand(B, L, device=device)  # per-token threshold
            apply = per_token < p_mask  # Bernoulli(p_mask) per token
            apply = apply & (~full_drop[:, None])  # skip already-fully-dropped samples
            states = torch.where(apply, null, states)

        return states

    def _apply_random_so3_aug(self, x: Tensor, seq_mask: Tensor, training: bool) -> Tensor:
        """Per-sample uniform random SO(3) rotation around the origin.

        Applied to ``x_1`` AFTER :func:`_center_geometric` and BEFORE
        :func:`_maybe_kabsch` / interpolation, so the rotated ``x_1``
        is what flows into the interpolant, the decoder input, and the
        loss target -- the augmentation is "baked in" rather than
        applied as an in-loss alignment. For centered coordinates this
        rotation preserves CoM = 0, so we don't re-center after.

        Why SO(3) (not full SE(3))?
        ---------------------------
        Translation is mooted by :func:`_center_geometric` and by the
        zero-CoM Gaussian prior used for ``x_0``: any added translation
        would just be removed at the next centering step (and the
        Kabsch noise-alignment step would still align ``x_0`` to the
        translated ``x_1``). Proteina's "SE(3) augmentation" is in
        practice SO(3) for the same reason.

        Why train-only?
        ---------------
        Val/test loss should compare like-to-like across runs. Keeping
        the canonical PDB orientation at eval means the absolute
        numbers stay directly comparable to the no-aug baseline. The
        Kabsch-aligned RMSD reported by ``FlowBackboneSampling`` is
        already rotation-invariant, so we don't lose any insight at
        eval; we only avoid adding sampling noise to the metric.

        Uniform sampling on SO(3)
        -------------------------
        1. ``M ~ N(0, I)_{3x3}`` (one per batch element).
        2. ``Q, R = QR(M)``.
        3. Multiply columns of ``Q`` by ``sign(diag(R))`` -> uniform on
           ``O(3)`` (Mezzadri 2007, "How to generate random matrices
           from the classical compact groups").
        4. If ``det(Q) < 0``, flip the first column of ``Q`` -> uniform
           on ``SO(3)`` (det = +1).

        Numerics
        --------
        Run in fp32 under an explicit ``autocast(enabled=False)`` block
        because ``torch.linalg.qr`` / ``torch.linalg.det`` do not have
        a CUDA bf16 kernel; mirrors the pattern in :func:`_maybe_kabsch`.
        Cost is ~one 3x3 QR + 3x3 det per sample -- effectively free.

        Parameters
        ----------
        x : Tensor
            Coords of shape ``(B, L, A, 3)``. Assumed already centered.
        seq_mask : Tensor
            Shape ``(B, L)``. Used to re-zero pad positions after the
            rotation (rotations of zero stay zero; this is cosmetic
            cleanup for any post-centering numerical fuzz at pads).
        training : bool
            Train-only gate. Returns ``x`` unchanged at val/test.
        """
        if not training or not self.random_so3_aug:
            return x

        B = x.shape[0]
        device = x.device
        orig_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            M = torch.randn(B, 3, 3, device=device, dtype=torch.float32)
            Q, R = torch.linalg.qr(M)
            # Uniform on O(3) via sign(diag(R)) trick.
            signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            # signs: (B, 3); unsqueeze(-2) broadcasts as column-wise sign.
            Q = Q * signs.unsqueeze(-2)
            # Project O(3) -> SO(3): flip first column where det = -1.
            det = torch.linalg.det(Q)
            col0_mult = (1.0 - 2.0 * (det < 0).to(torch.float32)).unsqueeze(-1)
            Q = Q.clone()
            Q[..., 0] = Q[..., 0] * col0_mult

            x_flat = x.reshape(B, -1, 3).to(torch.float32)
            x_rot = torch.bmm(x_flat, Q).reshape_as(x).to(orig_dtype)

        m = seq_mask[..., None, None].to(x_rot.dtype)
        return x_rot * m

    def _maybe_kabsch(self, x0: Tensor, x1: Tensor, seq_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Kabsch-align ``x0`` to ``x1`` per sample on flattened atoms.

        moco's KabschAugmentation expects ``(B, N, dim)`` with a single
        mask vector per sample, so we flatten the per-residue atoms into
        ``L*A`` nodes and expand the mask before calling and reshape back.

        CUDA's batched SVD (``svd_cuda_gesvdjBatched``) has no bf16 kernel,
        so under ``bf16-mixed`` autocast the alignment must temporarily
        upcast to fp32. The rotation matrix returned by Kabsch is exact
        regardless of dtype, so this is a 3x3 SVD per sample done in fp32
        -- effectively free vs. the model forward.
        """
        if self.flow_matcher.augmentation_type is None:
            return x0, x1
        B, L, A, D = x1.shape
        flat_mask = seq_mask.unsqueeze(-1).expand(-1, -1, A).reshape(B, L * A).float()
        orig_dtype = x1.dtype
        x0_flat = x0.reshape(B, L * A, D).float()
        x1_flat = x1.reshape(B, L * A, D).float()
        device_type = "cuda" if x1.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x0_flat, x1_flat = self.flow_matcher.apply_augmentation(x0_flat, x1_flat, mask=flat_mask)
        return (
            x0_flat.reshape(B, L, A, D).to(orig_dtype),
            x1_flat.reshape(B, L, A, D).to(orig_dtype),
        )

    def _moco_loss(
        self,
        model_pred: Tensor,
        x_1: Tensor,
        x_0: Tensor,
        t: Tensor,
        x_t: Tensor,
        seq_mask: Tensor,
    ) -> Tensor:
        """Per-atom flow-matching loss.

        Dispatches ``target_type`` from the moco interpolant's
        ``prediction_type`` so the velocity variant is a pure YAML flip:

        - ``prediction_type=DATA``: target=``x_1``, ``target_type=DATA``.
          moco applies the ``1/(1-t)^2`` reweight (Proteina's data-pred
          recipe).
        - ``prediction_type=VELOCITY``: target=``x_1 - x_0``,
          ``target_type=VELOCITY``. moco does NOT apply the reweight
          (Proteina's ``target_pred=v`` recipe).

        Reshapes to ``(B, L*A, D)`` so moco's ``mask.unsqueeze(-1)``
        broadcasts correctly over the atom axis.
        """
        B, L, A, D = x_1.shape
        flat_mask = seq_mask.unsqueeze(-1).expand(-1, -1, A).reshape(B, L * A).to(x_1.dtype)
        # `calculate_target` returns x_1 for DATA, x_1 - x_0 for VELOCITY.
        # Match `target_type` to the interpolant's `prediction_type` so the
        # natural Proteina-style pairing is the default for both runs.
        target = self.flow_matcher.calculate_target(x_1, x_0)
        target_type = self.flow_matcher.prediction_type
        per_sample = self.flow_matcher.loss(
            model_pred.reshape(B, L * A, D),
            target.reshape(B, L * A, D),
            t=t,
            xt=x_t.reshape(B, L * A, D),
            mask=flat_mask,
            target_type=target_type,
        )
        return per_sample.mean()

    def _single_step(self, batch: dict, split: str) -> dict:
        states, seq_mask, residue_index = self.featurize(batch)
        device = states.device

        x_1 = batch["coords_res"].to(device).to(torch.get_default_dtype())
        B, L, A, D = x_1.shape

        if self.center_x1:
            x_1 = _center_geometric(x_1, seq_mask, self.n_atoms)

        # Scale to the prior's units (Proteina: Angstrom -> nm, factor 0.1) so the
        # flow is well-conditioned (data std ~ unit Gaussian). Applied after
        # centering; commutes with the SO(3) rotation below. All downstream tensors
        # (x_0 prior, x_t, the loss target, the decoder I/O) live in this scaled space.
        x_1 = x_1 * self.coord_scale

        # Random SO(3) augmentation on GT (train-only). MUST come after
        # centering and BEFORE Kabsch+interp so the rotated x_1 is what
        # `x_t`, the decoder input, and the loss target all consume --
        # the loss is then naturally computed against the rotated GT.
        x_1 = self._apply_random_so3_aug(x_1, seq_mask, training=(split == "train"))

        states_in = self._maybe_cfg_dropout(states, training=(split == "train"))

        x_0 = self._sample_prior_centered((B, L, A, D), seq_mask, device)
        x_0, x_1 = self._maybe_kabsch(x_0, x_1, seq_mask)

        t = self.time_distribution.sample(B, device=device)

        # interpolate per-atom -- moco's interpolate is shape-agnostic for the
        # leading data dims, so we can feed the 4D tensor directly.
        t_expanded = t.view(B, 1, 1, 1)
        x_t = x_1 * t_expanded + x_0 * (1.0 - t_expanded)

        decoder_name, decoder = self._decoder

        # Self-conditioning warm forward (ESMFold2 / Karras EDM trick).
        # With probability `selfcond_train_prob`, run a no-grad forward
        # pass first to get a stale prediction, convert it to DATA space
        # (so the self-cond carry is always "model's previous estimate
        # of the clean x_1" regardless of `prediction_type`), then
        # condition the actual grad-bearing forward on the detached
        # data-space estimate. The other ``1 - selfcond_train_prob``
        # fraction trains the model with ``x_selfcond=None`` so it
        # remains robust to the cold-start case at inference step 0.
        # Costs ~+50% per-step compute on average (one warm forward at
        # no-grad ~ half cost of grad forward).
        #
        # Step U refinement: passing the data-space estimate instead of
        # the raw model output keeps the semantic of self-conditioning
        # consistent across DATA and VELOCITY parametrisations. For DATA
        # prediction, `process_data_prediction` is a no-op, so this is
        # byte-identical to the prior behavior (`flow_selfcond` is
        # unaffected). For VELOCITY prediction, the conversion is
        # `xt + (1-t)*v`, i.e. the "what would x_1 look like if I
        # straight-line integrated from x_t at time t".
        x_selfcond = None
        if self.use_self_conditioning and split == "train" and torch.rand((), device=device) < self.selfcond_train_prob:
            with torch.no_grad():
                prev_out = decoder(
                    states_in,
                    seq_mask,
                    residue_index=residue_index,
                    xt=x_t,
                    time_cond=t,
                    x_selfcond=None,
                )
                # Decoder may return a dict if a distogram head is enabled;
                # only the coords are used for self-conditioning.
                model_pred_prev, _, _ = _split_decoder_output(prev_out)
                x_selfcond = (
                    self.flow_matcher.process_data_prediction(
                        model_pred_prev.reshape(B, L * A, D),
                        x_t.reshape(B, L * A, D),
                        t=t,
                    )
                    .reshape(B, L, A, D)
                    .detach()
                )

        decoder_out = decoder(
            states_in,
            seq_mask,
            residue_index=residue_index,
            xt=x_t,
            time_cond=t,
            x_selfcond=x_selfcond,
        )
        model_pred, distogram_logits, three_di_logits = _split_decoder_output(decoder_out)

        # Data-space view of the model output, used for the aux pairwise
        # loss and the `x_recon` returned to BackboneReconstruction. No-op
        # for `prediction_type=DATA`; converts via `xt + (1-t)*v` for
        # `prediction_type=VELOCITY` (Step S).
        x_1_hat_data = self.flow_matcher.process_data_prediction(
            model_pred.reshape(B, L * A, D),
            x_t.reshape(B, L * A, D),
            t=t,
        ).reshape(B, L, A, D)

        fm_loss = self._moco_loss(model_pred, x_1, x_0, t, x_t, seq_mask)
        aux_loss = _pairwise_l2(x_1_hat_data, x_1, seq_mask)
        total_loss = fm_loss + self.aux_pairwise_l2_weight * aux_loss

        log_payload = {
            f"{split}_loss": total_loss,
            f"{split}_fm_loss": fm_loss,
            f"{split}_aux_pairwise_l2": aux_loss,
        }

        # Proteina-style distogram aux loss: cross-entropy on bucketised
        # GT Cα-Cα distances against the pair head's logits. Active only
        # when `t > aux_distogram_t_lim` (per-sample gate) and when the
        # decoder actually emits a distogram. `x_1` is in the scaled coord
        # space (`coord_scale` applied), so we bucket on a scaled bin edge.
        if self.aux_distogram_weight > 0.0 and distogram_logits is not None:
            gt_ca = x_1[:, :, 1, :]  # (B, L, 3) -- Cα at atom index 1
            gt_dists = torch.cdist(gt_ca, gt_ca, p=2)  # (B, L, L)
            max_dist_scaled = self.max_dist_a * self.coord_scale
            distogram_per_sample = _distogram_ce(
                distogram_logits,
                gt_dists,
                seq_mask,
                num_dist_buckets=self.num_dist_buckets,
                max_dist_scaled=max_dist_scaled,
            )  # (B,)
            t_gate = (t.detach().flatten() > self.aux_distogram_t_lim).to(distogram_per_sample.dtype)
            distogram_loss = (distogram_per_sample * t_gate).mean()
            distogram_loss_no_gate = distogram_per_sample.mean()
            total_loss = total_loss + self.aux_distogram_weight * distogram_loss
            log_payload[f"{split}_aux_distogram"] = distogram_loss
            log_payload[f"{split}_aux_distogram_no_gate"] = distogram_loss_no_gate
            # Reassign to surface the updated total in logs.
            log_payload[f"{split}_loss"] = total_loss

        # 3Di-token classification aux loss: cross-entropy from per-token
        # features back to the GT 3Di tokens. Targets are the PRE-CFG-dropout
        # tokens from `featurize` (`states`), so the head is supervised on
        # the original 3Di sequence even on samples whose decoder INPUT was
        # null'd via `p_uncond` -- those samples are exactly where the head
        # has to learn coord+time -> 3Di without the input-embedding skip
        # carrying the answer. `ignore_index=num_3di_classes` masks padding
        # / null targets out of the CE.
        if self.aux_3di_ce_weight > 0.0 and three_di_logits is not None:
            per_token_ce = F.cross_entropy(
                three_di_logits.permute(0, 2, 1),  # (B, K=20, L) for spatial CE
                states.long(),  # (B, L), targets in [0, 19] or num_3di_classes for pad
                reduction="none",
                ignore_index=self.num_3di_classes,
            )  # (B, L); zeros at ignore positions
            mask_valid = (states != self.num_3di_classes).to(per_token_ce.dtype) * seq_mask
            per_sample_ce = per_token_ce * mask_valid
            denom = mask_valid.sum(dim=1).clamp_min(1.0)
            per_sample_3di_ce = per_sample_ce.sum(dim=1) / denom  # (B,)
            if self.aux_3di_t_lim >= 0.0:
                t_gate_3di = (t.detach().flatten() > self.aux_3di_t_lim).to(per_sample_3di_ce.dtype)
                three_di_loss = (per_sample_3di_ce * t_gate_3di).mean()
            else:
                three_di_loss = per_sample_3di_ce.mean()
            three_di_loss_no_gate = per_sample_3di_ce.mean()
            total_loss = total_loss + self.aux_3di_ce_weight * three_di_loss
            # Mean per-token accuracy of the 3Di head, averaged over valid
            # positions only -- cheap diagnostic to see if the head is
            # actually solving the trivial cond-branch problem.
            with torch.no_grad():
                pred_3di = three_di_logits.argmax(dim=-1)  # (B, L)
                correct = (pred_3di == states) & (mask_valid > 0)
                acc_denom = mask_valid.sum().clamp_min(1.0)
                three_di_acc = correct.to(per_sample_3di_ce.dtype).sum() / acc_denom
            log_payload[f"{split}_aux_3di_ce"] = three_di_loss
            log_payload[f"{split}_aux_3di_ce_no_gate"] = three_di_loss_no_gate
            log_payload[f"{split}_aux_3di_acc"] = three_di_acc
            log_payload[f"{split}_loss"] = total_loss

        # 3Di-CE-FROM-COORDS aux loss: encode the predicted Cα geometry
        # through the (frozen) Foldseek mini3di pipeline and CE against
        # the GT 3Di tokens. Differentiable wrt Cα coords, so this term
        # directly shapes the coord head -- distinct from the per-token
        # `aux_3di_ce_weight` head which shapes the U-ViT features and
        # has a trivial cond-branch path through the input-embedding
        # skip. We freeze the partner-index from GT (computed once by
        # `Structure3diTransform` and surfaced in `batch["3di_partner_index"]`)
        # so the non-differentiable argmin over virtual-center distances
        # is sidestepped.
        if self.aux_3di_coord_ce_weight > 0.0 and self.mini3di_torch is not None and "3di_partner_index" in batch:
            gt_partner_index = batch["3di_partner_index"].to(device=device, dtype=torch.long)  # (B, L)
            gt_3di = states.long()  # (B, L), pre-CFG-dropout, num_3di_classes at pad
            # mini3di operates on Angstrom-scale coords (the Foldseek
            # VAE was trained on raw distances in A); our internal
            # `x_1_hat_data` is in scaled space (multiplied by
            # coord_scale, so 0.1 A i.e. nm-ish units). Un-scale before
            # encoding so the descriptor distances land in the model's
            # training distribution.
            ca_pred_a = x_1_hat_data[:, :, 1, :] / self.coord_scale  # (B, L, 3)

            # Per-sample loop: the descriptor pipeline uses
            # `index_select(0, J)` which is per-chain; vectorising via
            # `torch.gather` is doable but the per-sample cost is small
            # (a couple of small matmuls + the VAE forward), so the
            # loop stays simple here. The eager loop also keeps the
            # autograd graph clean: each sample's CE is summed into the
            # batch loss before backward is called.
            per_sample_3di_coord_ce = []
            for b in range(B):
                Lb = int(seq_mask[b].sum().item())
                if Lb < 3:
                    per_sample_3di_coord_ce.append(torch.zeros((), device=device, dtype=ca_pred_a.dtype))
                    continue
                ca_b = ca_pred_a[b, :Lb]  # (Lb, 3)
                pi_b = gt_partner_index[b, :Lb]  # (Lb,) long, J in [1, Lb-2]
                gt_b = gt_3di[b, :Lb]  # (Lb,) long
                # The torch encoder's `compute_descriptors` uses J-1 / J / J+1
                # which assumes J in [1, Lb-2]. The numpy `_find_residue_partners`
                # always satisfies this for the unpadded chain length we passed
                # at transform time, so `pi_b` is safe over [:Lb].
                logits_b = self.mini3di_torch(
                    ca_b,
                    partner_index=pi_b,
                    temperature=self.aux_3di_coord_ce_temperature,
                )  # (Lb, num_3di_classes=20)
                ce_b = F.cross_entropy(
                    logits_b,
                    gt_b,
                    reduction="mean",
                    ignore_index=self.num_3di_classes,
                )
                per_sample_3di_coord_ce.append(ce_b)
            per_sample_3di_coord_ce_t = torch.stack(per_sample_3di_coord_ce)  # (B,)
            if self.aux_3di_coord_ce_t_lim >= 0.0:
                t_gate_coord = (t.detach().flatten() > self.aux_3di_coord_ce_t_lim).to(per_sample_3di_coord_ce_t.dtype)
                three_di_coord_loss = (per_sample_3di_coord_ce_t * t_gate_coord).mean()
            else:
                three_di_coord_loss = per_sample_3di_coord_ce_t.mean()
            three_di_coord_loss_no_gate = per_sample_3di_coord_ce_t.mean()
            total_loss = total_loss + self.aux_3di_coord_ce_weight * three_di_coord_loss
            # Diagnostic: argmax accuracy of the predicted-coords-derived
            # 3Di tokens vs GT, averaged over valid positions across the
            # batch. Recovery climbs slowly while Kabsch is bad and
            # surges as the geometry locks in -- a more sensitive signal
            # than the loss for "is the prediction structural yet?".
            with torch.no_grad():
                acc_num = 0.0
                acc_den = 0.0
                for b in range(B):
                    Lb = int(seq_mask[b].sum().item())
                    if Lb < 3:
                        continue
                    ca_b = ca_pred_a[b, :Lb]
                    pi_b = gt_partner_index[b, :Lb]
                    logits_b = self.mini3di_torch(
                        ca_b,
                        partner_index=pi_b,
                        temperature=self.aux_3di_coord_ce_temperature,
                    )
                    pred_b = logits_b.argmax(dim=-1)
                    gt_b = gt_3di[b, :Lb]
                    valid_b = gt_b != self.num_3di_classes
                    acc_num += (pred_b[valid_b] == gt_b[valid_b]).sum().item()
                    acc_den += valid_b.sum().item()
                three_di_coord_acc = (
                    torch.tensor(acc_num / max(acc_den, 1.0), device=device)
                    if acc_den > 0
                    else torch.zeros((), device=device)
                )
            log_payload[f"{split}_aux_3di_coord_ce"] = three_di_coord_loss
            log_payload[f"{split}_aux_3di_coord_ce_no_gate"] = three_di_coord_loss_no_gate
            log_payload[f"{split}_aux_3di_coord_acc"] = three_di_coord_acc
            log_payload[f"{split}_loss"] = total_loss

        self.log_dict(log_payload, batch_size=B, sync_dist=True)

        if self.automatic_optimization is False:
            self.manual_backward(total_loss)
            self.trainer.optimizers[0].step()

        # `x_1_hat_data` is the data-space prediction in the scaled (nm) space used
        # for the loss above. The recon callback / `BackboneReconstruction` compare
        # `x_recon` against the Angstrom ground truth (`coords_res`), so de-scale the
        # *output* copy back to Angstrom (matching `sample()` and the deterministic
        # tokenizer). The loss is unaffected since it already consumed the raw
        # `model_pred` in the appropriate prediction-type space (DATA or VELOCITY).
        return {"loss": total_loss, "x_recon": {decoder_name: x_1_hat_data / self.coord_scale}}

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self._single_step(batch, split="train")

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self._single_step(batch, split="val")

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Tolerate ckpts saved before new auxiliary heads were added.

        Lightning's `Trainer.fit(ckpt_path=...)` calls
        `pl_module.load_state_dict(state, strict=True)` by default. When
        we resume an older ckpt into a variant that adds a new head
        (e.g. `decoder.distogram_head.*`), the saved state_dict is
        MISSING those keys and the strict load raises. This hook patches
        `state_dict` in place:

        - Missing keys get filled from the just-instantiated model's
          state_dict, so the new head starts at its init values
          (zero-init final layer for the distogram head; default init
          for the hidden layer and any future heads).
        - Unexpected keys are dropped with a warning -- reserved for
          future arch shrinks; never fired by the current code path.

        Both branches are silent at the Lightning level after this hook;
        the subsequent strict load only sees keys that match.
        """
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            return
        own_state = self.state_dict()
        missing = [k for k in own_state if k not in state]
        extra = [k for k in state if k not in own_state]
        if missing:
            logger.warning(
                "ckpt is missing %d key(s); filling from current init (first 5: %s%s)",
                len(missing),
                missing[:5],
                " ..." if len(missing) > 5 else "",
            )
            for k in missing:
                state[k] = own_state[k].clone()
        if extra:
            logger.warning(
                "ckpt has %d unexpected key(s); dropping (first 5: %s%s)",
                len(extra),
                extra[:5],
                " ..." if len(extra) > 5 else "",
            )
            for k in extra:
                del state[k]

    def configure_optimizers(self):
        optimizer = self.optim_factory(params=self.parameters())

        # Only forward the kwargs the configured scheduler actually accepts.
        # `get_cosine_schedule_with_warmup` wants `num_training_steps` (decays
        # to 0 over that horizon), but `get_constant_schedule_with_warmup`
        # (warmup -> flat LR) takes only `num_warmup_steps` and raises on the
        # extra kwarg. Filtering here lets both share one code path / yaml shape.
        scheduler_kwargs = {
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
        }
        scheduler_fn = self.lr_scheduler.func if isinstance(self.lr_scheduler, functools.partial) else self.lr_scheduler
        try:
            accepted = set(inspect.signature(scheduler_fn).parameters)
            scheduler_kwargs = {k: v for k, v in scheduler_kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            pass

        scheduler = self.lr_scheduler(optimizer=optimizer, **scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Inference helper: one-shot sample with default knobs."""
        states, seq_mask, residue_index = self.featurize(batch)
        return self.sample(states=states, seq_mask=seq_mask, residue_index=residue_index)

    @torch.no_grad()
    def sample(
        self,
        states: Tensor,
        seq_mask: Tensor,
        residue_index: Tensor | None = None,
        *,
        guidance_scale: float | None = None,
        n_steps: int | None = None,
        sampling_mode: str | None = None,
        sc_scale_noise: float | None = None,
        sc_scale_score: float | None = None,
        autoguidance_model: torch.nn.Module | None = None,
        autoguidance_ratio: float | None = None,
        schedule_type: str | None = None,
        schedule_exponent: float | None = None,
        min_t: float = 0.0,
        gt_mode: str = "tan",
        gt_p: float = 1.0,
        gt_clamp: float | None = None,
        t_lim_ode: float = 0.99,
        init_temperature: float = 1.0,
    ) -> Tensor:
        """Run an ODE (or SDE) integration to sample ``x_1`` conditional on 3Di tokens.

        Parameters
        ----------
        states, seq_mask, residue_index
            Output of :meth:`featurize` -- in inference the caller is
            responsible for the pad-class sentinel.
        guidance_scale
            CFG weight; if ``None`` defaults to ``self.guidance_scale``.
        n_steps
            Number of integration steps; if ``None`` defaults to
            ``self.n_sampling_steps``. If it differs from
            ``self.inference_schedule.nsteps`` a fresh
            :class:`LinearInferenceSchedule` is built on the fly.
        sampling_mode
            ``"ode"`` (Euler) or ``"sde"`` (score-stochastic). Default
            ``self.sampling_mode``.
        sc_scale_noise, sc_scale_score
            Knobs for the SDE branch (``noise_temperature`` and
            ``score_temperature`` in moco's
            :meth:`ContinuousFlowMatcher.step_score_stochastic`). Ignored
            when ``sampling_mode='ode'``.
        autoguidance_model, autoguidance_ratio
            Optional Proteina-style autoguidance hook. When both are
            null (the default) the sampling path is pure CFG; when a
            ``nn.Module`` is supplied, its prediction is blended into
            the guided combine as
            ``v_guided = w * v_cond + (1-w) * (ag_r * v_ag + (1-ag_r) * v_null)``.
        schedule_type
            Inference time-grid family: ``"linear"`` (default/uniform),
            ``"power"`` (steps ~ ``u**exponent``), or ``"log"`` (dense near
            ``t->1``). When ``None`` the existing ``self.inference_schedule``
            is reused if ``n_steps`` matches, else a uniform linear grid.
        schedule_exponent
            Exponent for ``power`` (default 2.0) / ``log`` (default -2.0).
        min_t
            Lower bound of the time grid -- start integration at ``t=min_t``
            instead of 0 (skips the highest-noise region).
        gt_mode, gt_p, gt_clamp, t_lim_ode
            SDE-only noise-injection schedule controls forwarded to moco's
            :meth:`ContinuousFlowMatcher.step_score_stochastic`
            (``gt_mode`` shape, power transform, clamp, and the ``t``
            above which the SDE switches to a pure ODE step).
        init_temperature
            Scale applied to the initial prior sample ``x_0`` (low-temperature
            init when ``< 1``). Integration units are unchanged.

        Notes
        -----
        SDE sampling (``sampling_mode='sde'``) requires
        ``flow_matcher.augmentation_type is None`` (moco forbids the
        vector-field->score conversion under OT/Kabsch coupling); callers
        must null it before sampling if the model was trained with Kabsch.
        """
        device = states.device
        B, L = states.shape
        A = self.n_atoms
        D = 3

        if residue_index is None:
            residue_index = torch.arange(L, device=device)[None].expand(B, -1)

        w = guidance_scale if guidance_scale is not None else self.guidance_scale
        n = n_steps if n_steps is not None else self.n_sampling_steps
        mode = sampling_mode if sampling_mode is not None else self.sampling_mode
        sc_n = sc_scale_noise if sc_scale_noise is not None else self.sc_scale_noise
        sc_s = sc_scale_score if sc_scale_score is not None else self.sc_scale_score
        ag_model = autoguidance_model if autoguidance_model is not None else self.autoguidance_model
        ag_ratio = autoguidance_ratio if autoguidance_ratio is not None else self.autoguidance_ratio

        x_t = self._sample_prior_centered((B, L, A, D), seq_mask, device)
        if init_temperature != 1.0:
            x_t = x_t * init_temperature

        if schedule_type is None and min_t == 0.0:
            # Back-compat fast path: reuse the configured schedule when n matches.
            sched = (
                self.inference_schedule if n == self.inference_schedule.nsteps else LinearInferenceSchedule(nsteps=n)
            )
        else:
            st = (schedule_type or "linear").lower()
            if st == "linear":
                sched = LinearInferenceSchedule(nsteps=n, min_t=min_t)
            elif st == "power":
                sched = PowerInferenceSchedule(
                    nsteps=n, exponent=schedule_exponent if schedule_exponent is not None else 2.0, min_t=min_t
                )
            elif st == "log":
                sched = LogInferenceSchedule(
                    nsteps=n, exponent=schedule_exponent if schedule_exponent is not None else -2.0, min_t=min_t
                )
            else:
                raise ValueError(f"Unknown schedule_type={schedule_type!r}; expected 'linear', 'power', or 'log'.")
        time_steps = sched.generate_schedule(device=device)
        dts = sched.discretize(device=device)

        flat_mask = seq_mask.unsqueeze(-1).expand(-1, -1, A).reshape(B, L * A).to(x_t.dtype)
        null_states = torch.full_like(states, self.num_3di_classes)
        decoder_name, decoder = self._decoder
        use_cfg = (w != 1.0) or (ag_model is not None)

        # Self-conditioning carry-over. ``None`` at step 0 (no prior
        # estimate exists), then re-bound to the previous step's guided
        # x_1 prediction. Tokenized this way (one tensor, shared between
        # cond / null / autoguidance branches) the cost is one extra
        # input projection per decoder call -- no extra forward.
        x_selfcond = None

        for t_scalar, dt_scalar in zip(time_steps, dts):
            t_batch = sched.pad_time(B, float(t_scalar.item()), device=device)
            dt_batch = torch.full((B,), float(dt_scalar.item()), device=device)

            x1_cond, _, _ = _split_decoder_output(
                decoder(
                    states,
                    seq_mask,
                    residue_index=residue_index,
                    xt=x_t,
                    time_cond=t_batch,
                    x_selfcond=x_selfcond,
                )
            )

            if use_cfg:
                x1_null, _, _ = _split_decoder_output(
                    decoder(
                        null_states,
                        seq_mask,
                        residue_index=residue_index,
                        xt=x_t,
                        time_cond=t_batch,
                        x_selfcond=x_selfcond,
                    )
                )
                if ag_model is not None and ag_ratio != 0.0:
                    x1_ag, _, _ = _split_decoder_output(
                        ag_model(
                            states,
                            seq_mask,
                            residue_index=residue_index,
                            xt=x_t,
                            time_cond=t_batch,
                            x_selfcond=x_selfcond,
                        )
                    )
                    x1_pred = w * x1_cond + (1.0 - w) * (ag_ratio * x1_ag + (1.0 - ag_ratio) * x1_null)
                else:
                    x1_pred = x1_null + w * (x1_cond - x1_null)
            else:
                x1_pred = x1_cond

            x_t_flat = x_t.reshape(B, L * A, D)
            x1_pred_flat = x1_pred.reshape(B, L * A, D)

            # Carry the guided prediction forward as the next step's
            # self-cond input. Convert to DATA space here -- BEFORE
            # `step()` overwrites `x_t_flat` -- so the carry uses the
            # pre-step `x_t` (which matches the prediction's reference
            # point). For DATA prediction this is a no-op; for VELOCITY
            # prediction it is `xt + (1-t)*v`. Step U refinement: keeps
            # the self-cond semantic ("model's previous estimate of x_1")
            # consistent across parametrisations. Detached defensively so
            # no gradient flows even under future enable_grad callers.
            if self.use_self_conditioning:
                x_selfcond = (
                    self.flow_matcher.process_data_prediction(x1_pred_flat, x_t_flat, t=t_batch)
                    .reshape(B, L, A, D)
                    .detach()
                )

            if mode == "ode":
                x_t_flat = self.flow_matcher.step(
                    x1_pred_flat, x_t_flat, dt_batch, t_batch, mask=flat_mask, center=False
                )
            elif mode == "sde":
                x_t_flat = self.flow_matcher.step_score_stochastic(
                    x1_pred_flat,
                    x_t_flat,
                    dt_batch,
                    t_batch,
                    mask=flat_mask,
                    gt_mode=gt_mode,
                    gt_p=gt_p,
                    gt_clamp=gt_clamp,
                    noise_temperature=sc_n,
                    score_temperature=sc_s if sc_s > 0 else 1.0,
                    t_lim_ode=t_lim_ode,
                    center=False,
                )
            else:
                raise ValueError(f"Unknown sampling_mode={mode!r}; expected 'ode' or 'sde'.")

            x_t = x_t_flat.reshape(B, L, A, D)
            # Always re-center intermediate states. This is a CoM-free geometric
            # flow (prior is centered, x_1 is centered, the target field is
            # CoM-free), so any CoM drift injected by a non-CoM-free x1_pred or by
            # the SDE noise is spurious and must be removed every step -- matching
            # La-Proteina, which centers bb_ca at every integration step. Done with
            # our geometric CoM (over all atoms) since `center=False` is passed to
            # moco above (moco's `center=True` would average over the atom axis).
            x_t = _center_geometric(x_t, seq_mask, self.n_atoms)

            # (`x_selfcond` was already updated above, BEFORE the moco
            # step() overwrote `x_t_flat`, so the data-space conversion
            # used the pre-step `x_t` which is the correct reference
            # point for converting v -> xt + (1-t)*v.)

        # Undo the train-time coordinate scaling so callers get coordinates back in
        # the original units (Angstrom). Integration happened entirely in the scaled
        # (nm) space the model was trained on; only the final output is rescaled.
        return x_t / self.coord_scale
