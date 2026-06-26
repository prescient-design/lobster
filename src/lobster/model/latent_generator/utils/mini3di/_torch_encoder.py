"""Differentiable torch port of mini3di's Encoder.

Reproduces the numpy pipeline in `_encoder.py` step-by-step in torch so
gradients flow from input Cα coordinates through to per-residue 3Di
state logits. Used by `Tokenizer3diInputFlow` to attach a
"3Di-CE-from-coords" auxiliary loss: the predicted Cα geometry is
encoded through the same Foldseek Markov-state pipeline that produces
the GT 3Di tokens, then trained with cross-entropy against those tokens.

Two non-differentiable steps in the numpy pipeline:

1. `PartnerIndexEncoder._find_residue_partners` (`argmin` over the
   pairwise virtual-center distance matrix). Sidestepped by accepting
   ``partner_index`` as an INPUT (precomputed from GT) rather than
   recomputing it on every prediction. Frozen-from-GT partner indices
   give a stronger supervision signal anyway: "match the GT 3Di state at
   the GT-defined partner pair", which is the exact target the eval
   computes.

2. `CentroidLayer.__call__` (`argmin` over the 20-class centroid
   distance matrix). Replaced with a softmax over ``-D / temperature``
   so the head emits 20-class logits suitable for `F.cross_entropy`.

Equivalence:
    With ``partner_index`` recomputed from VC (still differentiable for
    the geometry but with an internal argmin) and ``hard=True``, the
    forward returns the SAME state indices as the numpy `Encoder` on
    real protein structures. See
    `tests/lobster/model/latent_generator/utils/mini3di/test_torch_encoder_equivalence.py`
    for the bit-for-bit check.
"""

from __future__ import annotations

import math
from importlib.resources import files as resource_files
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lobster.model.latent_generator.utils.mini3di import _unkerasify
from lobster.model.latent_generator.utils.mini3di._layers import DenseLayer

# Same numerical constants as the numpy pipeline (see VirtualCenterEncoder
# / Encoder in `_encoder.py`).
_THETA_DEG_DEFAULT = 270.0
_TAU_DEG_DEFAULT = 0.0
_DISTANCE_ALPHA_V_DEFAULT = 2.0
_DISTANCE_ALPHA_BETA_DEFAULT = 1.5336

# 3Di centroids, NxK = (20, 2). Verbatim from `Encoder._CENTROIDS` in
# `_encoder.py`. Written out so the two encoders can never silently diverge
# if upstream centroids are edited.
_CENTROIDS: np.ndarray = np.array(
    [
        [-1.0729, -0.3600],
        [-0.1356, -1.8914],
        [0.4948, -0.4205],
        [-0.9874, 0.8128],
        [-1.6621, -0.4259],
        [2.1394, 0.0486],
        [1.5558, -0.1503],
        [2.9179, 1.1437],
        [-2.8814, 0.9956],
        [-1.1400, -2.0068],
        [3.2025, 1.7356],
        [1.7769, -1.3037],
        [0.6901, -1.2554],
        [-1.1061, -1.3397],
        [2.1495, -0.8030],
        [2.3060, -1.4988],
        [2.5522, 0.6046],
        [0.7786, -2.1660],
        [-2.3030, 0.3813],
        [1.0290, 0.8772],
    ],
    dtype=np.float32,
)
NUM_3DI_CLASSES = _CENTROIDS.shape[0]


def _load_kerasify_dense_layers() -> list[DenseLayer]:
    """Load the foldseek 3Di VAE Dense layers from the bundled kerasify file."""
    weights_path = None
    try:
        weights_path = resource_files("lobster.model.latent_generator.utils.mini3di").joinpath(
            "encoder_weights_3di.kerasify"
        )
        with open(str(weights_path), "rb") as f:
            return _unkerasify.load(f)
    except (AttributeError, FileNotFoundError):
        pass
    here = Path(__file__).resolve().parent
    fallback = here / "encoder_weights_3di.kerasify"
    if not fallback.exists():
        raise FileNotFoundError(f"could not locate encoder_weights_3di.kerasify (tried {weights_path}, {fallback})")
    with open(fallback, "rb") as f:
        return _unkerasify.load(f)


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Same convention as `_utils.normalize` (numpy): divide each row by
    its norm except where the norm is exactly zero (matches numpy's
    ``where=norm != 0`` clause).
    """
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


class MiniThreeDiEncoderTorch(nn.Module):
    """Torch port of `lobster.model.latent_generator.utils.mini3di.Encoder`.

    Operates on a single chain at a time: ``ca, cb, n`` are ``(L, 3)``.
    For batched calls, vmap or loop externally; the geometry is per-chain.

    Parameters mirror the numpy `VirtualCenterEncoder` defaults (Foldseek's
    selected values).

    Notes
    -----
    Weights of the Dense layers and the centroids are registered as
    buffers (not parameters): they're frozen Foldseek-trained values, not
    something we want gradient-descent to update.
    """

    def __init__(
        self,
        *,
        theta_deg: float = _THETA_DEG_DEFAULT,
        tau_deg: float = _TAU_DEG_DEFAULT,
        distance_alpha_v: float = _DISTANCE_ALPHA_V_DEFAULT,
        distance_alpha_beta: float = _DISTANCE_ALPHA_BETA_DEFAULT,
    ) -> None:
        super().__init__()
        self.theta_rad = math.radians(theta_deg)
        self.tau_rad = math.radians(tau_deg)
        self._cos_theta = math.cos(self.theta_rad)
        self._sin_theta = math.sin(self.theta_rad)
        self._cos_tau = math.cos(self.tau_rad)
        self._sin_tau = math.sin(self.tau_rad)
        self.distance_alpha_v = float(distance_alpha_v)
        self.distance_alpha_beta = float(distance_alpha_beta)

        # Load VAE Dense layers + register as buffers.
        layers = _load_kerasify_dense_layers()
        # Sanity: every layer is a DenseLayer with float32 weights.
        for i, layer in enumerate(layers):
            assert isinstance(layer, DenseLayer), f"layer {i} is not Dense"
        # Register weights / biases / activation flags. We expose them as a
        # ParameterList of buffers (registered tensors) so .to(device) works.
        self._n_layers = len(layers)
        self._activations: list[bool] = [bool(layer.activation) for layer in layers]
        for i, layer in enumerate(layers):
            self.register_buffer(f"_w{i}", torch.from_numpy(np.asarray(layer.weights, dtype=np.float32)))
            self.register_buffer(f"_b{i}", torch.from_numpy(np.asarray(layer.biases, dtype=np.float32)))

        # Centroids buffer.
        self.register_buffer("_centroids", torch.from_numpy(_CENTROIDS).clone())

    @property
    def num_classes(self) -> int:
        return NUM_3DI_CLASSES

    # ------------------------------------------------------------------
    # Geometry pipeline (matches `_encoder.py` step-by-step)
    # ------------------------------------------------------------------

    def compute_virtual_center(
        self,
        ca: torch.Tensor,
        cb: torch.Tensor,
        n: torch.Tensor,
    ) -> torch.Tensor:
        """Cα, Cβ, N -> per-residue virtual center V. ``(..., L, 3)``."""
        v = cb - ca
        a = cb - ca
        b = n - ca
        k = _normalize(torch.cross(a, b, dim=-1))
        v = (
            v * self._cos_theta
            + torch.cross(k, v, dim=-1) * self._sin_theta
            + k * (k * v).sum(dim=-1, keepdim=True) * (1.0 - self._cos_theta)
        )
        k2 = _normalize(n - ca)
        v = (
            v * self._cos_tau
            + torch.cross(k2, v, dim=-1) * self._sin_tau
            + k2 * (k2 * v).sum(dim=-1, keepdim=True) * (1.0 - self._cos_tau)
        )
        v = v * self.distance_alpha_v + ca
        return v

    def compute_partner_indices(self, vc: torch.Tensor) -> torch.Tensor:
        """Closest non-self virtual-center per residue (single chain).

        Mirrors `PartnerIndexEncoder._find_residue_partners`: endpoints
        (residues 0 and L-1) are masked out of the candidate set and
        the diagonal is set to +inf so each residue picks a different
        partner. Returns ``(L,)`` long.

        This op contains an ``argmin`` so it does NOT carry gradients
        through the partner identity. Use only at validation/test time
        or for the equivalence smoke-check; in training, supply
        ``partner_index`` from the precomputed GT.
        """
        if vc.dim() != 2 or vc.shape[-1] != 3:
            raise ValueError(f"compute_partner_indices expects (L, 3); got {tuple(vc.shape)}")
        L = vc.shape[0]
        # Pairwise squared distances mirroring the numpy version exactly.
        # Numpy does:
        #     r = (X*X).sum(-1).reshape(-1, 1)
        #     r[0] = r[-1] = nan
        #     D = r - 2 X @ X.T + r.T
        # Crucially, the SAME `r` (with NaN at endpoints) is used for both
        # the row-broadcast (r) and the column-broadcast (r.T). That
        # NaN-taints rows 0/L-1 AND columns 0/L-1 of D, so no residue can
        # pick an endpoint as its partner. Using a separate masked copy
        # for only the row would let argmin pick endpoints, which numpy
        # never does -- the partner-index drift from that subtle
        # mismatch was the source of the equivalence test's residual
        # descriptor / state diffs.
        r = (vc * vc).sum(dim=-1, keepdim=True)  # (L, 1)
        r_masked = r.clone()
        r_masked[0] = float("nan")
        r_masked[-1] = float("nan")
        D = r_masked - 2.0 * (vc @ vc.T) + r_masked.T  # (L, L)
        # Diagonal -> +inf so a residue can't pick itself.
        diag_idx = torch.arange(L, device=vc.device)
        D[diag_idx, diag_idx] = float("inf")
        # NaN -> +inf so masked endpoint rows/cols never win the argmin.
        D = torch.nan_to_num(D, nan=float("inf"))
        return D.argmin(dim=-1)

    def compute_descriptors(
        self,
        ca: torch.Tensor,
        partner_index: torch.Tensor,
    ) -> torch.Tensor:
        """Cα + partner index -> 10-D conformational descriptors.

        Returns ``(L, 10)`` float in ``ca.dtype``. Endpoints (residues
        0 and L-1) get all-zero descriptors, matching the numpy version's
        ``desc = numpy.zeros(...)`` initialiser.
        """
        if ca.dim() != 2 or ca.shape[-1] != 3:
            raise ValueError(f"compute_descriptors expects (L, 3); got {tuple(ca.shape)}")
        L = ca.shape[0]
        desc = torch.zeros(L, 10, dtype=ca.dtype, device=ca.device)
        if L < 3:
            return desc

        I = torch.arange(1, L - 1, device=ca.device)
        J = partner_index[I]
        # numpy's partner-index pipeline guarantees J in [1, L-2] (endpoint
        # residues are NaN-masked out of the candidate set), so J-1 / J+1
        # are valid indices into ca. We assume the input partner_index has
        # the same property here -- both compute_partner_indices() and the
        # GT-frozen indices we plumb through the datamodule will satisfy
        # this. If it ever doesn't, index_select will raise; better to
        # error loudly than silently compute wrong descriptors.
        u1 = _normalize(ca.index_select(0, I) - ca.index_select(0, I - 1))
        u2 = _normalize(ca.index_select(0, I + 1) - ca.index_select(0, I))
        u3 = _normalize(ca.index_select(0, J) - ca.index_select(0, J - 1))
        u4 = _normalize(ca.index_select(0, J + 1) - ca.index_select(0, J))
        u5 = _normalize(ca.index_select(0, J) - ca.index_select(0, I))

        # Per-feature in-place fill matches the numpy version's ordering;
        # the model weights expect this exact column order.
        desc[I, 0] = (u1 * u2).sum(dim=-1)
        desc[I, 1] = (u3 * u4).sum(dim=-1)
        desc[I, 2] = (u1 * u5).sum(dim=-1)
        desc[I, 3] = (u3 * u5).sum(dim=-1)
        desc[I, 4] = (u1 * u4).sum(dim=-1)
        desc[I, 5] = (u2 * u3).sum(dim=-1)
        desc[I, 6] = (u1 * u3).sum(dim=-1)
        desc[I, 7] = torch.linalg.norm(ca.index_select(0, I) - ca.index_select(0, J), dim=-1)
        diff = (J - I).to(desc.dtype)
        desc[I, 8] = diff.clamp(-4.0, 4.0)
        desc[I, 9] = torch.copysign(torch.log(diff.abs() + 1.0), diff)
        return desc

    # ------------------------------------------------------------------
    # VAE pipeline (Dense layers + centroid distance)
    # ------------------------------------------------------------------

    def vae_forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Run the (frozen) Foldseek VAE Dense layers. Input ``(L, 10)``,
        output ``(L, 2)``.
        """
        x = descriptors
        for i in range(self._n_layers):
            w = getattr(self, f"_w{i}")
            b = getattr(self, f"_b{i}")
            x = x @ w + b
            if self._activations[i]:
                x = F.relu(x)
        return x  # (L, 2)

    def centroid_logits(self, z: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        """Squared distances from each VAE embedding ``z`` (shape ``(L, 2)``)
        to the 20 centroids -> 20-class **logits** ``-D / temperature``.

        Substitutes the numpy `CentroidLayer.__call__` argmin: feeding
        these logits to ``F.cross_entropy`` gives a smooth, differentiable
        proxy for "the GT centroid is the closest one". With
        ``temperature -> 0`` the softmax over these logits collapses to
        the same one-hot the argmin would produce.
        """
        # Use squared distance (matches numpy `r1 - 2 X@C^T + r2`)
        # to avoid an unnecessary sqrt on a value we negate immediately.
        # z: (L, 2), centroids: (K=20, 2). Output: (L, K).
        # ||z - c||^2 = ||z||^2 - 2 z@c^T + ||c||^2
        z2 = (z * z).sum(dim=-1, keepdim=True)  # (L, 1)
        c2 = (self._centroids * self._centroids).sum(dim=-1)  # (K,)
        D2 = z2 - 2.0 * (z @ self._centroids.T) + c2  # (L, K)
        return -D2 / float(temperature)

    # ------------------------------------------------------------------
    # End-to-end forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ca: torch.Tensor,
        cb: torch.Tensor | None = None,
        n: torch.Tensor | None = None,
        partner_index: torch.Tensor | None = None,
        *,
        hard: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Run the (Cα, Cβ, N, partner_index) -> 3Di pipeline.

        Parameters
        ----------
        ca : Tensor of shape ``(L, 3)``
            Per-residue Cα coordinates.
        cb, n : Tensor of shape ``(L, 3)``, optional
            Per-residue Cβ and N coordinates. ONLY consumed when
            ``partner_index`` is None (the partner pipeline needs to
            recompute the virtual center). Pass them to reproduce the
            numpy `Encoder.encode_atoms` output end-to-end for the
            equivalence smoke-test; for training we precompute the
            partner index from GT coords and skip this branch.
        partner_index : Optional ``(L,)`` long
            If provided, used directly (recommended for training:
            descriptors only depend on Cα + partner_index, so gradients
            flow purely through Cα geometry without re-encoding Cβ/N).
            If None, recomputed via `compute_partner_indices` (which is
            non-differentiable on the index).
        hard : bool
            ``False`` (default) -> return ``(L, 20)`` logits. ``True`` ->
            return ``(L,)`` long state indices via argmin (matches numpy
            `Encoder.encode_atoms` for the equivalence smoke-test; not
            differentiable).
        temperature : float
            Softmax temperature for the centroid logits. ``1.0`` is
            standard; smaller values sharpen toward the argmin.
        """
        if partner_index is None:
            if cb is None or n is None:
                raise ValueError(
                    "MiniThreeDiEncoderTorch.forward needs either `partner_index` or both `cb` and `n` to recompute it."
                )
            vc = self.compute_virtual_center(ca, cb, n)
            partner_index = self.compute_partner_indices(vc)
        descriptors = self.compute_descriptors(ca, partner_index)
        z = self.vae_forward(descriptors)
        logits = self.centroid_logits(z, temperature=temperature)
        if hard:
            return logits.argmax(dim=-1)
        return logits
