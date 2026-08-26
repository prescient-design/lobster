"""Unit tests for the Protenix fold-consistency distillation prototype.

Two pieces are exercised:

1. The **structure-expert producer** (:mod:`lobster.rl_training.rewards._protenix_structure_expert`)
   — deriving 3Di tokens (τ*) from a backbone via the frozen mini3di encoder, deriving LG
   structure tokens (s*) via a codec callable, and assembling a binder-scoped expert from a
   parsed two-chain complex.
2. The **structure-track SFT loss** (:func:`lobster.rl_training._structure_sft.structure_sft_loss`)
   — the structural mirror of the CHORD sequence SFT. As in ``test_chord_sft.py`` the model's
   pure helpers (``_iter_traj_steps``, ``_expert_context_seq``, ``_sft_step_ce``) are bound onto
   a tiny stand-in model, so no real (heavy) model is built. The two guardrail properties are:
   invariance to the policy's revealed rollout tokens (expert-context overwrites them) and
   sensitivity to the expert target tokens.
"""

import numpy as np
import torch

from lobster.rl_training._structure_sft import structure_sft_loss
from lobster.rl_training.rewards import (
    INVALID_3DI_STATE,
    SFT_IGNORE_INDEX,
    assemble_structure_expert,
    build_struct_sft_targets,
    derive_3di_tokens,
    derive_structure_tokens,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _ideal_helix(n: int) -> np.ndarray:
    """(n,3,3) idealized alpha-helix backbone (N, CA, C), roughly physical bond geometry."""
    # Standard alpha-helix: ~100 deg/residue, 1.5 A rise, ~2.3 A radius.
    coords = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        th = np.deg2rad(100.0 * i)
        z = 1.5 * i
        ca = np.array([2.3 * np.cos(th), 2.3 * np.sin(th), z])
        # place N and C offset from CA along tangent-ish directions
        th_n = np.deg2rad(100.0 * i - 40.0)
        th_c = np.deg2rad(100.0 * i + 40.0)
        n_at = np.array([2.3 * np.cos(th_n), 2.3 * np.sin(th_n), z - 0.5])
        c_at = np.array([2.3 * np.cos(th_c), 2.3 * np.sin(th_c), z + 0.5])
        coords[i, 0] = n_at
        coords[i, 1] = ca
        coords[i, 2] = c_at
    return coords


# --------------------------------------------------------------------------------------
# 1. structure-expert producer
# --------------------------------------------------------------------------------------
def test_derive_3di_tokens_shape_range_determinism():
    coords = _ideal_helix(20)
    t1 = derive_3di_tokens(coords)
    t2 = derive_3di_tokens(coords)
    assert t1.shape == (20,)
    assert t1.dtype == np.int64
    # every state is a valid 3Di index (0..19) or the invalid sentinel
    assert set(np.unique(t1)).issubset(set(range(20)) | {INVALID_3DI_STATE})
    # interior residues should resolve to *some* real (non-invalid) state
    assert (t1[2:-2] != INVALID_3DI_STATE).any()
    # deterministic
    assert np.array_equal(t1, t2)


def test_derive_3di_tokens_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError):
        derive_3di_tokens(np.zeros((10, 4, 3), dtype=np.float32))


def _fake_codec(n_tokens: int = 8):
    """A fake ``encode_structure`` mapping position i -> token (i %% n_tokens) via a one-hot."""

    def encode_structure(x_gt, mask, residue_index):
        B, L = x_gt.shape[0], x_gt.shape[1]
        oh = torch.zeros(B, L, n_tokens)
        idx = (torch.arange(L) % n_tokens).view(1, L, 1).expand(B, L, 1)
        oh.scatter_(-1, idx, 1.0)
        return oh, oh, mask

    return encode_structure


def test_derive_structure_tokens_via_codec():
    coords = _ideal_helix(10)
    toks = derive_structure_tokens(_fake_codec(8), coords)
    assert toks.shape == (10,)
    assert toks.dtype == np.int64
    assert np.array_equal(toks, np.arange(10) % 8)


def _mask_strict_codec(n_tokens: int = 8):
    """A fake codec that mirrors the REAL LG codec's mask requirement.

    The real ViT attention forms ``attn_mask`` via an einsum and then does
    ``sim -= (1 - attn_mask) * 1e6``. That subtraction raises on a ``bool`` tensor, so a bool
    mask makes every codec call throw — which the trainer swallows, silently dropping the whole
    struct-SFT expert (``scored_frac=0``). This codec reproduces that contract so
    ``derive_structure_tokens`` is pinned to pass a FLOAT mask.
    """

    def encode_structure(x_gt, mask, residue_index):
        if mask.dtype == torch.bool:
            # Same failure the real codec raises on ``(1 - mask) * 1e6`` with a bool tensor.
            raise RuntimeError("Subtraction with a bool tensor is not supported")
        B, L = x_gt.shape[0], x_gt.shape[1]
        oh = torch.zeros(B, L, n_tokens)
        idx = (torch.arange(L) % n_tokens).view(1, L, 1).expand(B, L, 1)
        oh.scatter_(-1, idx, 1.0)
        return oh, oh, mask

    return encode_structure


def test_derive_structure_tokens_passes_float_mask_to_codec():
    """Regression: derive_structure_tokens must feed the codec a float (not bool) mask.

    A bool mask silently disabled the entire structure-CHORD SFT term (scored_frac=0) because the
    real codec raises on it and the trainer swallows the exception. This pins the contract.
    """
    coords = _ideal_helix(10)
    toks = derive_structure_tokens(_mask_strict_codec(8), coords)
    assert toks.shape == (10,)
    assert np.array_equal(toks, np.arange(10) % 8)


def test_assemble_structure_expert_binder_scoping():
    # 2-chain complex: 3 antigen (A) + 5 binder (B) residues.
    coords = _ideal_helix(8)
    chains = np.array(["A", "A", "A", "B", "B", "B", "B", "B"])
    aa1 = np.array(list("GGGAAAAA"))
    exp = assemble_structure_expert(coords, chains, aa1, encode_structure_fn=_fake_codec(8))
    assert exp["binder_chain"] == "B"
    assert exp["n_binder"] == 5
    assert exp["binder_mask"].tolist() == [False, False, False, True, True, True, True, True]
    assert exp["tri_tokens"].shape == (8,)
    assert exp["structure_tokens"].shape == (8,)
    # 3Di-only mode when no codec is passed
    exp2 = assemble_structure_expert(coords, chains, aa1)
    assert exp2["structure_tokens"] is None


def test_assemble_structure_expert_supervise_scope():
    """Default scope supervises the whole complex; 'binder' restricts to the binder chain."""
    import pytest

    coords = _ideal_helix(8)
    chains = np.array(["A", "A", "A", "B", "B", "B", "B", "B"])
    aa1 = np.array(list("GGGAAAAA"))

    # default = whole complex (antigen + binder). The gen-mask gate in the loss decides which
    # of these actually contribute CE, so the default is safe even with the antigen pinned.
    exp = assemble_structure_expert(coords, chains, aa1)
    assert exp["supervise_scope"] == "complex"
    assert exp["supervise_mask"].tolist() == [True] * 8
    # binder_mask is still reported independently of the supervised scope
    assert exp["binder_mask"].tolist() == [False, False, False, True, True, True, True, True]

    # explicit binder scope
    expb = assemble_structure_expert(coords, chains, aa1, supervise_scope="binder")
    assert expb["supervise_scope"] == "binder"
    assert expb["supervise_mask"].tolist() == expb["binder_mask"].tolist()

    with pytest.raises(ValueError):
        assemble_structure_expert(coords, chains, aa1, supervise_scope="bogus")


# --------------------------------------------------------------------------------------
# 2. structure-track SFT loss
# --------------------------------------------------------------------------------------
class _StubModel:
    """Bind the model's pure SFT helpers onto a stand-in with a controllable forward.

    ``forward`` produces per-track logits that peak at ``(x[i] + x[i-1]) %% V`` — i.e. each
    position's logits depend on BOTH its own and its neighbour's input token. This makes the
    invariance test non-trivial: a revealed policy token can only leave a masked neighbour's
    logits unchanged if ``structure_sft_loss`` overwrites it with the expert token before the
    forward (the expert-context fix).
    """

    from lobster.model.leflur._leflur_sequence_structure_encoder_lightning_module import (
        LeFlurSequenceStructureEncoderLightningModule as _M,
    )

    _sft_step_ce = _M._sft_step_ce
    _expert_context_seq = staticmethod(_M._expert_context_seq)
    _iter_traj_steps = _M._iter_traj_steps

    def __init__(self, vs: int, vt: int):
        self.vs = vs
        self.vt = vt

    @staticmethod
    def _mix(tokens, V):
        B, L = tokens.shape
        shifted = torch.roll(tokens, shifts=1, dims=1)
        comb = (tokens + shifted) % V
        # per-token ramp so CE is sensitive to *which* target token is chosen, plus a peak at
        # the neighbour-mixed input token so logits depend on the (expert-context) input.
        logits = (0.1 * torch.arange(V, dtype=torch.float32)).view(1, 1, V).expand(B, L, V).clone()
        logits.scatter_(-1, comb.unsqueeze(-1).long(), 3.0)
        return logits

    def forward(
        self,
        xt,
        mask,
        residue_index,
        conditioning_tensor,
        *,
        timesteps,
        chain_ids,
        template_structure_tokens,
        scalar_cond_bins,
    ):
        out = {"structure_logits": self._mix(xt["structure_tokens"], self.vs)}
        out["tri_logits"] = self._mix(xt["tri_tokens"], self.vt) if "tri_tokens" in xt else None
        return out


def _make_trajectory(struct_xt, tri_xt, *, mask_struc=7, mask_tri=20, gen_struc=None, gen_tri=None):
    L = struct_xt.shape[1]
    dummy = torch.zeros(1, L)
    if gen_struc is None:
        gen_struc = torch.tensor([[False, False, True, True, True, True]])
    if gen_tri is None:
        gen_tri = torch.tensor([[False, False, True, True, True, True]])
    static = {
        "mask": torch.ones(1, L, dtype=torch.bool),
        "residue_index": torch.arange(L).unsqueeze(0),
        "conditioning_tensor": dummy,
        "chain_ids": torch.zeros(1, L, dtype=torch.long),
        "template_structure_tokens": None,
        "scalar_cond_bins": None,
        "mask_index_struc": mask_struc,
        "mask_index_tri": mask_tri,
        "gen_mask_struc": gen_struc,
        "gen_mask_tri": gen_tri,
        "use_3di_track": True,
    }
    rec = {
        "xt": {
            "sequence_tokens": torch.zeros(1, L, dtype=torch.long),
            "structure_tokens": struct_xt.clone(),
            "tri_tokens": tri_xt.clone(),
        },
        "t_seq": torch.zeros(1),
        "t_struc": torch.zeros(1),
    }
    return {"static": static, "steps": [rec]}


def _fixture():
    # positions: 0,1 = antigen ; 2..5 = binder. binder mix of masked (2,4) & revealed (3,5).
    struct_xt = torch.tensor([[1, 2, 7, 3, 7, 5]])  # 7 = struct mask
    tri_xt = torch.tensor([[1, 2, 20, 3, 20, 5]])  # 20 = tri mask
    struct_tgt = torch.tensor([[-100, -100, 4, 3, 6, 5]])
    tri_tgt = torch.tensor([[-100, -100, 4, 3, 6, 5]])
    sup = torch.tensor([[False, False, True, True, True, True]])
    return struct_xt, tri_xt, struct_tgt, tri_tgt, sup


def test_structure_sft_loss_finite_and_zero_when_unsupervised():
    struct_xt, tri_xt, struct_tgt, tri_tgt, _ = _fixture()
    model = _StubModel(vs=8, vt=22)
    traj = _make_trajectory(struct_xt, tri_xt)
    no_sup = torch.zeros(1, 6, dtype=torch.bool)
    loss = structure_sft_loss(model, traj, struct_tgt, tri_tgt, no_sup)
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_structure_sft_loss_invariant_to_policy_revealed_tokens():
    """Perturbing a revealed BINDER policy token must not change the loss (expert-context fix)."""
    struct_xt, tri_xt, struct_tgt, tri_tgt, sup = _fixture()
    model = _StubModel(vs=8, vt=22)

    base = structure_sft_loss(
        model, _make_trajectory(struct_xt, tri_xt), struct_tgt, tri_tgt, sup, masked_only=False, use_phi=False
    )
    # pos 3 is a revealed binder position (target=3). Change the policy token there.
    struct_pert = struct_xt.clone()
    struct_pert[0, 3] = 0
    tri_pert = tri_xt.clone()
    tri_pert[0, 3] = 0
    pert = structure_sft_loss(
        model, _make_trajectory(struct_pert, tri_pert), struct_tgt, tri_tgt, sup, masked_only=False, use_phi=False
    )
    assert torch.allclose(base, pert, atol=1e-6)


def test_structure_sft_loss_sensitive_to_expert_target():
    struct_xt, tri_xt, struct_tgt, tri_tgt, sup = _fixture()
    model = _StubModel(vs=8, vt=22)
    base = structure_sft_loss(
        model, _make_trajectory(struct_xt, tri_xt), struct_tgt, tri_tgt, sup, masked_only=False, use_phi=False
    )
    # change the expert structure target at a masked supervised position (2)
    tgt2 = struct_tgt.clone()
    tgt2[0, 2] = 0
    pert = structure_sft_loss(
        model, _make_trajectory(struct_xt, tri_xt), tgt2, tri_tgt, sup, masked_only=False, use_phi=False
    )
    assert not torch.allclose(base, pert, atol=1e-6)


def test_structure_sft_loss_tri_track_contributes():
    struct_xt, tri_xt, struct_tgt, tri_tgt, sup = _fixture()
    model = _StubModel(vs=8, vt=22)
    traj = _make_trajectory(struct_xt, tri_xt)
    struct_only = structure_sft_loss(model, traj, struct_tgt, tri_tgt, sup, w_tri=0.0)
    with_tri = structure_sft_loss(model, traj, struct_tgt, tri_tgt, sup, w_tri=1.0)
    assert torch.isfinite(struct_only) and torch.isfinite(with_tri)
    assert not torch.allclose(struct_only, with_tri, atol=1e-6)


def test_structure_sft_loss_struct_only_mode():
    """tri_target=None (or w_tri=0) yields a valid structure-only distillation."""
    struct_xt, tri_xt, struct_tgt, _, sup = _fixture()
    model = _StubModel(vs=8, vt=22)
    traj = _make_trajectory(struct_xt, tri_xt)
    loss = structure_sft_loss(model, traj, struct_tgt, None, sup)
    assert torch.isfinite(loss) and float(loss) > 0.0


def test_complex_scope_is_noop_when_antigen_pinned():
    """Whole-complex supervision == binder-only when the antigen structure is NOT generated.

    The plain CHORD formulation gates CE by ``sup & gen_mask_struc``: with the antigen pinned
    (``gen_mask_struc`` False there) those positions never contribute, even with valid antigen
    targets and the complex-wide mask. The antigen structure contributes only when it is actually
    generated (template-target mode; see the next test).
    """
    model = _StubModel(vs=8, vt=22)
    # antigen (0,1) masked in xt and given VALID targets, so only the gen-mask gate can exclude them
    struct_xt = torch.tensor([[7, 7, 7, 3, 7, 5]])
    tri_xt = torch.tensor([[20, 20, 20, 3, 20, 5]])
    struct_tgt = torch.tensor([[0, 1, 4, 3, 6, 5]])
    tri_tgt = torch.tensor([[0, 1, 4, 3, 6, 5]])
    sup_binder = torch.tensor([[False, False, True, True, True, True]])
    sup_complex = torch.ones(1, 6, dtype=torch.bool)

    gen_pinned = torch.tensor([[False, False, True, True, True, True]])  # antigen pinned
    traj = _make_trajectory(struct_xt, tri_xt, gen_struc=gen_pinned, gen_tri=gen_pinned)
    l_binder = structure_sft_loss(model, traj, struct_tgt, tri_tgt, sup_binder)
    l_complex = structure_sft_loss(model, traj, struct_tgt, tri_tgt, sup_complex)
    assert torch.allclose(l_binder, l_complex, atol=1e-6)


def test_complex_scope_supervises_antigen_when_generated():
    """When the antigen structure IS generated (template-target mode), the whole-complex scope
    picks up the antigen positions and the loss changes vs binder-only."""
    model = _StubModel(vs=8, vt=22)
    struct_xt = torch.tensor([[7, 7, 7, 3, 7, 5]])
    tri_xt = torch.tensor([[20, 20, 20, 3, 20, 5]])
    struct_tgt = torch.tensor([[0, 1, 4, 3, 6, 5]])
    tri_tgt = torch.tensor([[0, 1, 4, 3, 6, 5]])
    sup_binder = torch.tensor([[False, False, True, True, True, True]])
    sup_complex = torch.ones(1, 6, dtype=torch.bool)

    gen_all = torch.ones(1, 6, dtype=torch.bool)  # antigen generated too
    traj = _make_trajectory(struct_xt, tri_xt, gen_struc=gen_all, gen_tri=gen_all)
    l_binder = structure_sft_loss(model, traj, struct_tgt, tri_tgt, sup_binder)
    l_complex = structure_sft_loss(model, traj, struct_tgt, tri_tgt, sup_complex)
    assert not torch.allclose(l_binder, l_complex, atol=1e-6)


# --------------------------------------------------------------------------------------
# 4. build_struct_sft_targets — trainer-side scatter onto the padded (G, L) layout
# --------------------------------------------------------------------------------------
def _expert(na: int, nb: int, *, with_struct: bool = True) -> dict:
    """Antigen-then-binder expert tokens with distinguishable per-chain values."""
    n = na + nb
    tri = np.arange(1, n + 1, dtype=np.int64)  # antigen 1..na, binder na+1..n
    ex = {"tri_tokens": tri, "supervise_mask": np.ones(n, dtype=bool)}
    if with_struct:
        ex["structure_tokens"] = np.arange(101, 101 + n, dtype=np.int64)
    return ex


def test_build_struct_sft_targets_scatter_mapping():
    """Antigen tokens land on antigen_idx, binder tokens on binder_idx; the split is at na."""
    # Interleaved layout: positions 0,2,4 = antigen; 1,3 = binder; 5 = padding.
    antigen_idx = np.array([0, 2, 4])
    binder_idx = np.array([1, 3])
    na, nb = 3, 2
    G, L = 1, 6
    out = build_struct_sft_targets([_expert(na, nb)], binder_idx, antigen_idx, G, L)

    tri = out["tri_target_ids"]
    st = out["struct_target_ids"]
    sup = out["supervise_mask"]
    assert out["n_scored"] == 1
    # antigen tokens (1..3) scattered to antigen_idx in order
    assert tri[0, 0] == 1 and tri[0, 2] == 2 and tri[0, 4] == 3
    # binder tokens (4..5) scattered to binder_idx in order
    assert tri[0, 1] == 4 and tri[0, 3] == 5
    assert st[0, 0] == 101 and st[0, 4] == 103 and st[0, 1] == 104
    assert sup[0, :5].all() and not sup[0, 5]  # position 5 (padding) untouched
    assert tri[0, 5] == SFT_IGNORE_INDEX and st[0, 5] == SFT_IGNORE_INDEX


def test_build_struct_sft_targets_none_expert_is_ignore_row():
    """A None expert (fold failed) yields an all-ignore row: nothing supervised."""
    antigen_idx = np.array([0, 1])
    binder_idx = np.array([2, 3])
    out = build_struct_sft_targets([None], binder_idx, antigen_idx, 1, 4)
    assert out["n_scored"] == 0
    assert not out["supervise_mask"].any()
    assert (out["tri_target_ids"] == SFT_IGNORE_INDEX).all()
    assert (out["struct_target_ids"] == SFT_IGNORE_INDEX).all()


def test_build_struct_sft_targets_length_mismatch_skipped():
    """An expert whose token span != na+nb is dropped to an all-ignore row (no partial scatter)."""
    antigen_idx = np.array([0, 1, 2])
    binder_idx = np.array([3, 4])  # na+nb == 5
    bad = {"tri_tokens": np.arange(4, dtype=np.int64), "supervise_mask": np.ones(4, dtype=bool)}
    out = build_struct_sft_targets([bad], binder_idx, antigen_idx, 1, 5)
    assert out["n_scored"] == 0
    assert not out["supervise_mask"].any()


def test_build_struct_sft_targets_3di_only_expert_leaves_struct_ignored():
    """A 3Di-only expert (no structure_tokens) supervises tri but leaves struct_target all-ignore."""
    antigen_idx = np.array([0, 1])
    binder_idx = np.array([2, 3])
    out = build_struct_sft_targets([_expert(2, 2, with_struct=False)], binder_idx, antigen_idx, 1, 4)
    assert out["n_scored"] == 1
    assert out["supervise_mask"].any()
    assert (out["tri_target_ids"][0, :4] != SFT_IGNORE_INDEX).all()
    assert (out["struct_target_ids"] == SFT_IGNORE_INDEX).all()


def test_build_struct_sft_targets_per_row_independence():
    """Mixed group: a valid expert and a None coexist without cross-contamination."""
    antigen_idx = np.array([0, 1])
    binder_idx = np.array([2, 3])
    out = build_struct_sft_targets([_expert(2, 2), None], binder_idx, antigen_idx, 2, 4)
    assert out["n_scored"] == 1
    assert out["supervise_mask"][0].any()
    assert not out["supervise_mask"][1].any()
