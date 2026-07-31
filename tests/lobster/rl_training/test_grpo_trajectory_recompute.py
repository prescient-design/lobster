"""Plumbing tests for the LeFlur GRPO trajectory-recompute methods.

The GRPO trainer reconstructs per-step transition log-probs (and the KL to a
frozen reference) from a captured rollout via
:meth:`LeFlurSequenceStructureEncoderLightningModule.logprob_over_trajectory` and
:meth:`~.kl_over_trajectory`. Those methods re-run ``forward`` (plus the CFG
second forward and the seq/3Di bias + diversity penalties) to reproduce the exact
biased logits, then call the differentiable DFM step kernel.

Instantiating the real module loads the LG codec, so these tests exercise the
recompute *plumbing* against a lightweight subclass with a deterministic fake
``forward`` (the numerical parity of the underlying kernel is covered by
``test_dfm_logprob.py``). They assert:

1. ``logprob_over_trajectory`` returns a finite ``(B,)`` tensor differentiable in
   the encoder parameters, and is deterministic across calls.
2. Log-prob is additive across steps — the full-trajectory value equals the sum of
   the per-step (``step_indices=[i]``) values (validates step accumulation +
   subsampling).
3. ``kl_over_trajectory`` against an identical reference is ~0 (validates the KL
   wiring and the CFG/bias/diversity reproduction being identical on both sides).
4. ``decode_endpoint_aa`` maps the sampled endpoint to standard-AA ids in [0, 22].
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lobster.model.leflur._leflur_sequence_structure_encoder_lightning_module import (
    LeFlurSequenceStructureEncoderLightningModule as _LM,
)

SEQ_VOCAB, SEQ_MASK = 33, 32
STRUC_VOCAB, STRUC_MASK = 6, 4
TRI_VOCAB, TRI_MASK = 22, 20


class _FakePolicy(_LM):
    """Subclass with a deterministic fake ``forward`` (skips the heavy codec init)."""

    def __init__(self, seed: int = 0) -> None:
        nn.Module.__init__(self)  # bypass _LM.__init__ (LG codec load) — we only need forward + params
        self.vocab_size = SEQ_VOCAB  # used by decode_endpoint_aa
        g = torch.Generator().manual_seed(seed)
        self.w_seq = nn.Parameter(torch.randn(SEQ_VOCAB, generator=g))
        self.w_struc = nn.Parameter(torch.randn(STRUC_VOCAB, generator=g))
        self.w_tri = nn.Parameter(torch.randn(TRI_VOCAB, generator=g))
        self.c_seq = nn.Parameter(torch.tensor(0.5))
        self.c_struc = nn.Parameter(torch.tensor(0.3))
        self.c_tri = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("_ar_seq", torch.arange(SEQ_VOCAB).float() * 0.01)
        self.register_buffer("_ar_struc", torch.arange(STRUC_VOCAB).float() * 0.01)
        self.register_buffer("_ar_tri", torch.arange(TRI_VOCAB).float() * 0.01)

    def forward(
        self,
        x_t,
        mask,
        residue_index,
        conditioning_tensor,
        timesteps=None,
        chain_ids=None,
        template_structure_tokens=None,
        scalar_cond_bins=None,
    ):
        cond = (
            conditioning_tensor.float().sum(dim=-1, keepdim=True)
            if conditioning_tensor is not None
            else torch.zeros_like(x_t["sequence_tokens"].float().unsqueeze(-1))
        )  # (B, L, 1) — makes the CFG (cond -> 0) path change the logits
        seq = x_t["sequence_tokens"].float().unsqueeze(-1)
        struc = x_t["structure_tokens"].float().unsqueeze(-1)
        out = {
            "sequence_logits": self.w_seq.view(1, 1, -1) * seq + self.c_seq * cond + self._ar_seq,
            "structure_logits": self.w_struc.view(1, 1, -1) * struc + self.c_struc * cond + self._ar_struc,
        }
        tri = x_t.get("tri_tokens")
        out["tri_logits"] = (
            self.w_tri.view(1, 1, -1) * tri.float().unsqueeze(-1) + self.c_tri * cond + self._ar_tri
            if tri is not None
            else None
        )
        return out


def _make_trajectory(batch_size: int = 2, length: int = 4, nsteps: int = 3, seed: int = 1) -> dict:
    """Build a self-consistent fake rollout store (static + steps + final_xt)."""
    g = torch.Generator().manual_seed(seed)

    def _tok(vocab: int) -> torch.Tensor:
        return torch.randint(0, vocab, (batch_size, length), generator=g, dtype=torch.int32)

    def _mask() -> torch.Tensor:
        m = torch.zeros(batch_size, length, dtype=torch.bool)
        m[:, 1:] = True  # position 0 fixed, rest generated
        return m

    gm_seq, gm_struc, gm_tri = _mask(), _mask(), _mask()
    static = {
        "mask": torch.ones(batch_size, length),
        "residue_index": torch.arange(length).unsqueeze(0).expand(batch_size, length).contiguous(),
        "conditioning_tensor": torch.randn(batch_size, length, 1, generator=g),
        "chain_ids": None,
        "template_structure_tokens": None,
        "scalar_cond_bins": None,
        "cfg_weight": 2.0,  # exercise the CFG second-forward path
        "use_3di_track": True,
        "sequence_logit_bias": torch.randn(SEQ_VOCAB, generator=g),
        "sequence_logit_bias_steps": 100,  # keep bias active for all steps
        "sequence_diversity_penalty": 0.5,
        "tri_logit_bias": torch.randn(TRI_VOCAB, generator=g),
        "tri_diversity_penalty": 0.3,
        "gen_mask_seq": gm_seq,
        "gen_mask_struc": gm_struc,
        "gen_mask_tri": gm_tri,
        "div_mask_seq": gm_seq.clone(),
        "div_mask_tri": gm_tri.clone(),
        "mask_index_seq": SEQ_MASK,
        "mask_index_struc": STRUC_MASK,
        "mask_index_tri": TRI_MASK,
    }

    steps = []
    for step_idx in range(nsteps):
        t_seq = torch.full((batch_size,), 0.2 + 0.2 * step_idx)
        t_struc = torch.full((batch_size,), 0.15 + 0.2 * step_idx)
        t_tri = torch.full((batch_size,), 0.25 + 0.2 * step_idx)
        rec = {
            "step_idx": step_idx,
            "xt": {
                "sequence_tokens": _tok(SEQ_VOCAB),
                "structure_tokens": _tok(STRUC_VOCAB),
                "tri_tokens": _tok(TRI_VOCAB),
            },
            "t_seq": t_seq,
            "t_struc": t_struc,
            "tracks": {
                "sequence_tokens": {
                    "xt": _tok(SEQ_VOCAB),
                    "x_next": _tok(SEQ_VOCAB),
                    "t": t_seq,
                    "dt": torch.tensor(0.05),
                    "temperature": 1.0,
                    "stochasticity": 1.0,
                    "gen_mask": gm_seq,
                },
                "structure_tokens": {
                    "xt": _tok(STRUC_VOCAB),
                    "x_next": _tok(STRUC_VOCAB),
                    "t": t_struc,
                    "dt": torch.tensor(0.05),
                    "temperature": 1.0,
                    "stochasticity": 1.0,
                    "gen_mask": gm_struc,
                },
                "tri_tokens": {
                    "xt": _tok(TRI_VOCAB),
                    "x_next": _tok(TRI_VOCAB),
                    "t": t_tri,
                    "dt": torch.tensor(0.05),
                    "temperature": 1.0,
                    "stochasticity": 1.0,
                    "gen_mask": gm_tri,
                },
            },
        }
        steps.append(rec)

    return {
        "static": static,
        "steps": steps,
        "final_xt": {
            "sequence_tokens": _tok(SEQ_VOCAB),
            "structure_tokens": _tok(STRUC_VOCAB),
            "tri_tokens": _tok(TRI_VOCAB),
        },
    }


def test_logprob_shape_finite_and_differentiable() -> None:
    policy = _FakePolicy()
    traj = _make_trajectory()

    lp = policy.logprob_over_trajectory(traj)
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()

    lp.sum().backward()
    assert policy.w_seq.grad is not None and torch.isfinite(policy.w_seq.grad).all()
    assert policy.w_tri.grad is not None and torch.isfinite(policy.w_tri.grad).all()


def test_logprob_deterministic() -> None:
    policy = _FakePolicy()
    traj = _make_trajectory()
    a = policy.logprob_over_trajectory(traj)
    b = policy.logprob_over_trajectory(traj)
    torch.testing.assert_close(a, b)


def test_logprob_additive_over_steps() -> None:
    """Full-trajectory log-prob == sum of per-step log-probs (accumulation + subsampling)."""
    policy = _FakePolicy()
    traj = _make_trajectory(nsteps=3)
    full = policy.logprob_over_trajectory(traj)
    per_step = sum(policy.logprob_over_trajectory(traj, step_indices=[i]) for i in range(3))
    torch.testing.assert_close(full, per_step, atol=1e-5, rtol=1e-5)


def test_seq_only_ablation_smaller() -> None:
    """Restricting tracks drops contributions (log-probs are negative, so |sum| shrinks)."""
    policy = _FakePolicy()
    traj = _make_trajectory()
    all_tracks = policy.logprob_over_trajectory(traj)
    seq_only = policy.logprob_over_trajectory(traj, tracks=("sequence_tokens",))
    assert seq_only.shape == all_tracks.shape
    # each track contributes a (negative) log-prob, so fewer tracks -> a larger (less negative) sum
    assert torch.all(seq_only >= all_tracks - 1e-6)


def test_kl_against_identical_reference_is_zero() -> None:
    policy = _FakePolicy(seed=0)
    ref = _FakePolicy(seed=0)  # identical weights
    ref.load_state_dict(policy.state_dict())
    traj = _make_trajectory()

    kl = policy.kl_over_trajectory(traj, ref)
    assert kl.shape == (2,)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-5, rtol=0)


def test_kl_against_different_reference_positive_and_differentiable() -> None:
    policy = _FakePolicy(seed=0)
    ref = _FakePolicy(seed=7)  # different weights -> nonzero KL
    traj = _make_trajectory()

    kl = policy.kl_over_trajectory(traj, ref)
    assert torch.all(kl >= -1e-6)  # KL is non-negative
    assert kl.sum() > 0
    kl.sum().backward()
    assert policy.w_seq.grad is not None and torch.isfinite(policy.w_seq.grad).all()
    assert ref.w_seq.grad is None  # reference is not updated


def test_decode_endpoint_aa_range() -> None:
    policy = _FakePolicy()
    traj = _make_trajectory()
    aa = policy.decode_endpoint_aa(traj)
    assert aa.shape == (2, 4)
    assert int(aa.min()) >= 0 and int(aa.max()) <= 22


def test_rollout_rejects_explicit_store() -> None:
    policy = _FakePolicy()
    with pytest.raises(ValueError, match="trajectory_store"):
        policy.rollout_with_logprobs(trajectory_store={})


# --- captured_logprob_per_step (inline behaviour-policy log-prob) --------------
#
# `captured_logprob_per_step` reads the per-track "logprob" the sampler stored
# inline (the faithful `old_lp` for GRPO), rather than recomputing via a second
# forward. It is a pure reader over the trajectory dict — no model state — so we
# build a hand-crafted store with known per-track values and assert the summation,
# track-selection, held-clean-step, and missing-key behaviours.


def _make_inline_trajectory(
    batch_size: int = 2,
    nsteps: int = 3,
    drop_tri_on_last: bool = False,
) -> dict:
    """Store with known inline per-track log-probs: seq/struc/tri = -1/-2/-4 per step."""
    steps = []
    for step_idx in range(nsteps):
        tracks = {
            "sequence_tokens": {"logprob": torch.full((batch_size,), -1.0)},
            "structure_tokens": {"logprob": torch.full((batch_size,), -2.0)},
            "tri_tokens": {"logprob": torch.full((batch_size,), -4.0)},
        }
        if drop_tri_on_last and step_idx == nsteps - 1:
            del tracks["tri_tokens"]  # track held clean this step (already resolved)
        steps.append({"step_idx": step_idx, "tracks": tracks})
    return {"static": {"mask": torch.ones(batch_size, 4)}, "steps": steps}


def test_captured_logprob_shape_and_sum_over_tracks() -> None:
    policy = _FakePolicy()
    traj = _make_inline_trajectory(batch_size=2, nsteps=3)
    lp = policy.captured_logprob_per_step(traj)
    assert lp.shape == (3, 2)  # (n_steps, B)
    # each step sums seq(-1) + struc(-2) + tri(-4) = -7
    torch.testing.assert_close(lp, torch.full((3, 2), -7.0))


def test_captured_logprob_seq_only_selection() -> None:
    policy = _FakePolicy()
    traj = _make_inline_trajectory(batch_size=2, nsteps=3)
    lp = policy.captured_logprob_per_step(traj, tracks=("sequence_tokens",))
    assert lp.shape == (3, 2)
    torch.testing.assert_close(lp, torch.full((3, 2), -1.0))  # seq track only


def test_captured_logprob_held_clean_step_contributes_zero() -> None:
    """A track absent from a step (held clean) adds nothing for that step."""
    policy = _FakePolicy()
    traj = _make_inline_trajectory(batch_size=2, nsteps=3, drop_tri_on_last=True)
    lp = policy.captured_logprob_per_step(traj)
    # steps 0,1 have all three (-7); step 2 drops tri (-1 + -2 = -3)
    expected = torch.tensor([[-7.0, -7.0], [-7.0, -7.0], [-3.0, -3.0]])
    torch.testing.assert_close(lp, expected)


def test_captured_logprob_missing_logprob_raises() -> None:
    """A rollout predating inline capture (no 'logprob' key) is a hard error."""
    policy = _FakePolicy()
    traj = _make_inline_trajectory(batch_size=2, nsteps=1)
    del traj["steps"][0]["tracks"]["sequence_tokens"]["logprob"]  # simulate old rollout
    with pytest.raises(KeyError, match="no inline 'logprob'"):
        policy.captured_logprob_per_step(traj)
