"""Parity + gradient tests for the differentiable absorbing-state DFM step kernel.

The GRPO trainer reconstructs per-step transition log-probabilities from stored
trajectories using :mod:`lobster.rl_training._dfm_logprob`. Those helpers must
reproduce bionemo's ``DiscreteFlowMatcher.step`` *exactly* (otherwise the policy
ratio is biased), be out-of-place (upstream mutates its input logits), and be
differentiable in the logits (GRPO backpropagates through them).

These tests:

1. ``test_step_prob_parity_*`` — capture the *real* ``step_prob`` that upstream
   feeds to ``torch.multinomial`` and assert our reimplementation matches to
   ``atol=1e-5`` at a mid step and the final step, for masked and unmasked
   positions.
2. ``test_no_inplace_mutation`` — our helper must not modify the input logits,
   whereas upstream does (documenting exactly why the reimplementation exists).
3. ``test_logprob_matches_captured_step_prob`` — the summed log-prob equals the
   hand-computed log of the captured golden ``step_prob`` at the sampled tokens.
4. ``test_gradcheck_logprob`` — double-precision ``gradcheck`` of the log-prob
   w.r.t. the logits.
"""

from __future__ import annotations

import pytest
import torch

from bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants import DiscreteFlowMatcher

from lobster.rl_training._dfm_logprob import dfm_step_logprob, dfm_step_prob

MASK_INDEX = 4  # inclusive masked prior -> mask lives at the last column of an S=5 vocab
VOCAB_SIZE = 5


def _make_interpolant() -> DiscreteFlowMatcher:
    """Build a real absorbing-state ``DiscreteFlowMatcher`` on CPU (matches LeFlur tracks)."""
    prior = DiscreteMaskedPrior(num_classes=VOCAB_SIZE, mask_dim=MASK_INDEX, inclusive=True)
    return DiscreteFlowMatcher(
        time_distribution=UniformTimeDistribution(),
        prior_distribution=prior,
        device="cpu",
    )


def _capture_golden_step_prob(
    interpolant: DiscreteFlowMatcher,
    logits: torch.Tensor,
    t: torch.Tensor,
    xt: torch.Tensor,
    dt: float,
    temperature: float,
    stochasticity: float,
    monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor:
    """Run upstream ``step`` and capture the exact ``step_prob`` it passes to multinomial.

    ``step`` mutates its ``logits`` argument in place, so we hand it a *clone*. We
    patch ``torch.multinomial`` to record its first positional argument (the
    ``(B*L, S)`` step-prob matrix) and return a valid dummy index so ``step``
    completes.
    """
    batch_size, length, num_classes = logits.shape
    captured: dict[str, torch.Tensor] = {}

    real_multinomial = torch.multinomial

    def _fake_multinomial(input, num_samples, *args, **kwargs):  # noqa: A002 - match torch signature
        captured["step_prob"] = input.detach().clone()
        return torch.zeros((input.shape[0], num_samples), dtype=torch.long, device=input.device)

    monkeypatch.setattr(torch, "multinomial", _fake_multinomial)
    try:
        interpolant.step(
            logits=logits.clone(),
            t=t,
            xt=xt,
            dt=dt,
            temperature=temperature,
            stochasticity=stochasticity,
        )
    finally:
        monkeypatch.setattr(torch, "multinomial", real_multinomial)

    return captured["step_prob"].view(batch_size, length, num_classes)


@pytest.mark.parametrize(
    "t_val,dt,desc",
    [
        (0.3, 0.05, "mid-step"),
        (0.98, 0.02, "final-step (t+dt==1 -> remask gate off)"),
    ],
)
@pytest.mark.parametrize("stochasticity", [0.0, 1.0, 2.0])
def test_step_prob_parity(t_val: float, dt: float, desc: str, stochasticity: float) -> None:
    """Our ``dfm_step_prob`` matches upstream ``step_prob`` to atol=1e-5."""
    monkeypatch = pytest.MonkeyPatch()
    interpolant = _make_interpolant()
    torch.manual_seed(0)

    batch_size, length = 2, 6
    logits = torch.randn(batch_size, length, VOCAB_SIZE, dtype=torch.float32)
    # Mix of masked and unmasked current states to exercise both step_prob terms.
    xt = torch.tensor(
        [[MASK_INDEX, 0, MASK_INDEX, 2, MASK_INDEX, 1], [1, MASK_INDEX, MASK_INDEX, MASK_INDEX, 0, 3]],
        dtype=torch.long,
    )
    t = torch.full((batch_size,), t_val, dtype=torch.float32)

    golden = _capture_golden_step_prob(
        interpolant, logits, t, xt, dt, temperature=1.0, stochasticity=stochasticity, monkeypatch=monkeypatch
    )
    mine = dfm_step_prob(logits, t, dt, xt, mask_index=MASK_INDEX, temperature=1.0, stochasticity=stochasticity)

    assert mine.shape == golden.shape
    torch.testing.assert_close(mine, golden, atol=1e-5, rtol=1e-5)
    # Rows are non-negative weights (multinomial normalizes internally, so they need
    # not sum to one — at the final step the 1/(1-t) factor can push a masked row
    # above one; that is faithfully reproduced and matched against `golden` above).
    assert torch.all(mine >= 0)


def test_step_prob_parity_with_temperature() -> None:
    """Parity holds when a non-unit softmax temperature is applied."""
    monkeypatch = pytest.MonkeyPatch()
    interpolant = _make_interpolant()
    torch.manual_seed(1)

    logits = torch.randn(1, 4, VOCAB_SIZE, dtype=torch.float32)
    xt = torch.tensor([[MASK_INDEX, MASK_INDEX, 0, MASK_INDEX]], dtype=torch.long)
    t = torch.full((1,), 0.4, dtype=torch.float32)

    golden = _capture_golden_step_prob(
        interpolant, logits, t, xt, dt=0.1, temperature=0.7, stochasticity=1.5, monkeypatch=monkeypatch
    )
    mine = dfm_step_prob(logits, t, 0.1, xt, mask_index=MASK_INDEX, temperature=0.7, stochasticity=1.5)
    torch.testing.assert_close(mine, golden, atol=1e-5, rtol=1e-5)


def test_no_inplace_mutation() -> None:
    """Our helper leaves the input logits untouched; upstream ``step`` mutates them."""
    interpolant = _make_interpolant()
    torch.manual_seed(2)
    logits = torch.randn(1, 3, VOCAB_SIZE, dtype=torch.float32)
    xt = torch.tensor([[MASK_INDEX, 0, MASK_INDEX]], dtype=torch.long)
    t = torch.full((1,), 0.3, dtype=torch.float32)

    before = logits.clone()
    dfm_step_prob(logits, t, 0.05, xt, mask_index=MASK_INDEX)
    torch.testing.assert_close(logits, before)  # ours: no mutation

    # Document the upstream in-place mutation that necessitates this module.
    upstream_logits = logits.clone()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            torch,
            "multinomial",
            lambda inp, num_samples, *a, **k: torch.zeros((inp.shape[0], num_samples), dtype=torch.long),
        )
        interpolant.step(logits=upstream_logits, t=t, xt=xt, dt=0.05)
    assert torch.any(upstream_logits[..., MASK_INDEX] == -1.0e9)


def test_logprob_matches_captured_step_prob() -> None:
    """Summed log-prob equals hand-computed log of the captured golden step_prob."""
    monkeypatch = pytest.MonkeyPatch()
    interpolant = _make_interpolant()
    torch.manual_seed(3)

    batch_size, length = 2, 5
    logits = torch.randn(batch_size, length, VOCAB_SIZE, dtype=torch.float32)
    xt = torch.tensor(
        [[MASK_INDEX, MASK_INDEX, 1, MASK_INDEX, 0], [MASK_INDEX, 2, MASK_INDEX, MASK_INDEX, MASK_INDEX]],
        dtype=torch.long,
    )
    x_next = torch.tensor(
        [[0, 3, 1, 2, 0], [1, 2, 0, 3, 2]],
        dtype=torch.long,
    )
    gen_mask = torch.tensor(
        [[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]],
        dtype=torch.bool,
    )
    t = torch.full((batch_size,), 0.35, dtype=torch.float32)

    golden = _capture_golden_step_prob(
        interpolant, logits, t, xt, dt=0.05, temperature=1.0, stochasticity=1.0, monkeypatch=monkeypatch
    )
    chosen = golden.gather(-1, x_next.unsqueeze(-1)).squeeze(-1)
    row_sum = golden.sum(dim=-1).clamp_min(1e-9)  # multinomial normalizes by the row sum
    logp = torch.log(chosen.clamp_min(1e-9)) - torch.log(row_sum)
    expected = (logp * gen_mask.float()).sum(dim=-1)

    got = dfm_step_logprob(
        logits, t, 0.05, xt, x_next, gen_mask, mask_index=MASK_INDEX, temperature=1.0, stochasticity=1.0
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_logprob_per_position_sums_to_default() -> None:
    """``per_position=True`` returns a masked ``(B, L)`` tensor that sums over L to the default.

    This is the contract the per-token structure-track advantage (e_lj / clash per-residue
    credit) relies on: routing credit per position, then summing, must not change the total.
    """
    torch.manual_seed(3)
    batch_size, length = 2, 5
    logits = torch.randn(batch_size, length, VOCAB_SIZE, dtype=torch.float32)
    xt = torch.tensor(
        [[MASK_INDEX, MASK_INDEX, 1, MASK_INDEX, 0], [MASK_INDEX, 2, MASK_INDEX, MASK_INDEX, MASK_INDEX]],
        dtype=torch.long,
    )
    x_next = torch.tensor([[0, 3, 1, 2, 0], [1, 2, 0, 3, 2]], dtype=torch.long)
    gen_mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)
    t = torch.full((batch_size,), 0.35, dtype=torch.float32)

    common = dict(mask_index=MASK_INDEX, temperature=1.0, stochasticity=1.0)
    summed = dfm_step_logprob(logits, t, 0.05, xt, x_next, gen_mask, **common)
    per_pos = dfm_step_logprob(logits, t, 0.05, xt, x_next, gen_mask, per_position=True, **common)

    assert per_pos.shape == (batch_size, length)
    # Fixed / non-generated positions carry exactly zero credit.
    assert torch.all(per_pos[~gen_mask] == 0.0)
    # Summing the per-position credit over L reproduces the default (B,) return.
    torch.testing.assert_close(per_pos.sum(dim=-1), summed, atol=1e-6, rtol=1e-6)


def test_gradcheck_logprob() -> None:
    """Double-precision gradcheck of the log-prob w.r.t. logits (interior probs)."""
    torch.manual_seed(4)
    batch_size, length = 1, 3
    logits = torch.randn(batch_size, length, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
    xt = torch.full((batch_size, length), MASK_INDEX, dtype=torch.long)  # all-mask -> interior unmask term
    x_next = torch.tensor([[0, 1, 2]], dtype=torch.long)
    gen_mask = torch.ones((batch_size, length), dtype=torch.bool)
    t = torch.full((batch_size,), 0.3, dtype=torch.float64)

    def fn(lg: torch.Tensor) -> torch.Tensor:
        return dfm_step_logprob(
            lg, t, 0.05, xt, x_next, gen_mask, mask_index=MASK_INDEX, temperature=1.0, stochasticity=1.0
        )

    assert torch.autograd.gradcheck(fn, (logits,), eps=1e-6, atol=1e-4)
