"""Unit tests for the LeFlur PLL scoring helpers.

These tests cover the pure / model-agnostic helpers in
:mod:`lobster.model.leflur._pll_scoring`. They run on CPU in well under a
second and do not require any LeFlur checkpoint.

End-to-end ``score_pll(...)`` tests with a real Lightning module are
exercised by :mod:`tests.lobster.cmdline.test_score_pll_dispatch`.
"""

from __future__ import annotations

import math

import pytest
import torch

from lobster.model.leflur._pll_scoring import (
    PROTEIN_LIGAND_VARIANTS,
    PROTEIN_VARIANTS,
    _validate_variants,
    absorbing_corrupt,
    ce_on_masked,
    stratified_t_samples,
)


# ---------------------------------------------------------------------------
# stratified_t_samples
# ---------------------------------------------------------------------------


def test_stratified_t_samples_in_bounds() -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    t = stratified_t_samples(K=16, eps=0.02, device=torch.device("cpu"), generator=g)
    assert t.shape == (16,)
    assert torch.all(t > 0.02 - 1e-6)
    assert torch.all(t < 1.0 - 0.02 + 1e-6)


def test_stratified_t_samples_one_per_stratum() -> None:
    """Each draw must fall in its own equal-width stratum (the defining property)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(1234)
    K = 8
    eps = 0.02
    edges = torch.linspace(eps, 1.0 - eps, steps=K + 1)
    t = stratified_t_samples(K=K, eps=eps, device=torch.device("cpu"), generator=g)
    for k in range(K):
        assert edges[k] - 1e-6 <= t[k] <= edges[k + 1] + 1e-6, (
            f"t[{k}] = {t[k]:.4f} not in stratum [{edges[k]:.4f}, {edges[k + 1]:.4f}]"
        )


def test_stratified_t_samples_deterministic_under_seed() -> None:
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(42)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(42)
    t1 = stratified_t_samples(K=12, device=torch.device("cpu"), generator=g1)
    t2 = stratified_t_samples(K=12, device=torch.device("cpu"), generator=g2)
    assert torch.equal(t1, t2)


def test_stratified_t_samples_rejects_zero_K() -> None:
    g = torch.Generator(device="cpu")
    with pytest.raises(ValueError, match="K must be >= 1"):
        stratified_t_samples(K=0, device=torch.device("cpu"), generator=g)


# ---------------------------------------------------------------------------
# absorbing_corrupt
# ---------------------------------------------------------------------------


def test_absorbing_corrupt_respects_t_rate() -> None:
    """At low t we should mask many positions; at high t very few."""
    K, L = 64, 32
    mask_id = 99
    device = torch.device("cpu")
    g = torch.Generator(device=device)
    g.manual_seed(0)

    clean = torch.arange(L).expand(K, L).contiguous()
    valid = torch.ones(K, L)

    t_low = torch.full((K,), 0.1)
    _, mp_low = absorbing_corrupt(clean, t_low, mask_id, valid, g)
    frac_low = mp_low.float().mean().item()
    # Expectation = 1 - t = 0.9; allow ±0.05 with K*L=2048 samples
    assert abs(frac_low - 0.9) < 0.05, f"low-t mask frac {frac_low:.3f} far from 0.9"

    t_high = torch.full((K,), 0.9)
    _, mp_high = absorbing_corrupt(clean, t_high, mask_id, valid, g)
    frac_high = mp_high.float().mean().item()
    assert abs(frac_high - 0.1) < 0.05, f"high-t mask frac {frac_high:.3f} far from 0.1"


def test_absorbing_corrupt_writes_mask_id() -> None:
    K, L = 4, 10
    mask_id = 77
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    clean = torch.arange(L).expand(K, L).contiguous()
    valid = torch.ones(K, L)
    t_values = torch.full((K,), 0.5)
    x_t, mp = absorbing_corrupt(clean, t_values, mask_id, valid, g)
    assert torch.all(x_t[mp] == mask_id)
    assert torch.all(x_t[~mp] == clean[~mp])


def test_absorbing_corrupt_honors_valid_mask() -> None:
    """Padding positions (valid_mask=0) must NEVER be marked as masked."""
    K, L = 4, 12
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    clean = torch.arange(L).expand(K, L).contiguous()
    valid = torch.ones(K, L)
    valid[:, 8:] = 0  # last 4 positions are padding
    t_values = torch.full((K,), 0.1)  # high mask rate, so padding stands out
    _, mp = absorbing_corrupt(clean, t_values, 99, valid, g)
    assert mp[:, 8:].sum().item() == 0


# ---------------------------------------------------------------------------
# ce_on_masked
# ---------------------------------------------------------------------------


def test_ce_on_masked_matches_manual_average() -> None:
    K, N, V = 2, 5, 4
    target = torch.tensor([[0, 1, 2, 3, 0], [3, 2, 1, 0, 3]])
    masked = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])

    # Build logits with uniform distribution -> CE = log(V).
    logits = torch.zeros(K, N, V)
    avg_ce, sum_ce, n_masked = ce_on_masked(logits, target, masked)
    assert torch.allclose(n_masked, torch.tensor([3.0, 3.0]))
    expected = math.log(V)
    assert torch.allclose(avg_ce, torch.full((K,), expected), atol=1e-6)
    assert torch.allclose(sum_ce, torch.full((K,), 3.0 * expected), atol=1e-6)


def test_ce_on_masked_zero_masked_no_divide_error() -> None:
    """When n_masked is 0 for some row, avg_ce must not raise."""
    target = torch.tensor([[0, 1, 2]])
    masked = torch.tensor([[False, False, False]])
    logits = torch.zeros(1, 3, 4)
    avg_ce, sum_ce, n_masked = ce_on_masked(logits, target, masked)
    assert n_masked.item() == 0
    assert sum_ce.item() == 0
    assert torch.isfinite(avg_ce).all()


# ---------------------------------------------------------------------------
# variant validation
# ---------------------------------------------------------------------------


def test_validate_variants_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown PLL variant"):
        _validate_variants(("seq", "totally_made_up"), allowed=PROTEIN_VARIANTS, kind="x")


def test_validate_variants_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _validate_variants((), allowed=PROTEIN_VARIANTS, kind="x")


def test_variant_constants_are_subset_of_one_another() -> None:
    """All protein-only variants must also be valid for PL (subset relationship)."""
    assert set(PROTEIN_VARIANTS).issubset(set(PROTEIN_LIGAND_VARIANTS) | {"joint_true_2"})
    # joint_true_2 is protein-only by design; joint_true_4 is PL-only
    assert "joint_true_4" in PROTEIN_LIGAND_VARIANTS
    assert "joint_true_4" not in PROTEIN_VARIANTS
    assert "joint_true_2" in PROTEIN_VARIANTS
    assert "joint_true_2" not in PROTEIN_LIGAND_VARIANTS


# ---------------------------------------------------------------------------
# End-to-end against a stub Lightning module
# ---------------------------------------------------------------------------


class _StubProteinModule:
    """Minimal stand-in for ``LeFlurSequenceStructureEncoderLightningModule``.

    Implements just enough of the contract used by
    :func:`score_protein_pll`: ``mask_token_id``, ``mask_index_struc_tokens``,
    ``training`` flag, ``eval()`` / ``train()`` no-ops, and a ``forward``
    method that returns uniform logits so every CE call evaluates to log(V)
    over the masked positions (predictable smoke check).
    """

    def __init__(self, vocab_seq: int = 32, vocab_struc: int = 1024):
        self.vocab_seq = vocab_seq
        self.vocab_struc = vocab_struc
        self.mask_token_id = vocab_seq - 1
        self.mask_index_struc_tokens = vocab_struc - 2
        self.training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward(self, x_t, mask, residue_index, conditioning_tensor, timesteps=None):
        K, L = x_t["sequence_tokens"].shape
        device = x_t["sequence_tokens"].device
        return {
            "sequence_logits": torch.zeros(K, L, self.vocab_seq, device=device),
            "structure_logits": torch.zeros(K, L, self.vocab_struc, device=device),
        }


def test_score_protein_pll_uniform_logits_gives_log_V() -> None:
    """Uniform logits ⇒ every CE = log(V); average across K and B is exact.

    Use ``L=128`` so the highest stratified-t stratum (midpoint ≈ 0.92) has
    ``prob(zero masks) ≈ 0.92^128 ≈ 3e-5`` — negligible. The canonical
    algorithm (matching the conference script) folds zero-mask draws into
    the K-mean as 0, which would bias this exactness check on small L.
    """
    from lobster.model.leflur._pll_scoring import score_protein_pll

    stub = _StubProteinModule(vocab_seq=20, vocab_struc=512)
    B, L = 3, 128
    torch.manual_seed(0)
    sequence = torch.randint(0, stub.vocab_seq - 1, (B, L))
    structure = torch.randint(0, stub.vocab_struc - 2, (B, L))
    mask = torch.ones(B, L)

    out = score_protein_pll(
        stub,
        sequence_tokens=sequence,
        structure_tokens=structure,
        mask=mask,
        K=8,
        seed=0,
        variants=("seq", "struc", "joint_protein"),
    )

    expected_seq = math.log(stub.vocab_seq)
    expected_struc = math.log(stub.vocab_struc)
    assert out["seq"].shape == (B,)
    assert out["struc"].shape == (B,)
    assert torch.allclose(out["seq"], torch.full((B,), expected_seq), atol=1e-3)
    assert torch.allclose(out["struc"], torch.full((B,), expected_struc), atol=1e-3)
    assert torch.allclose(out["joint_protein"], out["seq"] + out["struc"], atol=1e-6)


def test_score_protein_pll_rejects_pl_variant() -> None:
    from lobster.model.leflur._pll_scoring import score_protein_pll

    stub = _StubProteinModule()
    with pytest.raises(ValueError, match="Unknown PLL variant"):
        score_protein_pll(
            stub,
            sequence_tokens=torch.zeros(1, 4, dtype=torch.long),
            structure_tokens=torch.zeros(1, 4, dtype=torch.long),
            mask=torch.ones(1, 4),
            K=4,
            variants=("joint_true_4",),  # PL-only
        )
