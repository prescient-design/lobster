"""Tests for the SPARSE per-design e_lj advantage (``w_elj_scalar``).

The sparse twin of the dense ``per_token_pot`` LJ arm (see ``test_pot_reward.py``): instead of
decomposing the packed-complex interface Lennard-Jones potential per binder residue and routing it
as a per-position structure-track advantage, ``w_elj_scalar`` takes each design's TOTAL e_lj, z-scores
it within the rollout group (:meth:`LeFlurGRPOTrainer._group_zscore`), and ADDS it to the design-level
GRPO advantage — a single scalar that broadcasts identically to every generated token of every track.

The two arms are meant to be byte-identical except for credit GRANULARITY, so the self-normalized
group z-score here mirrors the per-position group normalization the dense arm applies.

Conventions asserted here (must match ``_collect_and_advantage`` / ``_repack_terms_for_group``):
* e_lj is a PENALTY (larger = worse), so the trainer z-scores ``-e_lj`` → the least-clashing design
  in the group gets the highest advantage;
* the group z-score uses population std (``unbiased=False``), matching :meth:`_advantages`, so the
  added term is on the same scale as the base GRPO advantage;
* a design whose pack failed (``NaN`` e_lj) contributes 0 advantage and does NOT corrupt the finite
  designs' standardization;
* a group with <=1 finite design has no spread → all-zero (neutral) advantage.

Lives here (not ``test_trainers.py``) so it avoids importing ``lobster.rl_training.trainers`` (which
pulls in ``trl``, absent in this env). ``_group_zscore`` is a ``@staticmethod`` — no instance needed.
"""

import numpy as np
import pytest
import torch

from lobster.rl_training import GRPOTrainerConfig, LeFlurGRPOTrainer, TargetSpec

zscore = LeFlurGRPOTrainer._group_zscore
EPS = 1e-4


class TestGroupZscore:
    """Pure-torch unit tests of the self-normalizing group z-score."""

    def test_clean_vector_zero_mean_unit_std_matches_numpy(self):
        x = torch.tensor([-50.0, -120.0, -80.0, -200.0, -30.0, -95.0])
        z = zscore(x, EPS)
        ref = (x.numpy() - x.numpy().mean()) / (x.numpy().std() + EPS)  # population std
        np.testing.assert_allclose(z.numpy(), ref, atol=1e-5)
        assert float(z.mean()) == pytest.approx(0.0, abs=1e-5)
        assert float(z.std(unbiased=False)) == pytest.approx(1.0, abs=1e-3)

    def test_uses_population_std_not_sample_std(self):
        # unbiased=False must match _advantages; a sample-std impl would be off by sqrt(n/(n-1)).
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        z = zscore(x, eps=0.0)
        ref_pop = (x - x.mean()) / x.std(unbiased=False)
        torch.testing.assert_close(z, ref_pop)
        assert not torch.allclose(z, (x - x.mean()) / x.std(unbiased=True))

    def test_nan_entry_maps_to_zero_and_does_not_corrupt_others(self):
        xn = torch.tensor([-50.0, float("nan"), -80.0, -200.0, -30.0, -95.0])
        zn = zscore(xn, EPS)
        assert float(zn[1]) == 0.0
        assert torch.isfinite(zn).all()  # the NaN did not leak into any finite entry
        fin = torch.isfinite(xn)
        vf = xn[fin].numpy()
        ref = (vf - vf.mean()) / (vf.std() + EPS)  # standardized over the FINITE entries only
        np.testing.assert_allclose(zn[fin].numpy(), ref, atol=1e-5)

    def test_penalty_sign_convention_negate_before_zscore(self):
        # The trainer z-scores -e_lj; verify worse (higher) e_lj → lower advantage after negation.
        elj = torch.tensor([-30.0, -200.0])  # design 0 has HIGHER (worse) e_lj than design 1
        adv = zscore(-elj, EPS)
        assert float(adv[0]) < float(adv[1])

    def test_degenerate_groups_are_neutral(self):
        # 0 or 1 finite entry → no spread → all zeros (a group that can't be standardized is skipped).
        torch.testing.assert_close(zscore(torch.full((4,), float("nan")), EPS), torch.zeros(4))
        torch.testing.assert_close(zscore(torch.tensor([-5.0, float("nan"), float("nan")]), EPS), torch.zeros(3))

    def test_constant_finite_vector_no_blowup(self):
        # Zero spread among finite entries: eps floor keeps it finite and ~0 (no div-by-zero).
        z = zscore(torch.tensor([-42.0, -42.0, -42.0]), EPS)
        assert torch.isfinite(z).all()
        torch.testing.assert_close(z, torch.zeros(3), atol=1e-3, rtol=0)

    def test_detaches_input(self):
        # The advantage must never carry a gradient back into the reward computation.
        x = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        z = zscore(x, EPS)
        assert not z.requires_grad


class TestConfigSmoke:
    """Config wiring: the sparse field exists, defaults off, and is mutually exclusive with dense."""

    def test_field_default_off(self):
        # Default keeps the sparse term inert → byte-identical to pre-w_elj_scalar behavior.
        c = GRPOTrainerConfig()
        assert c.w_elj_scalar == 0.0
        assert c.per_token_pot is False

    def test_dense_and_sparse_mutually_exclusive(self):
        # per_token_pot (dense) and w_elj_scalar (sparse) both route the SAME interface e_lj; enabling
        # both would double-count it. The guard fires in __init__ before the model is ever touched,
        # so a stub model is enough to reach it (targets non-empty is the only earlier requirement).
        tgt = TargetSpec(target_id="t", antigen_pdb="/x.pdb", target_chain="A", binder_length=10)
        cfg = GRPOTrainerConfig(per_token_pot=True, w_pt_lj=1.0, w_elj_scalar=1.0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LeFlurGRPOTrainer(
                model=object(),  # never reached — the guard raises first
                reward_client=None,
                targets=[tgt],
                config=cfg,
                device="cpu",
                gen_cfg=None,
            )
