"""Tests for the Protenix-free (`need_conf`) gate in LeFlurGRPOTrainer._compute_rewards.

When every confidence weight is 0 and no structure coords are needed (the *_noptx
distribution-reward ablation runs), the trainer must skip the Protenix oracle
entirely — `score_group` is never called and the reward reduces to the pure
distribution term. Lives in its own module so it does not import
`lobster.rl_training.trainers` (which pulls in `trl`, absent in this env).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import lobster.rl_training.rewards as _rewards
from lobster.rl_training import LeFlurGRPOTrainer


def _zero_conf_cfg(**overrides):
    """A GRPO config namespace with every confidence/structure weight at 0."""
    cfg = dict(
        w_iptm=0.0,
        w_ptm=0.0,
        w_abag_iptm=0.0,
        w_plddt=0.0,
        w_gpde=0.0,
        w_pae_global=0.0,
        w_pae_interface=0.0,
        w_sctm_binder=0.0,
        w_sctm_complex=0.0,
        log_struct_diagnostic=False,
        log_dist_diagnostic=False,
        per_token_clash=False,
        per_token_chainbreak=False,
        w_seq_diversity=0.0,
        w_struct_diversity=0.0,
        w_aa_dist=0.0,
        w_3di_dist=0.0,
        dist_binder_frac=0.0,
        w_clash_contact=0.0,
        w_chainbreak=0.0,
        w_shape=0.0,
        w_sc_clash=0.0,
        w_aar=0.0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestComputeRewardsProtenixGate:
    """The `need_conf` gate: a Protenix-free reward must never call the oracle."""

    def test_no_conf_weights_skips_score_group(self):
        """All conf weights 0 + no coords ⇒ score_group untouched, reward = dist term."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _zero_conf_cfg(w_3di_dist=1.0)  # 3Di-only ablation
        # score_group would block on an empty queue if ever called — make it loud so a
        # regressed gate fails the test loudly instead of hanging.
        trainer.reward_client = Mock()
        trainer.reward_client.score_group.side_effect = AssertionError("score_group must not be called")

        # gen_bb is now decoded once at the _compute_rewards level (shared by the dist
        # and clash helpers) whenever need_dist or need_clash — mock it so the gate test
        # does not hit the real CPU-VIT decoder.
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 4, 3, 3))

        dist_terms = [0.1, 0.7, 0.4]
        trainer._distribution_terms_for_group = Mock(return_value=(dist_terms, {"dist/tv_3di": 0.3}))

        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["ACDE", "FGHI", "KLMN"], tri_seqs=None, trajectory={}, comp={}
        )

        trainer.reward_client.score_group.assert_not_called()
        assert rewards.tolist() == pytest.approx(dist_terms)
        # No confidence contribution and the dist diagnostic flows through.
        assert metrics["reward/confidence_term_mean"] == 0.0
        assert metrics["conf/pass_rate"] == 0.0
        assert metrics["conf/scored_frac"] == 0.0
        assert metrics["dist/tv_3di"] == pytest.approx(0.3)

    def test_conf_weight_on_calls_score_group(self):
        """A nonzero conf weight ⇒ the oracle IS queried (no accidental skip)."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _zero_conf_cfg(w_abag_iptm=1.0)
        trainer.reward_client = Mock()
        trainer.reward_client.score_group.return_value = [None, None]

        rewards, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["ACDE", "FGHI"], tri_seqs=None, trajectory={}, comp={}
        )

        trainer.reward_client.score_group.assert_called_once()
        assert rewards.tolist() == pytest.approx([0.0, 0.0])


class TestInterfaceCollapsePenalty:
    """`dist_min_iface`/`dist_iface_penalty`: a collapsed interface gets the penalty."""

    def _harness(self, monkeypatch, cfg_over: dict):
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _zero_conf_cfg(w_3di_dist=1.0, dist_metric="tv", **cfg_over)
        trainer._dist_ref = {}
        # 3 designs, L=4; coords/masks are only consumed by the (patched) hist fn.
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 4, 3, 3))
        comp = {
            "mask": torch.ones(1, 4),
            "binder_positions": torch.ones(4),
        }
        monkeypatch.setattr(_rewards, "reference_for", lambda ref, tid: ({}, {}, {}, {}, "target"))
        # design 0 collapses (n_iface below threshold); 1 and 2 are healthy. Tuple is
        # (h_aa_i, h_3di_i, h_aa_b, h_3di_b, n_iface, n_binder).
        hists = iter(
            [
                (None, None, None, None, 2, 4),
                ({"A": 1.0}, {"A": 1.0}, {"A": 1.0}, {"A": 1.0}, 20, 80),
                ({"A": 1.0}, {"A": 1.0}, {"A": 1.0}, {"A": 1.0}, 25, 90),
            ]
        )
        monkeypatch.setattr(_rewards, "design_hists_scoped", lambda *a, **k: next(hists))
        monkeypatch.setattr(
            _rewards,
            "combined_distribution_terms",
            lambda *a, **k: (
                0.6,
                {
                    "tv_aa": None,
                    "tv_3di": 0.4,
                    "js_aa": None,
                    "js_3di": None,
                    "tv_aa_binder": None,
                    "tv_3di_binder": 0.3,
                    "js_aa_binder": None,
                    "js_3di_binder": None,
                },
            ),
        )
        return trainer, comp

    def test_negative_penalty_applied_to_collapsed_design(self, monkeypatch):
        trainer, comp = self._harness(monkeypatch, {"dist_min_iface": 4, "dist_iface_penalty": -1.0})
        weighted, metrics = trainer._distribution_terms_for_group(
            target_id="t0", seqs=["AAAA", "CDEF", "GHIK"], trajectory={}, comp=comp
        )
        assert weighted == pytest.approx([-1.0, 0.6, 0.6])  # collapsed -> penalty; others -> term
        assert metrics["dist/frac_penalized"] == pytest.approx(1 / 3)
        assert metrics["dist/n_iface_mean"] == pytest.approx((2 + 20 + 25) / 3)

    def test_default_reproduces_soft_zero(self, monkeypatch):
        """Defaults (min=4, penalty=0.0) reproduce the old skip->0 behaviour."""
        trainer, comp = self._harness(monkeypatch, {"dist_min_iface": 4, "dist_iface_penalty": 0.0})
        weighted, metrics = trainer._distribution_terms_for_group(
            target_id="t0", seqs=["AAAA", "CDEF", "GHIK"], trajectory={}, comp=comp
        )
        assert weighted == pytest.approx([0.0, 0.6, 0.6])
        assert metrics["dist/frac_penalized"] == pytest.approx(1 / 3)
