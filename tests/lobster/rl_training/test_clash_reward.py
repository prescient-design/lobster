"""Tests for the smooth clash + interface-contact reward.

Two layers:

* pure-numpy unit tests of :func:`clash_contact_reward` and the interface-fraction
  band :func:`_iface_frac_band` — clash separation, the "smooth & well-behaved"
  requirement (C¹ continuity + the band's interior optimum), boundedness, and the
  key new behaviour that an *over-large* interface (the diverged-GRPO pathology) is
  scored 0;
* trainer-wiring tests of ``_clash_terms_for_group`` / ``_compute_rewards`` — the
  term is weighted, aggregated, summed into the reward, and stays Protenix-free.

The trainer tests live here (not in ``test_trainers.py``) so they avoid importing
``lobster.rl_training.trainers`` (which pulls in ``trl``, absent in this env).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import lobster.rl_training.rewards as _rewards
from lobster.rl_training import LeFlurGRPOTrainer
from lobster.rl_training.rewards import clash_contact_reward
from lobster.rl_training.rewards._clash_reward import _iface_frac_band


# ------------------------------------------------------------------ fixtures
def _make_chain(n: int, origin: np.ndarray, ca_step: float = 3.8, axis: str = "x") -> np.ndarray:
    """A simple straight backbone of ``n`` residues (CA spaced ``ca_step`` along ``axis``).

    Each residue gets non-collinear N/CA/C so the virtual Cβ is finite.
    """
    coords = np.zeros((n, 3, 3))
    step = {"x": np.array([ca_step, 0.0, 0.0]), "y": np.array([0.0, ca_step, 0.0])}[axis]
    for i in range(n):
        base = origin + step * i
        coords[i, 0] = base + np.array([-1.2, 0.0, 0.0])  # N
        coords[i, 1] = base  # CA
        coords[i, 2] = base + np.array([0.6, 1.2, 0.0])  # C
    return coords


def _complex(antigen: np.ndarray, binder: np.ndarray):
    """Assemble (coords_full, valid_mask, binder_mask) from antigen+binder chains."""
    coords = np.concatenate([antigen, binder], axis=0)
    na, nb = antigen.shape[0], binder.shape[0]
    valid = np.ones(na + nb, dtype=bool)
    binder_mask = np.array([False] * na + [True] * nb, dtype=bool)
    return coords, valid, binder_mask


# --------------------------------------------------- unit: iface-fraction band
class TestIfaceFracBand:
    """The pure asymmetric raised-cosine window (0.1 / 0.2 / 0.4 by default)."""

    def test_peak_is_one(self):
        assert _iface_frac_band(0.2, 0.1, 0.2, 0.4) == pytest.approx(1.0)

    def test_zero_at_edges_and_outside(self):
        for x in (0.0, 0.05, 0.1, 0.4, 0.5, 0.63, 1.0):
            assert _iface_frac_band(x, 0.1, 0.2, 0.4) == pytest.approx(0.0, abs=1e-12)

    def test_positive_inside(self):
        assert _iface_frac_band(0.15, 0.1, 0.2, 0.4) > 0.0
        assert _iface_frac_band(0.3, 0.1, 0.2, 0.4) > 0.0

    def test_scalar_in_scalar_out_array_preserved(self):
        assert np.isscalar(_iface_frac_band(0.2, 0.1, 0.2, 0.4))
        arr = _iface_frac_band(np.array([0.1, 0.2, 0.4]), 0.1, 0.2, 0.4)
        assert arr.shape == (3,)
        assert arr[1] == pytest.approx(1.0)

    def test_monotone_rising_then_falling(self):
        rise = _iface_frac_band(np.linspace(0.1, 0.2, 40), 0.1, 0.2, 0.4)
        fall = _iface_frac_band(np.linspace(0.2, 0.4, 40), 0.1, 0.2, 0.4)
        assert np.all(np.diff(rise) >= -1e-12)
        assert np.all(np.diff(fall) <= 1e-12)

    def test_zero_slope_at_knots(self):
        """C¹: raised-cosine ⇒ derivative → 0 at lo/peak/hi (value change is O(h²))."""
        h = 1e-4
        assert _iface_frac_band(0.1 + h, 0.1, 0.2, 0.4) < 1e-5  # rising foot
        assert 1.0 - _iface_frac_band(0.2 - h, 0.1, 0.2, 0.4) < 1e-5  # peak (left)
        assert 1.0 - _iface_frac_band(0.2 + h, 0.1, 0.2, 0.4) < 1e-5  # peak (right)
        assert _iface_frac_band(0.4 - h, 0.1, 0.2, 0.4) < 1e-5  # falling foot

    def test_asymmetric_shoulders(self):
        """Falling shoulder (0.2→0.4) is wider than rising (0.1→0.2): decays slower.

        At equal distance 0.05 from the peak, the falling side sits higher."""
        below = _iface_frac_band(0.15, 0.1, 0.2, 0.4)  # 0.05 below peak (rising)
        above = _iface_frac_band(0.25, 0.1, 0.2, 0.4)  # 0.05 above peak (falling)
        assert above > below


# ------------------------------------------------------------- unit: clash
class TestClashScore:
    def test_overlapping_chains_clash(self):
        """Binder on top of the antigen ⇒ heavy overlap ⇒ clash_score strongly penalized.

        Threshold reflects the calibrated defaults (d_clash=2.2, clash_scale=50): a
        fully-coincident 8-mer has E_clash ≈ 122 ⇒ clash_score ≈ 0.09, far below the
        clash-free case (≈1). Kept as a comparative separation check rather than a
        scale-specific absolute so it tracks the default retune.
        """
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        bd = ag.copy()  # identical position => atoms coincide
        coords, valid, binder = _complex(ag, bd)
        _, diag = clash_contact_reward(coords, valid, binder)
        assert diag["clash_score"] < 0.15
        assert diag["E_clash"] > 10.0
        # heavy overlap must score far below a well-separated docked pair
        sep = _make_chain(8, np.array([0.0, 6.0, 0.0]))
        _, diag_sep = clash_contact_reward(*_complex(ag, sep))
        assert diag["clash_score"] < 0.2 * diag_sep["clash_score"]

    def test_separated_chains_clash_free(self):
        """A well-separated docked pair ⇒ no overlap ⇒ clash_score ≈ 1."""
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        bd = _make_chain(8, np.array([0.0, 6.0, 0.0]))  # 6 Å off in y
        coords, valid, binder = _complex(ag, bd)
        _, diag = clash_contact_reward(coords, valid, binder)
        assert diag["clash_score"] > 0.95

    def test_internal_binder_clash_detected(self):
        """Two binder residues folded onto each other clash even with no antigen contact."""
        bd = _make_chain(6, np.array([0.0, 0.0, 0.0]))
        bd[5] = bd[0]  # fold residue 5 back onto residue 0 (|i-j|=5 >= seq_sep)
        ag = _make_chain(4, np.array([0.0, 40.0, 0.0]))  # far away (no cross clash)
        coords, valid, binder = _complex(ag, bd)
        _, diag = clash_contact_reward(coords, valid, binder)
        assert diag["E_clash"] > 1.0
        assert diag["clash_score"] < 1.0


class TestClashPerResidue:
    """``return_eres`` yields a per-binder-residue clash energy that sums to ``E_clash``.

    This decomposition is what the per-token clash advantage routes to the structure track,
    so it must (a) have one entry per binder residue and (b) partition ``E_clash`` exactly
    (binder×antigen pairs credited to the binder residue; binder×binder pairs split 50/50).
    """

    def test_off_by_default_and_byte_identical(self):
        """Without ``return_eres`` the diag has no per-residue key and the score is unchanged."""
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        bd = ag.copy()
        coords, valid, binder = _complex(ag, bd)
        term_a, diag_a = clash_contact_reward(coords, valid, binder)
        term_b, diag_b = clash_contact_reward(coords, valid, binder, return_eres=True)
        assert "E_clash_res" not in diag_a
        assert term_a == term_b and diag_a["E_clash"] == pytest.approx(diag_b["E_clash"])

    def test_cross_clash_sums_to_total(self):
        """Binder overlapping the antigen ⇒ per-residue energies sum to E_clash."""
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        bd = ag.copy()  # coincident ⇒ heavy cross clash
        coords, valid, binder = _complex(ag, bd)
        _, diag = clash_contact_reward(coords, valid, binder, return_eres=True)
        e_res = diag["E_clash_res"]
        assert e_res.shape == (8,)  # one entry per binder residue
        assert e_res.sum() == pytest.approx(diag["E_clash"], rel=1e-6, abs=1e-6)
        assert np.all(e_res >= 0.0)

    def test_internal_binder_clash_split_sums_to_total(self):
        """Folded binder (no antigen contact) ⇒ 50/50 endpoint split still sums to E_clash."""
        bd = _make_chain(6, np.array([0.0, 0.0, 0.0]))
        bd[5] = bd[0]
        ag = _make_chain(4, np.array([0.0, 40.0, 0.0]))
        coords, valid, binder = _complex(ag, bd)
        _, diag = clash_contact_reward(coords, valid, binder, return_eres=True)
        e_res = diag["E_clash_res"]
        assert e_res.shape == (6,)
        assert e_res.sum() == pytest.approx(diag["E_clash"], rel=1e-6, abs=1e-6)


# ----------------------------------------------------------- unit: contact
class TestContactScore:
    def _partial(self):
        """Long antigen + a clash-free parallel binder slid off the end so only a
        *few* binder residues sit at the interface (partial interface, in-band)."""
        ag = _make_chain(24, np.array([0.0, 0.0, 0.0]))  # long antigen along x
        bd = _make_chain(10, np.array([84.0, 6.0, 0.0]))  # 6 Å off, hanging off end
        return _complex(ag, bd)

    def test_partial_interface_scores_high(self):
        """A native-like partial interface (~15-25% of binder in contact) ⇒ high band."""
        coords, valid, binder = self._partial()
        term, diag = clash_contact_reward(coords, valid, binder)
        assert 0.1 < diag["iface_frac"] < 0.4  # inside the band
        assert diag["contact_score"] > 0.4
        assert diag["clash_score"] > 0.9  # clash-free (6 Å off)
        assert term > 0.4

    def test_over_large_interface_penalized(self):
        """All binder residues packed against the antigen (iface_frac ≈ 1, the
        diverged-GRPO pathology) ⇒ contact_score ≈ 0 despite being in contact."""
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        bd = _make_chain(8, np.array([0.0, 6.0, 0.0]))  # parallel, all residues near
        coords, valid, binder = _complex(ag, bd)
        term, diag = clash_contact_reward(coords, valid, binder)
        assert diag["iface_frac"] > 0.4  # over-large interface
        assert diag["contact_score"] < 0.1
        assert term < 0.1

    def test_floating_binder_no_contact(self):
        """Binder translated far away ⇒ iface_frac ≈ 0 ⇒ contact_score 0 ⇒ term 0."""
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        bd = _make_chain(8, np.array([0.0, 60.0, 0.0]))
        coords, valid, binder = _complex(ag, bd)
        term, diag = clash_contact_reward(coords, valid, binder)
        assert diag["iface_frac"] < 0.05
        assert diag["contact_score"] < 0.05
        assert term < 0.05

    def test_partial_beats_full_and_floating(self):
        """The band's essence: a partial interface beats both an over-large and an
        absent one."""
        partial = clash_contact_reward(*self._partial())[1]["contact_score"]
        ag = _make_chain(8, np.array([0.0, 0.0, 0.0]))
        full = clash_contact_reward(*_complex(ag, _make_chain(8, np.array([0.0, 6.0, 0.0]))))[1]["contact_score"]
        floating = clash_contact_reward(*_complex(ag, _make_chain(8, np.array([0.0, 60.0, 0.0]))))[1]["contact_score"]
        assert partial > full
        assert partial > floating

    def test_no_antigen_zero_contact(self):
        """No valid antigen residue ⇒ no interface ⇒ contact_score 0, term 0."""
        bd = _make_chain(6, np.array([0.0, 0.0, 0.0]))
        coords = bd
        valid = np.ones(6, dtype=bool)
        binder = np.ones(6, dtype=bool)
        term, diag = clash_contact_reward(coords, valid, binder)
        assert diag["contact_score"] == 0.0
        assert term == 0.0


# ------------------------------------------------ unit: smooth & well-behaved
class TestSmoothAndWellBehaved:
    def _separation_sweep(self, offsets):
        """Parallel chains separating in y — probes the clash transition."""
        ag = _make_chain(10, np.array([0.0, 0.0, 0.0]))
        terms, clash, contact = [], [], []
        for dy in offsets:
            bd = _make_chain(10, np.array([0.0, float(dy), 0.0]))
            t, d = clash_contact_reward(*_complex(ag, bd))
            terms.append(t)
            clash.append(d["clash_score"])
            contact.append(d["contact_score"])
        return np.array(terms), np.array(clash), np.array(contact)

    def _slide_sweep(self, shifts):
        """Slide a clash-free parallel binder off the antigen's end — smoothly
        scans the binder interface fraction from ~1 down to 0 (the band's domain)
        while clash stays ≈ 1."""
        ag = _make_chain(24, np.array([0.0, 0.0, 0.0]))  # long antigen along x
        terms, clash, contact, ifrac = [], [], [], []
        for s in shifts:
            bd = _make_chain(10, np.array([float(s), 6.0, 0.0]))  # 6 Å off in y
            t, d = clash_contact_reward(*_complex(ag, bd))
            terms.append(t)
            clash.append(d["clash_score"])
            contact.append(d["contact_score"])
            ifrac.append(d["iface_frac"])
        return np.array(terms), np.array(clash), np.array(contact), np.array(ifrac)

    def test_bounded_unit_interval(self):
        for arrs in (
            self._separation_sweep(np.linspace(0.0, 30.0, 61))[:3],
            self._slide_sweep(np.linspace(0.0, 95.0, 61))[:3],
        ):
            for arr in arrs:
                assert arr.min() >= 0.0 - 1e-9
                assert arr.max() <= 1.0 + 1e-9

    def test_no_jump_discontinuity(self):
        """Continuity: refining the slide grid shrinks the max successive step.

        A jump discontinuity would leave a floor no matter how fine the grid; a
        smooth function's max step falls with the sampling step."""
        coarse = np.abs(np.diff(self._slide_sweep(np.arange(0.0, 95.0, 0.5))[0])).max()
        fine = np.abs(np.diff(self._slide_sweep(np.arange(0.0, 95.0, 0.1))[0])).max()
        assert fine < coarse  # refining the grid shrinks the step => continuous
        assert fine < 0.05  # and the fine-grid step is small in absolute terms

    def test_clash_monotone_improves_with_separation(self):
        """clash_score is monotone nondecreasing as the chains separate."""
        _, clash, _ = self._separation_sweep(np.linspace(0.5, 12.0, 40))
        assert np.all(np.diff(clash) >= -1e-9)

    def test_contact_band_interior_optimum(self):
        """contact_score peaks at an interior interface fraction, not at the
        over-large (slide≈0) or empty (slide large) extremes."""
        shifts = np.linspace(0.0, 95.0, 96)
        _, _, contact, ifrac = self._slide_sweep(shifts)
        best = int(np.argmax(contact))
        assert 0 < best < len(shifts) - 1
        assert contact[best] > contact[0]  # beats the over-large extreme
        assert contact[best] > contact[-1]  # beats the empty extreme
        assert 0.1 < ifrac[best] < 0.4  # optimum sits inside the band

    def test_term_interior_optimum(self):
        """The full clash·contact term is maximised at an interior configuration."""
        shifts = np.linspace(0.0, 95.0, 96)
        terms, _, _, _ = self._slide_sweep(shifts)
        best = int(np.argmax(terms))
        assert 0 < best < len(shifts) - 1
        assert terms[best] > terms[0]
        assert terms[best] > terms[-1]


# ---------------------------------------------------------- trainer wiring
def _clash_cfg(**overrides):
    cfg = dict(
        w_clash_contact=1.0,
        clash_d_clash=3.0,
        clash_soft=0.5,
        clash_scale=4.0,
        contact_d0=8.0,
        contact_soft=1.0,
        frac_lo=0.1,
        frac_peak=0.2,
        frac_hi=0.4,
        clash_seq_sep=2,
        clash_include_cb=True,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestClashTermsForGroup:
    def test_weights_and_aggregates(self, monkeypatch):
        """Term is weighted by w_clash_contact and clash/* diagnostics are averaged."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _clash_cfg(w_clash_contact=2.0)
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 5, 3, 3))
        comp = {"mask": torch.ones(1, 5), "binder_positions": torch.ones(5)}

        diags = iter(
            [
                (
                    0.5,
                    {
                        "clash_score": 0.9,
                        "contact_score": 0.55,
                        "E_clash": 0.4,
                        "soft_n_iface": 1.6,
                        "iface_frac": 0.20,
                    },
                ),
                (
                    0.2,
                    {"clash_score": 0.4, "contact_score": 0.5, "E_clash": 3.6, "soft_n_iface": 1.2, "iface_frac": 0.15},
                ),
                (
                    0.0,
                    {
                        "clash_score": 0.0,
                        "contact_score": 0.3,
                        "E_clash": 20.0,
                        "soft_n_iface": 4.0,
                        "iface_frac": 0.50,
                    },
                ),
            ]
        )
        monkeypatch.setattr(_rewards, "clash_contact_reward", lambda *a, **k: next(diags))

        weighted, metrics = trainer._clash_terms_for_group(trajectory={}, comp=comp)

        assert weighted == pytest.approx([1.0, 0.4, 0.0])  # 2.0 * term
        assert metrics["reward/clash_term_mean"] == pytest.approx((1.0 + 0.4 + 0.0) / 3)
        assert metrics["clash/clash_score_mean"] == pytest.approx((0.9 + 0.4 + 0.0) / 3)
        assert metrics["clash/contact_score_mean"] == pytest.approx((0.55 + 0.5 + 0.3) / 3)
        assert metrics["clash/E_clash_mean"] == pytest.approx((0.4 + 3.6 + 20.0) / 3)
        assert metrics["clash/soft_n_iface_mean"] == pytest.approx((1.6 + 1.2 + 4.0) / 3)
        assert metrics["clash/iface_frac_mean"] == pytest.approx((0.20 + 0.15 + 0.50) / 3)

    def test_uses_shared_gen_bb_without_decoding(self):
        """When gen_bb is provided, the helper does not decode again."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _clash_cfg()
        trainer._decode_backbone_coords = Mock(side_effect=AssertionError("must not decode"))
        comp = {"mask": torch.ones(1, 5), "binder_positions": torch.ones(5)}
        gen_bb = np.zeros((2, 5, 3, 3))
        weighted, _ = trainer._clash_terms_for_group(trajectory={}, comp=comp, gen_bb=gen_bb)
        assert len(weighted) == 2
        trainer._decode_backbone_coords.assert_not_called()


class TestClashTermsPerResidue:
    """``_clash_terms_for_group(return_eres=True)`` scatters per-residue energy to (G, L)."""

    def test_return_eres_scatters_and_sums(self, monkeypatch):
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _clash_cfg(w_clash_contact=1.0)
        # L=5: positions 0-1 antigen, 2-4 binder (all valid).
        comp = {"mask": torch.ones(1, 5), "binder_positions": torch.tensor([0, 0, 1, 1, 1])}
        gen_bb = np.zeros((2, 5, 3, 3), dtype=np.float32)
        diags = iter(
            [
                (
                    0.3,
                    {
                        "clash_score": 0.9,
                        "contact_score": 0.5,
                        "E_clash": 6.0,
                        "soft_n_iface": 1.0,
                        "iface_frac": 0.2,
                        "E_clash_res": np.array([1.0, 2.0, 3.0]),
                    },
                ),
                (
                    0.1,
                    {
                        "clash_score": 0.4,
                        "contact_score": 0.5,
                        "E_clash": 9.0,
                        "soft_n_iface": 1.0,
                        "iface_frac": 0.3,
                        "E_clash_res": np.array([4.0, 0.0, 5.0]),
                    },
                ),
            ]
        )
        monkeypatch.setattr(_rewards, "clash_contact_reward", lambda *a, **k: next(diags))

        weighted, metrics, e_res_full = trainer._clash_terms_for_group(
            trajectory={}, comp=comp, gen_bb=gen_bb, return_eres=True
        )
        assert e_res_full.shape == (2, 5)
        assert np.all(e_res_full[:, :2] == 0.0)  # antigen positions carry no binder clash
        np.testing.assert_allclose(e_res_full[0, 2:], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(e_res_full[1, 2:], [4.0, 0.0, 5.0])
        # per-design row-sum equals E_clash for that design
        np.testing.assert_allclose(e_res_full.sum(axis=1), [6.0, 9.0])
        assert weighted == pytest.approx([0.3, 0.1])  # w_clash_contact=1.0


class TestStructPosAdvantage:
    """Per-position structure advantage: design advantage + per-residue clash credit."""

    def _trainer(self, w_pt_clash: float = 1.0):
        t = object.__new__(LeFlurGRPOTrainer)
        t.device = "cpu"
        t.config = SimpleNamespace(w_pt_clash=w_pt_clash, adv_eps=1e-6)
        return t

    def test_zero_clash_reduces_to_design_adv(self):
        t = self._trainer()
        A = t._struct_pos_advantage(torch.zeros(3, 4), torch.tensor([1.0, -0.5, 0.25]), torch.ones(4))
        assert A.shape == (3, 4)
        torch.testing.assert_close(A, torch.tensor([1.0, -0.5, 0.25]).unsqueeze(1).expand(3, 4).contiguous())

    def test_lower_clash_gets_higher_advantage(self):
        t = self._trainer()
        clash = torch.tensor([[0.0, 0.0], [10.0, 0.0]])  # design 0 clashes less at position 0
        A = t._struct_pos_advantage(clash, torch.zeros(2), torch.ones(2))
        assert A[0, 0] > A[1, 0]  # less clash -> higher advantage
        assert A[0, 1] == pytest.approx(A[1, 1])  # equal clash at position 1 -> equal

    def test_offmask_position_has_no_clash_credit(self):
        t = self._trainer()
        clash = torch.tensor([[5.0, 1.0], [0.0, 2.0]])
        design_adv = torch.tensor([0.3, -0.2])
        A = t._struct_pos_advantage(clash, design_adv, torch.tensor([1.0, 0.0]))  # position 1 not generated
        torch.testing.assert_close(A[:, 1], design_adv)  # only broadcast design adv survives

    def test_w_pt_clash_scales_credit_linearly(self):
        clash = torch.tensor([[0.0, 0.0], [10.0, 4.0]])
        A1 = self._trainer(1.0)._struct_pos_advantage(clash, torch.zeros(2), torch.ones(2))
        A2 = self._trainer(2.0)._struct_pos_advantage(clash, torch.zeros(2), torch.ones(2))
        torch.testing.assert_close(A2, 2.0 * A1)  # design_adv=0 -> advantage is pure clash credit


class TestComputeRewardsClashGate:
    def _cfg(self, **overrides):
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
            w_clash_contact=1.0,
            w_chainbreak=0.0,
            w_rog=0.0,
            w_shape=0.0,
            w_sc_clash=0.0,
            w_aar=0.0,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def test_clash_summed_and_protenix_free(self):
        """Clash-only reward ⇒ score_group untouched, reward = clash term, metrics flow."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = self._cfg(w_clash_contact=1.0)
        trainer.reward_client = Mock()
        trainer.reward_client.score_group.side_effect = AssertionError("score_group must not be called")
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 4, 3, 3))
        clash_terms = [0.2, 0.6, 0.9]
        trainer._clash_terms_for_group = Mock(
            return_value=(clash_terms, {"reward/clash_term_mean": 0.5666, "clash/clash_score_mean": 0.7})
        )

        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["ACDE", "FGHI", "KLMN"], tri_seqs=None, trajectory={}, comp={}
        )

        trainer.reward_client.score_group.assert_not_called()
        # decode happens exactly once and is shared into the clash helper.
        trainer._decode_backbone_coords.assert_called_once()
        _, kwargs = trainer._clash_terms_for_group.call_args
        assert kwargs["gen_bb"] is not None
        assert rewards.tolist() == pytest.approx(clash_terms)
        assert metrics["clash/clash_score_mean"] == pytest.approx(0.7)
        assert metrics["reward/confidence_term_mean"] == 0.0
