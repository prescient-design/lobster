"""Tests for the per-token all-atom interface-potential structure advantage.

The "second reward set" (``per_token_pot``): three per-binder-residue potentials computed
on the LigandMPNN pack — bounded LJ energy (``lj_eres``), buried ΔSASA (``dsasa_eres``) and
interface H-bonds (``hb_eres``) — routed as a per-position advantage to the STRUCTURE track,
ADDITIVE with the per-token clash / chain-break arms. These validate the trainer wiring
(:meth:`LeFlurGRPOTrainer._struct_pos_advantage`) rather than the potentials themselves
(those are validated offline in ``scripts/_packed_potentials*``).

Conventions asserted here (must match ``_repack_terms_for_group`` collection):
* every ``*_eres`` is passed in "larger = worse" convention (``lj_eres`` is already a penalty;
  ``dsasa``/``n_hb`` are negated at collection), so :meth:`_pos_norm_adv` gives MORE advantage
  to residues with less penalty / more buried area / more H-bonds;
* each arm is scaled by its own weight (``w_pt_lj`` / ``w_pt_dsasa`` / ``w_pt_hb``) and summed;
* off-mask (non-generated) positions carry only the broadcast design advantage.

Lives here (not ``test_trainers.py``) to avoid importing ``lobster.rl_training.trainers``
(which pulls in ``trl``, absent in this env).
"""

from types import SimpleNamespace

import pytest
import torch

from lobster.rl_training import LeFlurGRPOTrainer


class TestStructPosAdvantagePotentials:
    """Per-position structure advantage with the interface-potential arms."""

    def _trainer(self, *, w_pt_lj=1.0, w_pt_dsasa=1.0, w_pt_hb=1.0, w_pt_clash=1.0, w_pt_chainbreak=1.0):
        t = object.__new__(LeFlurGRPOTrainer)
        t.device = "cpu"
        t.config = SimpleNamespace(
            w_pt_clash=w_pt_clash,
            w_pt_chainbreak=w_pt_chainbreak,
            w_pt_lj=w_pt_lj,
            w_pt_dsasa=w_pt_dsasa,
            w_pt_hb=w_pt_hb,
            adv_eps=1e-6,
        )
        return t

    def test_none_potentials_reduce_to_design_adv(self):
        # With no per-residue signal the advantage is the broadcast design advantage (G, 1).
        t = self._trainer()
        design_adv = torch.tensor([1.0, -0.5, 0.25])
        A = t._struct_pos_advantage(None, design_adv, torch.ones(4))
        assert A.shape == (3, 1)
        torch.testing.assert_close(A, design_adv.unsqueeze(1))

    def test_lower_lj_penalty_gets_higher_advantage(self):
        t = self._trainer()
        lj = torch.tensor([[0.0, 0.0], [10.0, 0.0]])  # design 0 has less LJ penalty at position 0
        A = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), lj_eres=lj)
        assert A[0, 0] > A[1, 0]  # less penalty -> higher advantage
        assert A[0, 1] == pytest.approx(A[1, 1])  # equal penalty at position 1 -> equal

    def test_more_buried_sasa_gets_higher_advantage(self):
        # dsasa_eres is stored NEGATED (larger buried area -> more negative eres -> less penalty).
        t = self._trainer()
        dsasa = torch.tensor([[-30.0, 0.0], [0.0, 0.0]])  # design 0 buries more area at position 0
        A = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), dsasa_eres=dsasa)
        assert A[0, 0] > A[1, 0]

    def test_more_hbonds_get_higher_advantage(self):
        # hb_eres is stored NEGATED (more H-bonds -> more negative eres -> less penalty).
        t = self._trainer()
        hb = torch.tensor([[-2.0, 0.0], [0.0, 0.0]])  # design 0 makes more H-bonds at position 0
        A = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), hb_eres=hb)
        assert A[0, 0] > A[1, 0]

    def test_arms_are_additive(self):
        t = self._trainer()
        lj = torch.tensor([[0.0, 0.0], [10.0, 4.0]])
        dsasa = torch.tensor([[-5.0, -1.0], [0.0, 0.0]])
        hb = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
        A_lj = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), lj_eres=lj)
        A_ds = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), dsasa_eres=dsasa)
        A_hb = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), hb_eres=hb)
        A_all = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), lj_eres=lj, dsasa_eres=dsasa, hb_eres=hb)
        torch.testing.assert_close(A_all, A_lj + A_ds + A_hb)

    def test_additive_with_clash_and_chainbreak(self):
        t = self._trainer()
        clash = torch.tensor([[0.0, 0.0], [8.0, 2.0]])
        cb = torch.tensor([[0.0, 0.0], [0.5, 0.1]])
        lj = torch.tensor([[0.0, 0.0], [10.0, 4.0]])
        A_c = t._struct_pos_advantage(clash, torch.zeros(2), torch.ones(2))
        A_cb = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), chainbreak_eres=cb)
        A_lj = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), lj_eres=lj)
        A_all = t._struct_pos_advantage(clash, torch.zeros(2), torch.ones(2), chainbreak_eres=cb, lj_eres=lj)
        torch.testing.assert_close(A_all, A_c + A_cb + A_lj)

    def test_weights_scale_each_arm_linearly(self):
        lj = torch.tensor([[0.0, 0.0], [10.0, 4.0]])
        A1 = self._trainer(w_pt_lj=1.0)._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), lj_eres=lj)
        A2 = self._trainer(w_pt_lj=2.0)._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), lj_eres=lj)
        torch.testing.assert_close(A2, 2.0 * A1)

    def test_offmask_position_has_no_potential_credit(self):
        t = self._trainer()
        lj = torch.tensor([[5.0, 1.0], [0.0, 2.0]])
        design_adv = torch.tensor([0.3, -0.2])
        A = t._struct_pos_advantage(None, design_adv, torch.tensor([1.0, 0.0]), lj_eres=lj)  # pos 1 not generated
        torch.testing.assert_close(A[:, 1], design_adv)
