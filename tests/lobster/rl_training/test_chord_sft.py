"""Unit tests for the CHORD SFT-distillation term.

Covers the alphabet remap (:func:`binder_letters_to_aa33`), the per-step φ-weighted CE
helper (``_sft_step_ce``) and the payload-assembly conventions the trainer relies on. The
model-level helpers are exercised through a tiny stand-in object because ``_sft_step_ce`` is
a pure function of its arguments (uses no model state), so we can bind it to a dummy.
"""

import numpy as np
import torch

from lobster.rl_training.rewards import SFT_IGNORE_INDEX, binder_letters_to_aa33


def test_binder_letters_to_aa33_roundtrip():
    from lobster.tokenization._amino_acid import AA_VOCAB

    seq = "ACDEFGHIKLMNPQRSTVWY"
    ids = binder_letters_to_aa33(seq)
    assert ids.shape == (20,)
    for c, i in zip(seq, ids):
        assert int(AA_VOCAB[c]) == int(i)
    # Non-standard / gap letters map to the ignore sentinel (< 0).
    assert (binder_letters_to_aa33("XZ*-") == SFT_IGNORE_INDEX).all()
    # Poly-Ala is a real (valid) target — not ignored.
    assert (binder_letters_to_aa33("AAAA") == int(AA_VOCAB["A"])).all()


class _Dummy:
    """Bind the unbound model methods so we can call them without building a model."""

    from lobster.model.leflur._leflur_sequence_structure_encoder_lightning_module import (
        LeFlurSequenceStructureEncoderLightningModule as _M,
    )

    _sft_step_ce = _M._sft_step_ce


def _ce_step(logits, target, sup, xt_seq, gen_mask_seq, mask_index_seq, **kw):
    return _Dummy()._sft_step_ce(logits, target, sup, xt_seq, gen_mask_seq, mask_index_seq, **kw)


def _base_kwargs():
    return dict(label="hard", soft_targets=None, temperature=1.0, masked_only=True, use_phi=True)


def test_sft_step_ce_finite_and_zero_when_nothing_supervised():
    B, L, V = 2, 5, 33
    logits = torch.randn(B, L, V)
    target = torch.full((B, L), SFT_IGNORE_INDEX, dtype=torch.long)
    sup = torch.zeros(B, L, dtype=torch.bool)
    xt = torch.zeros(B, L, dtype=torch.long)  # all "masked" == mask_index_seq
    gen = torch.ones(B, L, dtype=torch.bool)
    out = _ce_step(logits, target, sup, xt, gen, 0, **_base_kwargs())
    assert torch.isfinite(out)
    assert float(out) == 0.0  # nothing supervised => zero


def test_sft_step_ce_masked_only_restriction():
    B, L, V = 1, 4, 33
    logits = torch.randn(B, L, V)
    target = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    sup = torch.ones(B, L, dtype=torch.bool)
    gen = torch.ones(B, L, dtype=torch.bool)
    mask_id = 32
    # Only positions 0,2 are still masked; 1,3 already resolved -> excluded by masked_only.
    xt = torch.tensor([[mask_id, 4, mask_id, 4]], dtype=torch.long)
    kw = _base_kwargs()
    kw["use_phi"] = False  # plain mean CE so we can reason about which positions count
    out_masked = _ce_step(logits, target, sup, xt, gen, mask_id, **kw)
    # With masked_only off, all four positions count -> different value.
    kw2 = dict(kw)
    kw2["masked_only"] = False
    out_all = _ce_step(logits, target, sup, xt, gen, mask_id, **kw2)
    assert torch.isfinite(out_masked) and torch.isfinite(out_all)
    assert abs(float(out_masked) - float(out_all)) > 1e-6


def test_phi_zeros_confident_and_rejected_tokens():
    # φ = p_t(1-p_t): a position where the policy is near-certain of the target (p≈1) OR
    # near-certain against it (p≈0) must contribute ~0; a p≈0.5 position dominates.
    B, L, V = 1, 3, 33
    logits = torch.full((B, L, V), -20.0)
    target = torch.tensor([[5, 6, 7]], dtype=torch.long)
    # pos0: policy certain OF target 5 (p_t≈1) -> phi≈0
    logits[0, 0, 5] = 20.0
    # pos1: policy certain AGAINST target 6 (mass on 10) -> p_t≈0 -> phi≈0
    logits[0, 1, 10] = 20.0
    # pos2: policy split 50/50 between target 7 and 12 -> p_t≈0.5 -> phi≈0.25 (max)
    logits[0, 2, 7] = 0.0
    logits[0, 2, 12] = 0.0
    sup = torch.ones(B, L, dtype=torch.bool)
    gen = torch.ones(B, L, dtype=torch.bool)
    xt = torch.zeros(B, L, dtype=torch.long)
    # Recompute φ the same way the helper does, to assert the weighting profile.
    p = torch.softmax(logits, dim=-1)
    p_t = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    phi = (p_t * (1 - p_t))[0]
    assert phi[0] < 1e-3 and phi[1] < 1e-3  # confident + rejected -> ~0
    assert phi[2] > 0.2  # contested -> dominates
    out = _ce_step(logits, target, sup, xt, gen, 0, **_base_kwargs())
    assert torch.isfinite(out) and float(out) > 0.0


def test_sft_step_ce_soft_requires_targets():
    # sequence_sft_loss raises early if soft is requested without a distribution; the
    # per-step helper itself dereferences soft_targets, so passing None must not silently
    # fall through to a hard path.
    B, L, V = 1, 3, 33
    logits = torch.randn(B, L, V)
    target = torch.tensor([[5, 6, 7]], dtype=torch.long)
    sup = torch.ones(B, L, dtype=torch.bool)
    gen = torch.ones(B, L, dtype=torch.bool)
    xt = torch.zeros(B, L, dtype=torch.long)
    kw = dict(label="soft", soft_targets=None, temperature=1.0, masked_only=True, use_phi=True)
    try:
        _ce_step(logits, target, sup, xt, gen, 0, **kw)
    except (AttributeError, TypeError):
        return  # dereferencing None distribution is the expected failure
    raise AssertionError("soft label with soft_targets=None must not succeed")


def test_sft_step_ce_soft_onehot_equals_hard():
    # A one-hot soft target at the (mode) identity must reproduce the hard-CE value exactly
    # (φ off so both reduce to plain mask-normalized mean CE over the same positions).
    B, L, V = 1, 4, 33
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    target = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    sup = torch.ones(B, L, dtype=torch.bool)
    gen = torch.ones(B, L, dtype=torch.bool)
    xt = torch.zeros(B, L, dtype=torch.long)  # all masked
    onehot = torch.zeros(B, L, V)
    onehot.scatter_(-1, target.unsqueeze(-1), 1.0)

    hard = _ce_step(
        logits,
        target,
        sup,
        xt,
        gen,
        0,
        label="hard",
        soft_targets=None,
        temperature=1.0,
        masked_only=True,
        use_phi=False,
    )
    soft = _ce_step(
        logits,
        target,
        sup,
        xt,
        gen,
        0,
        label="soft",
        soft_targets=onehot,
        temperature=1.0,
        masked_only=True,
        use_phi=False,
    )
    assert torch.isfinite(soft)
    assert torch.allclose(hard, soft, atol=1e-5)


def test_sft_step_ce_soft_phi_keys_on_mode_target():
    # Soft distillation still keys φ on the hard `target` (the teacher mode). A diffuse
    # soft distribution over many AAs must not change the φ weighting, only the CE numerator.
    B, L, V = 1, 3, 33
    logits = torch.full((B, L, V), -20.0)
    target = torch.tensor([[5, 6, 7]], dtype=torch.long)
    logits[0, 0, 5] = 20.0  # policy certain OF mode -> phi≈0
    logits[0, 1, 10] = 20.0  # policy certain AGAINST mode -> phi≈0
    logits[0, 2, 7] = 0.0  # contested -> phi≈0.25
    logits[0, 2, 12] = 0.0
    # A soft target that spreads mass across a few AAs (mode still at `target`).
    soft = torch.zeros(B, L, V)
    for b_pos, mode in enumerate([5, 6, 7]):
        soft[0, b_pos, mode] = 0.6
        soft[0, b_pos, (mode + 1) % V] = 0.4
    sup = torch.ones(B, L, dtype=torch.bool)
    gen = torch.ones(B, L, dtype=torch.bool)
    xt = torch.zeros(B, L, dtype=torch.long)
    out = _ce_step(
        logits,
        target,
        sup,
        xt,
        gen,
        0,
        label="soft",
        soft_targets=soft,
        temperature=1.0,
        masked_only=True,
        use_phi=True,
    )
    assert torch.isfinite(out) and float(out) > 0.0


def test_expert_context_seq_uses_expert_at_revealed_ignores_policy():
    """_expert_context_seq: revealed+valid-target positions carry the EXPERT token (independent
    of the policy token there); masked positions keep the mask; no-expert positions keep xt."""
    M = _Dummy._M
    mask_id = 32
    #        pos: 0=masked     1=revealed 2=masked     3=revealed 4=revealed(no expert)
    xt = torch.tensor([[mask_id, 4, mask_id, 7, 9]], dtype=torch.long)
    target = torch.tensor([[5, 6, 7, 8, SFT_IGNORE_INDEX]], dtype=torch.long)
    ctx = M._expert_context_seq(xt, target, mask_id)
    assert ctx.tolist() == [[mask_id, 6, mask_id, 8, 9]]
    # Invariance: perturbing the POLICY token at a revealed+expert position does not change ctx
    # there; a revealed position with NO expert target (pos4) tracks the policy token.
    xt2 = xt.clone()
    xt2[0, 1] = 30  # revealed+expert -> still expert (6), independent of policy
    xt2[0, 3] = 2  # revealed+expert -> still expert (8), independent of policy
    xt2[0, 4] = 2  # revealed, no expert target -> tracks policy (2)
    ctx2 = M._expert_context_seq(xt2, target, mask_id)
    assert ctx2.tolist() == [[mask_id, 6, mask_id, 8, 2]]


def _sft_traj_and_policy(seed: int = 3):
    """A tiny 1-design, length-4 rollout (pos0 fixed antigen; pos1..3 generated binder) with
    a deterministic per-step seq mask pattern, plus a matching expert target/supervise mask."""
    from .test_grpo_trajectory_recompute import SEQ_MASK, _FakePolicy, _make_trajectory

    traj = _make_trajectory(batch_size=1, length=4, nsteps=2, seed=seed)
    # Deterministic seq states: pos0 revealed antigen(7); binder pos1..3 partially masked.
    traj["steps"][0]["xt"]["sequence_tokens"] = torch.tensor([[7, SEQ_MASK, SEQ_MASK, 9]], dtype=torch.int32)
    traj["steps"][1]["xt"]["sequence_tokens"] = torch.tensor([[7, 3, SEQ_MASK, 9]], dtype=torch.int32)
    # Expert (LigandMPNN) target: antigen pos0 ignored, whole binder pos1..3 valid.
    target = torch.tensor([[SFT_IGNORE_INDEX, 5, 6, 8]], dtype=torch.long)
    sup = torch.tensor([[False, True, True, True]], dtype=torch.bool)
    return _FakePolicy(seed=0), traj, target, sup


def test_sequence_sft_loss_invariant_to_policy_context_sensitive_to_expert():
    """End-to-end guardrail that the mixing bug is fixed: the SFT loss must NOT depend on the
    policy's own revealed tokens (they are replaced by the expert context) and MUST depend on
    the expert target identities."""
    policy, traj, target, sup = _sft_traj_and_policy()
    base = policy.sequence_sft_loss(traj, target, sup, use_phi=False)
    assert torch.isfinite(base) and float(base.detach()) > 0.0

    # (1) Invariance: perturb the policy's revealed binder tokens (pos1 @ step1, pos3 @ both
    # steps) — all revealed positions carry valid expert targets, so ctx is unchanged.
    traj_p = policy.sequence_sft_loss  # alias for readability
    traj["steps"][1]["xt"]["sequence_tokens"][0, 1] = 30  # was 3
    traj["steps"][0]["xt"]["sequence_tokens"][0, 3] = 2  # was 9
    traj["steps"][1]["xt"]["sequence_tokens"][0, 3] = 2  # was 9
    after_policy = traj_p(traj, target, sup, use_phi=False)
    assert torch.allclose(base, after_policy, atol=1e-6), "SFT loss leaked the policy's own context"

    # (2) Sensitivity: changing the expert target at a masked-and-supervised position must move it.
    target2 = target.clone()
    target2[0, 2] = 7  # was 6 (pos2 is masked in both steps -> supervised)
    after_expert = policy.sequence_sft_loss(traj, target2, sup, use_phi=False)
    assert not torch.allclose(base, after_expert, atol=1e-6), "SFT loss ignored the expert target"


def test_fused_sft_matches_separate_with_expert_context():
    """logprob_and_sft_over_trajectory's SFT term equals sequence_sft_loss (both expert-context),
    and its log-prob equals logprob_over_trajectory — across φ / masked_only / label settings."""
    tracks = ("sequence_tokens", "structure_tokens", "tri_tokens")
    for label in ("hard", "soft"):
        for use_phi in (True, False):
            for masked_only in (True, False):
                policy, traj, target, sup = _sft_traj_and_policy(seed=5)
                soft_targets = None
                if label == "soft":
                    V = 33
                    soft_targets = torch.zeros(target.shape[0], target.shape[1], V)
                    tgt_c = target.clamp(min=0)
                    soft_targets.scatter_(-1, tgt_c.unsqueeze(-1), 1.0)  # one-hot at the expert id
                kw = dict(label=label, soft_targets=soft_targets, masked_only=masked_only, use_phi=use_phi)
                sep_sft = policy.sequence_sft_loss(traj, target, sup, **kw)
                sep_lp = policy.logprob_over_trajectory(traj, tracks=tracks)
                lp, sft = policy.logprob_and_sft_over_trajectory(traj, target, sup, tracks=tracks, **kw)
                assert torch.allclose(sft, sep_sft, atol=1e-5), (
                    f"fused SFT != separate ({label},{use_phi},{masked_only})"
                )
                assert torch.allclose(lp, sep_lp, atol=1e-5), f"fused lp != separate ({label},{use_phi},{masked_only})"


def test_payload_binder_alignment_convention():
    # Mirror the trainer's payload build: designed letters (binder order) scatter onto the
    # binder positions of the full layout; supervise mask = binder ∩ iface ∩ valid.
    L = 8
    binder_mask = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)  # binder at idx 3,4,5
    binder_idx = np.nonzero(binder_mask)[0]
    letters = "AGX"  # third residue non-standard -> ignored target
    iface = [True, False, True]  # residues 3 and 5 at interface
    tgt_bd = binder_letters_to_aa33(letters)
    valid_bd = tgt_bd >= 0
    sel = np.asarray(iface, dtype=bool)
    sup_bd = sel & valid_bd

    target_ids = np.full(L, SFT_IGNORE_INDEX, dtype=np.int64)
    supervise = np.zeros(L, dtype=bool)
    target_ids[binder_idx] = tgt_bd
    supervise[binder_idx] = sup_bd

    from lobster.tokenization._amino_acid import AA_VOCAB

    assert target_ids[3] == int(AA_VOCAB["A"]) and target_ids[4] == int(AA_VOCAB["G"])
    assert target_ids[5] == SFT_IGNORE_INDEX  # 'X' ignored
    # supervise: idx3 (iface+valid) yes; idx4 (not iface) no; idx5 (iface but invalid) no.
    assert supervise[3] and not supervise[4] and not supervise[5]
    # antigen positions never supervised.
    assert not supervise[:3].any() and not supervise[6:].any()
