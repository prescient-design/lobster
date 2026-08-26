"""Standalone GRPO trainer for the LeFlur absorbing-state discrete flow binder policy.

This is a plain training loop (not a Lightning ``training_step``): the policy is a
generative sampler whose reward — Protenix co-folding interface-pTM — is slow,
non-differentiable, and served out-of-process, so the natural structure is an
explicit GRPO loop:

1. **Rollout** a group of ``group_size`` designs for a target with
   :meth:`~LeFlurSequenceStructureEncoderLightningModule.rollout_with_logprobs`
   (production sampler, all CFG/bias/diversity/schedules), capturing per-step
   transitions.
2. **Score** the sampled endpoint sequences with the reward oracle
   (:class:`~lobster.rl_training.ProtenixRewardClient`) and assemble the reward as a
   flat sum of four weighted terms — a per-metric-clipped confidence combo, a
   structure self-consistency TM-score, and two within-group k-mer-Jaccard novelty
   rewards (AA + 3Di) — see ``rewards/README.md``.
3. **Advantage** = group-relative standardization ``(r - mean) / (std + eps)`` with
   a variance floor so zero-variance groups don't explode.
4. **PPO update** — for ``mu`` inner iterations over a random subsample of steps
   (diffu-GRPO step-subsampling), recompute the differentiable log-prob
   (:meth:`~LeFlurSequenceStructureEncoderLightningModule.logprob_over_trajectory`)
   and KL-to-reference, form the clipped surrogate ``+ beta * KL``, and step AdamW
   over the encoder only. The *old* policy log-prob is snapshotted per-step under
   ``no_grad`` before the inner loop, so the importance ratio is against a fixed
   behaviour policy even as the weights move.

The reward is a slow external oracle and the policy holds the LG codec, so this
loop is exercised on GPU via ``lobster_rl_train`` rather than unit-tested; the
differentiable log-prob/KL kernels and the reward/diversity terms it composes are
each unit-tested (``tests/lobster/rl_training/``).

Notes
-----
This v1 targets the single-target overfit milestone and runs on one device. It is
written to be wrapped in DDP later (shard ``group_size`` across ranks, all-gather
rewards before standardizing); that is intentionally left as a follow-up so the
overfit signal can be established on one GPU first.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from lobster.rl_training._structure_sft import structure_sft_loss
from lobster.rl_training.rewards import (
    SFT_IGNORE_INDEX,
    ProtenixRewardClient,
    assemble_structure_expert,
    binder_letters_to_aa33,
    build_struct_sft_targets,
    confidence_components,
    continuous_ip,
    jaccard_novelty_group,
    lc_saturating_reward,
    passes,
    reward_from_aar,
    reward_from_confidence,
    reward_from_shape,
    sc_clash_reward,
    structure_terms,
)

logger = logging.getLogger(__name__)

# Standard-AA token id -> letter (0-19 amino acids; 20=X, 21/22 gaps). Mirrors
# residue_constants.restype_order_with_x_inv; non-standard ids are remapped to a
# neutral residue before scoring (the sampled binder should be all-standard).
_AA_LETTERS = "ARNDCQEGHILKMFPSTWYV"
# 3Di structural-token alphabet (mini3di), index-aligned with the model's tri track.
_TRI_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


@dataclass
class GRPOTrainerConfig:
    """Hyperparameters for :class:`LeFlurGRPOTrainer`.

    Attributes
    ----------
    group_size : int
        Designs sampled per GRPO group (the "G" over which advantages are
        standardized).
    num_steps : int
        Number of GRPO optimization steps (outer loop).
    rollout_nsteps : int
        Denoising steps per rollout. Reduced (~40) vs production (200) because the
        log-prob recompute cost scales with captured steps (diffu-GRPO).
    steps_per_update : int
        Random step-subsample size used per inner PPO update. ``0`` = use all steps.
    mu : int
        Inner PPO updates per GRPO step (reusing the old-policy snapshot).
    beta : float
        KL-to-reference coefficient.
    eps_clip : float
        PPO ratio clip half-width.
    lr : float
        AdamW learning rate (encoder only).
    adv_eps : float
        Added to the group reward std when standardizing advantages.
    adv_std_floor : float
        Minimum group std; groups flatter than this contribute zero advantage
        (skipped) to avoid amplifying reward noise.
    normalize_advantage : bool
        If ``True`` (GRPO default) advantages are standardized ``(r - mean) /
        (std + eps)``. If ``False`` (Dr. GRPO) they are only mean-centered
        ``r - mean`` — this removes the per-group ``1/std`` reweighting that, in
        low-diversity groups, inflates the gradient of near-identical designs and
        biases the update toward easy (low-variance) groups.
    w_iptm, w_ptm, w_abag_iptm, w_plddt, w_gpde, w_pae_global, w_pae_interface : float
        Confidence-term weights. Each scales the corresponding Protenix metric after
        it is oriented higher-is-better and clipped to ``[0,1]`` (``*tm`` as-is,
        ``plddt/100``, ``1 - gpde/2``, ``1 - pae/32``). Defaults recover the shipped
        M22 reward: ``w_abag_iptm=1.0, w_ptm=0.5`` and every other metric ``0``.
    w_sctm_binder, w_sctm_complex : float
        Structure self-consistency weights on the TM-score between the LeFlur-generated
        backbone and the Protenix-predicted backbone — binder chain and whole complex
        respectively. ``> 0`` triggers ``return_coords`` on the reward client so the
        predicted CA coordinates ride back with the confidences. Default ``0``.
    w_seq_diversity, w_struct_diversity : float
        Weights on the within-group mean pairwise k-mer-Jaccard novelty of the AA
        sequence and the 3Di structural-token string (both already in ``[0,1]``,
        higher = more novel). Default ``0``.
    w_aa_dist, w_3di_dist : float
        Weights on the interface-distribution closeness ``clip(1 - D(hist, ref), 0,
        1)`` of the design's binder-interface amino-acid / 3Di histogram to the
        per-target reference (``dist_reference``). ``w_3di_dist > 0`` re-encodes the
        generated backbone with mini3di (binder in isolation) to build its 3Di
        interface histogram — making ``structure_tokens`` the true lever, so
        ``tracks`` must include them. Default ``0`` (inert / byte-identical).
    dist_metric : str
        Distribution distance ``D``: ``"tv"`` (total variation ``½Σ|p-q|``, default)
        or ``"js"`` (Jensen-Shannon in bits). Both are always logged.
    dist_reference : str | None
        Path to the per-target reference JSON (schema in
        :func:`~lobster.rl_training.rewards.load_reference_table`). Loaded once at
        construction when any ``w_*_dist > 0``; required in that case.
    binder_length_min, binder_length_max : int | None
        When both set, each GRPO step samples one binder length ``L ~ U[min, max]`` and
        holds it constant across the whole group (per-group length sampling); otherwise
        the target's own ``spec.binder_length`` is used. Default ``None`` (fixed length).
    tracks : tuple[str, ...]
        Tracks included in the log-prob ratio / KL (seq-only ablation = just
        ``("sequence_tokens",)``).
    capture_old_lp_inline : bool
        Use the behaviour-policy log-prob captured inline during sampling (exact
        biased logits, zero extra forwards) for ``old_lp``, instead of a post-hoc
        recompute. ``True`` also enables the ``ppo/ratio_init ~ 1`` consistency
        check. Set ``False`` only for rollouts predating inline capture.
    grad_clip : float
        Max global grad norm (``0`` disables).
    rollout_kwargs : dict
        Extra keyword arguments forwarded to ``generate_sample`` (temperatures,
        stochasticity, cfg_weight, biases, diversity penalties, schedules).
    seed : int
        Base RNG seed (step subsampling + target cycling).
    log_every : int
        Log metrics every N GRPO steps.
    ckpt_dir : str | None
        Directory to save periodic policy checkpoints (None = disabled).
    ckpt_every : int
        Save a checkpoint every N GRPO steps (when ``ckpt_dir`` set).
    """

    group_size: int = 16
    num_steps: int = 500
    rollout_nsteps: int = 40
    steps_per_update: int = 4
    mu: int = 2
    beta: float = 0.02
    eps_clip: float = 0.2
    lr: float = 1e-6
    adv_eps: float = 1e-4
    adv_std_floor: float = 1e-3
    normalize_advantage: bool = True
    # Confidence-term weights (README defaults recover shipped M22: abag_iptm + 0.5*ptm).
    w_iptm: float = 0.0
    w_ptm: float = 0.5
    w_abag_iptm: float = 1.0
    w_plddt: float = 0.0
    w_gpde: float = 0.0
    w_pae_global: float = 0.0
    w_pae_interface: float = 0.0
    # Structure self-consistency (TM-score gen-vs-predicted); > 0 => request coords.
    w_sctm_binder: float = 0.0
    w_sctm_complex: float = 0.0
    # Always compute+log sctm (worker returns coords every step) even when its
    # reward weight is 0 — a pure diagnostic. Adds per-step coord-transfer cost.
    log_struct_diagnostic: bool = False
    # Within-group Jaccard-novelty diversity (AA + 3Di token string).
    w_seq_diversity: float = 0.0
    w_struct_diversity: float = 0.0
    # Within-sequence anti-degeneracy: per-design linguistic-complexity SATURATING REWARD.
    # reward += w_seq_complexity * r_i, r_i = clip(LC_i / lc_full, 0, 1): full credit once
    # LC_i >= lc_full (no pressure on already-complex designs — the reward saturates), ramping
    # to 0 at collapse. LC_i = prod_{k=1..3} cov^(k)_i (within-sequence k-mer coverage). See
    # ``rewards/_diversity_reward.lc_saturating_reward``. Default 0 (inert).
    w_seq_complexity: float = 0.0
    lc_full: float = 0.7
    # Interface-distribution distance reward (Protenix-free shaping): closeness of the
    # design's binder-interface AA / 3Di histogram to the per-target reference
    # (``dist_reference`` JSON). Each ``w_*_dist`` scales ``clip(1 - D(hist, ref),
    # 0, 1)`` where ``D`` is ``dist_metric`` (``"tv"`` total variation, default, or
    # ``"js"`` Jensen-Shannon in bits). Default ``0`` (inert, byte-identical). Any
    # weight ``> 0`` requires ``dist_reference`` and re-encodes the generated backbone
    # with mini3di (binder in isolation) to build the 3Di interface histogram.
    w_aa_dist: float = 0.0
    w_3di_dist: float = 0.0
    # Always compute+log the AA/3Di distribution diagnostics (both interface and
    # whole-binder scopes) even when both ``w_*_dist`` are 0 — a pure diagnostic that
    # tracks how the interface distribution drifts under a reward that no longer targets
    # it. Requires ``dist_reference`` (so the TV/JS-to-reference diagnostics exist). The
    # distribution term contributes exactly 0 to the reward in this mode (the interface-
    # collapse ``dist_iface_penalty`` is also suppressed), so the reward is byte-identical
    # to a run with no distribution term. Mirrors ``log_struct_diagnostic``.
    log_dist_diagnostic: bool = False
    dist_metric: str = "tv"
    dist_reference: str | None = None
    # Interface-size guardrail. A design whose binder interface has fewer than
    # ``dist_min_iface`` residues (min cross-chain Cα–Cα < 8 Å) gets its whole
    # distribution reward set to ``dist_iface_penalty`` — overriding the histogram
    # term and the internal ``MIN_IFACE`` skip — so interface collapse is an
    # actively-repelled NEGATIVE signal instead of a soft 0 (the collapse basin the
    # spu=1/normalize=true runs drifted into). Defaults (min=4 == module MIN_IFACE,
    # penalty=0.0) reproduce the prior behaviour byte-for-byte.
    dist_min_iface: int = 4
    dist_iface_penalty: float = 0.0
    # Interface vs whole-binder distribution blend. The distribution closeness score is
    # computed at two scopes -- the binder INTERFACE (min cross-chain Cα–Cα < 8 Å) and
    # the ENTIRE valid binder chain -- and blended as
    # ``(1 - dist_binder_frac)·s_iface + dist_binder_frac·s_binder`` per alphabet. 0.0
    # (default) == interface-only, byte-identical to the prior behaviour; 1.0 ==
    # whole-binder-only; 0.5 weights them equally. Matching the whole-binder fold-state
    # histogram (dense, ~73–170 residues) removes the "balloon the interface fraction
    # over the whole binder" escape hatch that the interface-only histogram is gamed by.
    # Requires a reference table carrying ``aa_binder``/``3di_binder`` for the whole-
    # binder scope; if those are absent the whole-binder scope falls back to interface-
    # only (no reward dilution).
    dist_binder_frac: float = 0.0
    # Smooth steric-clash + interface-contact geometry reward (Protenix-free shaping;
    # see ``rewards/_clash_reward.py``). Scores each design's decoded backbone with a
    # single ``[0,1]`` term = clash_score·contact_score, where ``clash_score`` is a
    # smooth soft-core over backbone+Cβ atom pairs (→1 clash-free, →0 heavy overlap)
    # and ``contact_score`` is a smooth asymmetric raised-cosine band on the binder
    # interface fraction (fraction of binder residues at the interface): →0 for a
    # floating binder (frac < frac_lo) AND →0 for an over-large / interpenetrating
    # interface (frac > frac_hi), peaking at frac_peak (native passing band ≈ 0.16).
    # ``w_clash_contact`` scales the term. Default 0 (inert, byte-identical). Any
    # weight > 0 decodes the generated backbone (shared with the distribution term
    # when both are on). The remaining ``clash_*`` / ``contact_*`` / ``frac_*``
    # fields are the smooth-shape parameters (Å / interface-fraction band) and
    # default to the module constants — the clash scales are deliberately tolerant of
    # codec reconstruction noise (~1–2.7 Å RMSD); the frac band is calibrated to the
    # passing Proteina-Complexa interface-fraction distribution.
    w_clash_contact: float = 0.0
    clash_d_clash: float = 2.2
    clash_soft: float = 0.5
    clash_scale: float = 50.0
    contact_d0: float = 8.0
    contact_soft: float = 1.0
    frac_lo: float = 0.05
    frac_peak: float = 0.16
    frac_hi: float = 0.4
    clash_seq_sep: int = 2
    clash_include_cb: bool = True
    # Backbone chain-break (peptide-bond integrity) reward (Protenix-free realism
    # regularizer; see ``rewards/_chainbreak_reward.py``). Scores each design's decoded
    # binder backbone with a single ``[0,1]`` term = mean_r·gate, where ``mean_r`` is the
    # mean per-bond ``C(i)–N(i+1)`` realism (``exp(−(excess/σ)²)`` with a ``tol`` deadband
    # and a ``cap`` that saturates a 70 Å break to a 3.3 Å one) and ``gate`` collapses the
    # reward when bonds are catastrophically severed (``exp(−n_break/gate_k)``). Its purpose
    # is upstream of pass: the OTHER structural rewards (clash / shape / 3Di-dist) are all
    # measured on the generated backbone, and ~63% of our rollouts carry ≥1 hard break —
    # this keeps those coordinates physically valid so those terms stay trustworthy. Offline
    # AUROC vs Protenix pass ≈ 0.44 (expected: Protenix refolds the sequence and discards the
    # generated coords), so it is a regularizer, not a pass predictor. ``w_chainbreak`` scales
    # the term; default 0 (inert, byte-identical). Any weight > 0 decodes the generated
    # backbone (shared with the distribution/clash/shape terms when they are on).
    #
    # ``chainbreak_gate`` selects the break count in the gate: "count" (default) =
    # ``Σ 1[d > break_hard]`` (discrete) or "soft" = ``Σ sigmoid((d − break_d0)/break_soft)``
    # (smooth, no cliff at the threshold). The offline gate compare (``scripts/
    # _chainbreak_gate_compare.py``) selected "count": it keeps native designs at ~1.0 (soft's
    # sigmoid bulk floor haircuts clean backbones ~7–13%) and gives larger within-pool spread
    # (0.302 vs 0.267) for the GRPO advantage; "soft" is retained as an opt-in knob.
    w_chainbreak: float = 0.0
    chainbreak_gate: str = "count"  # "count" (discrete) or "soft" (sigmoid, severity-aware)
    chainbreak_gate_k: float = 2.0  # break count that drops the gate to 1/e
    chainbreak_ideal: float = 1.33  # ideal peptide-bond C–N length (Å)
    chainbreak_tol: float = 0.10  # free deadband around ideal (Å)
    chainbreak_cap: float = 2.00  # deviation saturation (Å): 70 Å break == 3.3 Å break
    chainbreak_sigma: float = 0.50  # soft-core bond-well width (Å)
    chainbreak_break_hard: float = 2.0  # count mode: C–N above this is a hard break (Å)
    chainbreak_break_d0: float = 2.0  # soft mode: sigmoid center (Å)
    chainbreak_break_soft: float = 0.10  # soft mode: sigmoid width (Å)
    # Order-20 3D-Zernike interface shape-complementarity (SC) reward (Protenix-free
    # shaping; see ``rewards/_shape_reward.py``). SC needs FULL-ATOM side chains to carry
    # signal (backbone-only is chance, AUROC 0.533; LigandMPNN-repacked full-atom is
    # 0.650), so each design's decoded backbone (antigen + binder N/CA/C, with the
    # pinned antigen sequence and the sampled binder sequence) is shipped to a persistent
    # LigandMPNN-repack worker pool (``rewards/_shape_reward_pool.ShapeRewardClient`` +
    # ``scripts/ligandmpnn_repack_server.py``) that packs side chains (openfold, GPU) and
    # scores the interface ΔSASA-patch 3DZD complementarity (scipy, CPU), returning a
    # single ``[0,1]`` term (clipped Pearson of the two interface descriptors).
    # ``w_shape`` scales it; default 0 (inert, byte-identical, no queue/pool). Any weight
    # > 0 requires ``shape_queue_dir`` and decodes the generated backbone (shared with the
    # distribution/clash terms when they are on). A design the pool fails/times-out on
    # gets the floor reward 0. The remaining ``shape_*`` fields configure the queue
    # client (timeout, poll, cache, shard fan-out across the pool).
    w_shape: float = 0.0
    shape_queue_dir: str | None = None
    shape_timeout_s: float = 1800.0
    shape_poll_s: float = 2.0
    shape_cache: bool = True
    shape_n_shards: int = 1
    # Two further full-atom reward terms served by the SAME LigandMPNN-repack worker pool
    # (``shape_queue_dir`` / ``ShapeRewardClient``). Because the pool packs each design once
    # and can return any subset of {SC, clash, AAR} in a single round-trip (the ``want``
    # protocol), turning several of these on adds no extra packing — the group is shipped
    # once with the union metric-set. Both default 0 (inert; no queue/pool unless some
    # repack weight > 0), share all the ``shape_*`` client config, and require
    # ``shape_queue_dir`` when enabled. All three run on the CPU worker pool, so throughput
    # scales simply by adding more CPU workers (no GPU contention).
    #
    #   w_sc_clash — all-atom side-chain steric clash over the WHOLE binder (self-clash +
    #     binder↔antigen), reward ``exp(−E_clash/scale) ∈ (0,1]`` (``rewards/_sc_clash_reward``).
    #     A hard biophysical constraint (overlapping VDW spheres are forbidden), kept even
    #     though offline pass-correlation is weak — it keeps the policy on the physical manifold.
    #   w_aar — ProteinMPNN whole-binder amino-acid recovery / designability
    #     (``rewards/_aar_reward``). Offline it is *anti-predictive* of the Protenix pass label
    #     (AUROC ≈ 0.29–0.34); provided opt-in and documented (not gated out), enable its
    #     weight only as a deliberate, calibrated choice.
    w_sc_clash: float = 0.0
    # sc_clash energy mode. False (default) = absolute ``E_clash_total`` — gameable by
    # interface retraction (fewer contact atoms lowers E without de-clashing), the
    # cannibalization mode seen in the co-equal sc_clash+aar arm. True = per-residue
    # DENSITY (``E_clash_binder/n_res + E_clash_iface_res/n_iface_res``): numerator and
    # denominator shrink together under retraction, so the policy must de-clash retained
    # contacts to gain reward (``rewards/_sc_clash_reward.sc_clash_reward(density=True)``).
    sc_clash_density: bool = False
    w_aar: float = 0.0
    # --- CHORD SFT-distillation (dense per-token sequence supervision) -----------------
    # Blends a supervised cross-entropy toward the LigandMPNN-designed binder sequence into
    # the GRPO update as a dynamically token-weighted auxiliary loss (CHORD, arXiv:2508.11408):
    #   L = (1 - μ)·L_GRPO + μ·L_SFT-φ            (KL, when on, is added outside the blend)
    # The expert (LigandMPNN in DESIGN mode) is queried on the learner's OWN generated
    # backbones via the SAME repack pool that serves SC/clash/AAR (want includes "aar" +
    # return_seq); the SFT term needs the *designed sequence*, not the AAR scalar, so it is
    # independent of ``w_aar`` (set ``w_aar=0`` to keep AAR out of the reward while still
    # distilling). ``sft_mu > 0`` activates the term.
    #
    # WHY per-token + φ (the offline caveat that drives this design): whole-binder MPNN
    # agreement is ANTI-predictive of Protenix pass (AUROC 0.292) — naive whole-binder SFT
    # would steer into the generic buried-hydrophobic failing mode. Defenses: the CHORD φ weight
    # (zeros consensus AND rejected tokens per residue) and the optional reward gate. The SFT
    # forward conditions on the EXPERT sequence context (see _expert_context_seq in the model),
    # so whole-binder distillation teaches the coherent LigandMPNN conditional p_θ(y* | y*-context)
    # rather than an expert/policy chimera — this makes ``sft_scope='binder'`` (the default) sound.
    sft_mu: float = 0.0  # convex-blend weight on L_SFT (0 = off, byte-identical legacy path)
    # Optional μ schedule. None (default) = constant ``sft_mu`` (CHORD finding: with φ on, a
    # small fixed μ matches a decay schedule). "linear_decay" = anneal μ from ``sft_mu`` to 0
    # linearly over ``num_steps`` (DAgger-style hand-off to pure RL late in training).
    sft_mu_schedule: str | None = None
    sft_use_phi: bool = True  # CHORD φ_t = p_t(1-p_t) token weighting (detached)
    sft_scope: str = "binder"  # "binder" (whole binder, default) or "interface" (binder∩iface)
    sft_label: str = "hard"  # "hard" (argmax designed identities) or "soft" (distil MPNN's full per-position output distribution, design_logq)
    sft_temperature: float = 1.0  # soft-label softmax temperature (unused for hard labels)
    sft_masked_only: bool = True  # supervise a position at a step only while it is still masked
    # Optional reward gate: only distil designs whose group advantage passes a rule.
    # None (default) = distil every design. "positive_adv" = supervise only designs with
    # advantage > 0 (learn the expert sequence on the group's above-average backbones).
    sft_reward_gate: str | None = None
    # --- Protenix fold-consistency SFT (structure + 3Di track distillation) ------------
    # The STRUCTURAL DUAL of the CHORD sequence SFT above. Where CHORD folds the reward
    # through a SEQUENCE expert (LigandMPNN redesigns the policy's structure -> a*, distilled
    # into the seq track), this folds through a STRUCTURE expert: Protenix folds the policy's
    # OWN output sequence -> structure X*, from which BOTH the LG structure tokens (s*) and the
    # 3Di tokens (τ*) are derived over the whole complex and distilled into the policy's
    # structure + 3Di endpoints (rewards/_protenix_structure_expert.py + _structure_sft.py):
    #   L = (1 - μ_seq)·L_GRPO + μ_seq·L_SFT_seq + μ_struct·L_SFT_struct   (+ β·KL outside)
    # μ_struct is an ADDITIVE aux weight (it does not scale down L_GRPO), so it composes with
    # the sequence CHORD term (sft_mu) and pure GRPO alike. ``struct_sft_mu > 0`` activates it.
    #
    # WHY: eval shows the binder monomer fold is reproduced by Protenix (~1.8-2.4 Å) but the
    # DOCKED pose disagrees (~25-35 Å after target superpose) — Protenix re-docks the sequence.
    # No backbone/pack/3Di-histogram reward moves docking fidelity because none condition the
    # policy's OWN structure endpoint on a fold consistent with its emitted sequence. Distilling
    # (s*, τ*) closes that sequence<->structure loop. Protenix is a structure ORACLE here
    # (coords -> tokens); its confidence (pTM/ipTM/pLDDT) stays offline-only, never a reward.
    #
    # COST/INFRA: unlike the Protenix-free CHORD seq runs, struct_sft_mu>0 REQUIRES the Protenix
    # fold every step (it sets return_coords + return_backbone on the reward client), so a
    # struct-SFT run needs a live Protenix worker pool (N_WORKERS>0).
    struct_sft_mu: float = 0.0  # additive weight on L_SFT_struct (0 = off, no extra forward)
    struct_sft_mu_schedule: str | None = None  # None (constant) or "linear_decay"
    struct_sft_w_struct: float = 1.0  # LG structure-track (s*) weight inside the struct-SFT loss
    struct_sft_w_tri: float = 1.0  # 3Di-track (τ*) weight inside the struct-SFT loss
    struct_sft_use_phi: bool = True  # CHORD φ_t = p_t(1-p_t) token weighting (detached)
    struct_sft_masked_only: bool = True  # supervise a position at a step only while still masked
    struct_sft_scope: str = "complex"  # "complex" (antigen+binder, default) or "binder"
    struct_sft_reward_gate: str | None = None  # None or "positive_adv" (distil only adv>0 designs)
    # Per-group binder-length sampling (both set => L ~ U[min, max], constant per group).
    binder_length_min: int | None = None
    binder_length_max: int | None = None
    # Multi-target gradient accumulation (training-stabilization arm). Each optimizer
    # step aggregates the policy-gradient over ``accum_targets`` independently-rolled-out
    # targets: their per-target losses are averaged (each ``backward()`` scaled by
    # ``1/accum_targets``) and a single ``optimizer.step()`` is applied per inner (mu)
    # iteration — the cross-prompt averaging TRL uses to stabilize Dr.GRPO over multiple
    # environments. ``shuffle_targets`` reshuffles the round-robin target order each epoch
    # (decorrelates consecutive single-target pulls). The default is ``accum_targets=10``
    # (the training-stabilization arm's proven setting): each step averages the gradient
    # over 10 independently-rolled-out targets, which decorrelates the multi-target reward
    # and is the current recommended baseline. Set ``accum_targets=1`` +
    # ``shuffle_targets=False`` to reproduce the legacy single-target round-robin loop
    # byte-for-byte.
    accum_targets: int = 10
    shuffle_targets: bool = False
    tracks: tuple[str, ...] = ("sequence_tokens", "structure_tokens", "tri_tokens")
    capture_old_lp_inline: bool = True
    grad_clip: float = 1.0
    # Gradient-checkpoint each per-step log-prob recompute (torch.utils.checkpoint,
    # non-reentrant). Frees each step's forward activations after its forward and
    # recomputes them in backward, so peak update memory is ~group_size×1 regardless
    # of ``steps_per_update`` — the knob that lets spu>1 fit at G=64 on a b200
    # (~1.3–2× compute for the recompute). Default False = current behaviour.
    grad_checkpoint: bool = False
    rollout_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    log_every: int = 1
    ckpt_dir: str | None = None
    ckpt_every: int = 50
    # --- Per-token (per-residue) clash advantage -------------------------------------
    # When True, the backbone clash-contact reward's clash energy is decomposed per binder
    # residue (``clash_contact_reward(return_eres=True)``; each binder↔antigen pair fully to
    # its binder residue, each intra-binder pair split 50/50) and routed as a PER-POSITION
    # advantage to the STRUCTURE track only. The final structure-track advantage is
    #   A_struct[g, l] = A_design[g] + A_clash[g, l]
    # where A_design is the usual group-normalized scalar advantage (applied design-level to
    # the sequence/tri tracks) and A_clash is the group-mean-centered, globally-std-normalized
    # per-residue clash signal (``-E_clash_res``, so LESS clash => higher advantage). Requires
    # ``w_clash_contact > 0`` (that term supplies the per-residue energy). Default False =
    # byte-identical scalar-advantage path. See ``docs/leflur/grpo_per_token_advantage_plan.md``.
    per_token_clash: bool = False
    w_pt_clash: float = 1.0  # scale on the per-residue clash advantage A_clash
    # Per-token (per-residue) CHAIN-BREAK advantage — the exact analog of per_token_clash for
    # the chain-break reward. When True, ``chainbreak_reward(return_eres=True)`` decomposes the
    # per-bond penalty ``1 − r_bond`` per binder residue (each bond split 50/50 between its two
    # endpoints; sum == pen so ``mean_r = 1 − pen/n_bonds``) and routes ``−cb_break_res`` as a
    # per-position advantage to the STRUCTURE track (same centering/normalization as clash, and
    # ADDITIVE with A_clash when both are on):
    #   A_struct[g, l] = A_design[g] + A_clash[g, l] + A_chainbreak[g, l]
    # Requires ``w_chainbreak > 0`` (that term supplies the per-residue penalty). Default False =
    # byte-identical scalar-advantage path. The gate is a design-level transform kept only in the
    # scalar term (mirroring clash's exp), so the per-residue signal is the raw bond realism.
    per_token_chainbreak: bool = False
    w_pt_chainbreak: float = 1.0  # scale on the per-residue chain-break advantage A_chainbreak
    # Per-token (per-residue) all-atom INTERFACE-POTENTIAL advantage. When True, the same
    # LigandMPNN pack used for R_SC is scored per binder residue for the three potentials that
    # beat the 3DZD SC reward offline (bounded LJ energy e_lj, buried ΔSASA, interface H-bonds;
    # AUROC 0.778/0.757/0.742 vs 0.73). Each per-residue vector is routed as a per-position
    # advantage to the STRUCTURE track (same centering/normalization as clash), ADDITIVE with
    # A_clash / A_chainbreak when those are on:
    #   A_struct[g, l] = A_design[g] + A_clash + A_chainbreak + A_lj + A_dsasa + A_hb
    # ``e_lj`` is a penalty (larger = worse); ``dsasa``/``n_hb`` are negated at collection so all
    # three enter posnorm as "larger = worse". Adds the "pot" want to the repack round-trip (no
    # extra pack — reuses the R_SC pack). Default False = byte-identical scalar path.
    per_token_pot: bool = False
    w_pt_lj: float = 1.0  # scale on the per-residue bounded-LJ advantage A_lj
    w_pt_dsasa: float = 1.0  # scale on the per-residue buried-ΔSASA advantage A_dsasa
    w_pt_hb: float = 1.0  # scale on the per-residue interface-H-bond advantage A_hb
    pot_with_sasa: bool = True  # compute the (slower) buried-ΔSASA term; off => A_dsasa disabled
    # Which track(s) receive any per-position structure advantage (clash, chain-break and/or
    # interface potentials); excluded from the design-level PPO on the remaining tracks.
    # Structure only by default. (Name kept for config compatibility; applies to all per-token
    # structure rewards.)
    pt_clash_tracks: tuple[str, ...] = ("structure_tokens",)
    # Track (compute + log) the all-atom side-chain clash metrics WITHOUT putting them in the
    # reward/gradient (diagnostic-only, mirrors ``log_dist_diagnostic``). Forces "clash" into
    # the repack ``want`` set at weight 0, so ``sc_clash`` metrics are logged alongside the
    # active terms without any packing cost beyond the metric. Default False.
    log_sc_clash_diagnostic: bool = False


@dataclass
class TargetSpec:
    """One binder-design target for GRPO.

    Attributes
    ----------
    target_id : str
        Key into the reward manifest (``complexa_score_targets.csv``).
    antigen_pdb : str
        Path to the target/antigen PDB used to build the composite conditioning.
    target_chain : str
        Chain id of the target within ``antigen_pdb``.
    binder_length : int
        Binder chain length to design.
    epitope_indices : list[int] | None
        Target-local epitope residues (for epitope-anchored init + conditioning).
    """

    target_id: str
    antigen_pdb: str
    target_chain: str
    binder_length: int
    epitope_indices: list[int] | None = None


class LeFlurGRPOTrainer:
    """GRPO fine-tuning loop for the LeFlur binder policy (see module docstring).

    Parameters
    ----------
    model : LeFlurSequenceStructureEncoderLightningModule
        The policy to fine-tune (updated in place; encoder params only).
    reward_client : ProtenixRewardClient
        Blocking client for the Protenix reward pool.
    targets : list[TargetSpec]
        Targets to optimize; cycled round-robin across GRPO steps.
    config : GRPOTrainerConfig
        Hyperparameters.
    device : torch.device
        Device for the policy + rollouts.
    gen_cfg : object
        A ``cfg.generation``-like node with ``.get(name, default)`` (max_length,
        epitope/template flags, etc.) passed through to
        :func:`build_binder_static_cond`.
    ref_module : LeFlurSequenceStructureEncoderLightningModule | None
        Frozen reference policy for the KL term. Defaults to a deep copy of
        ``model`` taken at construction.
    wandb_run : object | None
        Optional wandb run for metric logging.
    """

    def __init__(
        self,
        model,
        reward_client: ProtenixRewardClient,
        targets: list[TargetSpec],
        config: GRPOTrainerConfig,
        device: torch.device,
        gen_cfg,
        ref_module=None,
        wandb_run=None,
    ) -> None:
        if not targets:
            raise ValueError("At least one target is required")
        if getattr(config, "per_token_clash", False):
            if config.w_clash_contact <= 0:
                raise ValueError("per_token_clash requires w_clash_contact > 0 (it supplies the per-residue energy)")
        if getattr(config, "per_token_chainbreak", False):
            if getattr(config, "w_chainbreak", 0.0) <= 0:
                raise ValueError("per_token_chainbreak requires w_chainbreak > 0 (it supplies the per-residue penalty)")
        if getattr(config, "chainbreak_gate", "count") not in ("count", "soft"):
            raise ValueError(f"chainbreak_gate must be 'count' or 'soft', got {config.chainbreak_gate!r}")
        if getattr(config, "per_token_pot", False):
            if not (config.w_pt_lj or config.w_pt_dsasa or config.w_pt_hb):
                raise ValueError("per_token_pot requires at least one of w_pt_lj / w_pt_dsasa / w_pt_hb > 0")
        if (
            getattr(config, "per_token_clash", False)
            or getattr(config, "per_token_chainbreak", False)
            or getattr(config, "per_token_pot", False)
        ):
            missing = [t for t in config.pt_clash_tracks if t not in config.tracks]
            if missing:
                raise ValueError(f"pt_clash_tracks {missing} must be a subset of tracks {config.tracks}")
        self.model = model.to(device)
        self.reward_client = reward_client
        self._struct_sft_warned = False  # one-shot guard for the struct-SFT silent-no-op warning
        # Protenix fold-consistency SFT folds the policy sequence with Protenix every step, so it
        # needs a live Protenix reward client (unlike the Protenix-free CHORD sequence SFT).
        if getattr(config, "struct_sft_mu", 0.0) > 0 and reward_client is None:
            raise ValueError(
                "struct_sft_mu > 0 requires a Protenix reward_client (the structure expert is "
                "derived by folding the policy sequence); pass one or set struct_sft_mu=0"
            )
        self.targets = targets
        self.config = config
        self.device = device
        self.gen_cfg = gen_cfg
        self.wandb_run = wandb_run

        # Frozen reference for the KL regularizer.
        self.ref_module = ref_module if ref_module is not None else copy.deepcopy(model)
        self.ref_module.to(device)
        self.ref_module.requires_grad_(False)
        self.ref_module.eval()

        # Permanent CPU copy of the VIT structure decoder. Initialized once here so
        # reward computation never moves the GPU decoder, avoiding CUDA deadlocks with
        # live rollout state on the GPU stream. Reused by the structure self-consistency
        # term to decode the generated backbone on CPU.
        self._cpu_vit_decoder = copy.deepcopy(model.decoder_factory.decoders["vit_decoder"])
        self._cpu_vit_decoder.cpu()
        self._cpu_vit_decoder.eval()
        self._cpu_vit_decoder.requires_grad_(False)

        # Interface-distribution reward reference table (per-target AA/3Di histograms).
        # Loaded once when any distance weight is on; None keeps the term inert.
        self._dist_ref = None
        if config.w_aa_dist > 0 or config.w_3di_dist > 0 or config.log_dist_diagnostic:
            if not config.dist_reference:
                raise ValueError(
                    "dist_reference path is required when w_aa_dist/w_3di_dist > 0 or log_dist_diagnostic is set"
                )
            from lobster.rl_training.rewards import load_reference_table

            self._dist_ref = load_reference_table(config.dist_reference)
            logger.info(
                "loaded interface-distribution reference: %d per-target refs (%s)",
                len(self._dist_ref["per_target"]),
                config.dist_reference,
            )

        # LigandMPNN-repack reward client — ONE pool serving all three full-atom terms
        # (SC shape-complementarity, all-atom clash, ProteinMPNN AAR). Constructed once when
        # any repack weight > 0; None keeps every repack term inert (no queue / worker pool).
        # The client transports each design's antigen+binder backbone clouds to the pool and
        # blocks for the returned per-metric scalars (see rewards/_shape_reward_pool.py); the
        # requested metric-set is passed per group via the ``want`` protocol, so multiple
        # terms cost one packing round-trip.
        self._shape_client = None
        # CHORD SFT distillation (sft_mu>0) also needs the pool (it draws the LigandMPNN-designed
        # sequence over the "aar" repack path), so it counts toward needing a client even when
        # every repack REWARD weight is 0.
        need_repack = (
            config.w_shape > 0
            or config.w_sc_clash > 0
            or config.w_aar > 0
            or getattr(config, "sft_mu", 0.0) > 0
            or getattr(config, "log_sc_clash_diagnostic", False)  # tracked-but-off SC clash needs the pool
        )
        if need_repack:
            if not config.shape_queue_dir:
                raise ValueError(
                    "shape_queue_dir is required when any repack reward (w_shape / w_sc_clash / w_aar) "
                    "or CHORD SFT distillation (sft_mu) > 0"
                )
            from lobster.rl_training.rewards import ShapeRewardClient

            self._shape_client = ShapeRewardClient(
                config.shape_queue_dir,
                timeout_s=config.shape_timeout_s,
                poll_s=config.shape_poll_s,
                cache=config.shape_cache,
                n_shards=config.shape_n_shards,
            )
            logger.info(
                "LigandMPNN-repack reward pool ON (w_shape=%.3g, w_sc_clash=%.3g, w_aar=%.3g, queue=%s, n_shards=%d)",
                config.w_shape,
                config.w_sc_clash,
                config.w_aar,
                config.shape_queue_dir,
                config.shape_n_shards,
            )

        self.optimizer = torch.optim.AdamW(self.model.encoder.parameters(), lr=config.lr)
        self._rng = random.Random(config.seed)
        # Separate RNG for per-epoch target shuffling — kept independent of ``self._rng`` so
        # that with ``shuffle_targets=False`` (the default) it is never drawn from and the
        # binder-length / step-subset draw order stays byte-identical to the legacy loop.
        self._sched_rng = random.Random(config.seed + 1000)
        # Cache the per-(target, binder-length) static conditioning (target chain +
        # composite masks) — only the sampled binder changes across rollouts. Keyed by
        # (target_id, L) since per-group length sampling cycles many lengths per target;
        # a bounded LRU keeps the cache from growing across ~2000 targets × variable L.
        self._static_cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._static_cache_cap = 16

    # ---------------------------------------------------------------- conditioning
    def _sample_binder_length(self, spec: TargetSpec) -> int:
        """Binder length for this group: per-group sample when configured, else fixed.

        With both ``binder_length_min`` and ``binder_length_max`` set, draws one
        ``L ~ U[min, max]`` from the trainer RNG and holds it constant across the whole
        group (per the design decision). Otherwise falls back to ``spec.binder_length``.
        """
        cfg = self.config
        if cfg.binder_length_min is not None and cfg.binder_length_max is not None:
            return self._rng.randint(cfg.binder_length_min, cfg.binder_length_max)
        return spec.binder_length

    def _target_static(self, spec: TargetSpec, binder_length: int) -> dict:
        """Build (and LRU-cache) the composite static conditioning for a target + length."""
        key = (spec.target_id, binder_length)
        if key in self._static_cache:
            self._static_cache.move_to_end(key)  # mark most-recently-used
            return self._static_cache[key]

        from lobster.cmdline.generate_modes._binders import (
            _build_scalar_cond_bins,
            build_binder_static_cond,
            load_binder_target,
        )
        from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

        max_length = self.gen_cfg.get("max_length", 512)
        loaded = load_binder_target(spec.antigen_pdb, spec.target_chain, max_length=max_length)
        if loaded is None:
            raise ValueError(f"Failed to load target {spec.target_id} from {spec.antigen_pdb}")

        tok = AminoAcidTokenizerTransform(max_length=max_length)
        comp = build_binder_static_cond(
            binder_length,
            loaded["target_data_filtered"],
            loaded["chains_key"],
            loaded["target_chain_idx"],
            model=self.model,
            gen_cfg=self.gen_cfg,
            device=self.device,
            tokenizer_transform=tok,
            epitope_indices=spec.epitope_indices,
            output_dir=None,
            structure_path=spec.antigen_pdb,
            save_initial=False,
        )
        if comp is None:
            raise ValueError(f"Composite for target {spec.target_id} exceeds max_length {max_length}")
        # Positions of the designed binder chain (for slicing decoded sequences).
        comp["binder_positions"] = comp["chains_ids"][0] == comp["binder_chain_idx"]
        # Optional per-target scalar conditioning bins (concat cond), placed on binder residues.
        comp["scalar_cond_bins"] = _build_scalar_cond_bins(
            self.gen_cfg, comp["chains_ids"][0], comp["binder_chain_idx"], comp["L_total"], self.device
        )
        self._static_cache[key] = comp
        self._static_cache.move_to_end(key)
        while len(self._static_cache) > self._static_cache_cap:
            self._static_cache.popitem(last=False)  # evict least-recently-used
        return comp

    def _build_gen_kwargs(self, comp: dict, group_size: int) -> dict:
        """Assemble ``generate_sample`` kwargs for a group rollout from static cond.

        Every conditioning tensor is expanded along the batch dim to ``group_size``
        so the sampler draws ``G`` independent designs from identical conditioning.
        """

        def _expand(t):
            if t is None:
                return None
            return t.expand(group_size, *t.shape[1:]).contiguous()

        cfg = self.config
        kwargs = dict(
            length=comp["L_total"],
            num_samples=group_size,
            nsteps=cfg.rollout_nsteps,
            inpainting=True,
            input_structure_coords=_expand(comp["coords_res"]),
            input_sequence_tokens=_expand(comp["sequence_tokenized"]),
            input_mask=_expand(comp["mask"]),
            input_indices=_expand(comp["indices"]),
            inpainting_mask_sequence=_expand(comp["mask_sequence"]),
            inpainting_mask_structure=_expand(comp["mask_structure"]),
            chain_ids=_expand(comp["chain_ids_emb"]),
            conditioning_tensor_override=_expand(comp["cond_tensor"]),
            template_structure_tokens=_expand(comp["template_arg"]),
        )
        if comp.get("scalar_cond_bins"):
            kwargs["scalar_cond_bins"] = {k: _expand(v) for k, v in comp["scalar_cond_bins"].items()}
        # Sampler knobs (temperatures, stochasticity, cfg_weight, biases, diversity,
        # schedules) come straight from config.rollout_kwargs, overriding nothing above.
        kwargs.update(cfg.rollout_kwargs)
        return kwargs

    # ------------------------------------------------------------------- decoding
    def _decode_binder_seqs(self, trajectory: dict, comp: dict) -> list[str]:
        """Sampled binder sequences (letters) for each design in the group."""
        aa = self.model.decode_endpoint_aa(trajectory)  # (G, L) standard-AA ids
        pos = comp["binder_positions"].to(aa.device)
        binder_ids = aa[:, pos].cpu().tolist()
        seqs = []
        for row in binder_ids:
            seqs.append("".join(_AA_LETTERS[i] if 0 <= i < 20 else "G" for i in row))
        return seqs

    def _decode_binder_tri(self, trajectory: dict, comp: dict) -> list[str] | None:
        """Per-design 3Di structural-token strings for the binder positions.

        Reads the sampled endpoint 3Di tokens (``final_xt["tri_tokens"]``) — the
        model's own structural alphabet ``"ACDEFGHIKLMNPQRSTVWYX"`` — and maps them
        to letters at the binder positions, mirroring :meth:`_decode_binder_seqs`.
        Returns ``None`` when the checkpoint has no 3Di track (no ``tri_tokens`` in
        the rollout), so the caller falls back to sequence-only diversity.
        """
        final_xt = trajectory.get("final_xt", {})
        tri = final_xt.get("tri_tokens")
        if tri is None:
            return None
        pos = comp["binder_positions"].cpu()
        tri_ids = tri.long()[:, pos].cpu().tolist()
        n = len(_TRI_ALPHABET)
        return ["".join(_TRI_ALPHABET[i] if 0 <= i < n else "X" for i in row) for row in tri_ids]

    # ------------------------------------------------------------- generated coords
    def _decode_backbone_coords(self, trajectory: dict, comp: dict) -> torch.Tensor:
        """Decode the sampled structure tokens to ``(G, L, 3, 3)`` backbone coords on CPU.

        Uses the permanent CPU copy of the VIT decoder (``self._cpu_vit_decoder``,
        built in ``__init__``) so no device moves happen here — avoids a CUDA deadlock
        with live GPU rollout state. The three decoded backbone atoms are ``(N, CA, C)``;
        CA is atom index 1. Reused by the structure self-consistency reward.
        """
        G = self.config.group_size
        struc_tokens = trajectory["final_xt"]["structure_tokens"].long().cpu()  # (G, L)
        n_tok = self.model.quantizer.n_tokens
        mask_cpu = comp["mask"].expand(G, -1).bool().cpu()  # (G, L)
        valid = mask_cpu & (struc_tokens < n_tok)
        with torch.no_grad():
            onehot = (
                torch.nn.functional.one_hot(struc_tokens.clamp(0, n_tok - 1), n_tok).float()
                * valid.unsqueeze(-1).float()
            )  # (G, L, n_tok)
            xyz = self._cpu_vit_decoder(onehot, valid.float())  # (G, L, 3, 3)
        return xyz

    def _structure_terms_for_group(
        self, trajectory: dict, comp: dict, confs: list[dict | None]
    ) -> tuple[list[float], list[float], list[float]]:
        """Per-design structure self-consistency: ``(weighted_term, sctm_binder, sctm_complex)``.

        Decodes the generated backbone once (CA at atom index 1), then for each design
        computes the TM-score of the generated vs Protenix-predicted CA — binder chain
        and whole complex (antigen-then-binder, matching the worker's chain A/B order).
        A missing/length-mismatched pair contributes ``0.0`` (see
        :func:`~lobster.rl_training.rewards.structure_terms`).
        """
        import numpy as np

        cfg = self.config
        gen_ca = self._decode_backbone_coords(trajectory, comp)[:, :, 1, :]  # (G, L, 3)
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        antigen_mask = valid & ~binder_mask
        weighted, sctm_b, sctm_c = [], [], []
        for i, conf in enumerate(confs):
            gca = gen_ca[i].numpy()
            gen_binder = gca[binder_mask]
            gen_complex = np.concatenate([gca[antigen_mask], gca[binder_mask]], axis=0)
            pred_binder = pred_complex = None
            if conf is not None:
                bx = conf.get("binder_xyz")
                ax = conf.get("antigen_xyz")
                if bx is not None:
                    pred_binder = np.asarray(bx, dtype=np.float64)
                if ax is not None and bx is not None:
                    pred_complex = np.concatenate(
                        [np.asarray(ax, dtype=np.float64), np.asarray(bx, dtype=np.float64)], axis=0
                    )
            st = structure_terms(gen_binder, pred_binder, gen_complex, pred_complex)
            sctm_b.append(st["sctm_binder"])
            sctm_c.append(st["sctm_complex"])
            weighted.append(cfg.w_sctm_binder * st["sctm_binder"] + cfg.w_sctm_complex * st["sctm_complex"])
        return weighted, sctm_b, sctm_c

    def _distribution_terms_for_group(
        self,
        target_id: str,
        trajectory: dict,
        comp: dict,
        seqs: list[str],
        *,
        gen_bb: np.ndarray | None = None,
    ) -> tuple[list[float], dict]:
        """Per-design distribution-distance reward (interface + whole-binder) + diagnostics.

        Re-encodes the *generated* backbone (binder chain in isolation) to 3Di and
        tallies the binder AA + 3Di histograms at **two scopes** — the interface
        (binder residues with min cross-chain Cα–Cα ``< 8 Å``) and the entire valid
        binder chain — sharing a single encode, then blends their closeness scores to
        the per-target reference by ``dist_binder_frac`` (α; 0 = interface-only,
        1 = whole-binder-only). Returns ``(weighted_terms (G,), metrics)``; designs
        whose interface has ``< dist_min_iface`` residues get ``dist_iface_penalty`` as
        their whole reward (collapse guardrail) instead of the blended term. Both scopes'
        TV/JS + mean interface/binder sizes are logged.

        ``gen_bb`` is the decoded ``(G, L, 3, 3)`` backbone; when ``None`` it is
        decoded here (the default, byte-identical to the old behaviour). The caller
        passes it in to share a single decode with the clash term.
        """
        import numpy as np

        from lobster.rl_training.rewards import (
            combined_distribution_terms,
            design_hists_scoped,
            reference_for,
        )

        cfg = self.config
        ref_aa_i, ref_3di_i, ref_aa_b, ref_3di_b, ref_src = reference_for(self._dist_ref, target_id)
        if gen_bb is None:
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        # Whether the distribution term feeds the reward at all. When both weights are 0
        # (log_dist_diagnostic-only), the term contributes exactly 0: histograms are still
        # computed and logged, but the interface-collapse penalty is suppressed too, so the
        # reward is byte-identical to a run with no distribution term.
        reward_on = cfg.w_aa_dist > 0 or cfg.w_3di_dist > 0
        # Compute 3Di histograms whenever the 3Di term is a reward OR we are tracking it.
        need_3di = cfg.w_3di_dist > 0 or cfg.log_dist_diagnostic

        weighted: list[float] = []
        # interface-scope diagnostics + whole-binder-scope diagnostics
        tv_aa, tv_3di, js_aa, js_3di = [], [], [], []
        tv_aa_b, tv_3di_b, js_aa_b, js_3di_b = [], [], [], []
        niface, nbinder = [], []
        n_skipped = 0
        n_penalized = 0
        for i, seq in enumerate(seqs):
            h_aa_i, h_3di_i, h_aa_b, h_3di_b, n_iface, n_binder = design_hists_scoped(
                gen_bb[i], valid, binder_mask, seq, need_aa=True, need_3di=need_3di
            )
            niface.append(n_iface)
            nbinder.append(n_binder)
            # Interface-size guardrail: a below-threshold (collapsed) interface gets a
            # fixed penalty as its whole reward instead of the histogram term / 0-skip,
            # so the collapse basin is repelled. Default dist_min_iface=4 == the module's
            # MIN_IFACE and penalty=0.0, which reproduces the old (skip -> 0) behaviour.
            # The whole-binder scope does NOT rescue a collapsed interface: an
            # interpenetrating/degenerate interface is still an escape hatch to repel.
            if n_iface < cfg.dist_min_iface:
                # Collapse guardrail is a reward signal; in diagnostic-only mode
                # (reward_on False) it must not inject the penalty into the reward.
                weighted.append(float(cfg.dist_iface_penalty) if reward_on else 0.0)
                n_penalized += 1
                if h_aa_i is None and h_3di_i is None and h_aa_b is None and h_3di_b is None:
                    n_skipped += 1
                continue
            term, diag = combined_distribution_terms(
                h_aa_i,
                h_3di_i,
                h_aa_b,
                h_3di_b,
                ref_aa_i,
                ref_3di_i,
                ref_aa_b,
                ref_3di_b,
                cfg.w_aa_dist,
                cfg.w_3di_dist,
                cfg.dist_binder_frac,
                cfg.dist_metric,
            )
            weighted.append(term)
            for lst, key in (
                (tv_aa, "tv_aa"),
                (tv_3di, "tv_3di"),
                (js_aa, "js_aa"),
                (js_3di, "js_3di"),
                (tv_aa_b, "tv_aa_binder"),
                (tv_3di_b, "tv_3di_binder"),
                (js_aa_b, "js_aa_binder"),
                (js_3di_b, "js_3di_binder"),
            ):
                if diag[key] is not None:
                    lst.append(diag[key])

        G = len(seqs)

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        metrics = {
            "reward/dist_term_mean": float(sum(weighted) / G),
            "dist/tv_aa": _mean(tv_aa),
            "dist/tv_3di": _mean(tv_3di),
            "dist/js_aa": _mean(js_aa),
            "dist/js_3di": _mean(js_3di),
            "dist/tv_aa_binder": _mean(tv_aa_b),
            "dist/tv_3di_binder": _mean(tv_3di_b),
            "dist/js_aa_binder": _mean(js_aa_b),
            "dist/js_3di_binder": _mean(js_3di_b),
            "dist/n_iface_mean": _mean(niface),
            "dist/n_binder_mean": _mean(nbinder),
            "dist/frac_skipped": float(n_skipped / G),
            "dist/frac_penalized": float(n_penalized / G),
            "dist/frac_pooled_ref": 1.0 if ref_src == "pooled" else 0.0,
        }
        return weighted, metrics

    def _clash_terms_for_group(
        self,
        trajectory: dict,
        comp: dict,
        *,
        gen_bb: np.ndarray | None = None,
        return_eres: bool = False,
    ) -> tuple[list[float], dict] | tuple[list[float], dict, np.ndarray]:
        """Per-design smooth clash + interface-contact geometry reward + diagnostics.

        Decodes the *generated* backbone and scores each design with
        :func:`~lobster.rl_training.rewards.clash_contact_reward` — a single
        ``[0,1]`` geometry term (``clash_score·contact_score``) weighted by
        ``w_clash_contact``. Penalizes both steric overlap (clashing backbones) and
        absence of interface contact (floating binders). Returns
        ``(weighted_terms (G,), metrics)``.

        ``gen_bb`` is the decoded ``(G, L, 3, 3)`` backbone; when ``None`` it is
        decoded here (shared with the distribution term when both are on).

        When ``return_eres`` is True, additionally returns ``E_clash_res (G, L)`` — the
        per-binder-residue clash energy scattered into the full padded length at the valid
        binder positions (zeros elsewhere), for the per-token clash advantage. Its per-design
        row-sum equals ``E_clash`` for that design.
        """
        import numpy as np

        from lobster.rl_training.rewards import clash_contact_reward

        cfg = self.config
        if gen_bb is None:
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()

        G = gen_bb.shape[0]
        L = valid.shape[0]
        # Full-length binder positions (order matches clash_contact_reward's E_clash_res).
        bpos = np.nonzero(valid & binder_mask)[0]
        e_res_full = np.zeros((G, L), dtype=np.float64) if return_eres else None

        weighted: list[float] = []
        clash_s, contact_s, e_clash, soft_n, ifrac = [], [], [], [], []
        for i in range(G):
            term, diag = clash_contact_reward(
                gen_bb[i],
                valid,
                binder_mask,
                d_clash=cfg.clash_d_clash,
                clash_soft=cfg.clash_soft,
                clash_scale=cfg.clash_scale,
                contact_d0=cfg.contact_d0,
                contact_soft=cfg.contact_soft,
                frac_lo=cfg.frac_lo,
                frac_peak=cfg.frac_peak,
                frac_hi=cfg.frac_hi,
                seq_sep=cfg.clash_seq_sep,
                include_cb=cfg.clash_include_cb,
                return_eres=return_eres,
            )
            weighted.append(cfg.w_clash_contact * term)
            clash_s.append(diag["clash_score"])
            contact_s.append(diag["contact_score"])
            e_clash.append(diag["E_clash"])
            soft_n.append(diag["soft_n_iface"])
            ifrac.append(diag["iface_frac"])
            if return_eres and bpos.size:
                e_res_full[i, bpos] = diag["E_clash_res"]

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        metrics = {
            "reward/clash_term_mean": float(sum(weighted) / G),
            "clash/clash_score_mean": _mean(clash_s),
            "clash/contact_score_mean": _mean(contact_s),
            "clash/E_clash_mean": _mean(e_clash),
            "clash/soft_n_iface_mean": _mean(soft_n),
            "clash/iface_frac_mean": _mean(ifrac),
        }
        if return_eres:
            return weighted, metrics, e_res_full
        return weighted, metrics

    def _chainbreak_terms_for_group(
        self,
        trajectory: dict,
        comp: dict,
        *,
        gen_bb: np.ndarray | None = None,
        return_eres: bool = False,
    ) -> tuple[list[float], dict] | tuple[list[float], dict, np.ndarray]:
        """Per-design backbone chain-break (peptide-bond integrity) reward + diagnostics.

        Decodes the *generated* backbone and scores each design's binder chain with
        :func:`~lobster.rl_training.rewards.chainbreak_reward` — a single ``[0,1]`` realism
        term (``mean_r·gate``) weighted by ``w_chainbreak``. Rewards an intact backbone so the
        other structure-track rewards (clash / shape / 3Di-dist), all measured on this
        backbone, stay trustworthy. Returns ``(weighted_terms (G,), metrics)``.

        ``gen_bb`` is the decoded ``(G, L, 3, 3)`` backbone; when ``None`` it is decoded here
        (shared with the distribution/clash terms when they are on).

        When ``return_eres`` is True, additionally returns ``cb_break_res (G, L)`` — the
        per-binder-residue break penalty scattered into the full padded length at the valid
        binder positions (zeros elsewhere), for the per-token chain-break advantage. Its
        per-design row-sum equals ``pen = Σ (1 − r_bond)`` for that design.
        """
        import numpy as np

        from lobster.rl_training.rewards import chainbreak_reward

        cfg = self.config
        if gen_bb is None:
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()

        G = gen_bb.shape[0]
        L = valid.shape[0]
        # Full-length binder positions (order matches chainbreak_reward's cb_break_res).
        bpos = np.nonzero(valid & binder_mask)[0]
        e_res_full = np.zeros((G, L), dtype=np.float64) if return_eres else None

        weighted: list[float] = []
        mean_r_l, gate_l, nbreak_l, nhard_l, maxcn_l = [], [], [], [], []
        for i in range(G):
            term, diag = chainbreak_reward(
                gen_bb[i],
                valid,
                binder_mask,
                ideal=cfg.chainbreak_ideal,
                tol=cfg.chainbreak_tol,
                cap=cfg.chainbreak_cap,
                sigma=cfg.chainbreak_sigma,
                gate_k=cfg.chainbreak_gate_k,
                gate_mode=cfg.chainbreak_gate,
                break_hard=cfg.chainbreak_break_hard,
                break_d0=cfg.chainbreak_break_d0,
                break_soft=cfg.chainbreak_break_soft,
                return_eres=return_eres,
            )
            weighted.append(cfg.w_chainbreak * term)
            mean_r_l.append(diag["mean_r"])
            gate_l.append(diag["gate"])
            nbreak_l.append(diag["n_break"])
            nhard_l.append(diag["n_hardbreak"])
            maxcn_l.append(diag["max_cn"])
            if return_eres and bpos.size:
                e_res_full[i, bpos] = diag["cb_break_res"]

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        metrics = {
            "reward/chainbreak_term_mean": float(sum(weighted) / G),
            "chainbreak/mean_r_mean": _mean(mean_r_l),
            "chainbreak/gate_mean": _mean(gate_l),
            "chainbreak/n_break_mean": _mean(nbreak_l),
            "chainbreak/n_hardbreak_mean": _mean(nhard_l),
            "chainbreak/max_cn_mean": _mean(maxcn_l),
            "chainbreak/frac_designs_broken": float(np.mean([1.0 if h > 0 else 0.0 for h in nhard_l])),
        }
        if return_eres:
            return weighted, metrics, e_res_full
        return weighted, metrics

    def _shape_terms_for_group(
        self,
        target_id: str,
        trajectory: dict,
        comp: dict,
        seqs: list[str],
        *,
        gen_bb: np.ndarray | None = None,
    ) -> tuple[list[float], dict]:
        """Per-design 3DZD interface shape-complementarity (SC) reward + diagnostics.

        Splits the decoded backbone into antigen (pinned) and binder (sampled) chains,
        pairs each with its AA sequence (antigen decoded from the pinned endpoint tokens,
        binder from ``seqs``), and ships the whole group to the LigandMPNN-repack pool via
        :class:`~lobster.rl_training.rewards.ShapeRewardClient`. The pool packs full-atom
        side chains and returns a per-design SC term ∈ ``[0,1]`` (clipped Pearson of the
        two interface ΔSASA-patch 3D-Zernike descriptors), weighted by ``w_shape``. A
        design the pool fails/times-out on contributes ``0.0``. Returns
        ``(weighted_terms (G,), metrics)``.

        ``gen_bb`` is the decoded ``(G, L, 3, 3)`` backbone (N, CA, C); when ``None`` it
        is decoded here (shared with the distribution/clash terms when they are on).
        """
        import numpy as np

        cfg = self.config
        if gen_bb is None:
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        antigen_mask = valid & ~binder_mask
        # Antigen sequence: decode the endpoint AA (antigen positions are pinned/clean, so
        # this returns the native antigen), same standard-AA alphabet as the binder seqs.
        aa = self.model.decode_endpoint_aa(trajectory).cpu().numpy()  # (G, L)

        designs: list[dict] = []
        for i in range(gen_bb.shape[0]):
            ag_ids = aa[i][antigen_mask]
            ag_seq = "".join(_AA_LETTERS[j] if 0 <= j < 20 else "G" for j in ag_ids)
            designs.append(
                {
                    "ag_bb": gen_bb[i][antigen_mask],  # (Na, 3, 3) N,CA,C
                    "ag_seq": ag_seq,
                    "bd_bb": gen_bb[i][binder_mask],  # (Nb, 3, 3)
                    "bd_seq": seqs[i],
                }
            )

        rewards_l, diags = self._shape_client.rewards_for_group(target_id, designs)
        weighted = [cfg.w_shape * r for r in rewards_l]

        G = gen_bb.shape[0]
        sc_vals = [d["sc"] for d in diags if d is not None and d.get("sc") is not None]
        npatch_a = [d["n_patch_a"] for d in diags if d is not None and d.get("n_patch_a") is not None]
        npatch_b = [d["n_patch_b"] for d in diags if d is not None and d.get("n_patch_b") is not None]
        n_scored = sum(1 for d in diags if d is not None)

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        metrics = {
            "reward/shape_term_mean": float(sum(weighted) / G),
            "shape/sc_mean": _mean(sc_vals),  # raw (pre-clip) interface Pearson
            "shape/n_patch_a_mean": _mean(npatch_a),
            "shape/n_patch_b_mean": _mean(npatch_b),
            "shape/scored_frac": float(n_scored / G),
        }
        return weighted, metrics

    def _build_repack_designs(
        self,
        trajectory: dict,
        comp: dict,
        seqs: list[str],
        gen_bb: np.ndarray,
    ) -> list[dict]:
        """Build the per-design antigen/binder backbone+sequence packets for the repack pool.

        Extracted from :meth:`_repack_terms_for_group` so the pipelined path can submit the
        repack round-trip (phase 1) before mapping its results (phase 2). Splits the decoded
        ``(G, L, 3, 3)`` backbone into the pinned antigen cloud (with its argmax-decoded
        sequence) and the sampled binder cloud (carrying the group's designed ``seqs``).

        Parameters
        ----------
        trajectory : dict
            The rollout trajectory (used only to decode the argmax antigen sequence).
        comp : dict
            The complex batch dict (``mask``, ``binder_positions``).
        seqs : list[str]
            The ``G`` decoded binder sequences (binder residue order).
        gen_bb : np.ndarray
            The decoded ``(G, L, 3, 3)`` N,CA,C backbone.

        Returns
        -------
        list[dict]
            One packet per design with keys ``ag_bb``, ``ag_seq``, ``bd_bb``, ``bd_seq``.
        """

        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        antigen_mask = valid & ~binder_mask
        aa = self.model.decode_endpoint_aa(trajectory).cpu().numpy()  # (G, L)

        designs: list[dict] = []
        for i in range(gen_bb.shape[0]):
            ag_ids = aa[i][antigen_mask]
            ag_seq = "".join(_AA_LETTERS[j] if 0 <= j < 20 else "G" for j in ag_ids)
            designs.append(
                {
                    "ag_bb": gen_bb[i][antigen_mask],  # (Na, 3, 3) N,CA,C
                    "ag_seq": ag_seq,
                    "bd_bb": gen_bb[i][binder_mask],  # (Nb, 3, 3)
                    "bd_seq": seqs[i],
                }
            )
        return designs

    def _repack_terms_for_group(
        self,
        target_id: str,
        trajectory: dict,
        comp: dict,
        seqs: list[str],
        want: tuple[str, ...],
        *,
        gen_bb: np.ndarray | None = None,
        return_seq: bool = False,
        want_pot: bool = False,
        precomputed_results: list[dict | None] | None = None,
    ) -> tuple[list[float], list[float], list[float], dict, dict | None, dict | None]:
        """Full-atom repack-pool reward terms (SC / all-atom clash / AAR) from ONE round-trip.

        Ships the group's antigen (pinned) + binder (sampled) backbone clouds to the shared
        LigandMPNN-repack worker pool a **single** time with the combined ``want`` metric-set,
        so a design is side-chain-packed once regardless of how many of the three full-atom
        terms are active. The pool returns per design a dict carrying the requested metrics
        (SC fields flat at the top level; ``clash`` / ``aar`` nested), which are mapped to
        scalars by :func:`~lobster.rl_training.rewards.reward_from_shape`,
        :func:`~lobster.rl_training.rewards.sc_clash_reward`, and
        :func:`~lobster.rl_training.rewards.reward_from_aar`. All three run on the CPU worker
        pool, so throughput scales by adding CPU workers.

        Design construction (antigen/binder split, sequences) is identical to
        :meth:`_shape_terms_for_group`. ``gen_bb`` is the decoded ``(G, L, 3, 3)`` backbone
        (shared with the distribution/clash terms when they are on); decoded here if ``None``.

        Returns ``(shape_terms, sc_clash_terms, aar_terms, metrics, sft_payload, pot_eres)`` —
        the first three each a length-``G`` list of the **weighted** contribution for that metric
        (all ``0.0`` for a metric not in ``want``), the merged diagnostics dict (only keys for the
        requested metrics), a ``sft_payload`` dict (see below) and — when ``want_pot`` (the
        per-token interface-potential arm is active) — a ``pot_eres`` dict carrying the dense
        per-binder-residue potential signals ``{"lj_eres","dsasa_eres","hb_eres"}`` as ``(G, L)``
        numpy penalty arrays (larger = worse) for the per-position structure advantage; ``None``
        otherwise. The ``sft_payload`` (present only when ``return_seq``, CHORD SFT active)
        carries the dense per-token expert target for the sequence-distillation term:

        * ``target_ids`` — ``(G, L)`` int64 ``numpy`` array of the LigandMPNN-designed binder
          identities in the 33-token ``AA_VOCAB`` space, placed at the binder positions of the
          full padded layout (antigen + non-generated positions and any non-standard designed
          residue set to :data:`SFT_IGNORE_INDEX`).
        * ``supervise_mask`` — ``(G, L)`` bool ``numpy`` array of positions to distil (binder ∩
          scope ∩ valid-designed-identity). ``scope='interface'`` restricts to designed binder
          residues at the interface (as reported by the repack worker's ``iface_binder``);
          ``scope='binder'`` supervises every designed binder residue.

        ``sft_payload`` is ``None`` when ``return_seq`` is false (byte-identical legacy path).
        The designed sequence + interface flags ride the SAME single repack round-trip as the
        reward metrics (``score_group(..., return_seq=True)`` sets the worker's ``aar`` dict's
        ``seq_design`` / ``iface_binder`` fields); no extra pack is performed.
        """
        import numpy as np

        cfg = self.config
        if gen_bb is None:
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        G = gen_bb.shape[0]

        if precomputed_results is not None:
            # Pipelined phase-2 path: the repack round-trip was submitted+collected by the
            # caller (see ``_rollout_and_submit`` / ``_collect_and_advantage``); skip the design
            # build + blocking ``score_group`` and map the given results. Byte-identical to the
            # inline path — ``submit_group``/``collect_group`` reproduce ``score_group`` exactly.
            results = precomputed_results
        else:
            designs = self._build_repack_designs(trajectory, comp, seqs, gen_bb)
            results = self._shape_client.score_group(target_id, designs, want=want, return_seq=return_seq)

        shape_terms = [0.0] * G
        sc_clash_terms = [0.0] * G
        aar_terms_l = [0.0] * G
        metrics: dict = {}

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        if "sc" in want:
            weighted = [cfg.w_shape * reward_from_shape(r) for r in results]
            shape_terms = weighted
            sc_vals = [r["sc"] for r in results if r is not None and r.get("sc") is not None]
            npatch_a = [r["n_patch_a"] for r in results if r is not None and r.get("n_patch_a") is not None]
            npatch_b = [r["n_patch_b"] for r in results if r is not None and r.get("n_patch_b") is not None]
            n_scored = sum(1 for r in results if r is not None and r.get("term") is not None)
            metrics.update(
                {
                    "reward/shape_term_mean": float(sum(weighted) / G),
                    "shape/sc_mean": _mean(sc_vals),
                    "shape/n_patch_a_mean": _mean(npatch_a),
                    "shape/n_patch_b_mean": _mean(npatch_b),
                    "shape/scored_frac": float(n_scored / G),
                }
            )

        if "clash" in want:
            clash_res = [r.get("clash") if r is not None else None for r in results]
            sc_density = bool(getattr(cfg, "sc_clash_density", False))
            weighted = [cfg.w_sc_clash * sc_clash_reward(cr, density=sc_density) for cr in clash_res]
            sc_clash_terms = weighted
            e_bd = [cr["E_clash_binder"] for cr in clash_res if cr is not None]
            e_if = [cr["E_clash_iface"] for cr in clash_res if cr is not None]
            e_tot = [cr["E_clash_total"] for cr in clash_res if cr is not None]
            e_if_res = [
                cr["E_clash_iface_res"]
                for cr in clash_res
                if cr is not None and cr.get("E_clash_iface_res") is not None
            ]
            n_if_res = [cr["n_iface_res"] for cr in clash_res if cr is not None and cr.get("n_iface_res") is not None]
            # per-residue DENSITY energy actually optimized in density mode (retraction-resistant):
            # e_clash_binder_norm (= E_clash_binder/n_res) + E_clash_iface_res/n_iface_res.
            e_density = [
                float(cr.get("e_clash_binder_norm", 0.0))
                + (
                    float(cr["E_clash_iface_res"]) / int(cr["n_iface_res"])
                    if int(cr.get("n_iface_res", 0) or 0) > 0
                    else 0.0
                )
                for cr in clash_res
                if cr is not None
            ]
            n_scored = sum(1 for cr in clash_res if cr is not None)
            metrics.update(
                {
                    "reward/sc_clash_term_mean": float(sum(weighted) / G),
                    "sc_clash/e_binder_mean": _mean(e_bd),  # whole-binder self-clash energy
                    "sc_clash/e_iface_mean": _mean(e_if),  # binder↔antigen clash energy
                    "sc_clash/e_total_mean": _mean(e_tot),
                    "sc_clash/e_iface_res_mean": _mean(e_if_res),  # interface-restricted (diagnostic)
                    "sc_clash/n_iface_res_mean": _mean(n_if_res),
                    "sc_clash/e_density_mean": _mean(e_density),  # per-residue density (density-mode energy)
                    "sc_clash/scored_frac": float(n_scored / G),
                }
            )

        if "aar" in want:
            aar_res = [r.get("aar") if r is not None else None for r in results]
            weighted = [cfg.w_aar * reward_from_aar(ar) for ar in aar_res]
            aar_terms_l = weighted

            def _finite(dicts: list[dict | None], key: str) -> list[float]:
                return [d[key] for d in dicts if d is not None and d.get(key) is not None and np.isfinite(d[key])]

            n_scored = sum(1 for ar in aar_res if ar is not None)
            metrics.update(
                {
                    "reward/aar_term_mean": float(sum(weighted) / G),
                    "aar/aar_mean": _mean(_finite(aar_res, "aar")),  # whole-binder recovery
                    "aar/aar_iface_mean": _mean(_finite(aar_res, "aar_iface")),  # diagnostic
                    "aar/c_mpnn_mean": _mean(_finite(aar_res, "c_mpnn")),
                    "aar/c_mpnn_iface_mean": _mean(_finite(aar_res, "c_mpnn_iface")),
                    "aar/scored_frac": float(n_scored / G),
                }
            )

        # --- Per-token interface potentials (second reward set) -----------------------------
        # The repack worker returns, per design, per-binder-residue vectors of the validated
        # all-atom interface potentials (``pot`` -> e_lj / n_hb / dsasa; see
        # ``scripts/_packed_potentials.potentials_per_residue``). Each is scattered onto the
        # (G, L) binder layout in the SAME "penalty" convention the per-token clash arm uses
        # (larger = worse, so ``_pos_norm_adv`` -> less penalty => higher advantage):
        #   e_lj    : penalty = e_lj              (bounded LJ energy; lower is better)
        #   dsasa   : penalty = -dsasa            (more buried area is better)
        #   n_hb    : penalty = -n_hb             (more interface H-bonds is better)
        # ``sum(vector)`` over binder residues == the validated scalar potential, so the mean
        # diagnostics below are directly comparable to the offline AUROC study.
        pot_eres: dict | None = None
        if want_pot:
            with_sasa = getattr(self.config, "pot_with_sasa", True)
            L = int(gen_bb.shape[1])
            binder_idx = np.nonzero(binder_mask)[0]
            nb = int(binder_idx.shape[0])
            lj_eres = np.zeros((G, L), dtype=np.float64)
            dsasa_eres = np.zeros((G, L), dtype=np.float64)
            hb_eres = np.zeros((G, L), dtype=np.float64)
            n_pot_scored = 0
            e_lj_sum, dsasa_sum, hb_sum = [], [], []
            for i, r in enumerate(results):
                p = r.get("pot") if r is not None else None
                if p is None:
                    continue
                lj = np.asarray(p.get("e_lj", []), dtype=np.float64)
                if lj.shape[0] != nb:
                    continue  # worker returned no / mismatched potential vector for this design
                lj_eres[i, binder_idx] = lj
                e_lj_sum.append(float(lj.sum()))
                hb = np.asarray(p.get("n_hb", []), dtype=np.float64)
                if hb.shape[0] == nb:
                    hb_eres[i, binder_idx] = -hb
                    hb_sum.append(float(hb.sum()))
                if with_sasa:
                    ds = np.asarray(p.get("dsasa", []), dtype=np.float64)
                    if ds.shape[0] == nb:
                        dsasa_eres[i, binder_idx] = -ds
                        dsasa_sum.append(float(ds.sum()))
                n_pot_scored += 1
            pot_eres = {"lj_eres": lj_eres, "hb_eres": hb_eres}
            if with_sasa:
                pot_eres["dsasa_eres"] = dsasa_eres  # only present when the SASA term is enabled
            metrics.update(
                {
                    "pot/scored_frac": float(n_pot_scored / G),
                    "pot/e_lj_mean": _mean(e_lj_sum),  # bounded LJ energy (lower = better)
                    "pot/n_hb_mean": _mean(hb_sum),  # interface H-bonds (higher = better)
                }
            )
            if with_sasa:
                metrics["pot/dsasa_mean"] = _mean(dsasa_sum)  # buried ΔSASA (higher = better)

        # --- CHORD SFT payload: dense per-token expert (LigandMPNN-designed) targets -------
        # Built from the SAME round-trip's ``aar`` dicts (seq_design + iface_binder). Each
        # design's designed binder identities are remapped to the 33-token AA_VOCAB space and
        # scattered onto the binder positions of the full padded layout; the supervise mask is
        # binder ∩ scope ∩ valid-designed-identity. Designs the worker failed to score
        # (aar dict missing / short) get an all-ignore row (nothing supervised).
        sft_payload: dict | None = None
        if return_seq:
            L = int(gen_bb.shape[1])
            binder_idx = np.nonzero(binder_mask)[0]  # full-layout indices of binder residues, in order
            nb = int(binder_idx.shape[0])
            interface_scope = str(getattr(cfg, "sft_scope", "interface")) == "interface"
            soft_label = str(getattr(cfg, "sft_label", "hard")) == "soft"
            target_ids = np.full((G, L), SFT_IGNORE_INDEX, dtype=np.int64)
            supervise_mask = np.zeros((G, L), dtype=bool)
            soft_targets = None
            if soft_label:
                # SOFT distillation: distil LigandMPNN's full per-position output distribution
                # (design_logq, cols = MPNN alphabet "…VWY"+X) instead of the argmax identity.
                # Map cols 0..19 -> AA_VOCAB ids and drop X (renormalize over the 20 canonical
                # AAs); φ still keys on the teacher MODE (argmax) token via target_ids.
                from lobster.rl_training.rewards._aar_reward import _STANDARD_AA1
                from lobster.tokenization._amino_acid import AA_VOCAB

                aa33_cols = binder_letters_to_aa33(_STANDARD_AA1)  # (20,) AA_VOCAB ids, MPNN col order
                soft_targets = np.zeros((G, L, len(AA_VOCAB)), dtype=np.float32)
            aar_res = [r.get("aar") if r is not None else None for r in results]
            n_sft_scored = 0
            teacher_conf_sum, teacher_conf_n = 0.0, 0
            for i, ar in enumerate(aar_res):
                if ar is None:
                    continue
                if interface_scope:
                    iface = ar.get("iface_binder")
                    sel = (
                        np.asarray(iface, dtype=bool)
                        if (iface is not None and len(iface) == nb)
                        else np.zeros(nb, dtype=bool)
                    )
                else:
                    sel = np.ones(nb, dtype=bool)
                if soft_label:
                    q21 = ar.get("design_logq")
                    if q21 is None or len(q21) != nb:
                        continue  # worker returned no / mismatched teacher distribution
                    q21 = np.asarray(q21, dtype=np.float32)  # (nb, 21) MPNN-alphabet probabilities
                    q20 = q21[:, :20]  # drop the X/unknown column
                    row_sum = q20.sum(axis=1, keepdims=True)  # (nb, 1)
                    valid_bd = row_sum[:, 0] > 1e-6  # rows carrying canonical-AA mass
                    q20 = q20 / np.clip(row_sum, 1e-6, None)  # renormalize over the 20 canonical AAs
                    q33 = np.zeros((nb, soft_targets.shape[2]), dtype=np.float32)
                    q33[:, aa33_cols] = q20
                    tgt_bd = np.where(valid_bd, aa33_cols[q20.argmax(axis=1)], SFT_IGNORE_INDEX).astype(np.int64)
                    soft_targets[i, binder_idx, :] = q33
                    if valid_bd.any():
                        teacher_conf_sum += float(q20[valid_bd].max(axis=1).sum())
                        teacher_conf_n += int(valid_bd.sum())
                else:
                    letters = ar.get("seq_design")
                    if not letters or len(letters) != nb:
                        continue  # worker returned no / mismatched designed sequence for this design
                    tgt_bd = binder_letters_to_aa33(letters)  # (nb,) 33-token ids, SFT_IGNORE_INDEX for non-std
                    valid_bd = tgt_bd >= 0
                sup_bd = sel & valid_bd
                target_ids[i, binder_idx] = tgt_bd
                supervise_mask[i, binder_idx] = sup_bd
                n_sft_scored += 1
            sft_payload = {"target_ids": target_ids, "supervise_mask": supervise_mask}
            if soft_label:
                sft_payload["soft_targets"] = soft_targets
            metrics.update(
                {
                    "sft/scored_frac": float(n_sft_scored / G),
                    "sft/supervised_tokens_mean": float(supervise_mask.sum() / G),
                }
            )
            if soft_label and teacher_conf_n > 0:
                metrics["sft/teacher_conf_mean"] = float(teacher_conf_sum / teacher_conf_n)

        return shape_terms, sc_clash_terms, aar_terms_l, metrics, sft_payload, pot_eres

    # -------------------------------------------------------------------- rewards
    def _struct_sft_payload_for_group(self, comp: dict, confs: list[dict | None]) -> tuple[dict, dict]:
        """Build the Protenix fold-consistency SFT target payload for one group.

        For each design, assembles the whole-complex backbone from the Protenix fold
        (``antigen_bb`` chain A then ``binder_bb`` chain B — the same antigen-then-binder order
        the sctm reward uses), derives the LG structure tokens (s*, via the policy FSQ codec
        ``encode_structure``) and 3Di tokens (τ*, via mini3di) over the whole complex, and
        scatters them onto the padded ``(G, L)`` layout at the antigen and binder positions
        (:func:`~lobster.rl_training.rewards.build_struct_sft_targets`). Designs the fold failed
        on (missing/short backbone, a predicted-length mismatch vs the layout, or a codec error)
        get an all-ignore row (nothing supervised).

        The token derivation runs under ``no_grad`` (targets, no graph). Returns
        ``(payload, metrics)`` with ``payload = {struct_target_ids (G,L), tri_target_ids (G,L),
        supervise_mask (G,L)}``.
        """
        import numpy as np

        cfg = self.config
        valid = comp["mask"][0].bool().cpu().numpy()
        binder_mask = comp["binder_positions"].cpu().bool().numpy()
        antigen_mask = valid & ~binder_mask
        binder_idx = np.nonzero(binder_mask)[0]
        antigen_idx = np.nonzero(antigen_mask)[0]
        G = len(confs)
        L = int(binder_mask.shape[0])
        na, nb = int(antigen_idx.shape[0]), int(binder_idx.shape[0])

        experts: list[dict | None] = []
        n_with_bb = 0  # designs whose fold returned a length-matched backbone (the gate at :1595)
        first_err: str | None = None  # first codec/assemble failure, surfaced if nothing gets scored
        with torch.no_grad():
            for conf in confs:
                ex = None
                if conf is not None:
                    ag_bb = conf.get("antigen_bb")
                    bd_bb = conf.get("binder_bb")
                    if ag_bb is not None and bd_bb is not None and len(ag_bb) == na and len(bd_bb) == nb:
                        n_with_bb += 1
                        coords_res = np.concatenate(
                            [np.asarray(ag_bb, dtype=np.float32), np.asarray(bd_bb, dtype=np.float32)], axis=0
                        )  # (na+nb, 3, 3) backbone (N, CA, C), antigen-then-binder
                        chains = np.array(["A"] * na + ["B"] * nb)
                        try:
                            ex = assemble_structure_expert(
                                coords_res,
                                chains,
                                encode_structure_fn=self.model.encode_structure,
                                binder_chain="B",
                                device=self.device,
                                supervise_scope=cfg.struct_sft_scope,
                            )
                        except Exception as e:  # noqa: BLE001 — a fold/codec failure just drops this design
                            ex = None
                            if first_err is None:
                                first_err = f"{type(e).__name__}: {e}"
                experts.append(ex)

        tgt = build_struct_sft_targets(experts, binder_idx, antigen_idx, G, L)
        # Diagnose the silent no-op: struct_sft_mu>0 but nothing got scored despite live folds means
        # the codec/assemble raised on every design (e.g. a bad mask dtype) and got swallowed above.
        # Warn ONCE with the first error so a 168h run can't train with zero struct-SFT gradient.
        if cfg.struct_sft_mu > 0 and tgt["n_scored"] == 0 and n_with_bb > 0 and not self._struct_sft_warned:
            self._struct_sft_warned = True
            logger.warning(
                "struct-SFT scored 0/%d designs despite %d folds with valid backbones — the structure "
                "expert is a silent no-op (zero gradient). First error: %s",
                G,
                n_with_bb,
                first_err or "(gate/length mismatch, no exception raised)",
            )
        payload = {
            "struct_target_ids": tgt["struct_target_ids"],
            "tri_target_ids": tgt["tri_target_ids"],
            "supervise_mask": tgt["supervise_mask"],
        }
        metrics = {
            "struct_sft/scored_frac": float(tgt["n_scored"] / G),
            "struct_sft/supervised_tokens_mean": float(tgt["supervise_mask"].sum() / G),
        }
        return payload, metrics

    def _compute_rewards(
        self,
        target_id: str,
        seqs: list[str],
        tri_seqs: list[str] | None,
        trajectory: dict,
        comp: dict,
        *,
        gen_bb: np.ndarray | None = None,
        precomputed_repack: list[dict | None] | None = None,
    ) -> tuple[torch.Tensor, dict, dict | None, dict | None, dict | None]:
        """Assemble the reward for a group.

        Returns ``(rewards (G,), metrics, sft_payload, struct_sft_payload, pt_extras)``.

        ``sft_payload`` is ``None`` unless CHORD sequence SFT distillation is active
        (``sft_mu>0``), in which case it carries the dense per-token LigandMPNN expert target
        tensors (see :meth:`_repack_terms_for_group`) consumed by :meth:`_ppo_update`.

        ``struct_sft_payload`` is ``None`` unless the Protenix fold-consistency SFT is active
        (``struct_sft_mu>0``), in which case it carries the dense structure (s*) + 3Di (τ*)
        expert target tensors (see :meth:`_struct_sft_payload_for_group`) consumed by
        :meth:`_ppo_update`.

        ``pt_extras`` is ``None`` unless the per-token clash advantage is active
        (``per_token_clash``), in which case it carries ``{"clash_eres": (G, L)}`` — the
        per-residue backbone clash energy used to build the per-position structure advantage.

        ``reward_i = confidence_i + structure_i + seq_diversity_i + struct_diversity_i``,
        each term a weighted, per-metric-clipped (``[0,1]``) contribution (see
        ``rewards/README.md``). Structure coords are fetched from the oracle only when a
        structure weight is non-zero.

        Pipelining (byte-identical): when ``gen_bb`` is supplied the shared decoded backbone
        is reused instead of re-decoded; when ``precomputed_repack`` is supplied the repack
        round-trip results (from a prior :meth:`~ShapeRewardClient.submit_group` /
        ``collect_group``) are mapped instead of blocking on ``score_group``. Both default to
        ``None`` — the legacy inline path — so all non-pipelined callers are unaffected.
        """
        cfg = self.config
        G = len(seqs)
        conf_weights = {
            "w_iptm": cfg.w_iptm,
            "w_ptm": cfg.w_ptm,
            "w_abag_iptm": cfg.w_abag_iptm,
            "w_plddt": cfg.w_plddt,
            "w_gpde": cfg.w_gpde,
            "w_pae_global": cfg.w_pae_global,
            "w_pae_interface": cfg.w_pae_interface,
        }
        need_struct = cfg.w_sctm_binder > 0 or cfg.w_sctm_complex > 0
        # Protenix fold-consistency SFT (struct_sft_mu>0) folds the policy sequence and derives
        # (s*, τ*) from the FULL (N, CA, C) backbone, so it needs the coords fetch with the
        # extra backbone payload (return_backbone). The CA-only sctm reward only needs coords.
        need_struct_sft = getattr(cfg, "struct_sft_mu", 0.0) > 0
        # Fetch coords + compute sctm whenever a structure weight is on OR the
        # diagnostic flag is set (so sctm is logged even at weight 0), OR the struct-SFT is on.
        need_coords = need_struct or cfg.log_struct_diagnostic or need_struct_sft
        # Protenix is only queried when a confidence/structure term actually needs it.
        # For a Protenix-free run (all conf weights 0, no structure/coords) we skip the
        # oracle entirely: confs=[None]*G makes every confidence helper contribute
        # 0/{}/False (see rewards/_protenix_reward.py), so the reward reduces to the
        # dense distribution term with no worker pool / queue involved.
        need_conf = any(w > 0 for w in conf_weights.values())
        if need_conf or need_coords:
            confs = self.reward_client.score_group(
                target_id, seqs, return_coords=need_coords, return_backbone=need_struct_sft
            )
        else:
            confs = [None] * G

        # 1. Confidence term — flat, per-metric-clipped weighted linear combo.
        conf_terms = [reward_from_confidence(c, conf_weights) for c in confs]

        # 2. Structure self-consistency term. Computed whenever coords are
        # available; the weighted contribution stays 0 when w_sctm_* are 0, so
        # with only the diagnostic flag on this feeds wandb but not the reward.
        if need_coords:
            struct_terms_l, sctm_b, sctm_c = self._structure_terms_for_group(trajectory, comp, confs)
        else:
            struct_terms_l = [0.0] * G
            sctm_b = sctm_c = [0.0] * G

        # 3./4. Diversity terms — mean pairwise k-mer-Jaccard novelty (AA + 3Di).
        seq_nov = jaccard_novelty_group(seqs)
        seq_div_terms = [cfg.w_seq_diversity * v for v in seq_nov]
        struct_nov = jaccard_novelty_group(tri_seqs) if tri_seqs is not None else [0.0] * G
        struct_div_terms = [cfg.w_struct_diversity * v for v in struct_nov]

        # 4b. Within-sequence anti-degeneracy — per-design saturating linguistic-complexity
        # reward (always computed so LC is tracked; term is 0 when the weight is off).
        w_seq_complexity = float(getattr(cfg, "w_seq_complexity", 0.0))
        lc_rew, lcs = lc_saturating_reward(seqs, lc_full=float(getattr(cfg, "lc_full", 0.7)))
        seq_complex_terms = [w_seq_complexity * v for v in lc_rew]

        # 5./6./7. Geometry + full-atom shaping terms (Protenix-free). All read the decoded
        # generated backbone, so decode it ONCE here and share it. Each is inert (0) and adds
        # no metrics when its weight is 0 — keeps the reward byte-identical.
        need_dist = cfg.w_aa_dist > 0 or cfg.w_3di_dist > 0 or cfg.log_dist_diagnostic
        need_clash = cfg.w_clash_contact > 0
        need_chainbreak = cfg.w_chainbreak > 0
        need_shape = cfg.w_shape > 0
        # All-atom side-chain clash: in the reward when w_sc_clash>0, OR tracked-but-off when
        # log_sc_clash_diagnostic is set (forces "clash" into the repack want at weight 0, so
        # its metrics are logged without entering the reward — mirrors log_dist_diagnostic).
        need_sc_clash = cfg.w_sc_clash > 0 or getattr(cfg, "log_sc_clash_diagnostic", False)
        # CHORD SFT distillation (sft_mu>0) needs the LigandMPNN-*designed* binder sequence,
        # which is produced by the "aar" repack path (with return_seq). It does NOT need AAR
        # in the reward — force the "aar" want when SFT is on even if w_aar==0 (the weighted
        # AAR term is then all-zero and the reward stays unchanged; only seq_design is used).
        need_sft = getattr(cfg, "sft_mu", 0.0) > 0
        need_aar = cfg.w_aar > 0 or need_sft
        # Per-token interface potentials (e_lj/dsasa/n_hb on the LigandMPNN pack) reuse the SAME
        # repack round-trip as R_SC; request them via the "pot" want when per_token_pot is on.
        need_pot = getattr(cfg, "per_token_pot", False)
        need_repack = need_shape or need_sc_clash or need_aar or need_pot
        # ``gen_bb`` may be supplied by a pipelined caller (decoded once in phase 1) — reuse it;
        # otherwise decode here when any backbone-reading term is active.
        if gen_bb is None and (need_dist or need_clash or need_chainbreak or need_repack):
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)

        # 5. Interface-distribution distance term.
        if need_dist:
            dist_terms_l, dist_metrics = self._distribution_terms_for_group(
                target_id, trajectory, comp, seqs, gen_bb=gen_bb
            )
        else:
            dist_terms_l = [0.0] * G
            dist_metrics = {}

        # 6. Smooth clash + interface-contact geometry term. When per-token clash is on, also
        # obtain the per-residue clash energy (G, L) to build the per-position structure advantage.
        clash_eres = None
        if need_clash:
            if cfg.per_token_clash:
                clash_terms_l, clash_metrics, clash_eres = self._clash_terms_for_group(
                    trajectory, comp, gen_bb=gen_bb, return_eres=True
                )
            else:
                clash_terms_l, clash_metrics = self._clash_terms_for_group(trajectory, comp, gen_bb=gen_bb)
        else:
            clash_terms_l = [0.0] * G
            clash_metrics = {}

        # 6b. Backbone chain-break (peptide-bond integrity) realism term. When per-token
        # chain-break is on, also obtain the per-residue break penalty (G, L) for the
        # per-position structure advantage.
        chainbreak_eres = None
        if need_chainbreak:
            if cfg.per_token_chainbreak:
                chainbreak_terms_l, chainbreak_metrics, chainbreak_eres = self._chainbreak_terms_for_group(
                    trajectory, comp, gen_bb=gen_bb, return_eres=True
                )
            else:
                chainbreak_terms_l, chainbreak_metrics = self._chainbreak_terms_for_group(
                    trajectory, comp, gen_bb=gen_bb
                )
        else:
            chainbreak_terms_l = [0.0] * G
            chainbreak_metrics = {}

        # 7. Full-atom LigandMPNN-repack terms (SC shape-complementarity, all-atom clash, AAR)
        # — one shared worker pool. The SC-only case routes through the original
        # _shape_terms_for_group (keeps the running SC arm byte-identical); any combination
        # that includes clash/aar routes through _repack_terms_for_group, which packs each
        # design ONCE and returns the union metric-set in a single round-trip.
        shape_terms_l = [0.0] * G
        sc_clash_terms_l = [0.0] * G
        aar_terms_l = [0.0] * G
        repack_metrics: dict = {}
        sft_payload: dict | None = None
        pot_eres: dict | None = None
        if need_shape and not (need_sc_clash or need_aar or need_pot):
            shape_terms_l, repack_metrics = self._shape_terms_for_group(
                target_id, trajectory, comp, seqs, gen_bb=gen_bb
            )
        elif need_repack:
            want = tuple(
                m
                for m, on in (("sc", need_shape), ("clash", need_sc_clash), ("aar", need_aar), ("pot", need_pot))
                if on
            )
            shape_terms_l, sc_clash_terms_l, aar_terms_l, repack_metrics, sft_payload, pot_eres = (
                self._repack_terms_for_group(
                    target_id,
                    trajectory,
                    comp,
                    seqs,
                    want,
                    gen_bb=gen_bb,
                    return_seq=need_sft,
                    want_pot=need_pot,
                    precomputed_results=precomputed_repack,
                )
            )

        # Protenix fold-consistency SFT payload (struct_sft_mu>0): dense (s*, τ*) structure +
        # 3Di expert targets derived by FOLDING the policy sequence (built from the same Protenix
        # ``confs`` fetched above with return_backbone). Independent of the reward — it is a
        # distillation target, not a reward term, so it does not enter ``rewards`` below.
        struct_sft_payload: dict | None = None
        struct_sft_metrics: dict = {}
        if need_struct_sft:
            struct_sft_payload, struct_sft_metrics = self._struct_sft_payload_for_group(comp, confs)

        rewards = torch.tensor(
            [
                c + s + sd + td + cx + dt + gt + cb + sh + scl + ar
                for c, s, sd, td, cx, dt, gt, cb, sh, scl, ar in zip(
                    conf_terms,
                    struct_terms_l,
                    seq_div_terms,
                    struct_div_terms,
                    seq_complex_terms,
                    dist_terms_l,
                    clash_terms_l,
                    chainbreak_terms_l,
                    shape_terms_l,
                    sc_clash_terms_l,
                    aar_terms_l,
                )
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Per-key mean/std over the designs that actually received that Protenix
        # field (None-scored designs are excluded from the raw-confidence stats).
        def _conf_stat(key: str) -> tuple[float, float]:
            vals = [c[key] for c in confs if c is not None and c.get(key) is not None]
            if not vals:
                return 0.0, 0.0
            t = torch.tensor(vals, dtype=torch.float32)
            return float(t.mean()), (float(t.std(unbiased=False)) if len(vals) > 1 else 0.0)

        ptm_mean, ptm_std = _conf_stat("ptm")
        iptm_mean, iptm_std = _conf_stat("iptm")
        abag_mean, abag_std = _conf_stat("abag_iptm")
        plddt_mean, _ = _conf_stat("plddt")
        gpde_mean, gpde_std = _conf_stat("gpde")
        pae_global_mean, pae_global_std = _conf_stat("pae_global")
        pae_iface_mean, pae_iface_std = _conf_stat("pae_interface")
        scored_frac = float(sum(continuous_ip(c) is not None for c in confs) / G)

        # Per-metric confidence contribution means (only active, present metrics).
        comp_sums: dict[str, float] = {}
        for c in confs:
            for wk, v in confidence_components(c, conf_weights).items():
                comp_sums[wk] = comp_sums.get(wk, 0.0) + v

        metrics = {
            # --- total reward + four-term decomposition (terms sum to reward/mean) ---
            "reward/mean": float(rewards.mean()),
            "reward/std": float(rewards.std(unbiased=False)),
            "reward/max": float(rewards.max()),
            "reward/confidence_term_mean": float(sum(conf_terms) / G),
            "reward/structure_term_mean": float(sum(struct_terms_l) / G),
            "reward/seq_diversity_term_mean": float(sum(seq_div_terms) / G),
            "reward/struct_diversity_term_mean": float(sum(struct_div_terms) / G),
            "reward/seq_complexity_term_mean": float(sum(seq_complex_terms) / G),
            "reward/struct_sctm_binder_term_mean": float(cfg.w_sctm_binder * sum(sctm_b) / G),
            "reward/struct_sctm_complex_term_mean": float(cfg.w_sctm_complex * sum(sctm_c) / G),
            # --- raw Protenix confidences (confidence module's field names) ---
            "conf/ptm_mean": ptm_mean,
            "conf/ptm_std": ptm_std,
            "conf/iptm_mean": iptm_mean,
            "conf/iptm_std": iptm_std,
            "conf/abag_iptm_mean": abag_mean,
            "conf/abag_iptm_std": abag_std,
            "conf/plddt_mean": plddt_mean,
            "conf/gpde_mean": gpde_mean,
            "conf/gpde_std": gpde_std,
            # PAE (predicted aligned error, Å) from token_pair_pae. global = mean over all
            # token pairs; interface = mean over cross-chain (A↔B) pairs.
            "conf/pae_global_mean": pae_global_mean,
            "conf/pae_global_std": pae_global_std,
            "conf/pae_interface_mean": pae_iface_mean,
            "conf/pae_interface_std": pae_iface_std,
            "conf/scored_frac": scored_frac,
            "conf/pass_rate": float(sum(passes(c) for c in confs) / G),
            # --- structure self-consistency (raw TM-scores) ---
            "struct/sctm_binder": float(sum(sctm_b) / G),
            "struct/sctm_complex": float(sum(sctm_c) / G),
            # --- diversity novelty (raw mean pairwise Jaccard distance) ---
            "diversity/seq_novelty_mean": float(sum(seq_nov) / G),
            "diversity/struct_novelty_mean": float(sum(struct_nov) / G),
            "diversity/lc_mean": float(sum(lcs) / G),
            "diversity/lc_degenerate_frac": float(sum(1.0 for lc in lcs if lc < 0.15) / G),
            "diversity/unique_frac": float(len(set(seqs)) / G),
        }
        # Per-metric confidence-term means (e.g. reward/conf_ptm_term_mean) — active only.
        for wk, total in comp_sums.items():
            metrics[f"reward/conf_{wk[2:]}_term_mean"] = float(total / G)
        if tri_seqs is not None:
            metrics["diversity/tri_unique_frac"] = float(len(set(tri_seqs)) / len(tri_seqs))
        # Interface-distribution reward diagnostics (only present when a dist weight is on).
        metrics.update(dist_metrics)
        # Clash + interface-contact geometry diagnostics (only when the weight is on).
        metrics.update(clash_metrics)
        # Backbone chain-break diagnostics (only when the weight is on).
        metrics.update(chainbreak_metrics)
        # Full-atom repack-pool diagnostics: SC shape / all-atom clash / AAR (only the
        # active metrics' keys are present).
        metrics.update(repack_metrics)
        # Protenix fold-consistency SFT diagnostics (only present when struct_sft_mu>0).
        metrics.update(struct_sft_metrics)

        # Per-token structure extras: per-residue clash energy and/or chain-break penalty
        # (G, L) for the per-position structure advantage. None unless a per-token structure
        # reward is on (byte-identical scalar path otherwise). Both signals can be present and
        # are combined additively in _struct_pos_advantage.
        pt_extras: dict | None = None
        _pot_on = need_pot and pot_eres is not None
        if (
            (cfg.per_token_clash and clash_eres is not None)
            or (cfg.per_token_chainbreak and chainbreak_eres is not None)
            or _pot_on
        ):
            pt_extras = {}
            if cfg.per_token_clash and clash_eres is not None:
                pt_extras["clash_eres"] = torch.as_tensor(clash_eres, dtype=torch.float32, device=self.device)  # (G, L)
            if cfg.per_token_chainbreak and chainbreak_eres is not None:
                pt_extras["chainbreak_eres"] = torch.as_tensor(
                    chainbreak_eres, dtype=torch.float32, device=self.device
                )  # (G, L)
            if _pot_on:
                # Per-residue interface potentials (G, L): e_lj (already a penalty, larger=worse),
                # and dsasa/n_hb already negated in _repack_terms_for_group so larger=worse for all.
                for _k in ("lj_eres", "dsasa_eres", "hb_eres"):
                    _v = pot_eres.get(_k)
                    if _v is not None:
                        pt_extras[_k] = torch.as_tensor(_v, dtype=torch.float32, device=self.device)  # (G, L)
        return rewards, metrics, sft_payload, struct_sft_payload, pt_extras

    @staticmethod
    def _advantages(
        rewards: torch.Tensor, eps: float, std_floor: float, normalize: bool = True
    ) -> tuple[torch.Tensor, float]:
        """Group-relative advantages; returns ``(A, std)``.

        With ``normalize=True`` (GRPO) advantages are standardized by the group std;
        with ``normalize=False`` (Dr. GRPO) they are only mean-centered, dropping the
        ``1/std`` reweighting. ``std`` is always the raw group reward std (used for the
        flat-group skip and logging), independent of the normalization mode.
        """
        std = float(rewards.std(unbiased=False))
        centered = rewards - rewards.mean()
        adv = centered / (rewards.std(unbiased=False) + eps) if normalize else centered
        return adv, std

    @staticmethod
    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        """Pearson correlation of two 1-D tensors; 0.0 if either is constant."""
        a = a.detach().float()
        b = b.detach().float()
        a = a - a.mean()
        b = b - b.mean()
        denom = a.norm() * b.norm()
        if float(denom) < 1e-12:
            return 0.0
        return float((a * b).sum() / denom)

    def _pos_norm_adv(self, eres: torch.Tensor, gen_mask_struc: torch.Tensor) -> torch.Tensor:
        """Per-position group-relative, globally-std-normalized per-residue credit.

        Shared core of the per-token structure advantage (clash and chain-break). For a
        per-residue signal ``eres (G, L)`` (energy/penalty; larger = worse), returns the
        UNWEIGHTED per-position advantage

            s = -eres                                          # less penalty → higher adv
            out[g, l] = (s[g, l] - mean_g s[·, l]) / (std + eps)   (masked, 0 off-mask)

        group-mean-centered **per position** and normalized by a SINGLE global std over all
        generated-structure positions (the signal is sparse, so a per-position std would blow
        up quiet positions). The caller scales by the per-signal weight and sums.
        """
        cfg = self.config
        s = -eres.to(self.device).float()  # (G, L)
        m = torch.as_tensor(gen_mask_struc, device=self.device).float()
        if m.dim() == 1:
            m = m.unsqueeze(0).expand_as(s)
        m = (m > 0).float()
        # Per-position group mean over the designs generated at that position.
        denom = m.sum(dim=0).clamp_min(1.0)  # (L,)
        pos_mean = (s * m).sum(dim=0) / denom  # (L,)
        centered = (s - pos_mean.unsqueeze(0)) * m  # (G, L), zero off-mask
        # Single global std over the masked entries.
        total = m.sum().clamp_min(1.0)
        std = ((centered.pow(2) * m).sum() / total).sqrt()
        return (centered / (std + cfg.adv_eps)) * m  # (G, L), zero off-mask

    def _struct_pos_advantage(
        self,
        clash_eres: torch.Tensor | None,
        design_adv: torch.Tensor,
        gen_mask_struc: torch.Tensor,
        *,
        chainbreak_eres: torch.Tensor | None = None,
        lj_eres: torch.Tensor | None = None,
        dsasa_eres: torch.Tensor | None = None,
        hb_eres: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-position structure-track advantage for the per-token structure arm(s).

        Combines the usual group-relative design advantage (broadcast over positions) with
        per-residue credit from the per-binder-residue backbone clash energy, chain-break
        penalty, and/or all-atom interface potentials (each independently centered/normalized
        by :meth:`_pos_norm_adv`, scaled by its weight, and summed):

            A_clash[g, l]      = w_pt_clash      · posnorm(-clash_eres)[g, l]
            A_chainbreak[g, l] = w_pt_chainbreak · posnorm(-chainbreak_eres)[g, l]
            A_lj[g, l]         = w_pt_lj         · posnorm(-lj_eres)[g, l]
            A_dsasa[g, l]      = w_pt_dsasa      · posnorm(-dsasa_eres)[g, l]
            A_hb[g, l]         = w_pt_hb         · posnorm(-hb_eres)[g, l]
            A_struct[g, l]     = design_adv[g] + Σ A_*[g, l]

        Any per-residue signal may be ``None`` (its arm off); with all ``None`` the result is
        just the broadcast design advantage. Every potential ``eres`` is passed to
        :meth:`_pos_norm_adv` in "larger = worse" convention (``lj_eres`` is already a penalty;
        ``dsasa_eres``/``hb_eres`` were negated at collection), so ``posnorm`` gives higher
        advantage to residues with less penalty / more buried area / more H-bonds. Credit is
        zeroed outside the generated-structure positions; those positions carry no PPO term.

        Parameters
        ----------
        clash_eres : torch.Tensor or None
            ``(G, L)`` per-residue clash energy (0 off the valid binder positions), or ``None``
            when the per-token clash arm is off.
        design_adv : torch.Tensor
            ``(G,)`` group-relative scalar advantage.
        gen_mask_struc : torch.Tensor
            Structure-track generation mask, ``(L,)`` or ``(G, L)`` (bool / numeric).
        chainbreak_eres : torch.Tensor or None, keyword-only
            ``(G, L)`` per-residue chain-break penalty, or ``None`` when the per-token
            chain-break arm is off.
        lj_eres, dsasa_eres, hb_eres : torch.Tensor or None, keyword-only
            ``(G, L)`` per-residue interface-potential penalties (bounded LJ energy, negated
            buried ΔSASA, negated interface H-bond count), or ``None`` when the per-token
            potential arm is off.

        Returns
        -------
        torch.Tensor
            ``(G, L)`` per-position structure advantage.
        """
        cfg = self.config
        a = design_adv.to(self.device).float().unsqueeze(1)  # (G, 1) → broadcasts over L
        if clash_eres is not None:
            a = a + cfg.w_pt_clash * self._pos_norm_adv(clash_eres, gen_mask_struc)
        if chainbreak_eres is not None:
            a = a + cfg.w_pt_chainbreak * self._pos_norm_adv(chainbreak_eres, gen_mask_struc)
        if lj_eres is not None:
            a = a + cfg.w_pt_lj * self._pos_norm_adv(lj_eres, gen_mask_struc)
        if dsasa_eres is not None:
            a = a + cfg.w_pt_dsasa * self._pos_norm_adv(dsasa_eres, gen_mask_struc)
        if hb_eres is not None:
            a = a + cfg.w_pt_hb * self._pos_norm_adv(hb_eres, gen_mask_struc)
        return a  # (G, L)

    # ---------------------------------------------------------------- GRPO update
    def _rollout_and_submit(self, spec: TargetSpec, step_idx: int) -> dict:
        """Phase 1 of a pipelined optimizer step: roll out one group and SUBMIT its repack.

        Samples the binder length, builds the complex, rolls out the group (no grad) and
        decodes its sequences; when the combined full-atom repack path is active it also
        decodes the shared backbone, builds the design packets and SUBMITS the repack
        round-trip to the a10g pool WITHOUT blocking
        (:meth:`~lobster.rl_training.rewards.ShapeRewardClient.submit_group`). The returned
        *pending* dict is consumed by :meth:`_collect_and_advantage`. Deferring the blocking
        collect lets the pool score earlier targets while later targets still roll out on the
        b200 — the single wall-clock win of lever A.

        Determinism (byte-identical to :meth:`_rollout_and_advantage`): the only RNG this
        draws are ``self._rng`` via :meth:`_sample_binder_length` and the torch sampling RNG
        inside ``rollout_with_logprobs`` — in the SAME per-target order as the un-pipelined
        loop (``len₀, roll₀, len₁, roll₁, …``), because the deferred reward/collect path
        draws no RNG at all. The step-subset draw (``_sample_step_subset``) still happens later
        in :meth:`_ppo_update`, after every packet is built, exactly as before.
        """
        cfg = self.config
        binder_length = self._sample_binder_length(spec)
        comp = self._target_static(spec, binder_length)

        # 1. Rollout (no grad — sampling only; grad enters via log-prob recompute).
        with torch.no_grad():
            trajectory = self.model.rollout_with_logprobs(**self._build_gen_kwargs(comp, cfg.group_size))
        seqs = self._decode_binder_seqs(trajectory, comp)
        tri_seqs = self._decode_binder_tri(trajectory, comp)

        # Mirror the reward-routing decision in _compute_rewards so we submit the SAME repack
        # round-trip it would otherwise run inline. SC-only routes through
        # _shape_terms_for_group (not split for pipelining) — only the combined
        # _repack_terms_for_group path is pre-submitted here.
        need_dist = cfg.w_aa_dist > 0 or cfg.w_3di_dist > 0 or cfg.log_dist_diagnostic
        need_clash = cfg.w_clash_contact > 0
        need_chainbreak = cfg.w_chainbreak > 0
        need_shape = cfg.w_shape > 0
        # Mirror _compute_rewards: log_sc_clash_diagnostic forces "clash" into want at weight 0.
        need_sc_clash = cfg.w_sc_clash > 0 or getattr(cfg, "log_sc_clash_diagnostic", False)
        need_sft = getattr(cfg, "sft_mu", 0.0) > 0
        need_aar = cfg.w_aar > 0 or need_sft
        need_pot = getattr(cfg, "per_token_pot", False)
        need_repack = need_shape or need_sc_clash or need_aar or need_pot
        repack_branch = need_repack and not (need_shape and not (need_sc_clash or need_aar or need_pot))

        gen_bb = None
        handle = None
        if need_dist or need_clash or need_chainbreak or need_repack:
            # Decode the shared backbone once here; reused by the phase-2 reward call (gen_bb=)
            # so it is not re-decoded — bit-identical to decoding inside _compute_rewards.
            gen_bb = self._decode_backbone_coords(trajectory, comp).numpy()  # (G, L, 3, 3)
        if repack_branch:
            want = tuple(
                m
                for m, on in (("sc", need_shape), ("clash", need_sc_clash), ("aar", need_aar), ("pot", need_pot))
                if on
            )
            designs = self._build_repack_designs(trajectory, comp, seqs, gen_bb)
            handle = self._shape_client.submit_group(spec.target_id, designs, want=want, return_seq=need_sft)

        return {
            "spec": spec,
            "step_idx": step_idx,
            "comp": comp,
            "trajectory": trajectory,
            "seqs": seqs,
            "tri_seqs": tri_seqs,
            "binder_length": binder_length,
            "gen_bb": gen_bb,
            "handle": handle,
        }

    def _collect_and_advantage(self, pending: dict, step_idx: int) -> dict:
        """Phase 2 of a pipelined optimizer step: collect the repack + finish the reward.

        Consumes a *pending* dict from :meth:`_rollout_and_submit`: collects the submitted
        repack results (blocking on ``collect_group`` only when a round-trip was submitted),
        then computes rewards, advantages and old log-probs and returns a *packet* dict in
        the same shape as :meth:`_rollout_and_advantage`. A packet always carries ``spec``,
        ``metrics`` and a ``flat`` flag; when ``flat`` is ``False`` it also carries the
        tensors the PPO inner loop needs (``trajectory``, ``advantages``,
        ``old_lp_per_step``, ``n_steps``). A flat group (reward std below the floor) is
        returned with ``flat=True`` and no update tensors so the caller can drop it.
        """
        cfg = self.config
        spec = pending["spec"]
        comp = pending["comp"]
        trajectory = pending["trajectory"]
        seqs = pending["seqs"]
        tri_seqs = pending["tri_seqs"]
        binder_length = pending["binder_length"]
        gen_bb = pending["gen_bb"]
        handle = pending["handle"]

        # Collect the (already-submitted) repack results; None ⇒ no combined repack path (the
        # SC-only / no-repack arms map their results inline in _compute_rewards as before).
        precomputed_repack = self._shape_client.collect_group(handle) if handle is not None else None

        # 2. Reward (four-term) + 3. advantages.
        rewards, metrics, sft_payload, struct_sft_payload, pt_extras = self._compute_rewards(
            spec.target_id,
            seqs,
            tri_seqs,
            trajectory,
            comp,
            gen_bb=gen_bb,
            precomputed_repack=precomputed_repack,
        )
        metrics["rollout/binder_length"] = float(binder_length)
        advantages, std = self._advantages(rewards, cfg.adv_eps, cfg.adv_std_floor, cfg.normalize_advantage)

        if std < cfg.adv_std_floor:
            logger.info("step %d target %s: group std %.4g < floor, skipping update", step_idx, spec.target_id, std)
            metrics["update/skipped_flat_group"] = 1.0
            return {"spec": spec, "metrics": metrics, "flat": True}

        n_steps = len(trajectory["steps"])
        # When any per-token structure arm is active (clash and/or chain-break), the STRUCTURE
        # track carries a per-position advantage (design-level advantage + per-residue clash
        # and/or chain-break credit) and is dropped from the design-level PPO; the remaining
        # (sequence/tri) tracks keep the scalar advantage.
        pt_active = cfg.per_token_clash or cfg.per_token_chainbreak or cfg.per_token_pot
        design_tracks = tuple(t for t in cfg.tracks if t not in cfg.pt_clash_tracks) if pt_active else cfg.tracks
        # 4a. OLD (behaviour) policy per-step log-prob, snapshotted under no_grad so the
        # importance ratio is against fixed weights across all inner updates. Log-prob is
        # additive over steps, so a subset's old log-prob is the sum of its per-step rows.
        # Prefer the INLINE captures (exact biased logits the sampler drew from, zero extra
        # forwards, faithful by construction); fall back to a post-hoc recompute for rollouts
        # without inline log-prob. With inline old_lp, the first inner iteration's ratio must
        # be ~1 (`ppo/ratio_init`) — a live check that the new_lp recompute mirrors the sampler.
        with torch.no_grad():
            if cfg.capture_old_lp_inline:
                old_lp_per_step = self.model.captured_logprob_per_step(trajectory, tracks=design_tracks)
            else:
                old_lp_per_step = torch.stack(
                    [
                        self.model.logprob_over_trajectory(trajectory, tracks=design_tracks, step_indices=[i])
                        for i in range(n_steps)
                    ]
                )  # (n_steps, G)

        # 4b. Per-token clash: build the per-position structure advantage (G, L) and snapshot the
        # per-position structure OLD log-prob per step (recompute — inline captures are summed
        # over positions so cannot serve, and the recompute gives struct_ratio_init == 1 exactly).
        struct_pos_advantage = None
        old_lp_struct_pos_per_step = None
        if pt_active:
            gen_mask_struc = trajectory["static"]["gen_mask_struc"]
            clash_eres = pt_extras.get("clash_eres") if pt_extras is not None else None
            chainbreak_eres = pt_extras.get("chainbreak_eres") if pt_extras is not None else None
            lj_eres = pt_extras.get("lj_eres") if pt_extras is not None else None
            dsasa_eres = pt_extras.get("dsasa_eres") if pt_extras is not None else None
            hb_eres = pt_extras.get("hb_eres") if pt_extras is not None else None
            struct_pos_advantage = self._struct_pos_advantage(
                clash_eres,
                advantages,
                gen_mask_struc,
                chainbreak_eres=chainbreak_eres,
                lj_eres=lj_eres,
                dsasa_eres=dsasa_eres,
                hb_eres=hb_eres,
            )  # (G, L)
            with torch.no_grad():
                old_lp_struct_pos_per_step = self.model.struct_pos_logprob_per_step(
                    trajectory, struct_tracks=cfg.pt_clash_tracks
                )  # (n_steps, G, L)
            metrics["ptclash/adv_abs_mean"] = float(struct_pos_advantage.abs().mean())
            metrics["ptclash/adv_max_abs"] = float(struct_pos_advantage.abs().max())

        metrics["advantage/std"] = std
        metrics["advantage/abs_mean"] = float(advantages.abs().mean())
        metrics["advantage/max_abs"] = float(advantages.abs().max())
        metrics["rollout/n_steps"] = float(n_steps)
        # CHORD SFT: convert the per-token expert targets to tensors once, on the model device,
        # so the PPO inner loop can call sequence_sft_loss without per-iteration host copies.
        sft_target_ids = sft_supervise_mask = sft_soft_targets = None
        if sft_payload is not None:
            sft_target_ids = torch.as_tensor(sft_payload["target_ids"], dtype=torch.long, device=self.device)
            sft_supervise_mask = torch.as_tensor(sft_payload["supervise_mask"], dtype=torch.bool, device=self.device)
            if "soft_targets" in sft_payload:
                sft_soft_targets = torch.as_tensor(sft_payload["soft_targets"], dtype=torch.float32, device=self.device)

        # Protenix fold-consistency SFT: whole-complex structure (s*) + 3Di (τ*) expert targets,
        # scattered onto the (G, L) layout (see ``_struct_sft_payload_for_group``). Moved to device
        # once here so the PPO inner loop calls ``structure_sft_loss`` without per-iteration copies.
        struct_sft_target_ids = struct_sft_tri_target_ids = struct_sft_supervise_mask = None
        if struct_sft_payload is not None:
            struct_sft_target_ids = torch.as_tensor(
                struct_sft_payload["struct_target_ids"], dtype=torch.long, device=self.device
            )
            struct_sft_tri_target_ids = torch.as_tensor(
                struct_sft_payload["tri_target_ids"], dtype=torch.long, device=self.device
            )
            struct_sft_supervise_mask = torch.as_tensor(
                struct_sft_payload["supervise_mask"], dtype=torch.bool, device=self.device
            )

        return {
            "spec": spec,
            "metrics": metrics,
            "flat": False,
            "trajectory": trajectory,
            "advantages": advantages,
            "old_lp_per_step": old_lp_per_step,
            "n_steps": n_steps,
            "sft_target_ids": sft_target_ids,
            "sft_supervise_mask": sft_supervise_mask,
            "sft_soft_targets": sft_soft_targets,
            "struct_sft_target_ids": struct_sft_target_ids,
            "struct_sft_tri_target_ids": struct_sft_tri_target_ids,
            "struct_sft_supervise_mask": struct_sft_supervise_mask,
            "design_tracks": design_tracks,
            "struct_pos_advantage": struct_pos_advantage,
            "old_lp_struct_pos_per_step": old_lp_struct_pos_per_step,
        }

    def _rollout_and_advantage(self, spec: TargetSpec, step_idx: int) -> dict:
        """Single-target rollout+reward+advantage (submit then collect, no overlap).

        Retained as a thin wrapper over :meth:`_rollout_and_submit` +
        :meth:`_collect_and_advantage` for single-target callers and tests. The two-phase
        ``train`` loop calls the halves directly so the repack wait overlaps later rollouts.
        Byte-identical to the pre-pipelining implementation.
        """
        return self._collect_and_advantage(self._rollout_and_submit(spec, step_idx), step_idx)

    def _chord_mu(self, step: int) -> float:
        """Effective CHORD SFT blend weight ``μ`` at optimizer ``step``.

        Returns ``0.0`` when SFT is off (``sft_mu == 0``). With ``sft_mu_schedule is None``
        (default) ``μ`` is the constant ``sft_mu`` (CHORD finding: with φ on, a small fixed μ
        matches a decayed schedule). ``"linear_decay"`` anneals ``μ`` from ``sft_mu`` down to
        ``0`` linearly over ``num_steps`` (a DAgger-style hand-off to pure RL late in training).
        """
        cfg = self.config
        mu0 = float(getattr(cfg, "sft_mu", 0.0))
        if mu0 <= 0.0:
            return 0.0
        sched = getattr(cfg, "sft_mu_schedule", None)
        if sched is None:
            return mu0
        if sched == "linear_decay":
            frac = min(max(step / max(1, cfg.num_steps - 1), 0.0), 1.0)
            return mu0 * (1.0 - frac)
        raise ValueError(f"unknown sft_mu_schedule: {sched!r}")

    def _struct_chord_mu(self, step: int) -> float:
        """Effective Protenix fold-consistency SFT blend weight ``μ_struct`` at optimizer ``step``.

        Mirrors :meth:`_chord_mu` but reads ``struct_sft_mu`` / ``struct_sft_mu_schedule``.
        Returns ``0.0`` when the structure SFT is off. Unlike the sequence CHORD ``μ`` — which
        trades off *against* the GRPO term — ``μ_struct`` is an **additive** auxiliary weight
        (see the blend in :meth:`_ppo_update`), so it composes with both pure GRPO and sequence
        CHORD without rescaling either.
        """
        cfg = self.config
        mu0 = float(getattr(cfg, "struct_sft_mu", 0.0))
        if mu0 <= 0.0:
            return 0.0
        sched = getattr(cfg, "struct_sft_mu_schedule", None)
        if sched is None:
            return mu0
        if sched == "linear_decay":
            frac = min(max(step / max(1, cfg.num_steps - 1), 0.0), 1.0)
            return mu0 * (1.0 - frac)
        raise ValueError(f"unknown struct_sft_mu_schedule: {sched!r}")

    def _ppo_update(self, packets: list[dict], step: int = 0) -> dict:
        """Run ``mu`` PPO inner updates over one or more rollout packets.

        With a single (non-flat) packet this is byte-identical to the legacy per-target
        update: one ``zero_grad``/``backward``/``step`` per inner iteration. With multiple
        packets the per-target policy-gradient losses are **averaged** — each ``backward()``
        is scaled by ``1/len(live)`` and gradients accumulate across targets before a single
        ``optimizer.step()`` per inner iteration. This is the cross-prompt gradient averaging
        TRL uses to stabilize Dr.GRPO over multiple environments: the optimizer sees the mean
        gradient over ``accum_targets`` targets instead of one noisy single-target estimate.

        When CHORD SFT distillation is active (``sft_mu > 0``) each target's loss becomes the
        convex blend ``(1 - μ)·pg_loss + μ·sft_ce`` (plus ``β·KL`` outside the blend), where
        ``sft_ce`` is the φ-weighted supervised CE toward the LigandMPNN-designed binder
        sequence (:meth:`~lobster.model.leflur...sequence_sft_loss`) evaluated on the same step
        subset. ``μ`` (:meth:`_chord_mu`) may be constant or annealed; ``μ==0`` reduces to the
        byte-identical legacy update.

        Returns the aggregated ``ppo/*`` (and ``sft/*``) update metrics (empty if every packet
        was flat). Per-target reward/rollout metrics stay on each packet for the caller to log.
        """
        cfg = self.config
        live = [p for p in packets if not p["flat"]]
        if not live:
            return {}
        n_live = len(live)

        ratios, clipfracs, kls, pg_losses = [], [], [], []
        dlp_means, dlp_corrs = [], []
        sft_losses: list[float] = []
        struct_sft_losses: list[float] = []  # Protenix fold-consistency SFT (structure + 3Di CE)
        struct_ratios: list[float] = []  # per-token clash: masked-mean per-position structure ratio
        struct_pg_losses: list[float] = []
        grad_norm = torch.tensor(0.0)
        # CHORD SFT blend weight for this optimizer step (0.0 => term off, legacy path).
        chord_mu = self._chord_mu(step)
        # Protenix fold-consistency SFT weight (additive auxiliary; 0.0 => term off).
        struct_chord_mu = self._struct_chord_mu(step)
        for _ in range(cfg.mu):
            # One accumulation window per inner iteration: zero grads, backward each target's
            # (mean-over-group) loss scaled by 1/n_live, then a single optimizer step.
            self.optimizer.zero_grad(set_to_none=True)
            for p in live:
                trajectory = p["trajectory"]
                advantages = p["advantages"]
                old_lp_per_step = p["old_lp_per_step"]
                n_steps = p["n_steps"]

                subset = self._sample_step_subset(n_steps)
                # Per-token clash: the STRUCTURE track uses a per-position advantage/PPO term; the
                # design-level PPO runs on the remaining tracks (seq/tri). The fused/plain forward
                # additionally returns the per-position structure log-prob (one shared forward).
                pt_on = (cfg.per_token_clash or cfg.per_token_chainbreak or cfg.per_token_pot) and p.get(
                    "struct_pos_advantage"
                ) is not None
                design_tracks = p.get("design_tracks", cfg.tracks)
                pos_tracks = cfg.pt_clash_tracks if pt_on else ()
                # CHORD SFT-distillation term: convex-blend a φ-weighted supervised CE toward the
                # LigandMPNN-designed binder sequence into the policy-gradient loss,
                #   L = (1 - μ)·L_GRPO + μ·L_SFT-φ   (+ β·KL, added outside the blend)
                # μ==0 or missing payload => byte-identical legacy update (no extra forward).
                use_sft = chord_mu > 0.0 and p.get("sft_target_ids") is not None
                use_struct_sft = struct_chord_mu > 0.0 and p.get("struct_sft_supervise_mask") is not None
                sft_ce = None
                struct_sft_ce = None
                new_lp_struct_pos = None
                # The GRPO policy log-prob (and, when per-token clash is on, the per-position
                # structure log-prob) always come from logprob_over_trajectory, which checkpoints
                # each step's forward on its own.
                out = self.model.logprob_over_trajectory(
                    trajectory,
                    tracks=design_tracks,
                    step_indices=subset,
                    grad_checkpoint=cfg.grad_checkpoint,
                    per_position_tracks=pos_tracks,
                )
                if pt_on:
                    new_lp, new_lp_struct_pos = out
                else:
                    new_lp = out
                if use_sft:
                    # Optional reward gate: distil only the group's above-average backbones.
                    row_mask = (
                        (advantages > 0).detach() if getattr(cfg, "sft_reward_gate", None) == "positive_adv" else None
                    )
                    # CHORD SFT-distillation CE, computed as a SEPARATE gradient-checkpointed pass
                    # from the GRPO log-prob above — deliberately NOT fused into one checkpoint unit.
                    # The expert-context SFT forward and the on-policy log-prob forward need different
                    # sequence inputs (expert vs policy tokens), so the old lever-B fusion ran BOTH
                    # forwards inside a single _step_pg_and_sft checkpoint segment; during backward
                    # recompute their activations (plus the CFG second forward) were co-resident, so
                    # peak memory = SUM of the two forwards and the full config OOM'd. Running them as
                    # two independently-checkpointed passes recomputes each at a different point in the
                    # backward, so peak = MAX not SUM. Numerically identical to the fused path
                    # (test_fused_sft_matches_separate_with_expert_context).
                    sft_ce = self.model.sequence_sft_loss(
                        trajectory,
                        p["sft_target_ids"],
                        p["sft_supervise_mask"],
                        step_indices=subset,
                        label=cfg.sft_label,
                        soft_targets=p.get("sft_soft_targets"),
                        temperature=cfg.sft_temperature,
                        masked_only=cfg.sft_masked_only,
                        use_phi=cfg.sft_use_phi,
                        row_mask=row_mask,
                        grad_checkpoint=cfg.grad_checkpoint,
                    )
                if use_struct_sft:
                    # Protenix fold-consistency SFT: distil the policy's whole-complex structure (s*)
                    # and 3Di (τ*) endpoints toward the tokens derived from the Protenix fold of the
                    # policy's OWN sequence — the structural dual of the sequence CHORD term above.
                    # Same optional positive-advantage reward gate.
                    struct_row_mask = (
                        (advantages > 0).detach()
                        if getattr(cfg, "struct_sft_reward_gate", None) == "positive_adv"
                        else None
                    )
                    struct_sft_ce = structure_sft_loss(
                        self.model,
                        trajectory,
                        p["struct_sft_target_ids"],
                        p["struct_sft_tri_target_ids"],
                        p["struct_sft_supervise_mask"],
                        w_struct=cfg.struct_sft_w_struct,
                        w_tri=cfg.struct_sft_w_tri,
                        step_indices=subset,
                        masked_only=cfg.struct_sft_masked_only,
                        use_phi=cfg.struct_sft_use_phi,
                        row_mask=struct_row_mask,
                        grad_checkpoint=cfg.grad_checkpoint,
                    )
                old_lp = old_lp_per_step[subset].sum(dim=0)
                # KL costs a full extra grad-forward over the subset (self + frozen ref). When
                # beta == 0 the term is multiplied by zero and contributes nothing to the loss, so
                # skip it entirely — this roughly halves the backward-graph memory (letting a larger
                # steps_per_update fit) and the compute. Log kl_mean = 0 in that case.
                if cfg.beta > 0:
                    kl = self.model.kl_over_trajectory(
                        trajectory, self.ref_module, tracks=cfg.tracks, step_indices=subset
                    )
                    kl_mean = kl.mean()
                else:
                    kl_mean = torch.zeros((), device=new_lp.device)

                ratio = torch.exp(new_lp - old_lp)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * advantages
                pg_loss = -torch.min(unclipped, clipped).mean()

                # Per-token clash: add the per-position structure PPO term. Uses the same clip on a
                # per-position ratio against the per-position structure advantage A_struct (which
                # already folds in the design-level advantage), masked-mean over generated
                # structure positions. old_lp_struct_pos is a no-grad recompute, so at the first
                # inner iteration ratio_pos == 1 exactly (struct_ratio_init check).
                if pt_on:
                    old_lp_struct_pos = p["old_lp_struct_pos_per_step"][subset].sum(dim=0)  # (G, L)
                    a_struct = p["struct_pos_advantage"]  # (G, L)
                    ratio_pos = torch.exp(new_lp_struct_pos - old_lp_struct_pos)
                    unclipped_pos = ratio_pos * a_struct
                    clipped_pos = torch.clamp(ratio_pos, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * a_struct
                    gm = torch.as_tensor(
                        trajectory["static"]["gen_mask_struc"], device=new_lp_struct_pos.device
                    ).float()
                    if gm.dim() == 1:
                        gm = gm.unsqueeze(0).expand_as(ratio_pos)
                    gm = (gm > 0).float()
                    gm_sum = gm.sum().clamp_min(1.0)
                    pg_loss_struct = -(torch.min(unclipped_pos, clipped_pos) * gm).sum() / gm_sum
                    pg_loss = pg_loss + pg_loss_struct
                    struct_pg_losses.append(float(pg_loss_struct.detach()))
                    struct_ratios.append(float(((ratio_pos * gm).sum() / gm_sum).detach()))

                # Blend the CHORD SFT CE (computed above in the fused forward) into the loss.
                if sft_ce is not None:
                    loss = (1.0 - chord_mu) * pg_loss + chord_mu * sft_ce + cfg.beta * kl_mean
                    sft_losses.append(float(sft_ce.detach()))
                else:
                    loss = pg_loss + cfg.beta * kl_mean
                # Protenix fold-consistency SFT is an ADDITIVE auxiliary term (does not rescale the
                # GRPO/sequence-SFT blend), so it composes with both pure GRPO and sequence CHORD.
                if struct_sft_ce is not None:
                    loss = loss + struct_chord_mu * struct_sft_ce
                    struct_sft_losses.append(float(struct_sft_ce.detach()))
                # Average across targets (n_live==1 => /1 is exact, byte-identical to legacy).
                (loss / n_live).backward()

                ratios.append(float(ratio.mean()))
                clipfracs.append(float(((ratio - 1.0).abs() > cfg.eps_clip).float().mean()))
                kls.append(float(kl_mean))
                pg_losses.append(float(pg_loss))
                # Common-mode vs advantage-differential diagnostic. dlp = new_lp - old_lp is the
                # per-design log-prob shift this update produced. Its GROUP MEAN is the common-mode
                # drift (all designs moving together, orthogonal to the mean-centered advantage); its
                # correlation with the advantage is the useful, differential signal. Flat reward with
                # large |dlp_mean| but ~0 dlp_adv_corr = the update is pushing the whole group in
                # lockstep instead of separating winners from losers (the M6-M12 failure mode).
                dlp = (new_lp - old_lp).detach()
                dlp_means.append(float(dlp.mean()))
                dlp_corrs.append(self._pearson(advantages, dlp))

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), cfg.grad_clip)
                if cfg.grad_clip > 0
                else torch.tensor(0.0)
            )
            self.optimizer.step()

        return {
            # First-recorded ratio: with inline old_lp this is a consistency check on the
            # new_lp recompute (should be ~1.0). Divergence flags a sampler/recompute mismatch.
            "ppo/ratio_init": ratios[0],
            "ppo/ratio_mean": sum(ratios) / len(ratios),
            "ppo/clip_frac": sum(clipfracs) / len(clipfracs),
            "ppo/kl_mean": sum(kls) / len(kls),
            "ppo/kl_term": cfg.beta * (sum(kls) / len(kls)),
            "ppo/pg_loss": sum(pg_losses) / len(pg_losses),
            "ppo/grad_norm": float(grad_norm),
            # CHORD SFT: blend weight this step + mean supervised CE (only when SFT is active).
            "sft/mu": chord_mu,
            "sft/ce_loss": (sum(sft_losses) / len(sft_losses)) if sft_losses else 0.0,
            # Protenix fold-consistency SFT: additive weight + mean (structure+3Di) CE.
            "struct_sft/mu": struct_chord_mu,
            "struct_sft/ce_loss": (sum(struct_sft_losses) / len(struct_sft_losses)) if struct_sft_losses else 0.0,
            # Diagnostics: common-mode drift and advantage-differential faithfulness.
            "ppo/dlp_mean": sum(dlp_means) / len(dlp_means),
            "ppo/dlp_adv_corr": sum(dlp_corrs) / len(dlp_corrs),
            "update/n_targets": float(n_live),
            # Per-token clash (only populated when per_token_clash is active). struct_ratio_init
            # ~1.0 confirms the per-position new/old log-prob recompute matches.
            **(
                {
                    "ppo/struct_ratio_init": struct_ratios[0],
                    "ppo/struct_ratio_mean": sum(struct_ratios) / len(struct_ratios),
                    "ppo/pg_loss_struct": sum(struct_pg_losses) / len(struct_pg_losses),
                }
                if struct_ratios
                else {}
            ),
        }

    def _sample_step_subset(self, n_steps: int) -> list[int]:
        k = self.config.steps_per_update
        if k <= 0 or k >= n_steps:
            return list(range(n_steps))
        return sorted(self._rng.sample(range(n_steps), k))

    def _iter_targets(self):
        """Infinite target schedule: round-robin, optionally reshuffled per epoch.

        With ``shuffle_targets=False`` (default) the order is a fixed round-robin over
        ``self.targets`` — for ``accum_targets=1`` the ``step``-th draw is exactly
        ``self.targets[step % len(self.targets)]`` (byte-identical to the legacy loop). With
        ``shuffle_targets=True`` the per-epoch order is reshuffled with ``self._sched_rng``
        (a dedicated RNG so the binder-length / step-subset draw order is never perturbed).
        """
        n = len(self.targets)
        order = list(range(n))
        while True:
            if self.config.shuffle_targets:
                self._sched_rng.shuffle(order)
            for i in order:
                yield self.targets[i]

    @staticmethod
    def _merge_step_metrics(packets: list[dict], update_metrics: dict) -> dict:
        """Combine per-target rollout metrics + shared PPO update metrics for logging.

        Single packet: passthrough (byte-identical keys to the legacy step). Multiple
        packets: numeric rollout/reward metrics are averaged across the batch's targets;
        the PPO update metrics (already aggregated over targets×mu) are attached once.
        """
        if len(packets) == 1:
            m = dict(packets[0]["metrics"])
            m.update(update_metrics)
            m["target"] = packets[0]["spec"].target_id
            return m
        # Multi-target: mean of every numeric per-target metric across the batch.
        keys: set[str] = set()
        for p in packets:
            keys.update(k for k, v in p["metrics"].items() if isinstance(v, (int, float)))
        merged: dict = {}
        for k in keys:
            vals = [p["metrics"][k] for p in packets if k in p["metrics"]]
            merged[k] = sum(vals) / len(vals)
        merged.update(update_metrics)
        merged["target"] = ",".join(p["spec"].target_id for p in packets)
        merged["update/batch_targets"] = float(len(packets))
        return merged

    # ------------------------------------------------------------------------ run
    def train(self) -> None:
        """Run the GRPO optimization loop for ``config.num_steps`` optimizer steps.

        Each step rolls out ``config.accum_targets`` targets and applies one accumulated
        optimizer update over them (see :meth:`_ppo_update`). ``num_steps`` counts optimizer
        updates, so ``accum_targets=1`` reproduces the legacy per-target loop exactly.
        """
        cfg = self.config
        logger.info(
            "Starting GRPO: %d steps, group_size=%d, rollout_nsteps=%d, mu=%d, beta=%g, lr=%g, "
            "accum_targets=%d, shuffle_targets=%s, targets=%s",
            cfg.num_steps,
            cfg.group_size,
            cfg.rollout_nsteps,
            cfg.mu,
            cfg.beta,
            cfg.lr,
            cfg.accum_targets,
            cfg.shuffle_targets,
            [t.target_id for t in self.targets],
        )
        # eval() — grad still flows (encoder params require grad); this only disables dropout /
        # BN-stat updates so the sampling forward and the log-prob recompute are deterministic and
        # bit-identical (the importance ratio depends on it). No train-only layers on the gen path.
        self.model.eval()
        sched = self._iter_targets()
        for step in range(cfg.num_steps):
            specs = [next(sched) for _ in range(max(1, cfg.accum_targets))]
            # Lever A — pipeline rollout ⟂ repack-wait: submit ALL targets' repack round-trips
            # up front (phase 1, non-blocking), so the a10g pool scores the earlier targets
            # while the b200 rolls out the later ones; then collect + finish reward/advantage
            # (phase 2). Byte-identical to the sequential path (same RNG draw order); only the
            # blocking collect is deferred so it overlaps subsequent rollouts.
            pendings = [self._rollout_and_submit(spec, step) for spec in specs]
            packets = [self._collect_and_advantage(pending, step) for pending in pendings]
            update_metrics = self._ppo_update(packets, step)
            metrics = self._merge_step_metrics(packets, update_metrics)
            metrics["step"] = step
            if step % cfg.log_every == 0:
                logger.info(
                    "step %d [%s] reward/mean=%.4f pass=%.3f ptm=%.3f iptm=%.3f "
                    "sctm_b=%.3f sctm_c=%.3f kl=%.4g ratio=%.3f "
                    "sft_mu=%.3g sft_ce=%.4g sft_tok=%.1f gnorm=%.4g",
                    step,
                    metrics.get("target", ""),
                    metrics.get("reward/mean", 0.0),
                    metrics.get("conf/pass_rate", 0.0),
                    metrics.get("conf/ptm_mean", 0.0),
                    metrics.get("conf/abag_iptm_mean", 0.0),
                    metrics.get("struct/sctm_binder", 0.0),
                    metrics.get("struct/sctm_complex", 0.0),
                    metrics.get("ppo/kl_mean", 0.0),
                    metrics.get("ppo/ratio_mean", 1.0),
                    metrics.get("sft/mu", 0.0),
                    metrics.get("sft/ce_loss", 0.0),
                    metrics.get("sft/supervised_tokens_mean", 0.0),
                    metrics.get("ppo/grad_norm", 0.0),
                )
                # Per-token clash guardrails (only when the arm is active): struct_ratio_init
                # must be ~1.0 (per-position old-lp recompute faithful) and the per-position
                # advantage magnitudes should be finite/non-degenerate.
                if "ppo/struct_ratio_init" in metrics:
                    logger.info(
                        "step %d [%s]   ptclash: struct_ratio_init=%.4f struct_ratio_mean=%.4f "
                        "adv_abs_mean=%.4g adv_max_abs=%.4g",
                        step,
                        metrics.get("target", ""),
                        metrics.get("ppo/struct_ratio_init", 1.0),
                        metrics.get("ppo/struct_ratio_mean", 1.0),
                        metrics.get("ptclash/adv_abs_mean", 0.0),
                        metrics.get("ptclash/adv_max_abs", 0.0),
                    )
            if self.wandb_run is not None:
                self.wandb_run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
            if cfg.ckpt_dir and cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
                self._save_checkpoint(step + 1)
        if cfg.ckpt_dir:
            self._save_checkpoint(cfg.num_steps)

    def _save_checkpoint(self, step: int) -> None:
        out = Path(self.config.ckpt_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"grpo_step_{step}.ckpt"
        # Lightning-style save so the result reloads via load_from_checkpoint.
        torch.save({"state_dict": self.model.state_dict(), "grpo_step": step}, path)
        logger.info("Saved GRPO checkpoint: %s", path)
