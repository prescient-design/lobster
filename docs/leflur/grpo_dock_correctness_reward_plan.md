# Plan: relabeled Protenix-structure CHORD SFT + a dock-correctness GRPO reward

Two coupled changes to the third reward set (Protenix fold-consistency SFT,
`[[protenix-foldconsistency-sft-reward]]`). They share one new primitive — an interface/epitope
derived from the Protenix-folded complex `X*` — so implement that once and reuse it.

## Motivation

Generation is conditioned on a specified epitope `E` (bidirectional hotspot conditioning +
epitope CFG, the docking lever). But Protenix routinely docks the policy sequence at a **different**
site `E' ≠ E`: the proposal already measures monomer fold reproduced at ~2.4 Å yet antigen-aligned
complex L-RMSD ~29.5 Å, and notes "the high ipTM is Protenix's confidence in *an* interface, not the
generated one" (`grpo_two_reward_sets_proposal.tex:570-578`).

The structure-SFT forward currently conditions on the frozen, originally-specified `E`
(`static["conditioning_tensor"]`, passed unchanged into `_structure_endpoint_logits`,
`src/lobster/rl_training/_structure_sft.py:60`; built once per target in `_target_static`,
`_leflur_grpo_trainer.py:684-730`). Distilling `X*` (which realizes `E'`) while conditioning on `E`
teaches an **inconsistent conditional** `p(structure@E' | cond=E)` — the structural dual of the
sequence-context chimera fixed in `[[chord-sft-expert-context-fix]]`.

**Decision (user):** do *not* gate the SFT down to only-correctly-docked designs (that discards
data). Instead **relabel** the SFT conditioning to the achieved interface, keeping every sample, and
push on-target docking through a **separate scalar GRPO reward**. Clean division of labor:

- **SFT (relabeled):** sequence↔structure coherence — dense/per-token, applied to *all* designs.
- **`R_dock` (scalar):** dock at the requested epitope — group-relative advantage, applied to all.

The two signals are orthogonal. A coherent-but-off-target design keeps full SFT credit but earns low
`R_dock`; an on-target-but-incoherent design earns high `R_dock` but low SFT. Together they pull
toward coherent *and* on-target.

## Shared primitive: interface derived from `X*`

Add `derive_interface_from_structure(coords, elements, chain_split, cutoff=5.0)` (heavy-atom
contacts; reuse the contact logic already used for the 3Di-TV interface). Returns:

- `epitope' (E')` — antigen residues with any heavy atom within `cutoff` of a binder heavy atom.
- `paratope' (P')` — the binder-side dual.

Place next to `derive_3di_tokens` / `derive_structure_tokens` in
`src/lobster/rl_training/rewards/_protenix_structure_expert.py` (currently derives only τ*/s*, no
contact map — confirmed).

## Change 1 — relabel the CHORD structure SFT conditioning

For each design fed to the structure SFT:

1. Derive `E'`, `P'` from `X*` (shared primitive).
2. Rebuild a conditioning tensor from `(E', P')`, identical construction to
   `src/lobster/cmdline/generate_modes/_binders.py:305-321` — `(1, L_total, 1)` float, `1.0` at
   `E'` antigen residues, bidirectional `P'` binder residues folded into the same channel.
3. Thread it in via a new `conditioning_override` arg on `_structure_endpoint_logits`
   (`_structure_sft.py:60` is the single edit point; today it reads `static["conditioning_tensor"]`
   directly). The sequence-SFT path is unaffected.

**Invariant:** always condition the SFT forward on the epitope/paratope the distilled structure
actually realizes. No design is dropped; the target is always self-consistent.

*Train/inference note:* SFT now trains `p(structure@E' | cond=E')` — a proper function of the
conditioning — which generalizes to the requested `E` at inference (`cond=E → structure@E`). This is
exactly why relabel beats keep-stale-`E`.

## Change 2 — `R_dock`: dock-correctness scalar reward

Reuse the **same** Protenix fold already produced for the SFT (zero extra folds).

For specified epitope residues `E` and the `X*` contact set:

- **Soft recall** `r_rec = mean_{e∈E} σ((d0 − d_e)/w)`, where `d_e` = min distance from antigen
  residue `e` to any binder heavy atom (`d0 ≈ 5–8 Å`, `w ≈ 1–2 Å`). Smooth contact indicator —
  dense gradient, not 0/1. Rewards actually touching the requested residues.
- **Soft precision** `r_prec = |contacts ∩ E| / |contacts|` — fraction of contacted antigen
  residues that lie in `E`. Anti-smear: stops the policy from wrapping the whole antigen to
  trivially cover `E`.
- **`R_dock = F1(r_rec, r_prec) = 2·r_rec·r_prec / (r_rec + r_prec)`** — penalizes both misses and
  smearing.

**Integration.** Standard scalar term in the GRPO objective, group-relative (Dr.GRPO), weight
`w_dock`, alongside clash/shape/3Di-TV. It is the on-target counterpart to ipTM: ipTM says "there is
a confident interface," `R_dock` says "it is the one we asked for." Epitope CFG biases inference
toward `E`; `R_dock` puts the same objective in the gradient so the policy internalizes it rather
than leaning on inference-time guidance. Pass (pTM>0.8 ∧ ipTM>0.7) stays an offline label.

**Reward-hacking guards.**
- Precision term vs the smear/interpenetration failure mode already seen in the clash story
  (`[[clash-reward-4pop-validation]]`).
- Co-active clash/shape rewards prevent "contact `E` by driving through the antigen."
- Optionally gate `R_dock` by a minimum pTM (and/or ipTM-weight it) so it never rewards a garbage
  fold that happens to touch `E`.

**Caveat — Protenix as judge.** `R_dock` trusts Protenix's docked pose and inherits its pose noise
(the ~29.5 Å L-RMSD finding). Mitigate with multi-seed Protenix averaging
(`[[grpo-reward-reproducibility]]`) and/or ipTM-weighting low-confidence poses.

**Tuning risk.** Because relabeling lets the SFT happily teach coherent *off-target* folds, `w_dock`
must be strong enough to supply the counter-pressure; too weak and the policy drifts to
"coherent but off-target." First smoke should confirm `R_dock` separates on- vs off-epitope gens
offline before it enters the gradient.

## Verification

1. Offline: `derive_interface_from_structure` on stored Protenix complexes reproduces the 3Di-TV
   interface set; `R_dock` on passing-Complexa vs off-epitope gens separates them (AUROC gate,
   like every reward before it enters the gradient).
2. Unit tests: relabel invariance (SFT loss depends on `E'`, not stale `E`); `R_dock` F1 edge cases
   (perfect dock → 1, disjoint → 0, smear → precision-limited).
3. Smoke (`sft_mu>0`, structure SFT on, `w_dock>0`): step 2 reached, `ratio_init ≈ 1.0` (GRPO path
   untouched), SFT sees relabeled conditioning, `R_dock` finite and in `[0,1]`.
4. Launch with a fresh OUTBASE; never cancel the live HARD run (`[[never-cancel-hard-chord-run]]`).

## Open items

- `d0`/`w`/`cutoff` values; whether to ipTM-weight `R_dock`.
- Whether `R_dock` should be F1 or recall-only when the epitope is small (few residues → precision
  noisy).
- Pin down Table 1's exact benchmark sampler recipe (schedule/checkpoint) vs the A8 ablation sampler
  — both share `sequence_diversity_penalty=2`; the gap is schedule + best-arm, not the penalty.
