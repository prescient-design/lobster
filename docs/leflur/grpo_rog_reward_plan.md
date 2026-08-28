# Plan: a radius-of-gyration (compactness) GRPO reward

**Date:** 2026-08-27 · **Status:** wired + launched (eight-term run, job 20460985) · companion to
[`grpo_reward_inventory.md`](grpo_reward_inventory.md) and the Table-6 geometry analysis
(`scripts/_grpo_bench_geom.py`).

## TL;DR

- **The gap.** Every existing structural reward measures a *local* property — steric clash and
  interface-contact fraction (`clash_contact_reward`), interface 3Di/AA distribution
  (`_distribution_reward`), 3DZD shape complementarity (`_shape_reward`), backbone peptide-bond
  integrity (`_chainbreak_reward`). **None sees the binder's global fold state**: whether the chain
  is a compact globule or an over-extended tangle. The cross-arm geometry sweep (Table 6) shows this
  is a real, **length-independent** gap — at essentially matched chain length the passing
  Proteina-Complexa references are compact while *every* trained arm is measurably over-extended.
- **Is it a lever?** **Plausibly yes, and it is now testable.** Compactness (below) separates the
  passing reference from the trained arms with a monotone, length-normalized ordering, and it is
  *orthogonal* to the terms already in the reward (clash/contact/chainbreak are all local). It is
  wired as an additive saturating term and turned on in the eight-term run; the verdict is the
  Protenix pass-rate / Rg shift of that run vs the six-term baseline.
- **The form.** A single `[0, 1]` **saturating** reward on the binder Cα radius of gyration:
  ```
  Rg_actual   = sqrt( mean_i ‖CA_i − mean(CA)‖² )     # binder Cα radius of gyration (Å)
  Rg_compact  = r0 · N^(1/3)                           # ideal globule Rg at N residues (r0=2.2)
  compactness = Rg_compact / Rg_actual                # ~0.7 globular, higher = more compact
  R           = clip(compactness / rog_full, 0, 1)     # rog_full = 0.76 (passing-Complexa target)
  ```
  `compactness` is length-normalized by construction (the `N^(1/3)` compact-globule scaling divides
  out chain length), so it targets **fold state, not size**. Saturating at the native-anchored target
  removes any pressure to collapse past the native fold.

---

## 1. What compactness measures and why it is the right normalization

The radius of gyration `Rg` scales with chain length as `Rg ∝ N^(1/3)` for a constant-density
globule (mass fills a ball whose radius grows as `N^(1/3)`). So raw `Rg` conflates two things we
must separate: **size** (bigger binders have bigger Rg — not a defect) and **fold state**
(over-extension at fixed size — the defect). Dividing by the compact-globule reference
`Rg_compact = r0 · N^(1/3)` removes the size term and leaves a scale-free fold-state score:

```
compactness = r0 · N^(1/3) / Rg_actual
```

A well-packed globule sits near a constant `~0.7` regardless of `N`; an extended or tangled chain
scores lower. `r0 = 2.2` is the codebase convention (`_grpo_bench_geom.py`'s `Rg_norm = Rg / (2.2 ·
N^(1/3))`, of which compactness is the reciprocal), calibrated so a typical globular protein scores
`~0.7`. This is exactly the mass-weighted `biotite.structure.gyration_radius` reduced to a Cα-only,
equal-mass selection (the centroid-RMS above); it is reimplemented in pure numpy in the reward
module to keep the term dependency-free and consistent with the other pure-numpy shaping terms.

## 2. Empirical separation (Table-6 geometry sweep)

At matched binder length the passing reference is compact and the trained arms are over-extended,
with a **monotone** ordering — the more the arm has been trained on the existing (local) reward set,
the *less* the extra CHORD-SFT-heavy arms recover compactness on their own:

| arm                     | compactness (r0=2.2) | Rg_norm = Rg/(2.2·N^(1/3)) |
|-------------------------|:--------------------:|:--------------------------:|
| passing Complexa (ref)  | **0.76**             | 1.31                       |
| six-term step275        | 0.67                 | 1.50                       |
| base                    | 0.63                 | 1.58                       |
| scalar-AAR              | 0.60                 | 1.67                       |
| CHORD-SFT step175       | 0.54                 | 1.85  (most over-extended) |

The gap is not chain length (Rg_norm is length-normalized) and not clash (the arms already de-clash
under the contact-band term) — it is genuine global over-extension that no local term penalizes.
`rog_full = 0.76` is set to the passing-Complexa compactness so the reward pulls the over-extended
arms toward the native fold state and is **inert on already-globular designs**.

## 3. Why saturate (anti-collapse)

Raw compactness is *unbounded above* — a collapsed dense blob scores arbitrarily high — so rewarding
it directly would invite a fold-collapse reward hack (crush the binder into a ball to farm reward).
The term instead **saturates** at the native-anchored target, exactly mirroring the linguistic-
complexity reward (`lc_saturating_reward`): full credit once `compactness ≥ rog_full`, ramping to 0
as the binder over-extends, and **no gradient to over-compact past native**. This is the same
"pull toward the native band, no force inside it" shape the LC-floor and contact-band terms use.

## 4. Implementation (wired)

- **Reward module** `src/plm_design_rl/rewards/_rog_reward.py` (+ back-compat shim at
  `src/lobster/rl_training/rewards/_rog_reward.py`, re-exported from both package `__init__`s):
  - `rog_compactness(ca, *, r0=2.2) -> float` — length-normalized compactness of `(N,3)` Cα coords.
  - `rog_compactness_reward(coords_full, valid_mask, binder_mask, *, r0=2.2, rog_full=0.76)
    -> (term, diag)` — scores the binder Cα (atom index 1) at `valid_mask & binder_mask`; returns
    the saturating `[0,1]` term and `{compactness, rg, n_res}`. Pure numpy, `< 2`-residue and
    zero-Rg guards, **scalar** (advantage-weighted log-prob, not back-propagated).
- **Trainer** `_leflur_grpo_trainer.py`:
  - `_rog_terms_for_group(trajectory, comp, *, gen_bb=None)` mirrors `_chainbreak_terms_for_group`
    — decodes the generated backbone once (shared `gen_bb` with the dist/clash/chainbreak terms) and
    loops the group, weighting by `w_rog`. Metrics: `reward/rog_term_mean`, `rog/compactness_mean`,
    `rog/rg_mean`, `rog/n_res_mean`, `rog/frac_saturated`.
  - `_compute_rewards` gates the decode on `need_rog = w_rog > 0` and sums the term with the others
    (both the pipelined and non-pipelined paths).
- **Config fields** (`GRPOTrainerConfig` + `cmdline/rl_train.py` mapping): `w_rog` (default 0.0 →
  inert), `rog_r0` (2.2), `rog_full` (0.76).
- **Tests** `tests/lobster/rl_training/test_rog_reward.py` — pure-numpy (compact > extended,
  length-normalization, saturation, binder-only masking, padding/`<2`/zero-Rg guards, shim identity)
  + trainer-wiring (weighting/aggregation, shared `gen_bb` without decoding).

## 5. Launch + validation

Turned on in the eight-term run (`w_rog=1.0`, alongside the new 3Di-LC term), config
`experiment/rl_leflur_binder_grpo_elj_chainbreak_chord_lc_jaccard_contactband_rog_lc3di`,
policy job **20460985** (Protenix-free; separate a10g repack pool for the e_lj/CHORD draw). The
term is additive and defaults to 0, so the live six-term run is byte-identically undisturbed.

**Verdict criteria** (six-term step-matched baseline vs eight-term):
1. **Does it move Rg?** `rog/compactness_mean` should climb toward `rog_full=0.76` and `rog/rg_mean`
   drop; `rog/frac_saturated` rise. (Necessary — confirms the gradient reaches the fold state.)
2. **Does it help pass rate?** The real test — Table-6 cold-A8 Protenix pass rate of the eight-term
   arm vs six-term at matched step. Compactness is a *lever* only if the pass rate rises (or holds
   while another reward-hack is removed); a compactness gain with flat pass rate means it is a
   correlate, not a cause (cf. the MPNN-consistency and n_hb/dsasa negative gates).
3. **No collapse hack.** Watch for `rog/compactness_mean` overshooting `rog_full` with a *falling*
   `contact_score` / rising self-clash — the saturation should prevent this; if it appears, lower
   `rog_full` or cap the term.
