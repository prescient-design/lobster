# Scope: distribution-matching GRPO reward (LeFlur ⟶ Proteina-Complexa)

**Status:** scoping / design only — no implementation yet.
**Implementation home:** the `complex_infra` worktree
(`src/lobster/rl_training/`), **not** this `3di_flow_decoder_pr` worktree. GRPO
lives there.
**Ask (verbatim):** *"revisit our grpo experiments but base the rewards on KL
difference between structure and sequence metrics between LeFlur and
Proteina-Complexa because it is a lot faster than waiting on Protenix and the
signal might be more consistent."*

---

## TL;DR / recommendation

Add a **fourth reward family** to the existing GRPO reward sum: a per-design
**interface-distribution distance** term that scores how close a design's
*binder-interface* amino-acid and 3Di histograms are to a **reference
distribution** (Proteina-Complexa's binders, and/or native), with **no Protenix
call**. It reuses tensors `_compute_rewards` already has in hand, so it is
essentially free and runs at policy speed.

Grounding: Tier-0 validated (target-difficulty-controlled) that a design whose
interface distribution is closer to a reference predicts passing. **D0 (done,
below) confirms this holds — and is *stronger* — against a Complexa reference,
and shows that with a Complexa reference the AA *and* 3Di angles both carry
independent, co-equal signal** (logistic z(AA) −0.214, z(3Di) −0.244), unlike the
native reference where 3Di was redundant. ⇒ include **both** terms, balanced, and
keep it strictly **interface-localized** (global whole-binder V-monotony is the
*failing* mode — `passing-binder-structural-signature`). See memories
`tier0-iface-distance-predicts-pass` (native) and this doc's D0 section (Complexa).

Three load-bearing caveats, expanded below:
1. **Use it as a dense *shaping* term, not a full Protenix replacement.** Tier-0
   effect sizes are modest (rho −0.08…−0.22). This buys a cheap, consistent,
   every-step gradient; Protenix stays as a **periodic validation gate** to
   detect reward-hacking. It is a faster/denser signal, not a more *accurate* one.
2. **Marginal matching is gameable** — safeguard with the diversity terms
   (already in the reward), a per-target reference, and the interface-only
   localization. Details in "Reward-hacking analysis".
3. **The reference source is a real decision** (Complexa-pooled vs
   native-per-target vs blend) — see "Reference source". The user named Complexa;
   native-per-target is arguably better-grounded but sparser. Recommendation:
   support both, default to **Complexa-pooled per-target with native fallback**.

---

## Why this is worth doing

- **Speed.** The Protenix confidence reward is the throughput bottleneck: ~80 s
  weight-load + ~25 s/design on an A10G, served out-of-process via a worker
  queue (`_protenix_reward.py`, memory `grpo-reward-worker-lifecycle`). A
  histogram-distance term is pure numpy over tensors already decoded on the
  policy side — microseconds/design. It gives gradient on **every** design every
  step, where ipTM is near-zero for all but the rare passer (README §1).
- **Consistency.** Single-seed Protenix ipTM is borderline-reproducible (seed std
  ≈ 0.064, heteroscedastic — memory `grpo-reward-reproducibility`); reward-chasing
  lucky draws was a repeated failure. An interface-histogram distance is
  deterministic given the design — zero oracle variance.
- **It is validated signal.** Tier-0 is the direct evidence that this quantity
  correlates with the true objective (pass), target-controlled, across all 7
  arms. We are turning a *validated predictor* into a *training signal*.
- **GRPO only needs monotonicity, not calibration.** Advantages are
  group-relative: `A_i = (r_i − mean(r_group)) / std(r_group)`. The distance term
  only has to *rank* the group by "nativeness" correctly; absolute scale and
  perfect calibration don't matter. A modest but correctly-signed predictor is
  exactly what GRPO can exploit.

---

## What we already have (the free lunch)

`LeFlurGRPOTrainer._compute_rewards(target_id, seqs, tri_seqs, trajectory, comp)`
(`_leflur_grpo_trainer.py:471`) is handed, per group of `G` designs, everything
the new term needs — **already decoded, no Protenix**:

| have | where | is |
|---|---|---|
| `seqs` | `_decode_binder_seqs` (`:381`) | per-design binder **AA** strings |
| `tri_seqs` | `_decode_binder_tri` (`:391`) | per-design binder **3Di** token strings |
| generated backbone | `_decode_backbone_coords` (`:410`, CPU VIT decoder) → `gen_ca` (G,L,3) (`:445`) | CA coords of the **generated** complex |
| antigen coords | `comp` / fixed target input | antigen is held fixed during generation ⇒ its backbone is known without Protenix |

So the interface (binder residues with min cross-chain Cα–Cα < 8 Å) is
computable **from the generated complex alone** — the same interface definition
Tier-0 used, just sourced from the generated backbone instead of the Protenix
cif. `seqs` and `tri_seqs` give the two histograms directly. The structure term
(`_structure_terms_for_group`, `:431`) already decodes `gen_ca` for every design,
so the interface computation can even share that decode.

**Consequence:** the new term slots in beside the existing three families with
the exact same shape — a per-design list summed into `reward_i` — and the
reward-hacking / weighting machinery (per-metric clip to [0,1], weight defaults 0)
carries over unchanged.

---

## The reward, precisely

### Metrics (two distributions, interface-local)

For each design *i*, over its **interface binder residues** only:
- `p_aa^i` — 20-bin amino-acid histogram (normalized).
- `p_3di^i` — 20-bin 3Di structural-state histogram (normalized; mini3di
  Foldseek encoder on the generated `(N,CA,C)`), **or** tally `tri_seqs[i]`
  restricted to interface positions (cheaper, no re-encode — `tri_seqs` is
  already decoded).

These are exactly the Tier-0 quantities (`_tier0_compute.py:design_hists`), so
the reference and the design histograms are computed by identical code paths.

### Reference distribution `q`

`q_aa^ref`, `q_3di^ref` — the target interface distributions to pull toward. Built
**once, offline** into a small lookup table (per-target + a pooled fallback),
loaded by the reward module. See "Reference source" for what goes in it.

### Distance → bounded reward

The reward framework's convention is *"orient higher-is-better, clip to [0,1]"*
so terms are comparable and none can blow up (README top). Raw KL is unbounded
and misbehaves on sparse design histograms (an interface has ~10–30 residues over
20 bins). Therefore:

- **Primary distance: total variation** `TV(p,q) = ½Σ|p−q| ∈ [0,1]` — symmetric,
  bounded, robust to sparsity; Tier-0's primary metric.
- **Reward:** `r_dist^i = clip(1 − TV(p^i, q^ref), 0, 1)` per histogram, so higher
  = closer to reference, already in [0,1] — a no-op under the shared `s(·)` clip.
- **If a KL flavor is wanted** (the user said "KL"): use **Jensen–Shannon
  divergence** `JS(p,q) ∈ [0,1]` (symmetric, bounded, finite on sparse p) rather
  than raw KL. Report both TV and JS in wandb; they should agree. Keep raw
  KL(p‖q) with additive smoothing only as a logged diagnostic, not the reward
  (it's what Tier-0 stored as `kl_*`).

### Term formulation (weighted — include BOTH, per D0)

```
dist_term_i = w_aa_dist  · clip(1 − D(p_aa^i , q_aa^ref ), 0, 1)
            + w_3di_dist · clip(1 − D(p_3di^i, q_3di^ref), 0, 1)
```

with `D = TV` (D0-B: TV ≈ JS, TV marginally wins on AA and is simpler/bounded/
smoothing-free; keep JS as a logged diagnostic).

**Weighting — updated by D0 (see "D0 results" below).** The original Tier-0
finding ("AA is the carrier; 3Di redundant") was **specific to the *native*
reference**. When D0 rebuilt the analysis against a **Complexa** reference, the
3Di angle carries *independent, co-equal* signal (within-target logistic:
z(3Di) −0.244, z(AA) −0.214 — vs native, where z(3Di) collapses to +0.05). The
sparse 1-structure-per-target native reference hid a real 3Di signal that the
dense Complexa reference reveals. **⇒ For a Complexa-referenced reward, include
both terms, roughly balanced** — start `w_aa_dist = w_3di_dist = 0.25` (sum 0.5).
Keep an AA-only arm (`w_3di_dist=0`) as an ablation, but AA-only is no longer the
default. This is the concrete knob D0 was run to set.

Add to the sum in `_compute_rewards`:
```
reward_i = confidence_term_i        # Protenix (may be down-weighted or made periodic)
         + structure_term_i         # scTM (needs Protenix backbone)
         + seq_diversity_term_i      # AA k-mer Jaccard novelty
         + struct_diversity_term_i   # 3Di k-mer Jaccard novelty
         + dist_term_i               # NEW: interface distribution → reference
```

Guard rails already present: `dist_term` is inert at `w_*_dist=0` (byte-identical
default), and each component is clip-bounded, so a degenerate histogram can't
blow up the reward. Follows the same "all weights default 0" pattern as the other
terms.

### An important, favorable property

A single design's interface histogram matching a **smooth** reference *requires
spread across bins*. A poly-Ala / poly-Ser interface is a delta at one bin ⇒ far
from a spread reference ⇒ high TV ⇒ **low** reward. So this term **intrinsically
penalizes interface degeneracy** — the exact failure mode (poly-Ala 0% pass,
poly-Ser collapse, distributed I/A/V collapse — memories
`degenerate-sequences-mark-failure`, `grpo-m14/m15`). It is a *distributional*
anti-degeneracy signal that the marginal-blind k-mer novelty terms miss. That is
a second reason to add it, beyond the direct pass correlation.

---

## Reference source (a real decision to make)

The user named **Proteina-Complexa**. Three options; recommend supporting all in
the offline builder and choosing per-run:

1. **Complexa-pooled-per-target** *(user's ask; recommended default).* For each
   target, pool the interface histograms of *all* Complexa designs on that target
   into a dense reference. **Pro:** dense (N designs ⇒ smooth 20-bin reference,
   good for histogram matching), model-achievable (a reachable target, not an
   idealized one), on disk already (`complexa_hbindomain` arm + Complexa 38-target
   benchmark). **Con:** Complexa is not ground truth — though memory
   `interface-3di-distribution` shows Complexa's interface is the *closest to
   native* of anything we have (TV 17.7%), so pulling toward Complexa ≈ pulling
   toward native. **Con:** GRPO trains on pinder-heteromer targets that may not
   overlap Complexa's target set ⇒ needs a fallback.

2. **Native-per-target** *(best-grounded, but sparse).* The pinder-heteromer
   training targets have **ground-truth complex PDBs** ⇒ a native interface
   histogram is directly computable per training target, and Tier-0 validated
   *native* specifically. **Pro:** ground truth, always available for training
   targets, per-target. **Con:** usually **one** native complex/target ⇒ ~15
   interface residues ⇒ a very sparse, noisy reference distribution to match
   against (this is the main reason to prefer pooled Complexa for the *reference*
   even while native is the *validation truth*).

3. **Blend / fallback.** Complexa-pooled where available, native-per-target
   otherwise; or native-pooled-across-targets as a target-agnostic prior. A
   convex blend `q = λ·q_complexa + (1−λ)·q_native` is trivial and worth an
   ablation.

**Recommendation:** build the table with **both** Complexa-pooled and
native-per-target; **default to Complexa-pooled-per-target with native-per-target
fallback**, and keep a pooled-across-all-targets global prior as last resort so
the term never NaNs on an unseen target. Log which reference each target used.

Builder = a thin generalization of `_tier0_compute.py:native_reference` /
`design_hists` that emits `{target_id: {aa: (20,), 3di: (20,)}}` JSON for each
source. Reuses `_iface_3di_distribution.py` verbatim.

---

## Reward-hacking analysis + safeguards

Marginal-distribution matching is a **necessary-not-sufficient** proxy; enumerate
how the policy could satisfy it *without* making real binders, and the guard for
each:

| hack | why the term allows it | guard |
|---|---|---|
| **Group collapse** to one well-matching design (all G identical) | distribution term is per-design; identical designs all score well | **diversity terms stay ON** (`w_seq_diversity`, `w_struct_diversity`): identical designs share all k-mers ⇒ novelty ~0 ⇒ penalized. This is why they exist (README §3). |
| **Match the marginal, wrong geometry** — right AA/3Di *counts*, residues not actually contacting antigen | marginal is contact-blind | interface is defined by the **generated** complex's cross-chain <8 Å; a non-docking binder has few/failed interface residues ⇒ `n_iface < 4` ⇒ term skipped/floored (Tier-0's own guard). Keep **scTM-complex** on to tie pose to fold. |
| **Global mimicry** — make the *whole binder* look native while the interface doesn't | if we scored global, not interface | **interface-only localization** (never global). Global whole-binder V-monotony is itself the failing mode (`passing-binder-structural-signature`); scoring global would reward the pathology. |
| **Proxy ↑ but pass ↓** — the modest correlation lets the policy climb the proxy off the pass manifold | Tier-0 rho is only −0.08…−0.22 | **periodic Protenix validation** (next section) is the backstop; **keep a non-zero confidence weight** so the true objective still has a vote every step (see "Two deployment modes"). |
| **Reference is Complexa's own biases** | Complexa ≠ native | validate against **native** and against **pass rate**, not against Complexa; optionally blend native into `q`. |

Design principles carried over from the M-run history (memory `grpo-mrun-ledger`,
`grpo-lever-diagnosis`): keep the anti-collapse safeguards independent of the new
term; `lr=1e-5` is the sole confirmed learning lever (don't co-vary it while
testing the reward); change **one reward variable at a time**.

---

## Protenix validation loop (the backstop)

The whole point is to *not* wait on Protenix every step — but we must still prove
the proxy tracks truth and catch hacking:

- Every `N` steps (e.g. 25–50), fold a **held-out sample** (e.g. 16 designs from
  the current policy) with the *existing* Protenix worker pool and log
  `val/pass_rate`, `val/iptm` alongside the proxy `reward/dist_term_mean`.
- **Health check:** proxy ↑ **and** Protenix pass ↑ ⇒ good. Proxy ↑ **while**
  Protenix pass flat/↓ ⇒ **reward-hacking detected** ⇒ stop / re-weight. This is
  the same divergence diagnostic that caught the serine/I-A-V collapses, just
  automated on a schedule instead of every step.
- Cost is bounded and off the critical path (async, on the idle worker pool),
  unlike the per-step confidence reward.

### Two deployment modes (pick per experiment)

- **A — Shaping (recommended first):** keep the confidence reward ON but the
  dist_term adds dense gradient. `w_abag_iptm=1.0, w_ptm=0.5` (shipped) +
  `w_aa_dist=0.5`. Tests whether the dense term *accelerates* / *stabilizes* the
  existing objective. Lowest risk.
- **B — Proxy-primary (the speed play):** confidence reward **periodic-only**
  (validation, not per-step) or heavily down-weighted; dist_term is the main
  per-step signal. This is the "much faster" run the user is after. Higher
  reward-hacking risk ⇒ only after A shows the term has correctly-signed traction,
  and only with the validation loop armed.

---

## Implementation plan (in `complex_infra`, when approved)

Mirrors the existing reward-module layout; **no `trl`/oracle deps** so it stays
import-light and unit-testable (like `_diversity_reward.py`):

1. **`rewards/_distribution_reward.py`** (new, pure numpy):
   `interface_histograms(gen_coords, chains, aa_seq, tri_seq) → (p_aa, p_3di,
   n_iface)`; `tv`, `js`; `distribution_terms(p_aa, p_3di, q_aa, q_3di, w_aa,
   w_3di, metric) → term`. Ports `_tier0_compute.py` interface/hist logic; reuses
   `_iface_3di_distribution.py` helpers (`get_interface_residues`, `encode_3di`,
   `IFACE_THRESH`).
2. **Reference table + builder** — offline script emitting per-target +
   pooled-fallback `q` JSON for Complexa and native sources; loaded once in the
   trainer `__init__` (like the CPU VIT decoder is at `:270`).
3. **Trainer wiring** — in `_compute_rewards` (`:471`): compute `dist_terms`
   from `seqs`/`tri_seqs`/`gen_ca` (reuse the `_structure_terms_for_group` decode
   at `:445`), add to `reward_i`; new config knobs `w_aa_dist`, `w_3di_dist`,
   `dist_metric`, `dist_reference` (defaults `0 / 0 / "tv" / "complexa"` ⇒
   byte-identical when off). wandb: `reward/dist_term_mean`,
   `reward/dist_aa_term_mean`, `reward/dist_3di_term_mean`, raw `dist/tv_aa`,
   `dist/tv_3di`, `dist/js_*`, `dist/n_iface_mean`.
4. **Periodic Protenix validation hook** in the trainer loop (mode-A/B switch).
5. **Config** — `rl_leflur_binder_grpo_dist.yaml` (D-series), following the m*
   experiment-yaml pattern.
6. **Tests** — `tests/lobster/rl_training/test_distribution_reward.py`:
   histogram sums to 1; TV/JS bounds [0,1] and symmetry; identical p,q ⇒ term = w;
   delta-histogram (poly-X) vs spread reference ⇒ near-0 reward (degeneracy
   penalty); `n_iface<4` ⇒ skipped; off-by-default byte-identity.

## D0 results (DONE — no training; this is what set the knobs)

Re-ran the Tier-0 target-controlled analysis on all 13,719 designs (re-computed
with per-design interface histograms) using a **per-target Complexa reference**
(built from the `complexa_hbindomain` arm on the same 20 HB in-domain targets;
19/20 targets had ≥25 Complexa designs, pooled fallback for the rest). Scripts:
`scripts/_tier0_compute.py` (now stores `h3di`/`haa`), `scripts/_tier0_d0_analyze.py`,
plot `scripts/_tier0_d0_complexa.png`.

- **D0-A — distance-to-Complexa predicts pass for LeFlur, target-controlled, and
  is *better* than distance-to-native.** Within-(arm,target) Spearman(dist, iptm),
  pooled LeFlur n=11,836 (10.8% pass), all Wilcoxon p<1e-4:
  - AA: Complexa med rho **−0.105** (71% neg) vs native −0.079. Pass-rate
    monotone by quartile, Q1(closest)→Q4: 18.2/9.5/9.8/5.6%.
  - 3Di: Complexa med rho **−0.082**, and now **cleanly monotone** —
    23.8/5.9/4.5/8.9% (Q1−Q4 = +14.9pp), vs the native 3Di quartile which was
    non-monotone/bimodal (+5.5pp). The dense Complexa ref fixes the 3Di angle.
- **D0-B — TV ≈ JS; use TV.** AA: TV −0.105 vs JS −0.104; 3Di: TV −0.082 vs JS
  −0.096 (JS marginally better for 3Di only). Differences are within noise ⇒ use
  TV (bounded, symmetric, no smoothing, Tier-0 precedent); log JS as a diagnostic.
- **D0-C — the reference choice flips the AA-vs-3Di story.** Within-target
  logistic `passes ~ z(dist_3di)+z(dist_aa)`:
  - **native** ref: z(AA) −0.222, z(3Di) **+0.046** → 3Di redundant, AA carries
    (the original Tier-0 conclusion).
  - **Complexa** ref: z(AA) −0.214, z(3Di) **−0.244** → **both independent and
    co-equal** (3Di marginally stronger). ⇒ **include both terms** in a
    Complexa-referenced reward (`w_aa_dist=w_3di_dist=0.25`), not AA-only.
- **Positive control (Complexa arm vs its own per-target ref; circular, flagged):**
  very strong — AA med rho −0.272, Q1−Q4 = +55pp — confirming the machinery.

Net: the Complexa reference is *validated and preferable* to native for this
reward, and it upgrades 3Di from "redundant shadow" to "co-equal independent
signal." The scope's weighting is updated accordingly above.

## Experiment ladder (D-series)

- **D0 — offline sanity (no training). ✅ DONE — see "D0 results" above.** Locked:
  Complexa per-target reference; **TV** distance; **both AA and 3Di** terms,
  balanced.
- **D1 — mode A, AA+3Di balanced** (`w_aa_dist=w_3di_dist=0.25`, confidence ON).
  Does dense shaping help/stabilize vs the M22 baseline?
- **D2 — ablation: AA-only vs 3Di-only vs balanced.** D0 predicts balanced ≥
  either alone; confirm under training.
- **D3 — mode B, proxy-primary** (confidence periodic-only). The speed play;
  validation loop armed. Compare wall-clock/step and pass-rate trajectory vs D1.
- **D4 — reference ablation:** Complexa-pooled vs native-per-target vs blend.

Single-target overfit first (the M6/M7 protocol) to confirm the term *moves* the
policy and the validation loop fires, then pinder-heteromer.

---

## Risks / open questions

- **Modest effect size (main risk).** rho −0.08…−0.22: the term may be too weak
  to lead on its own (mode B). Mitigation: lead with mode A; treat mode B as the
  stretch goal gated on D1 traction. Honest framing: *faster and more consistent,
  not more accurate* — it is a dense proxy, not a replacement for the pass oracle.
- **Reference mismatch on training targets** (Complexa vs pinder targets) — the
  fallback chain and native-per-target option handle this; log usage.
- **Sparse per-design interface** (~10–30 residues) makes any single histogram
  noisy — TV/JS chosen for sparsity-robustness; `n_iface≥4` floor; group
  averaging in the advantage smooths it.
- **Does "closer to Complexa" transfer off Complexa's targets?** D4 answers it;
  native-per-target is the hedge.
- **Interaction with scTM/diversity weights** — change one reward variable at a
  time (M-run discipline); hold `lr=1e-5`.

---

## One-paragraph summary for the user

Yes — this is a clean, cheap add. `_compute_rewards` already decodes, per design
with no Protenix, the binder AA sequence, the 3Di tokens, and the generated
backbone; from those we compute the design's **interface** AA + 3Di histograms
and score `1 − TV` (or `1 − JS`, the bounded stand-in for your "KL") against a
**Proteina-Complexa** reference distribution, weighting **AA above 3Di** because
Tier-0 showed AA is the carrier. It's a dense, deterministic, per-step signal —
much faster and more consistent than Protenix ipTM — but effect sizes are modest,
so I'd run it first as a *shaping* term on top of the confidence reward (mode A),
keep the diversity terms on and the scoring **interface-only** to block the known
collapse/monotony hacks, and fold a held-out sample with Protenix every ~25 steps
as a hacking backstop. Reference-source (Complexa-pooled vs native-per-target) is
the one real decision — I recommend supporting both and defaulting to
Complexa-pooled-per-target with a native fallback. Implementation is a new
`_distribution_reward.py` + reference builder + trainer wiring in the
`complex_infra` worktree, off by default (`w_*_dist=0`), starting with a
no-GPU D0 re-analysis to lock the knobs before any SLURM.
