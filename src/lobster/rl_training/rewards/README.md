# LeFlur GRPO reward terms

Reward oracles and shaping terms for GRPO fine-tuning of the `leflur-binder-3di`
policy. The trainer (`.._leflur_grpo_trainer.LeFlurGRPOTrainer._compute_rewards`)
scores a group of designs, sums the terms below into one scalar reward per design,
then computes group-relative advantages from it.

```
reward_i = confidence_term_i        # weighted linear combo of Protenix confidence metrics  (_protenix_reward.py)
         + structure_term_i         # weighted scTM: binder-chain + whole-complex            (_structure_reward.py)
         + seq_diversity_term_i      # k-mer Jaccard novelty on the AA sequence               (_diversity_reward.py)
         + struct_diversity_term_i   # k-mer Jaccard novelty on the 3Di tokens                (_diversity_reward.py)
```

Every term is a **weighted sum of metrics that have each been oriented so higher is
better and clipped to `[0,1]`** — so the weights are directly comparable across terms
and no single metric can blow up the reward. All weights default to `0` except the two
that reproduce the shipped behaviour; a missing/failed metric contributes `0`.

The **confidence term is a flat, weighted linear combination of the individual
Protenix confidence metrics**:

```
confidence_term_i = w_iptm  · s(iptm_i)
                  + w_ptm   · s(ptm_i)
                  + w_abag  · s(abag_iptm_i)
                  + w_plddt · s(plddt_i / 100)
                  + w_gpde  · s(1 − gpde_i / 2)             # lower gpde better ⇒ invert & scale
                  + w_pae_global · s(1 − pae_global_i / 32) # lower PAE better ⇒ invert & scale
                  + w_pae_iface  · s(1 − pae_interface_i / 32)
```

where `s(x) = clip(x, 0, 1)`.

**What we clip, and why.** Each metric is first *oriented* onto a higher-is-better
axis: the `*tm` metrics are already `[0,1]`; `plddt` is divided by 100; the
lower-is-better error metrics (`gpde`, `pae_*`, Å-like) are inverted and scaled by a
per-metric ceiling (`gpde/2`, `pae/32`) so "small error" maps to "near 1". `s(·)` then
**clips each oriented metric to `[0,1]` before weighting**. The clip is what keeps the
term bounded: a pathological Protenix value (e.g. a PAE above its 32 Å ceiling, giving
a negative pre-clip value, or a >1 rounding artifact) can't push a metric outside
`[0, w_m]` and dominate the sum. We clip *per oriented metric*, not the summed term, so
each component stays in its own `[0,1]` lane. Set any weight to 0 to drop that metric;
the current shipped behaviour is recovered with `w_abag=1, w_ptm=0.5` and all others 0.

The **structure term is the same clipped, weighted linear combination** — over
**self-consistency TM-scores** comparing the **LeFlur-generated backbone** (what the
policy sampled) against the **Protenix-predicted backbone** (what the oracle folds the
sequence into). It comes in two versions:

```
structure_term_i = w_sctm_binder  · sctm_binder_i     # binder chain only
                 + w_sctm_complex · sctm_complex_i     # whole binder+antigen complex
```

- `sctm_binder` — TM-score of the **binder chain only**: superpose the generated binder
  backbone onto the Protenix-predicted binder backbone, then TM-score. Isolates
  *fold* self-consistency — does the sequence fold to the shape the policy drew?
- `sctm_complex` — TM-score of the **whole complex** (binder + antigen together): with
  the antigen held fixed during generation, this additionally rewards the binder
  sitting in the same *pose* relative to the antigen as Protenix predicts, i.e. docking
  self-consistency on top of fold agreement.

Both are already `[0,1]` (TM-score) so `s(·)` is a no-op clip; a missing/failed
comparison contributes 0.

Everything here is pure Python / numpy / torch (no `trl`), so the reward terms
import on the policy side without pulling in the Protenix oracle's heavy deps, and
the scoring functions are unit-testable without GPUs or external binaries
(`tests/lobster/rl_training/test_{protenix,diversity,structure}_reward.py`).

---

## 1. Protenix confidence reward — `_protenix_reward.py`

**What.** The primary objective: does the designed binder actually dock the target?
We co-fold antigen + binder with Protenix-v2 and read its confidence module, then
combine the individual confidence metrics into one scalar via the weighted linear
combination at the top of this file. Each metric is an independent, separately
weighted reward component — no metric is privileged, no cross-metric normalization or
z-scoring; the term is simply their weighted sum.

- `reward_from_confidence(conf, weights)` → `Σ_m w_m · clip(orient_m(conf[m]), 0, 1)`,
  where `orient_m` maps each raw Protenix field onto a higher-is-better axis (`*tm`
  as-is, `plddt/100`, `1−gpde/2`, `1−pae/32`) and the clip bounds each oriented metric
  to `[0,1]`. A missing field contributes 0.
- `passes(conf)` → `ptm > 0.80 AND ip > 0.70` (with `ip = abag_iptm else iptm`) —
  the binder pass criterion, kept in sync with the eval
  (`scripts/_complexa_pertarget.py`) and logged as `conf/pass_rate`. This is the
  fixed evaluation gate; it does **not** change with the reward weights.

**Why a flat linear combination.** Earlier runs hand-special-cased individual
metrics — an additive `ptm_weight` bolt-on (M22) and a separately-coded z-scored
`gpde` rank bonus (M17) — which made the reward hard to reason about and each metric's
contribution incomparable. Flattening to one weighted sum of oriented metrics makes
the reward one legible knob-set: every metric has a weight in the same units, `0`
drops it, and the composition is obvious. Metric-specific behaviour (noise, spread,
direction) now lives only in the *choice of weight*, documented below, not in bespoke
code paths.

**What each metric buys you (for choosing weights).**
- `abag_iptm` / `iptm` — the docking signal itself. High-variance (seed std ≈ 0.064)
  and near-zero for most non-passing designs, so alone it gives gradient only on rare
  passers. `abag_iptm` is the antibody–antigen interface pTM; `iptm` is the fallback
  when `abag_iptm` is absent.
- `ptm` — lower-noise (seed std ≈ 0.023), real within-group spread on *every* design
  (≈0.65–0.82), and part of the pass criterion, so a modest `ptm` weight gives dense
  gradient toward the pass bar (this was the M22 rationale for `+0.5·ptm`). See memory
  `grpo-reward-reproducibility`.
- `plddt` — per-residue confidence in the folded structure (0–100 → `/100`); rewards
  foldability independent of docking.
- `gpde` — global predicted distance error, Å-like, **lower = better**; oriented as
  `1 − gpde/2` so a positive weight favours more-confident structures. In M19 it stayed
  flat (~1.40) under an indirect bonus — a direct weight is the cleaner test.
- `pae_global` / `pae_interface` — predicted aligned error (Å, **lower = better**),
  oriented as `1 − pae/32` (small error → near 1); interface PAE targets the docking
  interface specifically.

**Why served out-of-process.** Protenix is slow (~80 s weight load + ~25 s/design
on an A10G) and lives in its own py3.11 venv. `ProtenixRewardClient` submits a
group as queue jobs to a resident worker pool (`scripts/protenix_reward_server.py`)
over a shared filesystem, blocking for results; `n_shards` fans one group across
idle workers. Timeouts/failures floor only the affected designs (→ `None` → all
metrics contribute 0) rather than crashing the step. See memory
`grpo-reward-worker-lifecycle`.

**wandb.** `reward/confidence_term_mean` (the summed term) plus one
`reward/conf_<metric>_term_mean` per active metric so each component is visible;
raw fields under `conf/*` (`ptm`, `iptm`, `abag_iptm`, `plddt`, `gpde`, `pae_global`,
`pae_interface`).

---

## 2. Structure self-consistency reward — `_structure_reward.py`

**What.** A weighted combination of **self-consistency TM-scores** measuring how well
the sequence the policy designed folds back to the backbone the policy generated. The
two structures compared are:
- the **LeFlur-generated backbone** — the `xyz` the sampler decoded for this design
  (decoded on CPU from the sampled structure tokens via the VIT decoder), and
- the **Protenix-predicted backbone** — parsed from the co-folded complex the oracle
  already produces for the confidence reward (the `*_sample_0.cif`); no extra fold call.

```
structure_term_i = w_sctm_binder · sctm_binder_i + w_sctm_complex · sctm_complex_i
```

- `sctm_binder_i` — TM-score of the **binder chain only**: superpose the generated
  binder backbone onto the Protenix-predicted binder backbone (Kabsch), then TM-score
  (length-normalized by binder length). Isolates *fold* self-consistency — does the
  sequence fold to the shape the policy drew, independent of where it docks?
- `sctm_complex_i` — TM-score of the **whole complex** (binder + antigen). The antigen
  is held fixed during generation, so aligning the full generated complex to the full
  Protenix complex additionally rewards the binder occupying the same *pose* relative to
  the antigen — docking self-consistency layered on top of fold agreement.

Both are TM-scores already in `[0,1]` (higher = better), so the shared `s(·)=clip(·,0,1)`
is a no-op for them; a missing/failed comparison contributes 0.

**Why two versions.** `sctm_binder` and `sctm_complex` reward different failures.
Binder-only asks "is the fold realizable?" — a design can have a self-consistent fold
yet be docked into the wrong pocket. Complex asks "is the whole bound state
realizable?" — it subsumes pose agreement but can be dominated by the (fixed, large)
antigen and mask a bad binder fold. Exposing both as separate weights lets us reward
fold-consistency, pose-consistency, or a blend, rather than baking one choice in.

**Why at all.** Confidence metrics ask "is *some* good complex consistent with this
sequence?"; self-consistency asks "does the sequence fold to the structure *this policy
committed to*?" A high-ipTM design whose Protenix fold diverges from the generated
backbone is the policy getting lucky on the oracle, not learning to realize its own
structures — rewarding scTM ties the sequence and structure (3Di/LG) tracks together
and discourages that decoupling. It reuses the Protenix output already on disk.

**Implementation note (for when we build it).** The generated backbone lives on the
policy side (a tensor from the sampler); the Protenix backbone is produced in the
worker venv. The clean split is: the **worker** returns the predicted coordinates (or
precomputed scTM against a backbone the client passes in the job), and the
alignment/TM-score math lives in `_structure_reward.py` as pure numpy/torch
(unit-testable, no oracle deps) — mirroring how `_diversity_reward.py` stays
import-light. TM-score superposition is a Kabsch alignment;
we can vendor a small implementation rather than add a heavy dep.

**wandb.** `reward/structure_term_mean` plus `reward/struct_sctm_binder_term_mean` /
`reward/struct_sctm_complex_term_mean`; raw values under `struct/*` (`sctm_binder`,
`sctm_complex`).

---

## 3. Diversity rewards — `_diversity_reward.py`

**What.** Two independent novelty rewards, one per alphabet, each a design's
k-mer-Jaccard novelty vs *the rest of its group*:

```
seq_diversity_term_i    = w_seq_diversity   · jaccard_novelty(aa_seq_i,  other aa_seqs)
struct_diversity_term_i = w_struct_diversity · jaccard_novelty(tri_seq_i, other tri_seqs)
```

- **Sequence diversity** — mean pairwise k-mer-Jaccard *distance* (`1 − Jaccard`) of a
  design's AA sequence against every other design in the group. `1` = shares no k-mers
  with the group (maximally novel), `0` = identical to the group.
- **Structure diversity** — the same measure on the per-residue **3Di structural-token**
  string (`tri_seqs`), a cheap structural-novelty signal that needs no oracle.

Both are already in `[0,1]`, higher = better; each is inert (weight 0) by default and
requires its track (a 3Di track must exist for the structure term).

**Why this replaced the old bonus+penalty.** The previous term bundled a k-mer bonus, a
Foldseek TM-cluster rarity, *and* a two-sided anti-degeneracy hinge
(`max_aa_frac`/`entropy` floors) into one `diversity_adjustments(...)` scalar — three
mechanisms fighting to prevent the same failure (mode collapse to a degenerate design).
That was redundant and hard to tune. The Jaccard novelty already punishes degeneracy
directly: a poly-X or group-collapsed design shares all its k-mers with the group and
scores ~0, so a positive `w_*_diversity` pulls the group apart without a separate hinge
or the Foldseek dependency. Keeping AA and 3Di as **separate** rewards lets us weight
sequence-novelty and structure-novelty independently — the memories
`grpo-m14-serine-collapse-hinge` / `grpo-m15-distributed-collapse-m16` show the two
alphabets collapse independently, so they deserve independent knobs.

**wandb.** `reward/seq_diversity_term_mean`, `reward/struct_diversity_term_mean`; raw
values under `diversity/*` (`seq_novelty_mean`, `struct_novelty_mean`, `unique_frac`).

---

## Config knobs (`grpo.*` in the experiment YAML)

Every term is **one weight per metric** — flat, per-metric-clipped linear
combinations. Every weight defaults to `0`; set the ones you want. The shipped
default recovers the M22 behaviour with `w_abag_iptm=1.0, w_ptm=0.5` and everything
else (structure and diversity weights) off.

| knob | term | default | notes |
|------|------|---------|-------|
| `w_abag_iptm` | confidence | 1.0 | antibody–antigen interface pTM (docking) |
| `w_iptm` | confidence | 0 | interface pTM (docking, fallback signal) |
| `w_ptm` | confidence | 0.5 | low-noise, dense gradient toward the pass bar |
| `w_plddt` | confidence | 0 | oriented as `plddt/100` |
| `w_gpde` | confidence | 0 | oriented as `1 − gpde/2` (lower gpde ⇒ higher reward) |
| `w_pae_global` | confidence | 0 | oriented as `1 − pae_global/32` |
| `w_pae_interface` | confidence | 0 | oriented as `1 − pae_interface/32` |
| `w_sctm_binder` | structure | 0 | TM-score(gen binder, Protenix binder) ∈ [0,1] |
| `w_sctm_complex` | structure | 0 | TM-score(gen complex, Protenix complex) ∈ [0,1] |
| `w_seq_diversity` | diversity | 0 | AA k-mer-Jaccard novelty vs the group ∈ [0,1] |
| `w_struct_diversity` | diversity | 0 | 3Di k-mer-Jaccard novelty vs the group ∈ [0,1] |

Two more `grpo.*` knobs control **per-group binder-length sampling** (used by the
pinder-heteromer training set): `binder_length_min` / `binder_length_max`. When both
are set, each GRPO step samples one binder length `L ~ U[min, max]` and holds it
constant across the whole group (so advantages compare same-length designs); when
unset, the per-target `binder_length` from the manifest is used.

`reward.n_shards` (client) fans a group across the worker pool; it changes
throughput only, never reward values.
