# GRPO Binder Reward Shaping Plan

## Goal

Train LeFlur to generate binders that:
- **Pass Protenix co-fold**: `ptm > 0.80 AND abag_iptm > 0.70` (standard binder benchmark from `protenix_reward_server.py`)
- **Structurally diverse**: passers form distinct structural clusters, not a single degenerate mode

## Current M19 Reward (baseline)

M19 runs with `reward_clip_hi=1.0`, `reward_clip_lo=0.0`, so the raw dock signal is:

```
dock_i = clip(abag_iptm_i, 0.0, 1.0)     # clipped abag_iptm (or iptm if absent)
```

The full reward per design is:
```
r_i = dock_i
    + diversity_weight * novelty_i          # 0.2 × k-mer novelty vs group
    - degeneracy_weight * degeneracy_i      # 0.5 × max-AA-frac hinge + entropy hinge
    + gpde_weight * dock_std * z_gpde_i    # 0.3 × z-normalized gpde bonus
```

Where:
- `novelty_i` = `1 - max_j≠i jaccard(seq_i, seq_j)` (k-mer Jaccard novelty within group)
- `degeneracy_i` = `max(0, max_aa_frac - 0.3) + max(0, 0.8 - entropy)` (one-sided hinges)
- `z_gpde_i` = `(mean(gpde) - gpde_i) / std(gpde)` within group, scaled by `dock_std`

**Pass threshold used for logging**: `ptm > 0.80 AND abag_iptm > 0.70`  
**Pass rate in M19**: ~1–3% per step (1–2 passers per 64 designs per group), ~95% steps have 0 passers.

### What M19 taught us

- Group size 64 (vs 32) improved signal: more within-group ip variance → cleaner advantages
- ip drifted +0.04 over 115 steps (0.356 → 0.397), real but slow and decelerating
- ~95% of GRPO steps have zero passers → gradients are nearly all noise about "who is slightly less bad"
- ptm is much less noisy than abag_iptm (std ~0.023 vs ~0.064, 10-seed confirmed) — unused as reward signal
- M21 SS diagnostic confirmed gradients flow; the policy can move fast with dense signal

---

## Reward Experiments

### Exp A — M22: ptm + iptm continuous reward (priority: HIGH, cost: low)

**Hypothesis**: ptm provides a dense, low-noise second signal that gives gradient on every design, even non-passers.

**Reward change**:
```
dock_i = clip(abag_iptm_i, 0, 1) + α * clip(ptm_i, 0, 1)
```
with `α = 0.5` (ptm carries half the weight of iptm).

**Why ptm helps**:
- ptm std ~0.023 (10-seed) vs abag_iptm std ~0.064 → 3× less noisy per design
- ptm range in current binders: 0.65–0.82, with real variance across designs → non-zero gradient on every step
- ptm is a proxy for "does this sequence fold confidently" — correlated with passing but not identical
- No code change needed: `reward_from_confidence` currently returns `abag_iptm` only; adding `α*ptm` is a one-line change in `_protenix_reward.py`

**Implementation**: Add `ptm_weight: float = 0.0` to `GRPOTrainerConfig`, wire through `rl_train.py`, update `reward_from_confidence` to `abag_iptm + ptm_weight * ptm`.

**What to watch**: Does `reward/mean` climb faster? Does ip (abag_iptm) also climb even though we're not directly maximizing it?

---

### Exp B — M23: Multi-target easy subset (priority: HIGH, cost: moderate)

**Hypothesis**: Training on multiple targets simultaneously increases the number of passers per step dramatically, since even if target X has 0% pass rate, target Y might have 10%, giving clean signal.

**Target selection**: Use the 38-target complexa benchmark results. Pick the 8–10 targets with highest baseline pass rate (from the `summary.tsv` of prior campaigns). From memory, targets like `01_PD1` have low pass rate; others had >10% in the best arm.

**Implementation**: The trainer already supports `targets: [list]` in the config — just add more entries. Keep same reward, same group_size=64 per target, n_shards=8.

**Expected effect**: If 5 targets each have 3% pass rate independently, a mixed-target group effectively has ~15% passers → dense signal.

**Diversity bonus**: Multi-target naturally prevents mode collapse (can't collapse to PD1-specific sequences).

---

### Exp C — M24: ptm + iptm + multi-target combined (priority: MEDIUM, cost: moderate)

Run M22+M23 together once both are individually validated. The combination should give:
- Dense gradient on every step from ptm
- High pass rate from multi-target diversity
- Clean within-group advantages from group_size=64

---

### Exp D — M25: Distillation warm-start (priority: LOW, cost: high)

Use passing complexes from the distillation set (`/cv/scratch/u/lisanzas/grpo_distill_complexes/`) as a behavioral prior. Two approaches:
1. **Fine-tune checkpoint**: Run a short supervised fine-tune on passing binder sequences before GRPO — warm-starts the policy near the passing region
2. **DPO-style**: Add a KL term against a model fine-tuned on distilled passers instead of the frozen base model

Defer until M22/M23 have run and we understand whether the bottleneck is reward density or policy expressivity.

---

## Implementation order

```
M19 (running)  → baseline, arm C, group=64, single target, iptm-only dock
M22            → M19 + ptm_weight=0.5 (one config line, immediate)
M23            → M19 + multi-target easy subset (need target selection step)
M24            → M22 + M23 combined
M25            → distillation warm-start (later)
```

## Pass threshold clarification

From `protenix_reward_server.py`:
```python
PASS_PTM, PASS_IP = 0.80, 0.70
def _passes(conf): return conf["ptm"] > 0.80 and _continuous_ip(conf) > 0.70
```
`_continuous_ip` = `abag_iptm` when available (multi-chain antibody score), else `iptm`.

The reward function (`reward_from_confidence`) uses only `_continuous_ip` — **ptm is only used for the pass/fail logging metric, not for gradient**. M22 fixes this by adding ptm to the reward.
