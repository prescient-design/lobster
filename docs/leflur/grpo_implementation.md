# GRPO fine-tuning of the LeFlur binder policy

How we RL-fine-tune the LeFlur 3Di binder generator with GRPO. This is meant to be
read end-to-end and reproduced. It covers the outer loop, **how the likelihood is
computed for all three tracks** (sequence, structure/LG, 3Di), the reward terms
(including the 3Di-interface and sequence-based rewards), and — importantly —
**every place we deviate from the standard GRPO/PPO/diffusion-RL literature**.

Code lives in `src/lobster/rl_training/`:

| File | Role |
|---|---|
| `_leflur_grpo_trainer.py` | outer loop, reward assembly, advantages, PPO update |
| `_dfm_logprob.py` | differentiable per-step DFM transition log-prob (the likelihood kernel) |
| `rewards/_distribution_reward.py` | interface AA + 3Di histogram-distance reward |
| `rewards/_clash_reward.py` | smooth steric-clash + interface-contact geometry reward |
| `rewards/_diversity_reward.py` | within-group k-mer Jaccard novelty (seq + 3Di) |
| `rewards/_protenix_reward.py` | (optional) Protenix confidence reward |
| `model/leflur/..._lightning_module.py` | `rollout_with_logprobs`, `logprob_over_trajectory`, `captured_logprob_per_step`, `kl_over_trajectory` |

---

## 1. What the policy is

LeFlur generates a **binder** conditioned on a fixed **antigen/target** using
**discrete flow matching (DFM)** with an *absorbing (mask) prior*. Generation is a
sequence of denoising steps that progressively unmask tokens. There are **three
discrete tracks**, each with its own vocabulary and its own mask token:

- `sequence_tokens` — amino acids,
- `structure_tokens` — the LG (latent-generator) backbone-structure codebook,
- `tri_tokens` — 3Di structural-alphabet tokens.

At each denoising step `s` the model predicts, per track, "clean-token" logits
`x1_logits (B, L, S)` from the current partially-masked state `xt`. The DFM
transition kernel turns those logits into a per-position categorical over the next
state `x_{t+dt}`, and the sampler draws from it. **The policy log-probability of a
rollout is the sum, over steps and over tracks, of the log-prob of each sampled
transition** (Section 3).

For RL we only put advantage on the tracks the reward actually depends on. For the
current geometry/interface arm that is `tracks = [structure_tokens, tri_tokens]`
(both rewards are functions of structure); a sequence-reward arm would include
`sequence_tokens`. This is a config knob, not a code change.

---

## 2. The outer loop

One GRPO step operates on **one target** and a **group** of `G` rollouts for that
target (group-relative baseline). Pseudocode:

```
for step in range(num_steps):
    spec        = next_target()                 # round-robin over the target CSV
    L_binder    = sample_binder_length(spec)     # per-target length distribution
    comp        = build_static_conditioning(spec, L_binder)   # antigen, epitope, masks

    # 1. Rollout G designs (NO grad — sampling only)
    with torch.no_grad():
        traj = model.rollout_with_logprobs(**gen_kwargs(comp, group_size=G))
    seqs     = decode_binder_seqs(traj)          # G amino-acid strings
    tri_seqs = decode_binder_tri(traj)           # G 3Di strings

    # 2. Reward: one scalar per design
    rewards, metrics = compute_rewards(spec.target_id, seqs, tri_seqs, traj, comp)  # (G,)

    # 3. Group-relative advantages
    adv, std = advantages(rewards)               # (G,)
    if std < adv_std_floor: continue             # flat group -> no signal, skip

    # 4. OLD (behaviour) per-step log-prob, snapshotted under no_grad
    old_lp_per_step = model.captured_logprob_per_step(traj, tracks)   # (n_steps, G)

    # 5. mu PPO inner updates, each on a RANDOM subset of steps
    for _ in range(mu):
        subset = sample_step_subset(n_steps)     # k = steps_per_update random steps
        new_lp = model.logprob_over_trajectory(traj, tracks, step_indices=subset,
                                               grad_checkpoint=True)   # (G,), differentiable
        old_lp = old_lp_per_step[subset].sum(0)  # (G,)
        ratio  = exp(new_lp - old_lp)
        loss   = -min(ratio*adv, clip(ratio, 1±eps)*adv).mean() + beta*KL
        loss.backward(); clip_grad; optimizer.step()
```

The reward is used **only as a scalar** (it weights the log-prob via the
advantage). It is **never back-propagated**, so reward code can be plain numpy and
need only be numerically well-behaved — not autograd-differentiable. What *is*
differentiable is `new_lp` (Section 3).

Source: `_grpo_step` (`_leflur_grpo_trainer.py:904`).

---

## 3. The likelihood — how we compute log π(rollout) for all three tracks

This is the heart of the method and where most of our deviations live.

### 3.1 The DFM step distribution (per track, per step)

We reconstruct the exact absorbing-state DFM transition distribution used by the
sampler, but **out-of-place and differentiable in the logits**. For predicted
clean-token logits `logits (B,L,S)` at time `t` with step `dt`, current state
`xt`, mask index `m`:

```python
# _dfm_logprob.py:dfm_step_prob  (masked-prior branch)
x1_prob   = softmax(mask_out_col_m(logits) / temperature)          # P(clean token)
xt_is_mask = (xt == m)                                             # (B,L,1)
final_gate = (t + dt < 1)                                          # no remask on last step

step_prob = ( dt * x1_prob * ((1 + stochasticity*t)/(1 - t)) * xt_is_mask       # unmask
            + dt * (1 - xt_is_mask) * onehot(m) * stochasticity * final_gate )  # remask
step_prob = regularize(step_prob, xt)   # clamp[0,1], move leftover mass to the xt column
```

The per-step **transition log-prob over generated positions** is then

```python
# _dfm_logprob.py:dfm_step_logprob
row_sum = step_prob.sum(-1).clamp_min(eps)                 # (B,L)
chosen  = step_prob.gather(-1, x_next)                     # prob of the token actually drawn
logp    = log(chosen) - log(row_sum)                       # row-normalized categorical
return (logp * gen_mask).sum(-1)                           # (B,) — sum over generated positions
```

`gen_mask` restricts the sum to positions this design actually generates (fixed /
inpainted antigen positions contribute nothing).

**Why a reimplementation and not a call-through** (deviation): the upstream
bionemo `DiscreteFlowMatcher.step` (a) mutates the caller's logits in place
(`x_1_pred_logits[..., mask_index] = -1e9`), which breaks autograd and corrupts
the tensor we still need for the reference pass, and (b) regularizes with in-place
`scatter_`. We reproduce the `use_mask=True` branch **byte-for-byte in math** but
fully out-of-place. Only the masked-prior branch is implemented (all LeFlur tracks
use a `DiscreteMaskedPrior`); the uniform branch raises `NotImplementedError`.

**Row-normalization** (deviation): `torch.multinomial` treats its input as
*unnormalized weights* and divides by the row sum. At the final step the
`1/(1-t)` factor can push a masked row's mass above 1, so we divide by `row_sum`
to match the sampler's *actual* categorical exactly. In the common case each row
already sums to 1 and this is a no-op.

### 3.2 Summing across tracks and steps

`logprob_over_trajectory` (`..._lightning_module.py:1808`) reproduces the biased
per-track logits at each requested step (`_recompute_biased_step_logits` re-applies
CFG, logit biases, etc. in the same order the sampler used) and **accumulates the
per-step, per-track log-probs into a single `(B,)` policy log-prob**:

```python
total = 0
for step in requested_steps:
    biased = recompute_biased_step_logits(xt, t_seq, t_struc, step)   # dict[track] -> (B,L,S)
    for track in tracks:                      # e.g. [structure_tokens, tri_tokens]
        total += dfm_step_logprob(biased[track], t, dt, xt, x_next,
                                  gen_mask[track], mask_index[track],
                                  temperature, stochasticity)          # (B,)
return total                                                            # (B,)
```

So **log π(rollout) = Σ_steps Σ_tracks (per-position transition log-prob)**. A
track held clean at a given step (e.g. 3Di already fully resolved) simply
contributes nothing there. The identical machinery serves all three tracks — the
only difference between "3Di-interface reward" and "sequence reward" arms is
(a) which `tracks` receive advantage and (b) the reward function; the likelihood
computation is track-agnostic.

### 3.3 How many time-steps we evaluate the likelihood over

The rollout uses `rollout_nsteps = 40` denoising steps and we capture the
behaviour log-prob at **all 40** (inline, Section 3.4). But the **differentiable**
`new_lp` in each PPO inner update is computed over a **random subset of only
`k = steps_per_update` steps** (currently `k = 3`):

```python
def sample_step_subset(n_steps):
    k = steps_per_update
    if k <= 0 or k >= n_steps: return range(n_steps)
    return sorted(rng.sample(range(n_steps), k))
```

and the matching old log-prob is `old_lp = old_lp_per_step[subset].sum(0)`. This is
**diffu-GRPO step-subsampling** (deviation from vanilla GRPO): the log-prob is
additive over steps, so any subset gives an unbiased-in-expectation stochastic
estimate of the trajectory ratio, at a fraction of the compute/memory of the full
40-step backward. So: *behaviour* likelihood is exact over 40 steps; the *policy*
likelihood we differentiate is a random 3-of-40 estimate per µ-iteration.

### 3.4 Old (behaviour) log-prob: faithful inline capture

Standard PPO recomputes the behaviour log-prob with a second forward. We instead
**store the exact log-prob the sampler drew from, inline during the rollout**
(`captured_logprob_per_step`, `..._lightning_module.py:1903`), summed over tracks
to `(n_steps, G)`. Two benefits (deviation): zero extra forwards, and it uses the
*exact* biased logits the sampler used, so no sampler/recompute drift is possible.
As a live check, with inline `old_lp` the **first inner-iteration ratio must be
≈1** (`ppo/ratio_init`); a departure flags a recompute mismatch. A recompute
fallback (`capture_old_lp_inline: false`) exists for older rollouts.

### 3.5 Gradient checkpointing

With `grad_checkpoint: true`, each step's forward+log-prob is wrapped in
`torch.utils.checkpoint` (non-reentrant), so peak update memory is ~`O(group_size)`
instead of `O(group_size × steps_per_update)`. This is what lets `steps_per_update
> 1` fit at `group_size = 64`.

---

## 4. The rewards

`compute_rewards` (`_leflur_grpo_trainer.py:715`) sums up to **six** independent
per-design terms; the final reward is their sum. Every term is **inert (exactly 0,
no metrics) at weight 0**, so arms are pure config. Terms that read the generated
backbone share **one** decode (`gen_bb`), decoded once when any of them is active.

```
reward_i = confidence + structure_sc + seq_diversity + struct_diversity
                      + interface_distribution + clash_contact
```

The current Protenix-free geometry arm turns on only two:
`w_3di_dist = 1.0` and `w_clash_contact = 1.0` (all confidence/structure/diversity
weights 0). With all confidence weights 0 the Protenix oracle is **skipped
entirely** (`need_conf` false ⇒ `confs = [None]*G`), so the reward is fully local,
dense, and cheap (deviation from confidence-oracle-based binder RL: we replace the
expensive folding oracle with dense interface-distribution + geometry shaping).

### 4.1 Interface-distribution reward (3Di **and** sequence based)

File: `rewards/_distribution_reward.py`. This term rewards a design whose
**interface composition** matches a reference distribution measured on passing
Proteina-Complexa binders. It has an AA (sequence) sub-term and a 3Di sub-term —
this is where the *sequence-based* and *3Di-based* rewards both live:

1. **Interface residues.** Flag each binder residue whose min cross-chain Cα–Cα
   distance to the antigen is `< 8 Å` (`interface_binder_flags`, `IFACE_THRESH=8.0`).
2. **Histograms over interface residues.**
   - `aa_interface_hist(seq, flags)` → 20-bin amino-acid histogram (**sequence**),
   - `tridi_interface_hist(states, flags)` → 3Di-state histogram (**3Di**); 3Di
     states are computed from the decoded binder backbone.
3. **Distance to reference.** For each present histogram `h`, compute
   `D(h, h_ref)` with `D =` total variation (default, `tv`) or Jensen–Shannon
   (`js`), and turn it into a reward:

```python
# distribution_terms  (schematic)
term = w_aa_dist  * (1 - D(h_aa,  ref_aa))      # sequence-based
     + w_3di_dist * (1 - D(h_3di, ref_3di))     # 3Di-based
```

`D ∈ [0,1]` for TV, so each sub-term is a bounded closeness score. The reference
histograms are per-target (`grpo_dist_reference_complexa38.json`).

**Interface-collapse guardrail** (deviation): a degenerate escape hatch is to
shrink the interface until the histogram is trivially matchable. If a design has
fewer than `dist_min_iface = 4` interface residues, its **entire** distribution
reward is set to `dist_iface_penalty = -1.0` (a hard negative), which *repels* the
collapse mode rather than softly zeroing it.

Design decision (from the D0 analysis): against a **Complexa** reference the 3Di
term is co-equal and independent of the AA term (unlike against a *native*
reference, where 3Di is redundant once AA is controlled), which is why we can run a
3Di-only arm (`w_aa_dist=0, w_3di_dist=1`) and still get real signal.

### 4.2 Clash + interface-contact geometry reward

File: `rewards/_clash_reward.py`. The distribution reward matches interface
*composition* but is blind to 3-D geometry, so the policy can score well while
producing **clashing** backbones — or dodge clashes by **floating** out of
contact. This single smooth `[0,1]` term penalizes both. Everything is `C¹`-smooth
(no hard cutoffs) because the user explicitly required a well-behaved shaping term.

```python
term = clash_score * contact_score
```

- **Clash score** `∈ (0,1]` — soft-core over non-bonded heavy-atom pairs
  (binder×antigen + non-local binder×binder, backbone N/CA/C + virtual Cβ):
  ```
  p(d)    = 0.5*(1 - tanh((d - d_clash)/clash_soft))    # C¹, no cutoff
  clash   = exp(-Σ p(d) / clash_scale)
  ```
- **Contact score** `∈ [0,1]` — an **asymmetric raised-cosine band on the binder
  interface fraction** (the fraction of binder residues within `contact_d0=8 Å`
  of the antigen, via a smooth sigmoid soft-count):
  ```
  soft_n_iface = Σ_i sigmoid((contact_d0 - d_min_i)/contact_soft)
  iface_frac   = soft_n_iface / n_binder
  contact      = band(iface_frac; lo=0.1, peak=0.2, hi=0.4)
  ```
  `band(·)` is 0 outside `[lo, hi]`, peaks at 1.0 at `peak`, with **zero slope at
  all three knots**. It is calibrated to native passing-Complexa binders
  (`iface_frac` median ≈ 0.18, bulk 0.09–0.33) and scores 0 at the ≈0.63 fraction
  seen in the clashing diverged GRPO rollouts. So contact penalizes **no contact**
  (floating, `frac < 0.1`) *and* an **over-large / interpenetrating interface**
  (`frac > 0.4`).

Multiplicative coupling means a clashy-but-contacting or clean-but-floating design
both score ≈0; only a clean backbone in the native interface band scores high.
Diagnostics logged: `clash/clash_score_mean`, `clash/contact_score_mean`,
`clash/E_clash_mean`, `clash/soft_n_iface_mean`, `clash/iface_frac_mean`.

### 4.3 Diversity (sequence + 3Di) and confidence

- **Diversity** (`rewards/_diversity_reward.py`): mean within-group pairwise
  k-mer **Jaccard novelty** on AA strings (`w_seq_diversity`) and on 3Di strings
  (`w_struct_diversity`). Rewards a design for differing from its group-mates —
  an anti-mode-collapse term. Off in the current arm (the dist reward
  self-regularizes).
- **Confidence** (`rewards/_protenix_reward.py`): optional Protenix ipTM / pTM /
  abag-ipTM / pLDDT / PAE combo. Off in the Protenix-free arms; when any weight is
  >0 a worker pool folds the group and returns confidences (and coords, enabling
  the structure self-consistency term).

---

## 5. Advantages — Dr.GRPO

```python
# _advantages
std      = rewards.std(unbiased=False)
centered = rewards - rewards.mean()
adv      = centered / (std + eps)   if normalize_advantage else centered
```

We run with `normalize_advantage = false` — **Dr.GRPO**: advantages are
**mean-centered only, no `1/std`** (deviation from vanilla GRPO). This removes the
std-amplification that, in our earlier M-runs, let the policy chase lucky
high-variance groups. `std` is still computed and used for the **flat-group skip**
(`std < adv_std_floor` ⇒ no learnable signal ⇒ skip the update) and for logging.

---

## 6. The PPO update

Standard clipped surrogate, per µ-iteration, on the sampled step subset:

```python
ratio = exp(new_lp - old_lp)                                  # (G,)
loss  = -min(ratio*adv, clamp(ratio, 1-eps, 1+eps)*adv).mean() + beta*kl_mean
```

- **KL off** (`beta = 0.0`): we deliberately drop the KL-to-reference term for
  now. When `beta == 0` we **skip the KL forward entirely** (self + frozen ref),
  ~halving backward-graph memory and compute; `ppo/kl_mean` logs 0. KL is
  available (`kl_over_trajectory` uses a closed-form categorical KL over the
  vocabulary simplex, lower-variance than a single-sample estimate) but unused.
- `eps_clip = 0.2`, `mu = 2` inner iterations, `grad_clip = 1.0`, `lr = 1e-5`
  (the sole confirmed lever — `1e-6` freezes, `1e-4` diverges).

Diagnostics: `ppo/ratio_init` (≈1 consistency check), `ppo/clip_frac`,
`ppo/dlp_mean` (common-mode drift), `ppo/dlp_adv_corr` (advantage-differential
faithfulness — the useful signal separating winners from losers).

---

## 7. Deviations from standard literature — at a glance

| Deviation | What | Why |
|---|---|---|
| Out-of-place differentiable DFM kernel | reimplement bionemo `step` fully out-of-place | upstream mutates logits in place / uses in-place `scatter_` — breaks autograd & the ref pass |
| Row-normalized transition log-prob | divide `step_prob` by its row sum | matches `torch.multinomial`'s weight semantics; unbiased at the final step where mass can exceed 1 |
| diffu-GRPO step-subsampling | differentiate `new_lp` over `k=3` random of 40 steps | additive log-prob ⇒ unbiased estimate at a fraction of compute/memory |
| Inline faithful `old_lp` capture | store the sampler's exact per-step log-prob | zero extra forwards; no sampler/recompute drift; `ratio_init≈1` live check |
| Dr.GRPO advantages | mean-center only, no `1/std` | removes std-amplification (chasing lucky groups) |
| KL off | `beta=0`, KL forward skipped | ~halves update memory; enables larger `steps_per_update` |
| Gradient checkpointing per step | checkpoint each step forward+logprob | peak mem `O(G)` not `O(G·k)`; fits `G=64` |
| Protenix-free dense shaping | interface-distribution + clash/contact instead of a folding oracle | cheap, dense, local reward; no worker pool/queue |
| Interface-collapse hard penalty | `< dist_min_iface` residues ⇒ reward `= -1.0` | repels (not softly zeroes) the shrink-the-interface reward hack |
| Multiplicative smooth geometry term | `clash · contact`, both `C¹`, contact a raised-cosine band | one bounded well-behaved term that penalizes clash, floating, AND over-large interfaces |

---

## 8. Reproduce

Combined geometry arm config:
`experiment/rl_leflur_binder_grpo_dist_3di_clash_noptx_stab.yaml`. Key settings:

```yaml
model.ckpt_path: leflur-binder-3di        # warm start
grpo:
  group_size: 64
  rollout_nsteps: 40
  steps_per_update: 3        # diffu-GRPO step subset size (k)
  grad_checkpoint: true
  mu: 2
  beta: 0.0                  # KL off
  eps_clip: 0.2
  lr: 1.0e-5
  normalize_advantage: false # Dr.GRPO
  adv_std_floor: 1.0e-3      # flat-group skip
  # rewards
  w_3di_dist: 1.0            # 3Di interface-distribution reward (sequence: w_aa_dist)
  w_aa_dist: 0.0
  dist_metric: tv
  dist_min_iface: 4
  dist_iface_penalty: -1.0
  w_clash_contact: 1.0       # clash * contact geometry reward
  frac_lo: 0.1
  frac_peak: 0.2
  frac_hi: 0.4
  tracks: [structure_tokens, tri_tokens]   # which tracks get advantage
  capture_old_lp_inline: true
  # confidence + diversity all 0 -> Protenix skipped entirely
```

Launch (Protenix-free ⇒ `N_WORKERS=0` submits the policy job only):

```bash
OUTBASE=/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_dist_3di_clash_noptx_stab \
N_WORKERS=0 \
CONFIG=experiment/rl_leflur_binder_grpo_dist_3di_clash_noptx_stab \
  bash scripts/launch_leflur_grpo.slurm
```

To swap in a **sequence-based** reward instead of / alongside the 3Di one: set
`w_aa_dist > 0` and add `sequence_tokens` to `tracks` so the sequence track
receives advantage. The likelihood machinery (Section 3) is identical across
tracks; only the reward and the advantaged `tracks` change.
