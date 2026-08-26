# LeFlur GRPO Training Framework

A reference document for the GRPO RL fine-tuning setup for `leflur-binder-3di`.

---

## 1. The Big Picture

We're fine-tuning a **discrete flow matching (DFM)** protein binder design model using
**Group Relative Policy Optimization (GRPO)**, a variant of policy gradient RL.

**Goal**: increase the fraction of generated binders that pass a docking quality threshold
(ipTM > 0.5 when co-folded against the target by Protenix).

**Why it's hard**: the reward (Protenix ipTM) is slow (~50 s/design), non-differentiable,
and noisy (seed std ~0.065). We can't backpropagate through it. Instead we use policy gradient:
sample many sequences, score them, push up the probability of good ones.

---

## 2. What the Model Actually Generates

LeFlur is a **joint sequence + structure discrete flow matching model**. It generates:
- `sequence_tokens` — amino acid identities (20 AAs + mask token)
- `structure_tokens` — LG codec backbone tokens (4096-way vocab + mask)
- `tri_tokens` — 3Di structural alphabet (21-way vocab + mask)

The generation process is a **Markov chain** that starts from a fully-masked binder
(all positions = mask token) and iteratively uncovers tokens over `T` denoising steps:

```
t=0   [MASK MASK MASK ... MASK MASK MASK]   ← all binder positions masked
t=1   [MASK  A   MASK ... MASK  G   MASK]   ← some positions unmasked
t=2   [ A    A    E  ...  MASK  G    W  ]   ← more unmasked, some remasked
...
t=T   [ A    A    E  ...   F    G    W  ]   ← final sequence (scored by Protenix)
```

The target (antigen) residues are PINNED throughout — only the binder positions are
generated. The model's encoder attends jointly over target + noisy binder context
to predict the x₁ (clean) probability distribution at each step.

---

## 3. The GRPO Training Loop

One GRPO step:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTER GRPO STEP (one per ~2-5 min wall-clock)                          │
│                                                                         │
│  1. ROLLOUT (no grad)                                                   │
│     ┌──────────────────────────────────────────────────────────┐        │
│     │  Run generate_sample G=32 times (batched) for target     │        │
│     │  T=40 denoising steps, capturing:                        │        │
│     │    • xt at every step (the noisy state entering step t)  │        │
│     │    • x_next at every step (state after sampling)         │        │
│     │    • logprob(xt → x_next) under OLD policy (inline)      │        │
│     │    • final sequence (for Protenix scoring)               │        │
│     └──────────────────────────────────────────────────────────┘        │
│                                                                         │
│  2. SCORE (blocking on Protenix, ~2-5 min)                              │
│     ┌──────────────────────────────────────────────────────────┐        │
│     │  For each of G=32 sequences:                             │        │
│     │    r_dock  = clip(ipTM, [0,1])          ← Protenix       │        │
│     │    r_div   = 1 - max_kmer_Jaccard_to_peers ← novelty     │        │
│     │    r_degen = -hinge(max_aa_frac, entropy) ← anti-poly    │        │
│     │    r_gpde  = gpde_weight * dock_std * z_gpde ← struct    │        │
│     │    r_i     = r_dock + diversity_weight*r_div              │        │
│     │             + degeneracy_weight*r_degen + r_gpde          │        │
│     └──────────────────────────────────────────────────────────┘        │
│                                                                         │
│  3. ADVANTAGES                                                          │
│     ┌──────────────────────────────────────────────────────────┐        │
│     │  A_i = (r_i - mean(r)) / (std(r) + ε)                   │        │
│     │  [skip if std(r) < adv_std_floor — pure-noise group]     │        │
│     └──────────────────────────────────────────────────────────┘        │
│                                                                         │
│  4. PPO INNER LOOP (mu=2 iterations, with grad)                         │
│     ┌──────────────────────────────────────────────────────────┐        │
│     │  For each inner iteration:                               │        │
│     │    a. Sample 2 random step indices from {0,...,39}       │        │
│     │    b. RECOMPUTE logprob under NEW policy for those steps  │        │
│     │       new_lp = Σ_{t∈subset} log π_new(x_next_t | xt)    │        │
│     │    c. old_lp = Σ_{t∈subset} old_logprob[t]  (captured)  │        │
│     │    d. ratio = exp(new_lp - old_lp)   ← importance ratio  │        │
│     │    e. pg_loss = -min(ratio*A, clip(ratio,0.8,1.2)*A)     │        │
│     │    f. optimizer.step() on encoder parameters only         │        │
│     └──────────────────────────────────────────────────────────┘        │
│                                                                         │
│  → repeat for next GRPO step                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Denoising Step in Detail (Cause B)

This is the most important thing to understand about **why gradient is sparse**.

At each denoising step `t`, for each binder position `l`, the position is in one of two states:

```
State A: xt[l] = MASK TOKEN          State B: xt[l] = some amino acid
         (position not yet committed)          (position already committed)
```

The transition kernel `step_prob[l, :]` (the probability distribution over next tokens) is:

### State A — masked position (can unmask or stay masked)

```
step_prob[l, c] ∝ softmax(model_logits[l, c] / T)  × dt×(1+s×t)/(1-t)
                                                      ↑ unmasking rate
step_prob[l, MASK] = 1 - Σ_{c≠MASK} step_prob[l, c]  (stay probability)
```

→ **The probabilities DEPEND on the model's logits. Gradient flows here.**

### State B — already resolved position (can stay or get remasked)

```
step_prob[l, xt[l]] = 1 - dt × stochasticity       ← stay probability
step_prob[l, MASK]  =     dt × stochasticity        ← remask probability
step_prob[l, other] = 0
```

→ **These are CONSTANTS — they don't depend on model logits. Zero gradient.**

### What this means for our setup

With `stochasticity_seq=20` and `dt=1/40=0.025`:
```
remask probability per resolved position = 0.025 × 20 = 0.50
```

Every step, **50% of already-committed residues get remasked** and need to be regenerated.
This generates a lot of transitions, but the remasking itself is gradient-free.

**The only gradient-carrying events are: masked → amino acid transitions.**

```
Step t=0:  [M M M M M M M M M M M M ...]   ← all 80 positions masked
              ↓   ↓   ↓                       ~2 positions unmask (rate ≈ dt at t≈0)
Step t=1:  [M A M M V M M M M M M M ...]   ← 2 gradient-carrying transitions
            ↓ ↓ ↓ ↓ ↓ ↓                      several unmask, several re-mask
Step t=2:  [G A E M M M W M M M M M ...]   ← 5 gradient-carrying unmask events
...                                           (many re-masked positions also unmask)
Step t=T:  [G A E F R K W D L T N P ...]   ← final committed sequence
```

With `steps_per_update=2`, each PPO inner iteration gets gradient from roughly
**10-30 masked→unmasked positions** (the positions that were masked at the 2 chosen steps).

### Why high stochasticity hurts gradient signal

High stochasticity creates a lot of "churning" — positions are committed then remasked then
committed again. This adds trajectory log-prob variance (noise) without adding information
about binding quality. The model is getting gradient signals like "you committed position 47
to Ala at step 12, but then remasked it at step 13, and re-committed to Glu at step 14 —
this trajectory had high reward, so push up the probability of all those choices."

Most of those choices were random noise, not the reason the design passed.

**Lower stochasticity (stoch_seq=1-5)** would mean: once a position is committed, it stays
committed. The gradient would then only come from the INITIAL commitment, which is more
likely to carry structural information.

---

## 5. Where the Gradient Actually Comes From

The GRPO policy gradient is:

```
∇_θ J = E[ A_i × Σ_{t=0}^{T-1} Σ_{l ∈ binder, xt[l]=MASK} ∇_θ log softmax(model(xt)[l, x_next[l]]) ]
                  ↑                ↑
                  sum over steps   only masked→unmasked positions
```

Concretely, the gradient at step `t` for position `l` (when `xt[l] = MASK`) is:

```
∇_θ log p = ∇_θ [ log softmax(logits[l, x_next[l]]) ]
           = ∇_θ [ logits[l, x_next[l]] - log Σ_c exp(logits[l, c]) ]
           ≈ e_{x_next[l]} - softmax(logits[l])   (one-hot minus softmax)
```

The GRPO surrogate then SCALES this gradient by the advantage `A_i`:
- If design `i` passed (high `A_i > 0`): push UP the log-probability of `x_next[l]` (the token that was chosen)
- If design `i` failed (low `A_i < 0`): push DOWN the log-probability of `x_next[l]`

**The fundamental assumption**: the tokens chosen in gradient-carrying steps of high-reward
trajectories are systematically better for binding than those in low-reward trajectories.

This assumption is weak when:
1. Only 15-19% of steps have any design passing (most steps → random noise gradient)
2. The passing design is usually just 1 of 32 (stochastic win, not systematic superiority)
3. With 40 denoising steps, the model can't refine sequences carefully enough for the
   commitment decisions to reliably encode binding information

---

## 6. The Pass Rate Reality (corrected)

The statement "0% pass rate every step" was wrong. The actual distribution over M17 last 100 steps:

```
pass=0.000  (0/32 pass)  →  ~81% of steps
pass=0.031  (1/32 pass)  →  ~19% of steps
pass=0.062  (2/32 pass)  →  ~0.4% of steps
Mean pass rate: ~0.7%  (vs production 7.5% at 200 steps)
```

So ~85% of GRPO steps see ALL-FAIL groups (zero real positive advantage, pure noise gradient).
The ~15% of steps with 1 passer DO provide a signal: that 1 design gets large positive
advantage, the other 31 get small negative advantages. But:

1. The passing sequence varies step-to-step (different random wins each time)
2. What made it pass is largely stochastic (which tokens happened to be committed in which order)
3. 40 steps isn't enough to generate sequences that reliably pass — it's the difference between
   a rough draft and a polished sequence

**Root cause**: `rollout_nsteps=40` generates sequences at ~1/10 the quality of production
(200 steps). The model needs fine-grained resolution to generate sequences that consistently pass.

---

## 7. Why Cause E Is Actually Favorable (Discrete Flow + Seq-Only Advantage)

The literature warning from ScoRe-Flow (arXiv:2604.10962) — "deterministic ODE flows lack
tractable likelihoods" — does **NOT apply to us**. Here's why:

### We use discrete (absorbing-state) flow matching, not continuous ODE

LeFlur's sequence track uses a **discrete absorbing-state Markov chain** (CTMC), not a
continuous ODE. This has:
- Well-defined per-step transition probabilities (the categorical distributions in `dfm_step_prob`)
- Exact log-probability computation at each step (no change-of-variables Jacobian needed)
- Differentiable log-prob in the model logits (our `dfm_step_logprob`)

This is the same formulation as MDLM/SDLM (masked discrete language models), which have been
successfully trained with RL. The continuous ODE problem doesn't apply to us.

### Seq-only advantage is a simplification, not a bug

We use `tracks=["sequence_tokens"]`, meaning the GRPO ratio only accounts for the sequence
log-prob, ignoring structure (LG) and 3Di log-probs:

```
ratio = exp(Σ_{t∈subset} [log π_new(seq_t) - log π_old(seq_t)])
```

This is a valid (if approximate) objective because:
1. Protenix scores the SEQUENCE (folded into a structure) — the sequence is the primary action
2. The structure and 3Di tracks co-evolve during rollout but are NOT directly rewarded
3. Since the encoder is shared, seq-track gradient still updates weights that influence structure
4. It's a standard simplification in diffu-GRPO (cf. DiffusionDPO, DPOK)

The risk: the encoder might be pushed toward sequences that are "easier to generate" (higher
seq log-prob) rather than sequences that fold well (higher ipTM). But this is the fundamental
tradeoff of any policy-gradient method with proxy rewards — the proxy (seq log-prob advantage)
may not perfectly align with the true objective (ipTM improvement).

---

## 8. The Distogram Proxy: How to Add Intermediate Rewards

The distogram head predicts inter-residue distance distributions. It **already knows about
folds** (intra-chain contact F1 ~0.25) but is currently **blind to the docking interface**
(inter-chain contact F1 ~0.03). However, we can still use it as an intermediate-step proxy.

### What the distogram head produces

At each denoising step `t`, from the encoder's representation of the noisy complex:

```
distogram[i, j] = predicted Cβ-Cβ distance distribution between residues i and j
                  (64-way probability vector over distance bins 2-22 Å)
```

We have access to this via the `step_callback` mechanism (line ~1298 in generate_sample):

```python
if step_callback is not None:
    step_callback(step_idx, t_struc, unmasked_x, mask)
    #                                ↑
    #                                unmasked_x["distogram_logits"] (B, L, L, 64)
```

### Option 1: Binder compactness proxy (cheap, no reference needed)

At each step `t`, compute the fraction of binder-binder residue pairs predicted to be
in contact (<8 Å). This penalizes extended/non-compact binders — the "failing extended-helix
mode" identified in memory.

```
compactness_t = mean_{i,j ∈ binder, i<j} P(dist[i,j] < 8Å | distogram_logits_t)
```

Use this as a shaping reward at intermediate steps:
```
r_step_t = λ × compactness_t
```

This doesn't require the ground-truth structure and is differentiable (at training time
through soft predictions, at reward time as a scalar).

**Implementation**: add a `step_callback` to the rollout in `_grpo_step`, accumulate
per-step distogram stats, weight into the reward. Cost: ~5% extra compute per step.

### Option 2: Interface contact proxy (needs epitope residues, which we have)

We know the epitope residues (e.g., `[32, 94, 96, 101]` for PD-1). At each step `t`:

```
iface_contact_t = mean_{i ∈ binder, j ∈ epitope} P(dist[i,j] < 8Å | distogram_logits_t)
```

This measures how well the noisy binder (at step `t`) is predicted to contact the epitope.
A design that never develops inter-chain contacts at any point in the trajectory is
probably already failing.

**The key insight**: this is computed from the model's OWN distogram predictions, not from
Protenix. It's fast (no external oracle) and available at every step. It can serve as a
**dense intermediate reward** that provides signal at steps where Protenix would say "this
isn't good yet but it's heading somewhere."

### Option 3: Step-callback captured proxy + reward mixing

The full recipe for distogram-guided GRPO:

```python
# In _grpo_step, during rollout:
step_contacts = []

def distogram_step_callback(step_idx, t, unmasked_x, mask):
    disto_logits = unmasked_x.get("distogram_logits")  # (B, L, L, 64)
    if disto_logits is not None:
        # P(distance < 8Å) ≈ sum of first ~6 distance bins
        p_contact = disto_logits[..., :6].softmax(-1).sum(-1)  # (B, L, L)
        # Mean over binder-epitope pairs
        binder_mask = comp["binder_positions"]  # (L,)
        epi_mask = comp["epitope_mask"]         # (L,)
        iface = p_contact[:, binder_mask, :][:, :, epi_mask].mean(dim=(1, 2))  # (B,)
        step_contacts.append(iface.detach())

trajectory = self.model.rollout_with_logprobs(
    **self._build_gen_kwargs(comp, cfg.group_size),
    step_callback=distogram_step_callback,
)

# Shape the reward:
if step_contacts:
    mean_iface = torch.stack(step_contacts).mean(0)  # (B,) — time-averaged iface contact
    r_proxy = cfg.disto_weight * mean_iface
else:
    r_proxy = torch.zeros(cfg.group_size)

# Add to total reward before computing advantages
rewards = rewards + r_proxy.to(self.device)
```

**Expected effect**: the proxy provides gradient signal even in steps where Protenix says
all 32 fail (pass=0.000). Designs that develop better interface contacts during denoising
get higher proxy reward, even if they don't cross the ipTM>0.5 threshold. This broadens
the reward distribution in the "failing regime."

**Caveat from memory**: the distogram head has inter-chain contact F1 ~0.03 vs native
— it's not reliable for docking interface prediction. The proxy adds signal but not
necessarily the RIGHT signal. Should be validated first with a correlation study
(proxy at step t vs final Protenix ipTM).

---

## 9. Diagnosis Summary

```
WHY GRPO CAN'T MOVE ipTM:

┌──────────────────────────────────────────────────────────────────────┐
│ PRIMARY: rollout_nsteps=40 → 85% of steps have 0 passing designs    │
│                                                                      │
│   pass rate:  ~0.7%  vs  production ~7.5%  (10× worse)              │
│   cause: 40 steps is too few for C-form schedule to generate         │
│   binding-quality sequences                                          │
│   effect: 85% of GRPO steps have pure-noise gradient                │
│           15% have 1 passer → real signal, but stochastic win        │
├──────────────────────────────────────────────────────────────────────┤
│ SECONDARY: Gradient only from masked→unmasked transitions            │
│                                                                      │
│   stoch_seq=20 → 50% remask probability per step                    │
│   → lots of positional "churning" (commit, remask, recommit)        │
│   → gradient-carrying events are drowned in remask noise            │
│   → model gets gradient signal like "you committed A at step 12,    │
│     then remasked it, then committed E at step 14 — push these up"  │
│     (the E was probably random, not the reason it passed)            │
├──────────────────────────────────────────────────────────────────────┤
│ TERTIARY: Single-target G=32 → low reward variance                  │
│                                                                      │
│   ipTM std ≈ 0.083 within group, noise ≈ 0.065                      │
│   SNR ≈ 0.82 (K=1), 1.42 (K=3) — below reliable GRPO threshold     │
│   26% of groups are pure-noise even after K=3 averaging              │
└──────────────────────────────────────────────────────────────────────┘

FIXES (ordered):

  1. rollout_nsteps: 40 → 150    → 3-5 designs pass per step → real signal
  2. stochasticity_seq: 20 → 3   → cleaner trajectories, gradient less noisy
  3. multi-target (3-5 targets)  → wider reward landscape per step
  4. distogram proxy reward       → dense per-step signal in the failing regime
  5. steps_per_update: 2 → 8     → more gradient-carrying positions per update
```

---

## 10. Key Config Values (M17/M18)

| Parameter | Current value | Notes |
|---|---|---|
| `rollout_nsteps` | 40 | **Too few** — production uses 200 |
| `group_size` | 32 | OK for throughput, small for SNR |
| `stochasticity_seq` | 20 | **Very high** — causes positional churning |
| `stochasticity_struc` | 60 | Same issue |
| `stochasticity_tri` | 80 | Same issue |
| `steps_per_update` | 2 | Very few — only 2 of 40 steps get gradient |
| `mu` | 2 | 2 inner PPO updates per GRPO step |
| `lr` | 1e-5 | Confirmed stable |
| `beta` | 0.0 | No KL regularization |
| `tracks` | `[sequence_tokens]` | Seq-only advantage — correct simplification |
| `gpde_weight` | 0.3 | Adds ~0.02 std to reward — small contribution |
| `entropy_floor` | 0.80 | Anti-degeneracy hinge |
