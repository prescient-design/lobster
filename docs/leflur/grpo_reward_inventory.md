# LeFlur GRPO — Reward Inventory

Catalog of every reward term wired into the GRPO binder trainer, plus the rewards we
have discussed/scoped but not yet built (candidates for the next set of experiments)
and the reward-shaping levers around them.

_Last updated: 2026-08-14._

The final GRPO reward is a weighted sum of the implemented terms below; each has a
`w_*` weight knob in `GRPOTrainerConfig` (`src/lobster/rl_training/_leflur_grpo_trainer.py`).
All reward modules live in `src/lobster/rl_training/rewards/` and are pure (no `trl`
dependency) so the policy side can import them without the oracle's heavy deps.

---

## A. Implemented & wired into the trainer

| # | Reward | Module | Knob(s) | What it scores | Cost | Status / verdict |
|---|--------|--------|---------|----------------|------|------------------|
| 1 | **Protenix co-folding confidence** | `_protenix_reward.py` | `w_iptm, w_ptm, w_abag_iptm, w_plddt, w_gpde, w_pae_global, w_pae_interface` | Weighted, per-metric-clipped combo of Protenix confidence metrics; also defines the pass rule (`ptm>0.8 & iptm>0.7`) | Expensive (GPU worker pool + queue) | The "real" oracle. M22 = `w_abag_iptm=1, w_ptm=0.5`. Gradients work but ipTM stays flat → reward-quality, not gradient-flow, limitation |
| 2 | **Structure self-consistency (scTM)** | `_structure_reward.py` | `w_sctm_binder, w_sctm_complex` | Kabsch + TM-score of the LeFlur-generated backbone vs the Protenix-refold backbone (binder + whole complex) | Rides Protenix refold (needs coords back from worker) | Implemented. Q1 study: gen-vs-Protenix RMSD sharply separates pass/fail (complex ~4.5 vs ~14 Å) → strong self-consistency signal |
| 3 | **Within-group diversity / anti-degeneracy** | `_diversity_reward.py` | `w_seq_diversity, w_struct_diversity` | Within-group k-mer-Jaccard novelty on the AA sequence + the 3Di token string; plus max-AA / entropy-floor hinges (M14/M15 lineage) | Cheap (Protenix-free) | Kills poly-Ser / poly-Ala / distributed-collapse modes. Gen-time `sequence_diversity_penalty=2` is already a shipped default |
| 4 | **Interface + whole-binder distribution distance** | `_distribution_reward.py` | `w_aa_dist, w_3di_dist, dist_binder_frac (α)`, `dist_iface_penalty`, `dist_min_iface`, `dist_metric` | TV/JS distance of a design's AA + 3Di histograms to a per-target Complexa reference; α blends interface (0) ↔ whole-binder (1); interface-collapse guardrail | Cheap (Protenix-free) | Validated Protenix-free lever; **peaks ~s125, diverges by s425**. `aa2x` (w_aa=2) + `accum10` arms currently running |
| 5 | **Clash + interface-contact geometry** | `_clash_reward.py` | `w_clash_contact` (+ `clash_d_clash, clash_soft, clash_scale, contact_d0, contact_soft, frac_lo, frac_peak, frac_hi, clash_seq_sep, clash_include_cb`) | Soft-core steric clash over backbone+Cβ atom pairs × soft-count of contacting residues, banded to the native interface-fraction window | Cheap (Protenix-free) | Validated: de-clashes (E_clash 66→15) + pulls iface_frac toward native; fixes the interpenetration/gaming mode the pure-dist arm exhibited |
| 6 | **3DZD interface shape-complementarity (SC)** | `_shape_reward.py` + `_shape_reward_pool.py` | `w_shape` | Order-20 3D-Zernike SC (raw-Pearson) of ΔSASA interface contact patches; full-atom via LigandMPNN-repack worker pool (backbone-only variant also available) | Medium (SASA per design; rides the shared CPU repack pool) | Validated discriminator (AUROC 0.730). **Real but small, in-domain-only** lever: best on HB s125 (11.5% vs 9.1% base); no gain on Complexa (base already at native SC) |
| 7 | **All-atom side-chain clash (SC-clash)** | `_sc_clash_reward.py` | `w_sc_clash` (+ `sc_clash_scale`) | Soft-core Bondi-VDW clash over the **full packed side-chain cloud**: whole-binder self-clash (`|i−j|>seq_sep`) + binder↔antigen clash over all binder atoms; `reward = exp(−E_clash_total/scale) ∈ (0,1]` | Cheap-ish (rides the shared CPU repack pool) | **Biophysically grounded hard constraint**, not a correlation-gated proxy — overlapping VDW spheres are forbidden regardless of pass-rate correlation. Reward over the **whole binder**; interface-restricted clash tracked as a diagnostic only. Default `w_sc_clash=0` (opt-in) |
| 8 | **LigandMPNN amino-acid recovery (AAR)** | `_aar_reward.py` | `w_aar` | Whole-binder AAR = mean over binder residues of ProteinMPNN (`v_48_020`) self-redesign agreement on the design's own backbone; `reward = aar ∈ [0,1]` | Cheap-ish (rides the shared CPU repack pool) | Opt-in with a **documented caveat, not gated out**: offline whole-binder AAR is **anti-predictive** of the Protenix pass (AUROC ≈ 0.29–0.34; high MPNN agreement marks generic/degenerate seqs). Reward over the whole binder; `aar_iface`/`c_mpnn`/`c_mpnn_iface` tracked as diagnostics. Default `w_aar=0` |

**Shared CPU repack pool (terms 6–8) — scale by adding CPU workers.** All three
full-atom terms — SC shape (#6), SC-clash (#7), and AAR (#8) — are computed from the
**same** LigandMPNN side-chain repack, served by one shared-filesystem worker queue
(`_shape_reward_pool.ShapeRewardClient` ↔ the repack server). When more than one of the
three weights is on, the trainer sends the group **once** with the union metric-set
(`want=("sc","clash","aar")` canonicalized) and each design is packed a **single** time
regardless of how many of the three terms are active (`_repack_terms_for_group`); the
SC-only case still routes through the original `_shape_terms_for_group` byte-identically.
The packer (3 denoising steps × 4 von-Mises samples) is a small net that runs fine on
**CPU** (benchmarked ~0.66 s/design @ 8 threads, all outputs valid — see memory
`ligandmpnn-packer-cpu-viable`), so this pool is CPU-bound and **throughput scales
simply by adding CPU workers** (`--account=llm` for the big CPU quota) rather than
competing for scarce GPUs. Nothing about the reward *values* changes with worker count.

---

## B. Discussed / scoped but NOT yet built — candidates for the next experiments

> **Now built & wired (moved to section A):** All-atom side-chain clash (#7,
> `_sc_clash_reward.py`, `w_sc_clash`) and LigandMPNN AAR (#8, `_aar_reward.py`,
> `w_aar`) graduated from offline scoping into trainer reward terms this cycle, both
> riding the shared CPU repack pool (terms 6–8). Both default to weight 0.

| Candidate | Origin | Offline signal | Recommendation |
|-----------|--------|----------------|----------------|
| **MPNN seq↔struct consistency (`C_mpnn`)** | Part 3 of governing request; `grpo_mpnn_consistency_reward_plan.md` | Whole-binder gate **FAILED** — AUROC 0.292, anti-predictive (high MPNN confidence = generic/degenerate seq = failing mode). **Interface-scope FLIPS to weakly predictive (AUROC 0.579).** Now emitted for free as `c_mpnn`/`c_mpnn_iface` diagnostics by the wired AAR term (#8) | Only defensible as **interface-scoped**, and even then weak. If pursued, add as a small-weight interface term reading the `c_mpnn_iface` the AAR pass already returns — no new pool call |
| **Rosetta ddG interface energy** | Governing request (LigandMPNN-sidechain rewards) | **Deferred** (Rosetta slow); doc-only scope in `grpo_ligandmpnn_sidechain_rewards_plan` | Build only if clash and/or AAR clear the offline bar |
| **Per-term std-equalization (Option B)** | `aa2x` config note | — | Divide each reward term by its own group std so the AA/3Di balance stays adaptive as the 3Di spread collapses mid-training. The adaptive alternative to the static `w_aa=2` rebalance. **Strong next candidate** |
| **Compactness / Rg reward** | `binder-structural-pathologies`, `passing-binder-structural-signature` | Passing binders compact/short (Rg ~12.5 vs failing ~19); compactness predicts pass | Cheap, Protenix-free geometric term; not yet wired. **Strong next candidate** |
| **gen-vs-refold RMSD as a direct reward** | `q1-gen-vs-protenix-rmsd-agreement` | Complex CA-RMSD ~4.5 (pass) vs ~14 (fail); binder pose ~10 vs ~28 Å | Needs a fold in the loop (expensive) → scTM (#2) is the cheap wired proxy already |
| **Multi-seed Protenix averaging (K=3)** | `grpo-reward-reproducibility` | Single-seed ipTM std 0.065, heteroscedastic | A *lever* on reward #1 (already plumbed via `REWARD_SEEDS`), not a new term. Cuts noise ~1/√K at K× cost |
| **abag_iptm term** | SabDab SC vs abag_iptm study | Not emitted by this Protenix build | Knob exists (`w_abag_iptm`) but inert on this build; only usable if we switch to a build that emits abag_iptm |

---

## C. Reward-shaping levers (tuning surface, not new reward terms)

| Lever | Where | Effect |
|-------|-------|--------|
| `normalize_advantage` | grpo config | Per-group /std (== TRL `scale_rewards="group"`) |
| `accum_targets` | grpo config | Average the gradient over N targets per optimizer step (smooths jumpy reward; currently testing 4→10) |
| `shuffle_targets` | grpo config | Reshuffle target order each epoch |
| `dist_metric` | dist reward | TV vs JS |
| `dist_binder_frac (α)` | dist reward | Interface (0) ↔ whole-binder (1) blend |
| reference choice | dist reward | Complexa vs native (Complexa ref makes 3Di co-equal-independent, z(3Di)=−0.244 vs z(AA)=−0.214) |
| clash band params | clash reward | `d_clash, clash_scale, frac_lo/peak/hi` set the steric + interface-fraction window |
| `REWARD_SEEDS` | launcher env | K Protenix seeds averaged per design |

---

## Suggested next-experiment priority

1. **Option B (per-term std-equalization)** — adaptive fix for the AA-vs-3Di imbalance the static `w_aa=2` arm is testing; likely more robust than hand-tuned weights. Small addition to the existing distribution-reward plumbing.
2. **Compactness / Rg reward** — cheap, Protenix-free, and compactness is one of the most consistent pass predictors we have; nothing like it is wired yet.
3. **Interface-scoped MPNN consistency at small weight** — only to close out Part 3; offline signal is weak (0.579), so keep expectations low.
