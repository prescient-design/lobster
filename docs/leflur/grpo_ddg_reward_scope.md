# Scoping: an interface ΔΔG (binding-energy) GRPO reward

**Date:** 2026-08-26 · **Status:** scoping / not wired · companion to
[`grpo_reward_inventory.md`](grpo_reward_inventory.md) (where ddG was previously "Deferred —
Rosetta slow; doc-only scope") and the packed-potentials reward set
(`scripts/_packed_potentials.py`).

## TL;DR

- **Do we already have most of it?** Yes. `scripts/_packed_potentials.py` already computes, on the
  LigandMPNN-packed complex, the **cross-chain interface interaction energy** — `e_lj_atr`
  (fa_atr proxy), `e_lj_rep` (fa_rep proxy), `e_elec` (Coulomb w/ distance-dependent dielectric),
  H-bond / salt-bridge counts, `dsasa`, `n_buns` — each **per-binder-residue decomposed** for dense
  per-token credit. Under a **rigid** bound→unbound separation, intra-chain terms cancel exactly, so
  this cross-chain sum *is* the enthalpic core of an interface ΔΔG. `e_lj` is already our live
  primary reward.
- **What's physically missing** vs a real Rosetta interface ddG: (1) **solvation/desolvation**
  (`fa_sol`/`lk_ball`) — the term that most separates a true ΔΔG from a contact count; our `dsasa` is
  a geometric proxy, not an energy; (2) a **repacked/relaxed unbound reference** (flex_ddG relaxes
  the separated partners + backrub-averages); (3) **Rosetta-calibrated atom-typed weights** (we use a
  uniform LJ well depth `EPS=0.2` and count — don't energy-weight — H-bonds).
- **Is there a Python/fast-Rosetta implementation?** **Yes — [tmol](https://github.com/uw-ipd/tmol)
  (uw-ipd).** It is a GPU-accelerated PyTorch reimplementation of Rosetta's `beta_nov2016` energy
  with the **full term set** (`ljlk` = fa_atr/fa_rep/**fa_sol**, `lk_ball`, `elec`, `hbond`,
  `dunbrack`, backbone torsions, `cartbonded`, `ref`, `disulfide`), **differentiable** (autograd →
  gradient to coordinates), CUDA-kernel accelerated, **Apache-2.0**, pip-installable.
- **Does tmol ship a "ddG form"?** **No packaged InterfaceAnalyzer / ddG mover** — but it ships
  **every primitive to build one**, including a **GPU rotamer packer** (`tmol/pack`, simulated
  annealing, CPU+CUDA) and an **LBFGS minimizer** (`tmol/optimization`). So a Rosetta-faithful
  interface ddG (`E(bound) − E(A) − E(B)`, optionally repacking the unbound state) is a **thin
  wrapper** we write on top of tmol — ~score twice and translate one chain apart.
- **Recommendation:** adopt tmol as the ddG engine; ship the **rigid single-state ΔΔG** first
  (drop-in upgrade of `_packed_potentials` to true beta_nov2016 terms **including cross-chain
  desolvation**, same per-residue decomposition, one extra score call), validate AUROC-vs-pass on the
  existing offline harness, and only add the **repacked-unbound (flex_ddG-like)** flavor if the rigid
  form underperforms. **Skip PyRosetta** (license, CPU-bound minutes/design, non-differentiable).

---

## 1. What "interface ΔΔG of binding" means

The binding free energy of a two-body complex is
```
ΔG_bind = G(complex) − G(A_unbound) − G(B_unbound)
```
Rosetta operationalizes this two ways, both of which we can mirror:

- **InterfaceAnalyzer** (`InterfaceAnalyzerMover`): score the complex, computationally **separate**
  the two chains (translate one ~far apart), **repack** the newly-exposed interface side chains,
  re-score; `dG_separated` is the difference. Seconds/design on CPU, no backrub. This is the standard
  "interface energy" reported for de-novo binders.
- **flex_ddG**: for point-mutation ΔΔG — backrub-sample the interface, repack + minimize bound and
  unbound ensembles, average. Minutes/design. Overkill for a whole-design reward and far too slow for
  an inner-loop RL oracle.

For a GRPO reward we want the **whole-complex interface energy** (InterfaceAnalyzer-style), not a
per-mutation ΔΔG, and we want it **cheap and ideally per-residue-decomposable**.

## 2. What we already compute (the enthalpic core)

`scripts/_packed_potentials.py` runs on the side-chain-packed complex (chain A = antigen, B = binder)
and returns, restricted to the cross-chain interface:

| term | what it is | vs Rosetta |
|---|---|---|
| `e_lj_atr` / `e_lj_rep` | 12-6 LJ split at `r0=r_i+r_j`, rep soft-capped | **fa_atr / fa_rep proxy** (uniform `EPS`, Bondi radii — *not* atom-typed) |
| `e_elec` | `332·Σ qᵢqⱼ/(4d²)` on formal SC charges | **fa_elec proxy** (linear DDD, formal not partial charges) |
| `n_hb`, `n_hb_sc`, `n_saltbridge` | geometric H-bond / salt-bridge **counts** | **not** hbond *energy* (unweighted) |
| `dsasa` | `SASA(A)+SASA(B)−SASA(AB)` | geometric burial — **proxy for**, not equal to, `fa_sol` |
| `n_buns` | buried unsatisfied polar atoms | a penalty Rosetta folds into hbond/sol |
| `n_clash`, `packing` | overlaps / neighbor density | quality diagnostics |

**Key point:** every pairwise term is **per-binder-residue decomposed** (`potentials_per_residue`,
sums validated == scalars), which is exactly what feeds our **dense per-token structure-track
advantage**. A rigid ΔΔG preserves this decomposition; a repacked ΔΔG partly breaks it (the unbound
repack is not attributable to a single binder residue).

Because a **rigid** separation leaves both monomers' internal geometry unchanged, all intra-A and
intra-B energy terms cancel in `E(bound) − E(A) − E(B)`, leaving only the **cross-chain pair sum** —
i.e. *what we already compute*. So today's `e_lj + e_elec` is already a **rigid ΔΔG in proxy units**;
the upgrade is (a) real beta_nov2016 terms, (b) the missing **cross-chain desolvation**, (c)
optionally the unbound repack.

## 3. tmol assessment

Repo: `uw-ipd/tmol` (default branch `master`), Apache-2.0. README:
*"GPU-accelerated reimplementation of the Rosetta molecular modeling energy function
(`beta_nov2016_cart`) in PyTorch with custom C++/CUDA kernels … supports gradient-based
minimization, enabling ML models to incorporate biophysical scoring during training."*

- **Score terms present** (`tmol/score/*`): `ljlk` (fa_atr/fa_rep/**fa_sol**), `lk_ball`, `elec`,
  `hbond`, `dunbrack`, `backbone_torsion` (rama/omega), `cartbonded`, `genbonded`, `ref`,
  `disulfide`, `constraint`, `na_torsion`. → a **complete beta_nov2016**, notably including the
  **solvation** terms our numpy potentials lack.
- **Differentiable:** yes — `E.backward()` in the README; autograd to coordinates. If our decoded
  backbone is differentiable, ΔΔG could in principle be a *differentiable* reward (not just a scalar),
  though our current path uses rewards as scalars/per-token advantages, so this is upside, not a
  requirement.
- **Packing:** `tmol/pack` — full **rotamer packer with simulated annealing** (`_pack_rotamers.py`,
  `_simulated_annealing.py`, CPU + CUDA kernels) + rotamer builders. → tmol can produce the
  **repacked unbound state itself**, so a flex_ddG-like reward needs **no** external packer
  (independent of LigandMPNN).
- **Minimizer:** `tmol/optimization/_minimizers.py` (LBFGS/Armijo).
- **No packaged ddG/interface mover** (no `interface`/`ddg`/`bind`/`separate` module) — we write the
  bound−unbound wrapper (translate binder along an axis by ~500 Å; re-score).
- **Install / deployment:** prebuilt wheels on GitHub Releases + PyPI sdist; wheel tags select
  Python×torch×CUDA. **Our env fits an existing lane:** Python **3.12**, torch **2.8.0+cu128**,
  glibc **2.34** (≥ 2.28 required) → the `+cu128torch2.8 cp312` wheel (built `sm_75`, "also covers
  A100/L4"; PTX-forward-compatible to a10g `sm_86`). A **CPU wheel** exists for offline validation.
  HPC gotcha called out in the README: if `import tmol` throws `GLIBCXX_3.4.xx not found`, the
  system `libstdc++` is too old — fix with `TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .` (source
  build) or a newer GCC/`libstdcxx-ng`. Neither tmol nor PyRosetta is currently installed in the
  blessed venv.

## 4. Options

**A. Rigid single-state ΔΔG via tmol (recommended first).**
Pack the complex (reuse the existing LigandMPNN repack we already run for `e_lj`/SC/sc_clash), tmol-
score the bound complex, translate the binder ~500 Å, tmol-score the separated state, ΔΔG = bound −
unbound. With rigid separation this equals the **cross-chain term sum** → keeps exact per-residue
decomposition (dense per-token credit) and costs **one extra score call** over what we already pay.
Upgrade over today: real atom-typed fa_atr/fa_rep/fa_elec/hbond **plus cross-chain fa_sol/lk_ball
desolvation** — the physics our proxy omits.

**B. Repacked-unbound (InterfaceAnalyzer / flex_ddG-like).**
As A, but **repack the separated partners** (tmol's packer) before scoring the unbound state
(optionally minimize/backrub-average). More faithful (captures strain relief on unbinding), heavier
(a pack per partner per design), and only **partly** per-residue-decomposable. Add only if A's
AUROC-vs-pass underperforms `e_lj`.

**C. Pure-numpy extension (no new dependency).**
Add a Lazaridis–Karplus-style `fa_sol` desolvation term to `_packed_potentials.py` (we already have
per-atom SASA and radii). Cheapest, no deployment risk, but re-implements + re-calibrates physics
tmol already ships validated — worth it only if tmol proves un-deployable on the cluster.

## 5. Why not PyRosetta

Same energy function, but: **license** (RosettaCommons academic/commercial terms vs tmol Apache-2.0),
**CPU-bound & slow** (InterfaceAnalyzer ~seconds/design + process overhead; flex_ddG ~minutes/design)
— untenable at `G=64` rollouts — and **non-differentiable**. tmol is strictly better for an
inner-loop RL reward.

## 6. Recommended plan

1. **Deployability spike (blocking, cheap).** Install the CPU wheel in the blessed venv, confirm
   `import tmol`, score one packed Complexa complex, and reproduce a known-sign interface energy.
   (Guard against the `GLIBCXX`/wheel-lane gotchas above.) Then confirm the `+cu128torch2.8 cp312`
   GPU wheel imports on an a10g. **Confirm-before-SLURM / before any install.**
2. **Rigid ΔΔG oracle (Option A).** New reward module `plm_design_rl/rewards/_ddg_reward.py`
   consuming the **same packed atom14 clouds** the `sc_clash`/`shape` rewards already use; tmol behind
   a new optional extra (`[tmol]`), raising a clear install error at construction (mirrors the
   `[ligandmpnn]`/`[protenix]` pattern in the package proposal). Keep per-binder-residue decomposition
   for per-token credit.
3. **Offline validation (same bar as every prior reward).** Score an existing design pool, compute
   **AUROC of ΔΔG vs the Protenix pass label** on the `_packed_potentials_analyze.py` harness, and
   compare against the current `e_lj` (0.778), `dsasa` (0.757), 3DZD SC (0.73). Ship the ddG term as
   a reward only if it **beats `e_lj`** (else `e_lj` stays primary and ddG is documented, not wired —
   same discipline as the dropped MPNN-consistency / distribution rewards).
4. **Only if warranted:** add Option B (repacked unbound) and re-run the AUROC gate.

## 7. Environment facts (blessed venv, 2026-08-26)

```
python 3.12 · torch 2.8.0+cu128 · cuda 12.8 · glibc 2.34
tmol: not installed   pyrosetta: not installed
```
Matching tmol wheel lane: `tmol-<ver>+cu128torch2.8-cp312-cp312-manylinux_2_28_x86_64.whl`
(GPU, sm_75 build, forward-compatible to a10g/b200) or the `+cpu` cp312 wheel for offline scoring on
the defq CPU pool (`--account=llm`).
