# LeFlur De-novo Binder Design

LeFlur's two complex-trained checkpoints design **de-novo protein binders**
against a target antigen: given a target structure and a handful of epitope
residues, they generate a novel binder sequence **and** its structure, docked
against the requested epitope. This is the sixth LeFlur inference mode and the
only one that ships its own dedicated checkpoints.

| Checkpoint | Track | Recommended config | Default |
|---|---|---|---|
| **`leflur-binder-3di`** | sequence + LG structure + **3Di** | `experiment/generate_binder_3di` | ✅ 3Di framework |
| **`leflur-binder-disto`** | sequence + LG structure | `experiment/generate_binder_disto` | ✅ non-3Di framework |

Both are ~5.7 GiB and live on
[`Sidney-Lisanza/leflur`](https://huggingface.co/Sidney-Lisanza/leflur),
downloaded on first use by `resolve_checkpoint`.

- **`leflur-binder-3di`** adds a third generative track over the Foldseek 3Di
  structural alphabet on top of the sequence + latent-structure tracks. The
  3Di track gives an additional per-residue structural handle that widens
  target coverage on the Complexa benchmark (works on more targets), at a
  sampler recipe tuned on the 38-target sweep.
- **`leflur-binder-disto`** is the two-track (sequence + latent structure)
  complex checkpoint with a distogram auxiliary head, run with the base
  sampler schedules. Slightly higher aggregate pass rate, narrower coverage.

When in doubt, start with **`leflur-binder-3di`** (the default).

---

## 1. Install

```bash
uv sync --extra mgm --extra struct-cpu   # add --extra flash on GPU
```

No HuggingFace token is required — both the checkpoint repo and the benchmark
dataset are public.

## 2. Design a binder against one target

The configs ship the full production sampler recipe baked in, so a single
target run is one command. Point `generation.input_structures` at your target
PDB, name the target chain, and list the epitope residues (0-indexed into that
chain's `coords_res`):

```bash
uv run lobster_generate \
    --config-name experiment/generate_binder_3di paths=public \
    generation.input_structures=/path/to/target.pdb \
    generation.target_chain=A \
    generation.epitope_indices="[32,94,96,101]" \
    generation.binder_length="[80,150]" \
    generation.n_designs_per_structure=10 \
    output_dir=./out/binder_one
```

Each design writes a decoded binder+target complex PDB and a row in the
metrics CSV (sequence, ESMFold self-consistency, per-design timing). Swap
`generate_binder_3di` → `generate_binder_disto` to run the non-3Di arm.

`binder_length` accepts either a single int (`generation.binder_length=100`)
or a `[min,max]` range that is resampled per design.

## 3. Run the full Complexa benchmark

The **Complexa 38-target benchmark** is the canonical binder-design evaluation:
38 therapeutic target antigens, each with epitope residues, a deep MSA, and a
binder-length range. Fetch it once from HuggingFace, then loop it with the
portable runner:

```bash
# Fetch the benchmark (38 target PDBs + MSAs + manifests, ~60 MiB)
lobster_leflur_benchmarks fetch complexa-binder

# Smoke test — first target, one design
uv run python examples/run_complexa_binder.py --limit 1 --n-designs 1

# Full 38-target 3Di run, 100 designs/target
uv run python examples/run_complexa_binder.py \
    --n-designs 100 --out-dir ./out/complexa_3di

# Non-3Di (disto) arm
uv run python examples/run_complexa_binder.py \
    --config experiment/generate_binder_disto \
    --n-designs 100 --out-dir ./out/complexa_disto
```

`examples/run_complexa_binder.py` reads the fetched
`complexa_gen_targets.csv`, resolves each target's PDB relative to the
benchmark dir, and calls `lobster_generate` per target with that target's
`epitope_indices` / `binder_length` / `target_chain`. It replaces the
cluster-only slurm driver for non-cluster users; run
`--help` for `--targets`, `--seed`, `--dry-run`, and `--extra` pass-through.

The benchmark manifest schema (relative paths, portable):

| Column | Meaning |
|---|---|
| `target_id` | Stable id, e.g. `01_PD1` |
| `pdb_path` | Target antigen PDB, relative to the benchmark dir (`pdbs/{id}.pdb`) |
| `target_chain` | Antigen chain the binder is designed against |
| `epitope_indices` | Comma-separated epitope residues (0-indexed into `coords_res`) |
| `binder_len_min`, `binder_len_max` | Per-target binder length range |

## 4. Score with Protenix (the eval)

Generation is only half the benchmark. A design **PASSES** when
[Protenix](https://github.com/bytedance/Protenix) co-folding of the
binder+antigen complex gives:

> **pTM > 0.80 AND ipTM > 0.70**

Protenix runs in a **separate, heavy environment** (its own venv + weights)
and is intentionally *not* a hard dependency of LeFlur. The scoring driver
used for the paper is `scripts/_score_sabdab_minibinders.py` (kept locally,
not shipped); it co-folds each generated binder against its antigen (using the
antigen sequence + a3m from `complexa_score_targets.csv`) and writes per-design
`ptm` / `iptm` columns. Any co-folding oracle (Protenix, AlphaFold3-family,
Boltz) works — the PASS rule is oracle-agnostic.

Aggregate three numbers per model over the 38 targets:

- **pass rate** — fraction of all designs that PASS,
- **target coverage** — targets with ≥ 1 passing design (out of 38),
- **unique folds / covered target** — distinct Foldseek clusters (TM > 0.5)
  among a target's passing binders (structural diversity of the working
  designs).

## 5. Results (100 designs/target)

Reproduced from the scored Complexa runs. The **Complexa** row is the Complexa
model's own binder benchmark result (its published ceiling on this set; note
3 of the 38 targets OOM'd at 24 GB for that model and are excluded from its
denominator).

| Model | Config | Pass rate | Coverage | Unique folds / covered |
|---|---|---:|---:|---:|
| **`leflur-binder-3di`** (default) | `experiment/generate_binder_3di` | 6.05% | 36 / 38 | 4.03 |
| **`leflur-binder-disto`** | `experiment/generate_binder_disto` | 6.37% | 33 / 38 | 4.30 |
| Complexa (reference) | — | 28.80% | 35 / 35 † | 11.51 |

† 3 of the 38 targets (`25_CbAgo`, `35_H1`, `38_TNFalpha`) OOM'd for the
Complexa model at 24 GB and are excluded from its denominator. PASS = Protenix
pTM > 0.80 AND ipTM > 0.70; 100 designs/target.

**Reading the table.** The two LeFlur arms are close on aggregate pass rate;
`leflur-binder-3di` trades a fraction of a point for **wider target coverage**
(the 3Di track finds a working design on more antigens). Both trail the
Complexa model substantially — Complexa remains the stronger binder generator
on its own benchmark — so these numbers are an honest, reproducible baseline,
not a state-of-the-art claim.

Regenerate the table with the per-target analysis script (needs the scored
CSVs + a Foldseek binary):

```bash
uv run python scripts/_complexa_pertarget.py \
    stoch_a8_tri80_n100:leflur-binder-3di \
    disto_last:leflur-binder-disto \
    complexa_complexabench:complexa
```

## 6. The sampler recipe

The recipe below is baked into `experiment/generate_binder_3di.yaml`; you only
override the per-target target/epitope/length. It is the strongest
binder-docking arm found on the 38-target Complexa sweep.

| Knob | 3Di (`generate_binder_3di`) | disto (`generate_binder_disto`) |
|---|---|---|
| sequence schedule | `PowerInferenceSchedule` (exp 2) | `LogInferenceSchedule` |
| structure schedule | `LinearInferenceSchedule` | `LinearInferenceSchedule` |
| 3Di schedule | `LogInferenceSchedule` | — |
| `sequence_diversity_penalty` | 2 | 2 |
| `tri_diversity_penalty` | 8 | — |
| `stochasticity_struc` | 60 | 60 |
| `stochasticity_tri` | 80 | — |
| `nsteps` | 400 | 400 |

Two design decisions worth calling out:

- **Diversity penalties** subtract `penalty × (running per-design token
  frequency)` from the logits, so a single design cannot collapse to one
  residue / one 3Di state. `sequence_diversity_penalty=2` lifts Complexa pass
  rate materially by removing degenerate (poly-residue) sequences;
  `tri_diversity_penalty=8` downregulates the failure-correlated monotonous-3Di
  mode.
- **Per-track schedules.** The 3Di track denoises on a `Log` schedule while
  sequence uses `Power(2)` and structure stays `Linear` — the arm that
  maximised docking success in the sweep.

`use_epitope_conditioning=true` feeds the epitope residues into the model's
hotspot channel (the complex/epitope-trained docking prior), in addition to
seeding binder placement. Turn it off (`generation.use_epitope_conditioning=false`)
for unconditioned placement.

## 7. Bring your own target

You don't need to register a benchmark to design against your own antigen —
point the config at any PDB:

```bash
uv run lobster_generate \
    --config-name experiment/generate_binder_3di paths=public \
    generation.input_structures=/abs/path/to/my_antigen.pdb \
    generation.target_chain=B \
    generation.epitope_indices="[10,12,44,45,47]" \
    generation.binder_length="[70,120]" \
    generation.n_designs_per_structure=50 \
    output_dir=./out/my_target
```

Epitope indices are 0-indexed positions into the target chain's residue array
(`coords_res`), not PDB author numbering. If unsure, pick the surface residues
you want the binder centred on and verify placement in the first few decoded
complexes.

## See also

- [`checkpoints.md`](checkpoints.md) — the two binder checkpoints in the
  registry, how to `fetch` / `inspect` them.
- [`benchmarks.md`](benchmarks.md) — the `complexa-binder` dataset schema,
  citation, and license.
- [`cli.md`](cli.md) — full `lobster_generate` / `lobster_leflur_benchmarks`
  reference.
- The LeFlur [README](../../src/lobster/model/leflur/README.md) benchmark
  section for the summarised results table.
