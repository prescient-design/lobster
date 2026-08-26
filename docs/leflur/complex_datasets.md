# LeFlur Complex / Multimer Training Datasets

This is the consolidated reference for the **protein–protein complex and
multimer** data LeFlur is trained on: what each source is, how big it is
(clusters vs. structures), how the interface pairs were filtered, and which
Hydra data config combines them into a training mix.

All counts below were read directly from the Hydra data-config headers under
[`src/lobster/hydra_config/data/`](../../src/lobster/hydra_config/data/) **and
verified against the on-disk cluster files** (`torch.load`) on 2026-07-27. Where
a config-header comment and the loaded file disagree, both numbers are shown.

> **Note on `.pt` cluster files.** Every `*_clusters*.pt` used here is a
> `{member_id: cluster_id}` map (values are `int`). So `len(file)` is the number
> of **members** (individual structures / dimers), and the number of **clusters**
> is the count of *unique values*. Training samples **cluster-uniformly** (one
> random member per cluster per epoch), so the cluster count — not the member
> count — is what controls a source's effective diversity in a balanced mix.

---

## 1. Source datasets

### Complex / multimer sources (the interface data)

| Source | Interface class | Clusters | Structures | Filter / provenance | atom14 train path |
|---|---|---:|---:|---|---|
| **Pinder** | Hetero protein–protein | 8,631 | 267,416 | Pinder curated protein–protein complexes; clustered by Pinder `system_id`. | `/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pinder_atom14/train_pdb_processed` |
| **SAbDab** | Antibody–antigen | 3,095 | 12,850 | SAbDab antibody–antigen complexes (VH+VL + antigen). | `/cv/scratch/u/lisanzas/sabdab_atom14/train_pdb_processed` |
| **teddymer** | Intra-protein domain–domain | 587,687 | 587,687 | Non-singleton TED domain–domain dimers (representative set; 1 member = 1 cluster). | `/cv/scratch/u/lisanzas/teddymer_atom14/train_pdb_processed` |
| **afdb-heterodimer** | Predicted heterodimer | 80,144 | 80,144 | AFDB high-confidence heterodimers, `ipSAE ≥ 0.6 & pDockQ2 ≥ 0.23` (repr set; 1:1). | `/cv/scratch/u/lisanzas/afdb_hetero_atom14/train_pdb_processed` |
| **afdb-homodimer** | Predicted homodimer | 241,664 | 1,879,497 | AFDB high-confidence homodimers, `max_ipSAE ≥ 0.6 & max_pDockQ2 ≥ 0.23`; seqid40 clusters. | `/cv/scratch/u/lisanzas/afdb_homo_atom14/train_pdb_processed` |
| **denovo-binder** | Generated binder–target | — | 9,274¹ | De-novo mini-binder–target complexes from the complexa / boltzgen generators; atom14 with an `epitope` field. Only in the `_denovobinder` mix. | `/cv/scratch/u/lisanzas/denovo_binder_atom14/train_pdb_processed` |

¹ From the config header; the `denovo-binder` source appears only in
`…_pinder1x_sabdab1x_len640_denovobinder.yaml`.

For **teddymer** and **afdb-heterodimer**, clusters == structures: they are
already representative (deduplicated) sets, so each entry is its own cluster. For
**afdb-homodimer**, the header states **1,927,836** high-confidence homodimers
before atom14 processing; the loaded training file holds **1,879,497** members in
**241,664** seqid40 clusters (the header's "241,664 clusters" figure matches
exactly). The teddymer and AFDB dimer sources were assembled from TED domains
and AFDB predictions respectively, filtered on the interface-confidence
thresholds above (foldseek / EBI download pipeline).

### Monomer base sources (context — carried from the LeFlur pretraining mix)

The complex mixes are built **on top of** the monomer pretraining sources so the
model does not forget single-chain folding. These are single-chain (not
interface) data:

| Source | Clusters | Members | atom14 train path |
|---|---:|---:|---|
| PDB (seqid40, SS-balanced) | 50,124 | 280,586 | `/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/train.pt` |
| AFDB monomer (SwissProt, SS-balanced) | 84,816 | 220,354 | `/cv/data/ai4dd/data2/lisanzas/AFDB/train_processed` |
| denovo (generated monomers) | 26,406 | 772,439 | `/cv/scratch/u/lisanzas/denovo_dataset/ume_dataset/train_processed_pt` |
| TED | — | — | `/cv/scratch/u/lisanzas/ted_ume_dataset/ume_dataset/train_processed_pt` — **dead weight, see §4** |
| CATH | — | — | `/cv/scratch/u/lisanzas/cath_dataset/ume_dataset/train_processed_pt` — **dead weight, see §4** |

### Cluster files (full paths)

| Source | Cluster file |
|---|---|
| PDB | `/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/pdb_seqid40_clusters_ss_balanced.pt` |
| AFDB monomer | `/cv/data/ai4dd/data2/lisanzas/AFDB/pdb_swissprot_clusters_ss_balanced.pt` |
| denovo | `/cv/scratch/u/lisanzas/denovo_dataset/ume_dataset/denovo_clusters_ss_balanced_v2.pt` |
| TED | `/cv/scratch/u/lisanzas/ted_ume_dataset/ume_dataset/ted_clusters_ss_balanced.pt` |
| CATH | `/cv/scratch/u/lisanzas/cath_dataset/ume_dataset/cath_clusters_ss_balanced.pt` |
| Pinder | `/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pinder_atom14/pinder_systemid_clusters_train.pt` |
| SAbDab | `/cv/scratch/u/lisanzas/sabdab_atom14/sabdab_clusters_train.pt` |
| teddymer | `/cv/scratch/u/lisanzas/teddymer_atom14/teddymer_clusters_train.pt` |
| afdb-heterodimer | `/cv/scratch/u/lisanzas/afdb_hetero_atom14/afdb_hetero_clusters_train.pt` |
| afdb-homodimer | `/cv/scratch/u/lisanzas/afdb_homo/afdb_homo_clusters_train.pt` |
| denovo-binder | `/cv/scratch/u/lisanzas/denovo_binder_atom14/binder_clusters_train.pt` |

---

## 2. Training mixes

Each mix is one Hydra data config. `balance_datasets: true` means every listed
train source contributes an **equal nominal share** (1 / N sources); per-source
caps and replicate overrides then reshape the realized per-epoch proportions.

| Config (`data=…`) | max_len | Train sources | Complex sources added |
|---|---:|---|---|
| `structure_leflur_complex_teddymer_afdbhetero_len640` | 640 | 9 | Pinder, SAbDab, teddymer, afdb-hetero |
| `…_teddymer_afdbhetero_len640_mono30k` | 640 | 9 (monomers capped 30k) | same as above |
| `…_teddymer_afdbhetero_homo_len640` | 640 | +afdb-homo | + afdb-homodimer |
| `…_teddymer_afdbhetero_homo_len640_mono30k` | 640 | 8 real + afdb-homo (TED/CATH dropped) | + afdb-homodimer |
| `structure_leflur_p_plus_pinder_complex_chainembed_pinder1x_sabdab1x_len640` | 640 | 7 | Pinder, SAbDab |
| `…_pinder1x_sabdab1x_len640_denovobinder` | 640 | 8 | Pinder, SAbDab, denovo-binder |
| `…_pinder2x` / `…_pinder3x` / `…_pinder2x_sabdab2x` | 512 | 7 | Pinder (2×/3×), SAbDab (1×/2×) upsampling variants |

### Realized per-epoch composition

**`…_teddymer_afdbhetero_len640`** (9 sources; ~419.7k clusters/epoch after caps).
The three giants (AFDB-monomer, teddymer, afdb-hetero) are capped at **60k
clusters/epoch** with a fresh random subset each epoch (rotation → full coverage
over training; requires `trainer.reload_dataloaders_every_n_epochs=1`). SAbDab is
replicated **10×** so it is not the smallest source.

| PDB | AFDB | denovo | teddymer | afdb-hetero | Pinder | SAbDab | CATH | TED |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14.3% | 14.3% | 14.3% | 14.3% | 14.3% | 10.3% | 7.4% | 5.6% | 5.2% |

**`…_teddymer_afdbhetero_homo_len640_mono30k`** (8 real sources + afdb-homo; TED &
CATH dropped). Monomers (PDB / AFDB / denovo) capped **30k**, replicate 1×;
teddymer / afdb-hetero / afdb-homo capped **60k** (rotating); SAbDab 10×. Net
**complex/interface 74.6% vs. monomer 25.4%**:

| teddymer | afdb-hetero | afdb-homo | Pinder | SAbDab | PDB | AFDB | denovo |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17.6% | 17.6% | 17.6% | 12.7% | 9.1% | 8.8% | 8.8% | 7.8% |

**`…_pinder1x_sabdab1x_len640`** (7 sources, equal ~14.3% each: PDB / AFDB /
denovo / TED / CATH / Pinder / SAbDab). This is the antibody/complex **finetune**
mix; `max_length` was raised 512 → 640 so full VH+VL (~230) + antigen (up to
~400) complexes are no longer truncated. The model uses RoPE (NeoBERT), so no
positional table is resized and 512-trained weights load cleanly. The
`_denovobinder` variant adds the 9,274 de-novo binder complexes as an 8th source
(~12.5% each).

---

## 3. Validation sources

All complex mixes share the same held-out val sources (appended after the train
paths, so they are excluded from training):

- **CAMEO** (`/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed`) — monomer
  forward/inverse folding.
- **MultiFlow test** (`/cv/data/ai4dd/data2/lisanzas/AFDB/test_multiflow_processed`)
  — monomer.
- **Pinder val**
  (`/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pinder_atom14/val_pdb_processed`)
  — protein–protein; homo-heavy (~1003 homo / ~187 hetero). Used for the
  forward-folding DockQ eval.
- **SAbDab val** (`/cv/scratch/u/lisanzas/sabdab_atom14/val_pdb_processed`) —
  antibody–antigen.

---

## 4. Caveats (read before quoting numbers)

- **TED & CATH are dead weight in the `teddymer_afdbhetero` configs.** Their
  cluster files (`ted/cath_clusters_ss_balanced.pt`) key on IDs that match **none**
  of the processed `.pt` filenames, so the datamodule loads **0** data points for
  them → they contribute **0 clusters** even in the base and `mono30k` configs
  (the ~5% shares shown for the base config are nominal pre-load). The
  `…_homo_len640_mono30k` config **drops them entirely** — this both matches the
  effective mix and fixes a crash: an empty PyG dataset makes `files_exist([])`
  False, re-runs `process()` every launch, and races `pre_transform.pt` writes
  across DDP ranks on NFS (crashed runs 17688573 / 17688902 / 17694140).
- **afdb_homo cluster path.** The homo config's cluster file lives under
  `afdb_homo/` (not `afdb_homo_atom14/`, where the structures are) — a naming
  asymmetry worth checking if you rewire paths.
- **Per-source caps only rotate** if `trainer.reload_dataloaders_every_n_epochs=1`
  is set; otherwise the first random 60k/30k subset is frozen for the whole run.
- **Counts are cluster-uniform-relevant.** teddymer's 587,687 "clusters" are 1
  member each, so its realized diversity per epoch is bounded by its 60k cap, not
  by 587k.

---

## 5. Where the raw numbers live

Source of truth = the config-header comment blocks in
`/cv/home/lisanzas/lobster/src/lobster/hydra_config/data/`:

- `/cv/home/lisanzas/lobster/src/lobster/hydra_config/data/structure_leflur_complex_teddymer_afdbhetero_len640.yaml`
- `/cv/home/lisanzas/lobster/src/lobster/hydra_config/data/structure_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k.yaml`
- `/cv/home/lisanzas/lobster/src/lobster/hydra_config/data/structure_leflur_p_plus_pinder_complex_chainembed_pinder1x_sabdab1x_len640.yaml`
  (+ `_denovobinder`, `_pinder2x`, `_pinder3x`, `_pinder2x_sabdab2x` variants)

Cluster counts in the tables above were regenerated with `torch.load` over each
`cluster_file_list` entry (unique-value count for clusters, key count for
members).
