# /data2/ Recovery Survey & Backup Strategy

> **Date Created:** December 18, 2025  
> **Context:** /data2/ was deleted; backup available from ~2 weeks ago  
> **Purpose:** Track what needs to be recreated and establish backup procedures

---

## Table of Contents
1. [Datasets to Recreate](#datasets-to-recreate)
2. [Checkpoints to Retrain](#checkpoints-to-retrain)
3. [Safe Checkpoints (Already on HuggingFace)](#safe-checkpoints-already-on-huggingface)
4. [Historical Run/Log Directories](#historical-runlog-directories)
5. [Recovery Priority Order](#recovery-priority-order)
6. [Backup Strategy Plan](#backup-strategy-plan)

---

## Datasets to Recreate

### 1. PDB Data (Latent Generator)

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/latent_generator_files/pdb_data/split_data/train.pt` | PDB training split | Multiple configs |
| `/data2/lisanzas/latent_generator_files/pdb_data/split_data/validation.pt` | PDB validation split | Multiple configs |
| `/data2/lisanzas/latent_generator_files/pdb_data/split_data/test.pt` | PDB test split | Multiple configs |
| `/data2/lisanzas/latent_generator_files/pdb_data/pdb_seqid40_clusters.pt` | Sequence clustering file | Multiple configs |

### 2. AFDB SwissProt Data (Gen-UME)

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/AFDB/train_processed/` | AFDB training set | `structure_afdb_swissprot.yaml` |
| `/data2/lisanzas/AFDB/valid_cameo_processed/` | CAMEO validation set | `structure_afdb_swissprot.yaml`, callbacks |
| `/data2/lisanzas/AFDB/test_multiflow_processed/` | MultiFlow test set | `structure_afdb_swissprot.yaml` |
| `/data2/lisanzas/AFDB/pdb_swissprot_clusters.pt` | SwissProt clustering | `structure_afdb_swissprot.yaml` |

### 3. ESM Atlas Data (Gen-UME)

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/ume/simplefold_dataset/train_processed/` | Processed ESM Atlas structures | `structure_esm_atlas_afdb_swissprot.yaml` |
| `/data2/ume/simplefold_dataset/esm_atlas/` | Raw ESM Atlas data | Processing scripts |

### 4. Ligand Datasets (Latent Generator)

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/pdb_bind/train/` | PDBBind train (old) | `structure_ligand_pdb.yaml` |
| `/data2/lisanzas/pdb_bind/val/` | PDBBind val (old) | `structure_ligand_pdb.yaml` |
| `/data2/lisanzas/pdb_bind/test/` | PDBBind test (old) | `structure_ligand_pdb.yaml` |
| `/data2/lisanzas/pdb_bind_12_15_25/train/` | PDBBind train (new, with bond_matrix) | `structure_ligand_pdb_sair_bond.yaml` |
| `/data2/lisanzas/pdb_bind_12_15_25/val/` | PDBBind val (new) | `structure_ligand_pdb_sair_bond.yaml` |
| `/data2/lisanzas/pdb_bind_12_15_25/test/` | PDBBind test (new) | `structure_ligand_pdb_sair_bond.yaml` |
| `/data2/lisanzas/geom_12_15_25/train/` | GEOM ligands (with bond_matrix) | `structure_ligand_pdb_sair_bond.yaml` |
| `/data2/lisanzas/sair_12_15_25/train/` | SAIR protein-ligand (with bond_matrix) | `structure_ligand_pdb_sair_bond.yaml` |
| `/data2/lisanzas/sair_protein_ligand/train/` | SAIR (old) | `structure_ligand_pdb_sair.yaml` |

### 5. CATH Data

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/CATH_v4_3/processed_structures_pt/train/cath_train.pt` | CATH train | `structure_cath.yaml` |
| `/data2/lisanzas/CATH_v4_3/processed_structures_pt/val/cath_val.pt` | CATH val | `structure_cath.yaml` |
| `/data2/lisanzas/CATH_v4_3/processed_structures_pt/test/cath_test.pt` | CATH test | `structure_cath.yaml` |

### 6. SAbDab (Antibody) Data

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/sabdab/train_denovo_processed_pt/train_denovo_data.pt` | SAbDab train | `structure_sabdab.yaml` |
| `/data2/lisanzas/sabdab/val_denovo_processed_pt/val_denovo_data.pt` | SAbDab val | `structure_sabdab.yaml` |
| `/data2/lisanzas/sabdab/test_denovo_processed_pt/test_dummy_denovo_data.pt` | SAbDab test | `structure_sabdab.yaml` |

### 7. ESM-C Embeddings (Latent Generator)

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/latent_generator_files/esm_c_300m_embeddings_iterable_sampler/train/` | ESM-C train embeddings | `structure_pinder_esm.yaml` |
| `/data2/lisanzas/latent_generator_files/esm_c_300m_embeddings_iterable_sampler/val/` | ESM-C val embeddings | `structure_pinder_esm.yaml` |
| `/data2/lisanzas/latent_generator_files/esm_c_300m_embeddings_iterable_sampler/test/` | ESM-C test embeddings | `structure_pinder_esm.yaml` |

### 8. AFDB Genie Data

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/latent_generator_files/afdb_data/processed_pt/train_afdb_genie2_data.pt` | AFDB Genie train | `structure_afdb_genie.yaml` |

### 9. MultiFlow Test Set

| Path | Description | Used In |
|------|-------------|---------|
| `/data2/lisanzas/multi_flow_data/test_set_filtered_pt/` | MultiFlow test set | Generation experiments |

---

## Checkpoints to Retrain

### Latent Generator Checkpoints (Local paths - NOT on HuggingFace)

| Path | Model Name | Priority |
|------|------------|----------|
| `/data2/ume/latent_generator_/runs//2025-11-09T14-23-55/last.ckpt` | LG Ligand | Medium |
| `/data2/ume/latent_generator_/runs//2025-11-06T00-40-11/last.ckpt` | LG full attention 2 | Medium |
| `/data2/ume/latent_generator_/runs//2025-12-07T22-38-42/epoch=830-step=88917-val_loss=16.5010.ckpt` | LG Protein Ligand | **HIGH** |
| `/data2/ume/latent_generator_/runs//2025-12-13T16-34-07/epoch=240-step=25787-val_loss=16.4510.ckpt` | LG Protein Ligand fsq 4375 | **HIGH** |
| `/data2/ume/latent_generator_/runs//2025-12-13T14-57-53/epoch=210-step=22577-val_loss=17.2066.ckpt` | LG Protein Ligand fsq 1000 | **HIGH** |
| `/data2/lisanzas/latent_generator/studies/outputs/train/dev/runs/2025-11-09_22-19-12/checkpoints/last.ckpt` | LG full attention 512 PDB Pinder FSQ | Low |

### Gen-UME Checkpoints

| Path | Model Size | Priority |
|------|------------|----------|
| `/data2/ume/gen_ume/runs//2025-11-17T20-31-05/last.ckpt` | 750M model | **HIGH** (used in many experiments) |
| `/data2/ume/gen_ume/runs//2025-11-07T13-19-11/last.ckpt` | 450M model | **HIGH** |
| `/data2/lisanzas/gen_ume/runs//2025-12-05T16-48-13/last.ckpt` | ESM Atlas trained | High |
| `/data2/lisanzas/gen_ume/runs//2025-12-17T20-25-52/epoch=28-step=20985-val_loss=5.0925.ckpt` | Latest large resume | **HIGH** |

---

## Safe Checkpoints (Already on HuggingFace)

These are hosted on HuggingFace and **don't need retraining**:

- ✅ `LG Ligand 20A`
- ✅ `LG Ligand 20A 512 1024`
- ✅ `LG Ligand 20A 512 1024 element`
- ✅ `LG Ligand 20A continuous`
- ✅ `LG Ligand 20A seq 3di Aux`
- ✅ `LG 20A seq Aux`
- ✅ `LG 20A seq 3di c6d Aux`
- ✅ `LG 20A seq 3di c6d Aux Pinder`
- ✅ `LG 20A seq 3di c6d Aux PDB`
- ✅ `LG 20A seq 3di c6d Aux PDB Pinder`
- ✅ `LG 20A seq 3di c6d Aux PDB Pinder Finetune`
- ✅ `LG 20A`
- ✅ `LG 10A`
- ✅ `LG full attention`

---

## Historical Run/Log Directories

These are training runs and logs - may be acceptable to lose:

- `/data2/ume/latent_generator_/slurm/logs/`
- `/data2/ume/gen_ume/slurm/logs/`
- `/data2/ume/latent_generator_/runs/` (except specific checkpoints above)
- `/data2/ume/.cache2/`
- `/data2/lisanzas/.cache/`
- `/data2/lisanzas/gen_ume/tmp/`

---

## Recovery Priority Order

### Tier 1 - Critical (Blocks current work)

1. ⬜ PDB training/validation/test splits + cluster file
2. ⬜ AFDB SwissProt processed datasets (train/val/test)
3. ⬜ Gen-UME 750M checkpoint (`2025-11-17T20-31-05`)
4. ⬜ Latest Latent Generator protein-ligand checkpoints

### Tier 2 - High (Needed for experiments)

1. ⬜ PDBBind/GEOM/SAIR with bond_matrix (12_15_25 versions)
2. ⬜ ESM Atlas processed structures
3. ⬜ Gen-UME 450M checkpoint
4. ⬜ CAMEO/MultiFlow test sets

### Tier 3 - Medium (Nice to have)

1. ⬜ CATH datasets
2. ⬜ SAbDab datasets
3. ⬜ ESM-C embeddings
4. ⬜ AFDB Genie data

---

## Backup Strategy Plan

### Part 1: Automatic Checkpoint Backup

#### Option A: S3 Bucket Backup (Recommended)

**Setup:**
```bash
# S3 bucket structure
s3://prescient-pcluster-data/gen_ume/
├── checkpoints/
│   ├── latent_generator/
│   │   ├── LG_Protein_Ligand_v1.ckpt
│   │   ├── LG_Protein_Ligand_fsq_4375_v1.ckpt
│   │   └── ...
│   └── gen_ume/
│       ├── gen_ume_750M_v1.ckpt
│       ├── gen_ume_450M_v1.ckpt
│       └── ...
└── datasets/
    ├── pdb/
    ├── afdb/
    ├── ligand/
    └── ...
```

**Automatic Upload Callback:**

Create a new callback that uploads checkpoints to S3 after each save:

```python
# src/lobster/callbacks/_s3_checkpoint_callback.py
import boto3
import os
from pathlib import Path
from pytorch_lightning.callbacks import Callback

class S3CheckpointBackupCallback(Callback):
    """Automatically backup checkpoints to S3 after saving."""
    
    def __init__(
        self,
        s3_bucket: str = "prescient-lobster",
        s3_prefix: str = "checkpoints",
        project_name: str = "latent_generator",
        upload_every_n_epochs: int = 10,
        upload_best_only: bool = False,
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.project_name = project_name
        self.upload_every_n_epochs = upload_every_n_epochs
        self.upload_best_only = upload_best_only
        self.s3_client = boto3.client("s3")
    
    def _upload_to_s3(self, local_path: str, s3_key: str):
        """Upload a file to S3."""
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            print(f"✅ Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {e}")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when a checkpoint is saved."""
        # Get the checkpoint path
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return
        
        # Upload best checkpoint
        if self.upload_best_only and ckpt_callback.best_model_path:
            best_path = ckpt_callback.best_model_path
            if os.path.exists(best_path):
                filename = Path(best_path).name
                s3_key = f"{self.s3_prefix}/{self.project_name}/best/{filename}"
                self._upload_to_s3(best_path, s3_key)
        
        # Upload periodic checkpoints
        if trainer.current_epoch % self.upload_every_n_epochs == 0:
            last_path = ckpt_callback.last_model_path
            if last_path and os.path.exists(last_path):
                filename = Path(last_path).name
                s3_key = f"{self.s3_prefix}/{self.project_name}/periodic/{filename}"
                self._upload_to_s3(last_path, s3_key)
```

**Hydra Config:**
```yaml
# src/lobster/hydra_config/callbacks/s3_backup.yaml
s3_backup:
  _target_: lobster.callbacks._s3_checkpoint_callback.S3CheckpointBackupCallback
  s3_bucket: "prescient-pcluster-data"
  s3_prefix: "gen_ume/checkpoints"
  project_name: ${logger.project}
  upload_every_n_epochs: 10
  upload_best_only: false
```

#### Option B: HuggingFace Hub Backup

**Automatic Upload Callback:**

```python
# src/lobster/callbacks/_hf_checkpoint_callback.py
from huggingface_hub import HfApi, upload_file
from pytorch_lightning.callbacks import Callback
import os

class HuggingFaceCheckpointCallback(Callback):
    """Automatically upload checkpoints to HuggingFace Hub."""
    
    def __init__(
        self,
        repo_id: str = "Sidney-Lisanza/latent_generator",
        upload_every_n_epochs: int = 50,
        upload_best_only: bool = True,
    ):
        self.repo_id = repo_id
        self.upload_every_n_epochs = upload_every_n_epochs
        self.upload_best_only = upload_best_only
        self.api = HfApi()
    
    def _upload_to_hf(self, local_path: str, path_in_repo: str):
        """Upload a file to HuggingFace Hub."""
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
            )
            print(f"✅ Uploaded to HuggingFace: {self.repo_id}/{path_in_repo}")
        except Exception as e:
            print(f"❌ Failed to upload to HuggingFace: {e}")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return
        
        # Upload best checkpoint to HuggingFace
        if self.upload_best_only and ckpt_callback.best_model_path:
            best_path = ckpt_callback.best_model_path
            if os.path.exists(best_path):
                # Create descriptive name
                model_name = os.environ.get("MODEL_NAME", "model")
                path_in_repo = f"checkpoints_for_lg/{model_name}.ckpt"
                self._upload_to_hf(best_path, path_in_repo)
```

### Part 2: Dataset Backup to S3

#### Initial Upload Script

```bash
#!/bin/bash
# scripts/backup_datasets_to_s3.sh

S3_BUCKET="s3://prescient-pcluster-data/gen_ume/datasets"

# PDB Data
echo "Uploading PDB data..."
aws s3 sync /data2/lisanzas/latent_generator_files/pdb_data/ \
    ${S3_BUCKET}/latent_generator/pdb_data/ \
    --exclude "*.log"

# AFDB SwissProt
echo "Uploading AFDB SwissProt..."
aws s3 sync /data2/lisanzas/AFDB/ \
    ${S3_BUCKET}/afdb/ \
    --exclude "*.log"

# Ligand datasets
echo "Uploading ligand datasets..."
aws s3 sync /data2/lisanzas/pdb_bind_12_15_25/ \
    ${S3_BUCKET}/ligand/pdb_bind_12_15_25/

aws s3 sync /data2/lisanzas/geom_12_15_25/ \
    ${S3_BUCKET}/ligand/geom_12_15_25/

aws s3 sync /data2/lisanzas/sair_12_15_25/ \
    ${S3_BUCKET}/ligand/sair_12_15_25/

# ESM Atlas
echo "Uploading ESM Atlas..."
aws s3 sync /data2/ume/simplefold_dataset/train_processed/ \
    ${S3_BUCKET}/esm_atlas/train_processed/

# CATH
echo "Uploading CATH..."
aws s3 sync /data2/lisanzas/CATH_v4_3/ \
    ${S3_BUCKET}/cath/

# SAbDab
echo "Uploading SAbDab..."
aws s3 sync /data2/lisanzas/sabdab/ \
    ${S3_BUCKET}/sabdab/

echo "✅ Dataset backup complete!"
```

#### Dataset Sync Utility

```python
# scripts/sync_datasets.py
"""
Utility to sync datasets between local storage and S3.

Usage:
    # Download datasets from S3
    python scripts/sync_datasets.py download --dataset pdb
    
    # Upload datasets to S3
    python scripts/sync_datasets.py upload --dataset all
    
    # List available datasets
    python scripts/sync_datasets.py list
"""

import argparse
import subprocess
from pathlib import Path

DATASETS = {
    "pdb": {
        "local": "/data2/lisanzas/latent_generator_files/pdb_data/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/latent_generator/pdb_data/",
    },
    "afdb": {
        "local": "/data2/lisanzas/AFDB/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/afdb/",
    },
    "pdb_bind": {
        "local": "/data2/lisanzas/pdb_bind_12_15_25/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/ligand/pdb_bind_12_15_25/",
    },
    "geom": {
        "local": "/data2/lisanzas/geom_12_15_25/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/ligand/geom_12_15_25/",
    },
    "sair": {
        "local": "/data2/lisanzas/sair_12_15_25/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/ligand/sair_12_15_25/",
    },
    "esm_atlas": {
        "local": "/data2/ume/simplefold_dataset/train_processed/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/esm_atlas/train_processed/",
    },
    "cath": {
        "local": "/data2/lisanzas/CATH_v4_3/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/cath/",
    },
    "sabdab": {
        "local": "/data2/lisanzas/sabdab/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/sabdab/",
    },
    "multiflow": {
        "local": "/data2/lisanzas/multi_flow_data/",
        "s3": "s3://prescient-pcluster-data/gen_ume/datasets/multiflow/",
    },
}

def sync(source: str, dest: str, dry_run: bool = False):
    """Sync files between source and destination."""
    cmd = ["aws", "s3", "sync", source, dest]
    if dry_run:
        cmd.append("--dryrun")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def download(dataset: str, dry_run: bool = False):
    """Download dataset from S3."""
    if dataset == "all":
        for name, paths in DATASETS.items():
            print(f"\n📥 Downloading {name}...")
            sync(paths["s3"], paths["local"], dry_run)
    else:
        paths = DATASETS[dataset]
        sync(paths["s3"], paths["local"], dry_run)

def upload(dataset: str, dry_run: bool = False):
    """Upload dataset to S3."""
    if dataset == "all":
        for name, paths in DATASETS.items():
            print(f"\n📤 Uploading {name}...")
            sync(paths["local"], paths["s3"], dry_run)
    else:
        paths = DATASETS[dataset]
        sync(paths["local"], paths["s3"], dry_run)

def list_datasets():
    """List available datasets."""
    print("\nAvailable datasets:")
    print("-" * 60)
    for name, paths in DATASETS.items():
        print(f"  {name}:")
        print(f"    Local: {paths['local']}")
        print(f"    S3:    {paths['s3']}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync datasets between local and S3")
    subparsers = parser.add_subparsers(dest="command")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download from S3")
    dl_parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()) + ["all"])
    dl_parser.add_argument("--dry-run", action="store_true")
    
    # Upload command
    ul_parser = subparsers.add_parser("upload", help="Upload to S3")
    ul_parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()) + ["all"])
    ul_parser.add_argument("--dry-run", action="store_true")
    
    # List command
    subparsers.add_parser("list", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download(args.dataset, args.dry_run)
    elif args.command == "upload":
        upload(args.dataset, args.dry_run)
    elif args.command == "list":
        list_datasets()
    else:
        parser.print_help()
```

### Part 3: Updated Training Scripts

Update SLURM scripts to use S3 backup:

```bash
# Add to slurm/scripts/train_*.sh

# Enable S3 checkpoint backup
export S3_CHECKPOINT_BUCKET="prescient-lobster"
export S3_CHECKPOINT_PREFIX="checkpoints"

# Add callback to training command
srun ... \
    lobster_train \
    experiment=train_gen_ume \
    +callbacks.s3_backup.s3_bucket=${S3_CHECKPOINT_BUCKET} \
    +callbacks.s3_backup.s3_prefix=${S3_CHECKPOINT_PREFIX} \
    ...
```

### Part 4: Recovery Checklist

After implementing backup:

- [ ] Create S3 bucket `prescient-lobster` (or verify existing)
- [ ] Run initial dataset backup script
- [ ] Add S3CheckpointBackupCallback to all training configs
- [ ] Update experiment configs to use S3 paths as fallback
- [ ] Set up weekly verification of S3 backups
- [ ] Document recovery procedure

---

## Recovery Status Tracker

**Last Survey: January 5, 2026**

### DATASETS

| Item | Status | Notes |
|------|--------|-------|
| PDB splits (train/val/test) | ✅ Recovered | `/data2/lisanzas/latent_generator_files/pdb_data/split_data/` |
| PDB cluster file | ✅ Recovered | `pdb_seqid40_clusters.pt` |
| AFDB SwissProt (train) | ✅ Recovered | `/data2/lisanzas/AFDB/train_processed/` |
| AFDB SwissProt (val/test) | ✅ Recovered | `valid_cameo_processed/`, `test_multiflow_processed/` |
| AFDB cluster file | ✅ Recovered | `pdb_swissprot_clusters.pt` |
| ESM Atlas | ✅ Recovered | `/data2/ume/simplefold_dataset/train_processed/` |
| PDBBind (old) | ✅ Recovered | `/data2/lisanzas/pdb_bind/` |
| PDBBind 12_15_25 | ✅ **COMPLETE** | `/data2/lisanzas/pdb_bind_12_15_25/` - 27,294 files with bond_matrix |
| GEOM 12_15_25 | ✅ **COMPLETE** | `/data2/lisanzas/geom_12_15_25/` - 246,840 train, 30,953 val, 30,936 test |
| SAIR 12_15_25 | ✅ **REPROCESSED** | `/data2/lisanzas/sair_12_15_25/` - 279,963 train, 38,611 val, 80,343 test |
| SAIR (old) | ⏸️ Not needed | Replaced by SAIR 12_15_25 |
| CATH | ✅ Recovered | `/data2/lisanzas/CATH_v4_3/` |
| SAbDab | ✅ Recovered | `/data2/lisanzas/sabdab/` |
| MultiFlow Test Set | ✅ Recovered | `/data2/lisanzas/multi_flow_data/` |
| AFDB Genie | ❌ Missing | `/data2/lisanzas/latent_generator_files/afdb_data/processed_pt/` - low priority |
| ESM-C Embeddings | ✅ Recovered | `/data2/lisanzas/latent_generator_files/esm_c_300m_embeddings_iterable_sampler/` |

### CHECKPOINTS

| Item | Status | Notes |
|------|--------|-------|
| Gen-UME 90M (PDB) | ✅ S3 Backup | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_90M_PDB.ckpt` (1.1 GiB) |
| Gen-UME 750M (2025-11-17) | ✅ S3 Backup | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_2025-11-17_*.ckpt` (8.3 GiB) |
| Gen-UME 450M (2025-11-07) | ✅ S3 Backup | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_450M_2025-11-07_*.ckpt` (5.3 GiB) |
| Gen-UME 750M ESM Atlas (2026-01-04) | ✅ S3 Backup | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_ESM_Atlas_2026-01-04_*.ckpt` (8.3 GiB) |
| Gen-UME Latest Large (2025-12-17) | ❌ Missing | Low priority - experimental training |
| **LG Protein Ligand 4096** (2026-01-05) | ✅ **NEW S3** | `LG_Protein_Ligand_4096_2026-01-05.ckpt` (292.9 MiB) |
| **LG Protein Ligand fsq 4375** (2026-01-05) | ✅ **NEW S3** | `LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt` (295.8 MiB) |
| **LG Protein Ligand fsq 4375/15360** (2026-01-07) | ✅ **NEW S3** | `LG_Protein_Ligand_fsq_4375_15360_2026-01-07.ckpt` (360.2 MiB) |
| LG Protein Ligand (2025-12-07) | ❌ **LOST** | Original 512-token SLQ model - needs retraining |
| LG Protein Ligand fsq 1000 (2025-12-13) | ❌ **LOST** | 1000-token FSQ model - needs retraining |
| LG Ligand (2025-11-09) | ✅ S3 Backup | `LG_Ligand_2025-11-09.ckpt` (250.5 MiB) |
| LG full attention 2 | ✅ S3 Backup | `LG_full_attention_2_2025-11-06.ckpt` (245.3 MiB) |

### Available Latent Generator runs in `/data2/ume/latent_generator_/runs/`:

Runs with checkpoints available (Nov 2025):
- 2025-11-30T16-50-54
- 2025-11-28T17-38-46
- 2025-11-26T15-51-49
- 2025-11-25T14-42-33
- 2025-11-21T23-30-09
- 2025-11-20T17-28-11
- (and more from Nov 17-21)

---

## Summary

### ✅ Recovered/Reprocessed (14 datasets):
- PDB data + clusters
- AFDB SwissProt + clusters  
- ESM Atlas
- PDBBind (old)
- **SAIR 12_15_25** ✅ (279,963 train, 38,611 val, 80,343 test)
- **PDBBind 12_15_25** ✅ (21,835 train, 2,729 val, 2,730 test)
- **GEOM 12_15_25** ✅ (246,840 train, 30,953 val, 30,936 test)
- CATH, SAbDab, MultiFlow
- ESM-C Embeddings

### 🔄 Currently Processing:
None - all datasets ready!

### ❌ Not Recovered (low priority):
- **AFDB Genie** - can reprocess if needed
- **SAIR (old)** - replaced by SAIR 12_15_25

### ✅ Recovered Checkpoints (backed up to S3):

**Gen-UME Models:**
| Model | Local Path | S3 Path | Size |
|-------|-----------|---------|------|
| Gen-UME 90M (PDB) | (from old S3 bucket) | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_90M_PDB.ckpt` | 1.1 GiB |
| Gen-UME 450M | `/data2/ume/gen_ume/runs/2025-11-07T13-19-11/` | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_450M_2025-11-07_*.ckpt` | 5.3 GiB |
| Gen-UME 750M | `/data2/ume/gen_ume/runs/2025-11-17T20-31-05/` | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_2025-11-17_*.ckpt` | 8.3 GiB |
| Gen-UME 750M ESM Atlas | `/data2/lisanzas/gen_ume/runs/2026-01-04T19-10-12/` | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_ESM_Atlas_2026-01-04_*.ckpt` | 8.3 GiB |

**Latent Generator Models:**
| Model | Local Path | S3 Path | Size |
|-------|-----------|---------|------|
| LG Ligand | `/data2/ume/latent_generator_/runs/2025-11-09T14-23-55/last.ckpt` | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Ligand_2025-11-09.ckpt` | 250.5 MiB |
| **LG Protein Ligand 4096** | `/data2/ume/latent_generator_/runs/2026-01-05T16-48-02/last.ckpt` | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_4096_2026-01-05.ckpt` | 292.9 MiB |
| **LG Protein Ligand fsq 4375** | `/data2/ume/latent_generator_/runs/2026-01-05T16-13-19/last.ckpt` | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt` | 295.8 MiB |
| **LG Protein Ligand fsq 4375/15360** | `/data2/ume/latent_generator_/runs/2026-01-07T02-17-14/last.ckpt` | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_15360_2026-01-07.ckpt` | 360.2 MiB |
| LG full attention 2 | `/data2/ume/latent_generator_/runs/2025-11-06T00-40-11/last.ckpt` | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_full_attention_2_2025-11-06.ckpt` | 245.3 MiB |

### ❌ Missing Checkpoints (Lost - need retraining):
| Model | Original Path | Status |
|-------|--------------|--------|
| LG Protein Ligand (2025-12-07) | `/data2/ume/latent_generator_/runs/2025-12-07T22-38-42/` | **LOST** - needs retraining |
| LG Protein Ligand fsq 1000 (2025-12-13) | `/data2/ume/latent_generator_/runs/2025-12-13T14-57-53/` | **LOST** - needs retraining |

---

## Processing Commands

### Check Processing Status
```bash
# Check job status
squeue -u $USER

# Check output counts
echo "PDBBind 12_15_25:"; find /data2/lisanzas/pdb_bind_12_15_25 -name "*_ligand.pt" 2>/dev/null | wc -l
echo "GEOM 12_15_25:"; find /data2/lisanzas/geom_12_15_25 -name "*.pt" 2>/dev/null | wc -l
echo "SAIR 12_15_25:"; find /data2/lisanzas/sair_12_15_25 -name "*_protein.pt" 2>/dev/null | wc -l
```

### Submit Processing Jobs
```bash
# PDBBind (fast, ~1-2 hours)
sbatch slurm/scripts/process_pdbbind_bond_matrix_array.sh

# GEOM (slower, ~4-8 hours due to S3)
sbatch slurm/scripts/process_geom_bond_matrix_array.sh
```

### Train LG Protein-Ligand Models (after datasets ready)
```bash
# SLQ quantization (512 tokens)
sbatch slurm/scripts/train_latent_generator_protein_ligand_sair.sh

# FSQ quantization (4375 tokens)
sbatch slurm/scripts/train_latent_generator_protein_ligand_fsq_ligand_4375.sh

# FSQ quantization (1000 tokens)
sbatch slurm/scripts/train_latent_generator_protein_ligand_fsq_ligand_1000.sh
```

---

*Last updated: January 8, 2026 (added Gen-UME 90M, 750M ESM Atlas, and LG Protein Ligand fsq 4375/15360 checkpoints to S3)*

