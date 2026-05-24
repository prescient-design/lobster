# Protein-Ligand Training Pipeline Schematic

## Overview
This document provides a comprehensive schematic of the protein-ligand training pipeline in the Lobster codebase.

---

## 1. Data Loading & Processing Pipeline

### 1.1 Three Dataset Modes

The pipeline supports three modes based on what data is available:

```
MODE 1: PROTEIN-ONLY (e.g., StructureDataset)
    dataset_filenames = ["protein1.pt", "protein2.pt", ...]
    __getitem__(idx) returns: {coords_res, mask, sequence, ...}

MODE 2: LIGAND-ONLY (e.g., LigandDataset with no proteins)
    dataset_filenames = ["ligand1.pt", "ligand2.pt", ...]
    __getitem__(idx) returns: {"protein": None, "ligand": ligand_dict}

MODE 3: PROTEIN-LIGAND PAIRS (e.g., LigandDataset with matching pairs)
    dataset_filenames = [("protein1.pt", "ligand1.pt"), 
                         ("protein2.pt", "ligand2.pt"), ...]
    __getitem__(idx) returns: {"protein": protein_dict, "ligand": ligand_dict}
```

### 1.2 Data Loading Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LOADING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

Dataset Files (*.pt)
    ├── Protein: {coords_res, mask, indices, sequence, chains}
    └── Ligand: {atom_coords, mask, atom_indices, element_indices}
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LigandDataset (__getitem__)  [datasets/_ligand_dataset.py:93]        │
│                                                                        │
│  if isinstance(dataset_filenames[idx], tuple):                        │
│      # MODE 3: Protein-Ligand Pair                                    │
│      x_protein = torch.load(dataset_filenames[idx][1])  # protein.pt  │
│      x_ligand = torch.load(dataset_filenames[idx][0])   # ligand.pt   │
│      if transform_protein: x_protein = transform_protein(x_protein)   │
│  else:                                                                 │
│      # MODE 2: Ligand-Only                                            │
│      x_protein = None                                                  │
│      x_ligand = torch.load(dataset_filenames[idx])                    │
│                                                                        │
│  if transform_ligand: x_ligand = transform_ligand(x_ligand)           │
│                                                                        │
│  return {"protein": x_protein, "ligand": x_ligand}                    │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Transforms (Optional)                                                 │
│  - StructureBackboneTransform: Crops/pads protein to max_length       │
│  - StructureLigandTransform: Processes ligand coordinates              │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  DataLoader (Batching)                                                 │
│  - Collects N samples                                                  │
│  - Passes to collate function                                          │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  COLLATE FUNCTION ROUTING (data/_collate_structure.py)                │
│                                                                        │
│  collate_fn_backbone() - Used for MODES 2 & 3                         │
│    ├─► Handles protein-ligand pairs (MODE 3)                          │
│    └─► Handles ligand-only (MODE 2)                                   │
│                                                                        │
│  default_collate() - Used for MODE 1                                  │
│    └─► Standard PyTorch collation for protein-only                    │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  collate_fn_backbone (data/_collate_structure.py:125)                 │
│                                                                        │
│  Input: batch = [{"protein": p_dict, "ligand": l_dict}, ...]          │
│         or      [{"protein": None, "ligand": l_dict}, ...]  (MODE 2)  │
│                                                                        │
│  Step 1: Separate protein and ligand                                  │
│    protein_batch = [item["protein"] for item in batch]                │
│    ligand_batch = [item["ligand"] for item in batch]                  │
│                                                                        │
│  Step 2: Check what data we have                                      │
│    has_proteins = any(p is not None for p in protein_batch)           │
│    has_ligands = any(l is not None for l in ligand_batch)             │
│                                                                        │
│  Step 3: Collate proteins (if has_proteins)                           │
│    if has_proteins:                                                    │
│      - Find max_length across batch                                   │
│      - Pad coords_res: [B, max_L, n_atoms, 3]                        │
│      - Pad mask: [B, max_L]                                           │
│      - Pad indices, sequence, chains                                  │
│      → protein_collated = {coords_res, mask, indices, sequence, ...}  │
│    else:                                                               │
│      protein_collated = {}                                             │
│                                                                        │
│  Step 4: Collate ligands (if has_ligands)                            │
│    if has_ligands:                                                     │
│      ligand_collated = collate_fn_ligand(ligand_batch)                │
│      - Find max_length across ligand atoms                            │
│      - Pad ligand_coords: [B, max_L_ligand, 3]                       │
│      - Pad ligand_mask: [B, max_L_ligand]                            │
│      - Pad ligand_indices, element_indices                            │
│      → ligand_collated = {ligand_coords, ligand_mask, ...}            │
│    else:                                                               │
│      ligand_collated = {}                                              │
│                                                                        │
│  Step 5: Merge into unified batch                                     │
│    batch = {**protein_collated, **ligand_collated}                    │
│                                                                        │
│  OUTPUT DEPENDS ON MODE:                                               │
│    MODE 1 (protein-only): {coords_res, mask, sequence, ...}           │
│    MODE 2 (ligand-only):  {ligand_coords, ligand_mask, ...}           │
│    MODE 3 (both):         {coords_res, mask, ligand_coords, ...}      │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    UNIFIED BATCH (structure depends on mode)
    
    MODE 1 (protein-only):
    {
        coords_res: [B, L, n_atoms, 3],
        mask: [B, L],
        indices: [B, L],
        sequence: [B, L],
        chains: [B, L]
    }
    
    MODE 2 (ligand-only):
    {
        ligand_coords: [B, L_ligand, 3],
        ligand_mask: [B, L_ligand],
        ligand_indices: [B, L_ligand],
        ligand_element_indices: [B, L_ligand],
        ligand_atomic_numbers: [B, L_ligand]
    }
    
    MODE 3 (protein-ligand):
    {
        coords_res: [B, L, n_atoms, 3],
        mask: [B, L],
        indices: [B, L],
        sequence: [B, L],
        chains: [B, L],
        ligand_coords: [B, L_ligand, 3],
        ligand_mask: [B, L_ligand],
        ligand_indices: [B, L_ligand],
        ligand_element_indices: [B, L_ligand],
        ligand_atomic_numbers: [B, L_ligand]
    }
```

---

## 2. Three Dataset Modes - Complete Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THREE DATASET MODES SUMMARY                           │
└─────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════╗
║ MODE 1: PROTEIN-ONLY                                                  ║
╚═══════════════════════════════════════════════════════════════════════╝

DATASET:
  - StructureDataset or BackboneDataset
  - Files: ["protein1.pt", "protein2.pt", ...]
  - Config: data=structure_pdb

COLLATE:
  - Uses: default_collate() (standard PyTorch)
  - Output: {coords_res, mask, sequence, indices, chains}

FEATURIZE:
  - has_proteins = True, has_ligands = False
  - coords = batch["coords_res"]  # [B, L, n_atoms, 3]
  - NO concatenation
  - SE(3) applied to protein only
  - Returns: (coords, seq_mask, residue_index, sequence)
            ↑ 4 elements

ENCODER:
  - Detects MODE 1 by: len(x_feat) == 4
  - Processes: coords [B, L, n_atoms, 3]
  - Output: emb [B, L, embed_dim]

DECODER:
  - Reconstructs: protein_coords [B, L, n_atoms, 3]
  - No ligand reconstruction

USE CASE:
  - Training on protein structure databases (PDB, AlphaFold)
  - Learning protein backbone representations


╔═══════════════════════════════════════════════════════════════════════╗
║ MODE 2: LIGAND-ONLY                                                   ║
╚═══════════════════════════════════════════════════════════════════════╝

DATASET:
  - LigandDataset (with no protein files)
  - Files: ["ligand1.pt", "ligand2.pt", ...]
  - Config: data=structure_ligand (with protein_filenames=null)

COLLATE:
  - Uses: collate_fn_backbone()
  - Detects: all protein_batch items are None
  - Output: {ligand_coords, ligand_mask, ligand_indices, 
             ligand_element_indices, ligand_atomic_numbers}

FEATURIZE:
  - has_proteins = False, has_ligands = True
  - coords = batch["ligand_coords"]  # [B, L_ligand, 3]
  - seq_mask = batch["ligand_mask"]  # [B, L_ligand]
  - NO concatenation (ligand becomes "coords")
  - SE(3) applied to ligand only
  - Returns: (None, None, None, None,
             ligand_coords, ligand_mask, ligand_residue_index,
             ligand_atomic_numbers)
            ↑ 8 elements (first 4 are None)

ENCODER:
  - Detects MODE 2 by: len(x_feat) == 8 and coords is None
  - Processes: ligand_coords [B, L_ligand, 3]
  - Output: emb [B, L_ligand, embed_dim]

DECODER:
  - Reconstructs: ligand_coords [B, L_ligand, 3]
  - Predicts: element_types [B, L_ligand, num_elements]
  - No protein reconstruction

USE CASE:
  - Training on small molecule databases
  - Learning ligand representations
  - Molecular generation


╔═══════════════════════════════════════════════════════════════════════╗
║ MODE 3: PROTEIN-LIGAND PAIRS                                          ║
╚═══════════════════════════════════════════════════════════════════════╝

DATASET:
  - LigandDataset (with protein-ligand pairs)
  - Files: [("protein1.pt", "ligand1.pt"), 
            ("protein2.pt", "ligand2.pt"), ...]
  - Config: data=structure_ligand_pdb

COLLATE:
  - Uses: collate_fn_backbone()
  - Detects: both protein and ligand data present
  - Output: {coords_res, mask, sequence, indices, chains,
             ligand_coords, ligand_mask, ligand_indices,
             ligand_element_indices, ligand_atomic_numbers}

FEATURIZE:
  - has_proteins = True, has_ligands = True
  - Step 1: Extract both
      protein_coords = batch["coords_res"]     # [B, L, n_atoms, 3]
      ligand_coords = batch["ligand_coords"]   # [B, L_ligand, 3]
  
  - Step 2: CONCATENATE (key step!)
      coords_flat = protein_coords.reshape(B, L*n_atoms, 3)
      coords_combined = cat([coords_flat, ligand_coords], dim=1)
      # → [B, L*n_atoms + L_ligand, 3]
      
      mask_combined = cat([protein_mask_flat, ligand_mask], dim=1)
      # → [B, L*n_atoms + L_ligand]
  
  - Step 3: SE(3) applied to ENTIRE COMPLEX together
      coords_combined = apply_random_se3_batched(coords_combined, ...)
      # Protein and ligand rotate/translate together!
  
  - Step 4: SPLIT back
      ligand_coords = coords_combined[:, L*n_atoms:, :]
      protein_coords = coords_combined[:, :L*n_atoms, :]
  
  - Returns: (coords, seq_mask, residue_index, sequence,
             ligand_coords, ligand_mask, ligand_residue_index,
             ligand_atomic_numbers)
            ↑ 8 elements (all valid)

ENCODER:
  - Detects MODE 3 by: len(x_feat) == 8 and coords is not None
  - Processes: protein [B, L, n_atoms, 3] + ligand [B, L_ligand, 3]
  - Output: emb [B, L + L_ligand, embed_dim]

DECODER:
  - Reconstructs: protein_coords [B, L, n_atoms, 3]
  - Reconstructs: ligand_coords [B, L_ligand, 3]
  - Predicts: element_types [B, L_ligand, num_elements]

USE CASE:
  - Training on protein-ligand complexes (PDBBind, etc.)
  - Learning binding site representations
  - Drug design applications
  - SE(3) equivariance for the entire complex


╔═══════════════════════════════════════════════════════════════════════╗
║ KEY INSIGHT: MODE DETECTION                                           ║
╚═══════════════════════════════════════════════════════════════════════╝

The pipeline automatically detects which mode to use based on:

1. DATASET LEVEL:
   - What files are provided (protein, ligand, or both)

2. COLLATE LEVEL:
   - What keys are present in batch dict
   - has_proteins = "sequence" in batch
   - has_ligands = "ligand_coords" in batch

3. FEATURIZE LEVEL:
   - Returns 4 elements (MODE 1) or 8 elements (MODE 2/3)
   - First 4 elements None → MODE 2
   - First 4 elements valid → MODE 3

4. ENCODER LEVEL:
   - len(x_feat) == 4 → MODE 1
   - len(x_feat) == 8 and coords is None → MODE 2
   - len(x_feat) == 8 and coords is not None → MODE 3

This allows seamless switching between modes via config!


╔═══════════════════════════════════════════════════════════════════════╗
║ QUICK COMPARISON TABLE                                                ║
╚═══════════════════════════════════════════════════════════════════════╝

┌──────────────────┬────────────────┬────────────────┬─────────────────┐
│ Aspect           │ MODE 1         │ MODE 2         │ MODE 3          │
│                  │ Protein-Only   │ Ligand-Only    │ Protein-Ligand  │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Dataset          │ StructureData  │ LigandDataset  │ LigandDataset   │
│                  │                │ (no proteins)  │ (with pairs)    │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Config           │ structure_pdb  │ structure_     │ structure_      │
│                  │                │ ligand         │ ligand_pdb      │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Collate Fn       │ default_       │ collate_fn_    │ collate_fn_     │
│                  │ collate        │ backbone       │ backbone        │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Batch Keys       │ coords_res,    │ ligand_coords, │ coords_res,     │
│                  │ mask,          │ ligand_mask,   │ mask,           │
│                  │ sequence       │ ligand_*       │ sequence,       │
│                  │                │                │ ligand_coords,  │
│                  │                │                │ ligand_mask     │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Featurize        │ No concat      │ No concat      │ CONCATENATE     │
│ Strategy         │                │ (ligand only)  │ protein+ligand  │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ SE(3) Applied To │ Protein        │ Ligand         │ Complex         │
│                  │                │                │ (together!)     │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Featurize Return │ 4 elements     │ 8 elements     │ 8 elements      │
│                  │                │ (first 4 None) │ (all valid)     │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Encoder Input    │ [B,L,n_a,3]    │ [B,L_lig,3]    │ [B,L,n_a,3] +   │
│                  │                │                │ [B,L_lig,3]     │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Encoder Output   │ [B,L,D]        │ [B,L_lig,D]    │ [B,L+L_lig,D]   │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Decoder Output   │ protein_coords │ ligand_coords, │ protein_coords, │
│                  │                │ elements       │ ligand_coords,  │
│                  │                │                │ elements        │
├──────────────────┼────────────────┼────────────────┼─────────────────┤
│ Use Case         │ Protein        │ Small molecule │ Drug design,    │
│                  │ structure      │ generation     │ binding site    │
│                  │ learning       │                │ learning        │
└──────────────────┴────────────────┴────────────────┴─────────────────┘
```

---

## 3. Model Forward Pass Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL FORWARD PASS                               │
└─────────────────────────────────────────────────────────────────────────┘

UNIFIED BATCH
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TokenizerMulti.single_step()                                          │
│  (model/latent_generator/tokenizer/_tokenizer_multi.py:395)            │
│                                                                        │
│  Detects mode by checking length of featurized output:                │
│    x_feat = self.encoder.featurize(batch, ...)                        │
│                                                                        │
│    if len(x_feat) == 8:  # MODE 3 (protein-ligand)                    │
│      coords, seq_mask, residue_index, sequence,                       │
│      ligand_coords, ligand_mask, ligand_residue_index,                │
│      ligand_atom_types = x_feat                                       │
│                                                                        │
│    elif len(x_feat) == 4:  # MODE 1 (protein-only)                    │
│      coords, seq_mask, residue_index, sequence = x_feat               │
│      ligand_coords = None                                              │
│      ligand_mask = None                                                │
│      ligand_residue_index = None                                       │
│      ligand_atom_types = None                                          │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ViTEncoder.featurize()                                                │
│  (model/latent_generator/structure_encoder/_vit_encoder.py:120)        │
│                                                                        │
│  Step 1: Determine what data we have (MODE detection)                 │
│    has_proteins = "sequence" in batch                                  │
│    has_ligands = "ligand_coords" in batch                              │
│                                                                        │
│  Step 2: Extract protein data (if has_proteins)                       │
│    if has_proteins:                                                    │
│      coords = batch["coords_res"].clone()        # [B, L, n_atoms, 3] │
│      seq_mask = batch["mask"].clone()            # [B, L]             │
│      residue_index = batch["indices"].clone()    # [B, L]             │
│      sequence = batch["sequence"].clone()        # [B, L]             │
│    else:                                                               │
│      coords = None                                                     │
│      seq_mask = None                                                   │
│      residue_index = None                                              │
│      sequence = None                                                   │
│                                                                        │
│  Step 3: Extract ligand data (if has_ligands)                         │
│    if has_ligands:                                                     │
│      ligand_coords = batch["ligand_coords"].clone() # [B, L_lig, 3]   │
│      ligand_mask = batch["ligand_mask"].clone()     # [B, L_lig]      │
│      ligand_residue_index = batch["ligand_indices"].clone()           │
│      ligand_atomic_numbers = batch["ligand_atomic_numbers"].clone()   │
│    else:                                                               │
│      ligand_coords = None                                              │
│      ligand_mask = None                                                │
│      ligand_residue_index = None                                       │
│      ligand_atomic_numbers = None                                      │
│                                                                        │
│  Step 4: Handle concatenation based on mode                           │
│                                                                        │
│    MODE 1 (protein-only): coords stays as [B, L, n_atoms, 3]          │
│                                                                        │
│    MODE 2 (ligand-only):                                               │
│      coords = ligand_coords      # [B, L_ligand, 3]                   │
│      seq_mask = ligand_mask      # [B, L_ligand]                      │
│      residue_index = ligand_residue_index                              │
│                                                                        │
│    MODE 3 (protein-ligand): Concatenate!                              │
│      coords_flat = coords.reshape(B, L*n_atoms, 3)                    │
│      coords_combined = cat([coords_flat, ligand_coords], dim=1)       │
│      # → [B, L*n_atoms + L_ligand, 3]                                 │
│                                                                        │
│      seq_mask_flat = seq_mask.expand(...).reshape(B, L*n_atoms)       │
│      mask_combined = cat([seq_mask_flat, ligand_mask], dim=1)         │
│      # → [B, L*n_atoms + L_ligand]                                    │
│                                                                        │
│      coords = coords_combined                                          │
│      seq_mask = mask_combined                                          │
│                                                                        │
│  Step 5: Apply SE(3) transformations (if random_se3=True)             │
│    if coords is not None and seq_mask.any():                          │
│      coords = apply_random_se3_batched(                                │
│          coords,                                                       │
│          atom_mask=seq_mask,                                           │
│          translation_scale=translation_scale                           │
│      )                                                                 │
│    # For MODE 3: transforms protein+ligand together as complex        │
│    # For MODE 1: transforms protein only                              │
│    # For MODE 2: transforms ligand only                               │
│                                                                        │
│  Step 6: Apply frame alignment (if frame_type is set)                 │
│    if frame_type is not None and coords is not None:                  │
│      coords = apply_global_frame_to_coords(                            │
│          coords,                                                       │
│          frame_type=frame_type,  # e.g., "pca_frame"                  │
│          mask=seq_mask                                                 │
│      )                                                                 │
│                                                                        │
│  Step 7: Split back based on mode                                     │
│                                                                        │
│    MODE 3 (protein-ligand): Split concatenated coords                 │
│      ligand_coords = coords[:, L*n_atoms:, :]                         │
│      coords = coords[:, :L*n_atoms, :]                                │
│      coords = coords.reshape(B, L, n_atoms, 3)                        │
│      seq_mask = seq_mask[:, :L*n_atoms].reshape(B, L, n_atoms)        │
│      seq_mask = seq_mask.sum(dim=-1) > 0  # Back to residue mask     │
│                                                                        │
│      Return: (coords, seq_mask, residue_index, sequence,              │
│               ligand_coords, ligand_mask, ligand_residue_index,       │
│               ligand_atomic_numbers)                                   │
│                                                                        │
│    MODE 2 (ligand-only):                                               │
│      Return: (None, None, None, None,                                 │
│               ligand_coords, ligand_mask, ligand_residue_index,       │
│               ligand_atomic_numbers)                                   │
│                                                                        │
│    MODE 1 (protein-only):                                              │
│      Return: (coords, seq_mask, residue_index, sequence)              │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ViTEncoder.forward()                                                  │
│  (model/latent_generator/structure_encoder/_vit_encoder.py:291)        │
│                                                                        │
│  Input: Featurized coordinates from above (mode-dependent)            │
│                                                                        │
│  Step 1: Determine dimensions                                         │
│    if coords is not None:  # MODE 1 or MODE 3                         │
│      B, L, _, _ = coords.shape                                        │
│    else:                   # MODE 2 (ligand-only)                     │
│      B, _, _ = ligand_coords.shape                                    │
│                                                                        │
│  Step 2: Process through TimeCondUViTEncoder                          │
│    emb = self.net(                                                     │
│        coords,              # [B, L, n_atoms, 3] or None              │
│        seq_mask,            # [B, L] or None                          │
│        ligand_coords,       # [B, L_ligand, 3] or None                │
│        ligand_mask,         # [B, L_ligand] or None                   │
│        ligand_atom_types,   # [B, L_ligand] or None                   │
│        ...                                                             │
│    )                                                                   │
│                                                                        │
│    Output shape depends on mode:                                      │
│      MODE 1: [B, L, embed_dim]                                        │
│      MODE 2: [B, L_ligand, embed_dim]                                 │
│      MODE 3: [B, L + L_ligand, embed_dim]                             │
│                                                                        │
│  Step 3: Concatenate masks for output (MODE 3 only)                   │
│    if ligand_coords is not None:                                      │
│      if coords is not None:  # MODE 3                                 │
│        seq_mask = cat([seq_mask, ligand_mask], dim=-1)                │
│      else:                   # MODE 2                                 │
│        seq_mask = ligand_mask                                          │
│                                                                        │
│  Step 4: Apply mask to embeddings                                     │
│    emb *= expand(seq_mask, emb)                                        │
│    # Zero out embeddings for padded positions                         │
│                                                                        │
│  Return: emb (shape depends on mode)                                  │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Quantizer (Optional)                                                  │
│  - Quantizes embeddings into discrete tokens                          │
│  - Splits into protein_tokens and ligand_tokens                       │
│  Return: {protein_tokens: [B, L, D], ligand_tokens: [B, L_lig, D]}   │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Decoder Factory                                                       │
│                                                                        │
│  For each decoder (vit_decoder, element_decoder, etc.):               │
│                                                                        │
│  ViT Decoder:                                                          │
│    Input: tokens (or embeddings)                                      │
│    Output: {                                                           │
│        protein_coords: [B, L, n_atoms, 3],                            │
│        ligand_coords: [B, L_ligand, 3]                                │
│    }                                                                   │
│                                                                        │
│  Element Decoder (for ligands):                                       │
│    Input: ligand_tokens                                               │
│    Output: element_logits [B, L_ligand, num_elements]                │
│                                                                        │
│  Return: x_recon = {                                                   │
│      "vit_decoder": {protein_coords, ligand_coords},                  │
│      "element_decoder": element_logits                                │
│  }                                                                     │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Loss Computation                                                      │
│                                                                        │
│  For each loss function:                                               │
│    - Coordinate loss: MSE(predicted_coords, true_coords)              │
│    - Element loss: CrossEntropy(element_logits, true_elements)        │
│    - Apply masks to ignore padded positions                           │
│                                                                        │
│  Total loss = weighted sum of all losses                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Data Structures

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY DATA STRUCTURES                              │
└─────────────────────────────────────────────────────────────────────────┘

PROTEIN DATA:
├── coords_res: [B, L, n_atoms, 3]
│   └── Residue-level coordinates (e.g., N, CA, C atoms per residue)
├── mask: [B, L]
│   └── Boolean mask: True for valid residues, False for padding
├── indices: [B, L]
│   └── Residue indices in original structure
├── sequence: [B, L]
│   └── Amino acid type indices (0-19 for 20 amino acids)
└── chains: [B, L]
    └── Chain IDs for multi-chain proteins

LIGAND DATA:
├── ligand_coords: [B, L_ligand, 3]
│   └── Atom-level coordinates (xyz for each atom)
├── ligand_mask: [B, L_ligand]
│   └── Boolean mask: True for valid atoms, False for padding
├── ligand_indices: [B, L_ligand]
│   └── Atom indices
└── ligand_element_indices: [B, L_ligand]
    └── Element type indices (maps to ELEMENT_VOCAB)

ELEMENT_VOCAB = ["PAD", "B", "Bi", "Br", "C", "Cl", "F", "H", 
                 "I", "N", "O", "P", "S", "Si"]

EMBEDDINGS:
├── After Encoder: [B, L + L_ligand, embed_dim]
│   └── Combined protein + ligand embeddings
└── After Quantizer: 
    ├── protein_tokens: [B, L, token_dim]
    └── ligand_tokens: [B, L_ligand, token_dim]

RECONSTRUCTIONS:
└── x_recon = {
    "vit_decoder": {
        "protein_coords": [B, L, n_atoms, 3],
        "ligand_coords": [B, L_ligand, 3]
    },
    "element_decoder": [B, L_ligand, num_elements]
}
```

---

## 5. Critical Code Locations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CRITICAL CODE LOCATIONS                             │
└─────────────────────────────────────────────────────────────────────────┘

DATA LOADING:
├── Dataset: src/lobster/datasets/_ligand_dataset.py
│   └── LigandDataset.__getitem__()
├── Collation: src/lobster/data/_collate_structure.py
│   ├── collate_fn_backbone()  (handles protein+ligand)
│   └── collate_fn_ligand()    (handles ligand-only)
└── Transforms: src/lobster/transforms/_structure_transforms.py

ENCODING:
├── Encoder: src/lobster/model/latent_generator/structure_encoder/
│   └── _vit_encoder.py
│       ├── ViTEncoder.featurize()  (Lines 123-292)
│       │   ├── Extracts protein/ligand from batch
│       │   ├── Concatenates coordinates
│       │   ├── Applies SE(3) transformations
│       │   └── Splits back into protein/ligand
│       └── ViTEncoder.forward()    (Lines 294-343)
│           └── Processes through U-ViT network
└── U-ViT Network: src/lobster/model/latent_generator/models/vit/
    └── _vit_utils.py
        └── TimeCondUViTEncoder

QUANTIZATION:
└── Quantizer: src/lobster/model/latent_generator/quantizer/
    ├── _fsq.py  (Finite Scalar Quantization)
    └── _ligand_tokenizer.py

DECODING:
└── Decoders: src/lobster/model/latent_generator/structure_decoder/
    ├── _vit_decoder.py      (Reconstructs coordinates)
    └── _element_decoder.py  (Predicts ligand elements)

TRAINING:
├── Main Loop: src/lobster/model/latent_generator/tokenizer/
│   └── _tokenizer_multi.py
│       ├── TokenizerMulti.single_step()     (Lines 395-536)
│       ├── TokenizerMulti.training_step()   (Line 538)
│       └── TokenizerMulti.validation_step() (Line 542)
└── Callbacks: src/lobster/model/latent_generator/callbacks/
    └── _backbone_reconstruction.py
        └── Saves PDB files for visualization

CONFIGURATION:
├── Experiment: src/lobster/hydra_config/experiment/
│   └── train_latent_generator.yaml
├── Data: src/lobster/hydra_config/data/
│   └── structure_ligand_pdb.yaml
└── Model: src/lobster/hydra_config/model/
    └── latent_generator_ligand.yaml
```

---

## 6. SE(3) Transformation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SE(3) TRANSFORMATION FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════╗
║ MODE 1: PROTEIN-ONLY                                                  ║
╚═══════════════════════════════════════════════════════════════════════╝

BEFORE SE(3):
    coords: [B, L, n_atoms, 3]  (in original frame)
    mask:   [B, L]
         │
         ▼
    FLATTEN (for SE(3) application):
    coords_flat: [B, L*n_atoms, 3]
    mask_flat:   [B, L*n_atoms]
         │
         ▼
    APPLY RANDOM SE(3):
    coords_flat = apply_random_se3_batched(coords_flat, atom_mask=mask_flat)
         │
         ▼
    RESHAPE BACK:
    coords: [B, L, n_atoms, 3]  (transformed)
         │
         ▼
    FEED TO NETWORK


╔═══════════════════════════════════════════════════════════════════════╗
║ MODE 2: LIGAND-ONLY                                                   ║
╚═══════════════════════════════════════════════════════════════════════╝

BEFORE SE(3):
    coords: [B, L_ligand, 3]  (actually ligand_coords, but stored as coords)
    mask:   [B, L_ligand]     (actually ligand_mask)
         │
         ▼
    APPLY RANDOM SE(3):
    coords = apply_random_se3_batched(coords, atom_mask=mask)
         │
         ▼
    coords: [B, L_ligand, 3]  (transformed)
         │
         ▼
    FEED TO NETWORK


╔═══════════════════════════════════════════════════════════════════════╗
║ MODE 3: PROTEIN-LIGAND (KEY MODE!)                                    ║
╚═══════════════════════════════════════════════════════════════════════╝

BEFORE SE(3):
    Protein: [B, L, n_atoms, 3]  (in original frame)
    Ligand:  [B, L_ligand, 3]    (in original frame)
         │
         ▼
    CONCATENATE (CRITICAL STEP!):
    coords_flat = protein.reshape(B, L*n_atoms, 3)
    coords_combined = cat([coords_flat, ligand], dim=1)
    # → [B, L*n_atoms + L_ligand, 3]
    
    mask_flat = protein_mask.expand(...).reshape(B, L*n_atoms)
    mask_combined = cat([mask_flat, ligand_mask], dim=1)
    # → [B, L*n_atoms + L_ligand]
         │
         ▼
    APPLY RANDOM SE(3) TO ENTIRE COMPLEX:
    ┌─────────────────────────────────────────────────────────────────┐
    │ apply_random_se3_batched(coords_combined, atom_mask=mask_combined) │
    │                                                                 │
    │ For each sample in batch:                                       │
    │   1. Generate random rotation R (3x3 matrix)                    │
    │   2. Generate random translation t (3D vector)                  │
    │   3. Transform ALL atoms together: x' = R @ x + t               │
    │   4. Only transform valid atoms (where mask == True)            │
    │                                                                 │
    │ KEY: Protein and ligand get SAME R and t!                       │
    │      This preserves their relative geometry!                    │
    └─────────────────────────────────────────────────────────────────┘
         │
         ▼
    TRANSFORMED:
    coords_combined: [B, L*n_atoms + L_ligand, 3]  (in random frame)
         │
         ▼
    APPLY FRAME ALIGNMENT (optional):
    ┌─────────────────────────────────────────┐
    │ apply_global_frame_to_coords()          │
    │                                         │
    │ Options:                                │
    │   - "pca_frame": Align to PCA axes      │
    │   - "norm_frame": Normalize             │
    │   - "mol_frame": Molecular frame        │
    │                                         │
    │ Applied to entire complex together!     │
    └─────────────────────────────────────────┘
         │
         ▼
    FINAL TRANSFORMED:
    coords_combined: [B, L*n_atoms + L_ligand, 3]
         │
         ▼
    SPLIT BACK:
    ligand_coords = coords_combined[:, L*n_atoms:, :]
    protein_coords = coords_combined[:, :L*n_atoms, :]
    protein_coords = protein_coords.reshape(B, L, n_atoms, 3)
         │
         ▼
    Protein: [B, L, n_atoms, 3]  (transformed)
    Ligand:  [B, L_ligand, 3]    (transformed)
         │
         ▼
    FEED TO NETWORK


╔═══════════════════════════════════════════════════════════════════════╗
║ KEY INSIGHTS                                                          ║
╚═══════════════════════════════════════════════════════════════════════╝

1. WHY CONCATENATE FOR MODE 3?
   - Ensures protein and ligand undergo IDENTICAL transformation
   - Preserves relative spatial relationship (binding geometry)
   - Critical for learning protein-ligand interactions

2. DATA AUGMENTATION:
   - Random rotations: Model learns rotation invariance
   - Random translations: Model learns translation invariance
   - Together: SE(3) equivariance

3. MASKING:
   - Only valid (non-padded) atoms are transformed
   - Padded atoms remain as zeros
   - Prevents learning from padding artifacts

4. FRAME ALIGNMENT:
   - Optional post-processing after SE(3)
   - Can normalize or align to canonical frame
   - Also applied to entire complex in MODE 3
```

---

## 7. Masking Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MASKING STRATEGY                                 │
└─────────────────────────────────────────────────────────────────────────┘

WHY MASKING?
    - Batches have variable-length proteins and ligands
    - Padding is added to make tensors rectangular
    - Masks indicate which positions are real vs. padded

PROTEIN MASKING:
    mask: [B, L]
    ├── True:  Valid residue (has real coordinates)
    └── False: Padded residue (zeros)
    
    Example for batch of 2 proteins (lengths 5 and 3, padded to 5):
    mask = [[1, 1, 1, 1, 1],    # Protein 1: all 5 residues valid
            [1, 1, 1, 0, 0]]    # Protein 2: 3 valid, 2 padded

LIGAND MASKING:
    ligand_mask: [B, L_ligand]
    ├── True:  Valid atom (has real coordinates)
    └── False: Padded atom (zeros)
    
    Example for batch of 2 ligands (10 and 7 atoms, padded to 10):
    ligand_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ligand 1: 10 atoms
                   [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]  # Ligand 2: 7 atoms

COMBINED MASKING (during featurization):
    When concatenating protein + ligand:
    
    seq_mask_flat = seq_mask.expand(-1, -1, n_atoms).reshape(B, L*n_atoms)
    mask_combined = cat([seq_mask_flat, ligand_mask], dim=1)
    # → [B, L*n_atoms + L_ligand]
    
    This ensures:
    ├── SE(3) transformations only applied to valid atoms
    ├── Network attention only on valid positions
    └── Loss computation ignores padded positions

MASK USAGE:
    1. During SE(3): apply_random_se3_batched(coords, atom_mask=mask)
    2. During encoding: emb *= expand(mask, emb)
    3. During loss: loss = loss * mask.unsqueeze(-1)
    4. During PDB save: coords[mask.bool()] (only valid atoms)
```

---

## 8. Training Loop Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                    │
└─────────────────────────────────────────────────────────────────────────┘

FOR EACH BATCH:
    │
    ├─► 1. Load & Collate Data
    │   └─► Unified batch: {protein data, ligand data, masks}
    │
    ├─► 2. Featurize (ViTEncoder.featurize)
    │   ├─► Extract protein & ligand coordinates
    │   ├─► Concatenate into combined representation
    │   ├─► Apply SE(3) transformations (rotation + translation)
    │   ├─► Apply frame alignment (optional)
    │   └─► Split back into protein & ligand
    │
    ├─► 3. Encode (ViTEncoder.forward)
    │   ├─► Process through U-ViT network
    │   ├─► Generate embeddings for all atoms/residues
    │   └─► Apply masks to zero out padding
    │
    ├─► 4. Quantize (optional)
    │   └─► Convert continuous embeddings to discrete tokens
    │
    ├─► 5. Decode
    │   ├─► Reconstruct protein coordinates
    │   ├─► Reconstruct ligand coordinates
    │   └─► Predict ligand element types
    │
    ├─► 6. Compute Loss
    │   ├─► Coordinate reconstruction loss (MSE)
    │   ├─► Element prediction loss (CrossEntropy)
    │   └─► Apply masks to ignore padding
    │
    ├─► 7. Backpropagation
    │   └─► Update model parameters
    │
    └─► 8. Save Structures (every N batches)
        ├─► Save reconstructed PDBs
        ├─► Save ground truth PDBs
        └─► Cleanup old files (keep max_total_files)

EVERY N STEPS:
    └─► Log metrics to W&B
    └─► Save checkpoint
    └─► Run validation
```

---

## 9. Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CONFIGURATION HIERARCHY                              │
└─────────────────────────────────────────────────────────────────────────┘

hydra_config/
├── experiment/train_latent_generator.yaml
│   ├── Sets: data, model, callbacks, trainer defaults
│   └── Override with: experiment=train_latent_generator
│
├── data/structure_ligand_pdb.yaml
│   ├── Dataset paths and parameters
│   ├── Collate function: collate_fn_backbone
│   ├── Transforms: StructureBackboneTransform, StructureLigandTransform
│   └── Override with: data=structure_ligand_pdb
│
├── model/latent_generator_ligand.yaml
│   ├── structure_encoder:
│   │   ├── encode_ligand: true
│   │   ├── embed_dim: 256
│   │   └── SE(3) parameters
│   ├── quantizer:
│   │   └── ligand_n_tokens: 512
│   └── decoder_factory:
│       ├── vit_decoder (reconstructs coordinates)
│       └── element_decoder (predicts elements)
│
├── callbacks/backbone_reconstruction.yaml
│   ├── save_every_n: 10000
│   └── max_total_files: 1000
│
└── trainer/default.yaml
    ├── max_epochs, devices, strategy
    └── DDP configuration

COMMAND LINE OVERRIDE:
    lobster_train \
        experiment=train_latent_generator \
        data=structure_ligand_pdb \
        model.structure_encoder.encode_ligand=true \
        callbacks.backbone_reconstruction.max_total_files=500
```

---

## 10. Debugging & Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DEBUGGING & VISUALIZATION                              │
└─────────────────────────────────────────────────────────────────────────┘

SAVED PDB FILES (from BackboneReconstruction callback):

Location: {output_dir}/structures/recon/

Files per batch (for batch_size=8, max_structures_per_batch=None):
├── struc_{batch_idx}_{step}_gen_item0.pdb      (reconstructed)
├── struc_{batch_idx}_{step}_gt_item0.pdb       (ground truth)
├── struc_{batch_idx}_{step}_gen_item1.pdb
├── struc_{batch_idx}_{step}_gt_item1.pdb
├── ...
└── struc_{batch_idx}_{step}_gt_item7.pdb

Each PDB contains:
├── Protein chain (A): Backbone atoms (N, CA, C)
└── Ligand chain (L): All atoms with element names

AUTOMATIC CLEANUP:
    - Keeps only max_total_files most recent PDBs
    - Deletes oldest files when limit exceeded
    - Default: 1000 files

VISUALIZATION:
    Use PyMOL, ChimeraX, or other PDB viewers:
    ```bash
    pymol struc_0_1000_gen_item0.pdb struc_0_1000_gt_item0.pdb
    ```
    
    Compare:
    ├── Reconstructed vs. Ground Truth
    ├── Protein-ligand binding pose
    └── Element predictions

LOGGING:
    W&B Dashboard shows:
    ├── Loss curves (coordinate, element)
    ├── Learning rate schedule
    └── Training metrics
```

---

## 11. Key Insights & Design Decisions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  KEY INSIGHTS & DESIGN DECISIONS                         │
└─────────────────────────────────────────────────────────────────────────┘

1. UNIFIED BATCH APPROACH
   ✓ Protein and ligand in same batch (not separate dataloaders)
   ✓ Enables joint SE(3) transformations
   ✓ Maintains spatial relationships in complex

2. CONCATENATION STRATEGY
   ✓ Flatten protein: [B, L, n_atoms, 3] → [B, L*n_atoms, 3]
   ✓ Concatenate with ligand: [B, L*n_atoms + L_ligand, 3]
   ✓ Process together through transformations
   ✓ Split back before network input

3. SE(3) EQUIVARIANCE
   ✓ Apply same transformation to protein + ligand
   ✓ Preserves relative geometry
   ✓ Data augmentation for rotation/translation invariance

4. MASKING DISCIPLINE
   ✓ Masks propagate through entire pipeline
   ✓ Prevents learning from padding artifacts
   ✓ Essential for variable-length sequences

5. MODULAR DESIGN
   ✓ Encoder, Quantizer, Decoder are independent
   ✓ Easy to swap components (e.g., different quantizers)
   ✓ Decoders can be added/removed via config

6. MEMORY EFFICIENCY
   ✓ Limit total PDB files saved
   ✓ Automatic cleanup of old files
   ✓ Configurable save frequency

7. DEBUGGING SUPPORT
   ✓ Save both reconstructed and ground truth
   ✓ Save all batch items (not just first)
   ✓ Apply masks before saving (no padding in PDBs)
```

---

## End of Schematic

For questions or clarifications, refer to the specific code files mentioned in Section 4.

