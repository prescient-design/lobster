# Bond Matrix Integration for Latent Generator

## Summary
This document outlines observations and a plan to add bond matrix support to the protein-ligand Latent Generator encoder, allowing the model to incorporate ligand topology information (which atoms are bonded to which) when creating structure tokens.

---

## Observations

### 1. Current Ligand Encoding in Latent Generator

**File:** `structure_encoder/_vit_encoder.py` → `TimeCondUViTEncoder` in `models/vit/_vit_utils.py`

Currently, ligands are encoded using:
- **3D Coordinates**: `ligand_coords` → embedded via `ligand_to_embedding` (MLP: 3 → hidden_dim)
- **Atom Types (optional)**: `ligand_atom_types` → embedded via `ligand_atom_type_embedding` (Embedding layer using `ELEMENT_VOCAB`)

```python
# Current ligand embedding (lines 717-723 in _vit_utils.py)
if self.encode_ligand and ligand_coords is not None:
    ligand_embedding = self.ligand_to_embedding(ligand_coords)
    
    if self.ligand_atom_embedding and ligand_atom_types is not None:
        ligand_type_embedding = self.ligand_atom_type_embedding(ligand_atom_types)
        ligand_embedding = ligand_embedding + ligand_type_embedding
```

**Gap:** No bond matrix information is used. The encoder only knows atom positions and types, but not which atoms are bonded.

---

### 2. Bond Matrix Support in Gen-UME Protein-Ligand Encoder

**Files:** `_gen_ume_protein_ligand_encoder.py`, `_bond_embedding.py`, `_bond_prediction.py`

The downstream Gen-UME model **already fully supports** bond matrices:

#### BondMatrixEmbedding (encoder side)
```python
# From _bond_embedding.py
class BondMatrixEmbedding(nn.Module):
    """Embed bond matrix information into atom features."""
    
    def forward(self, atom_embeddings, bond_matrix, atom_mask):
        # 1. Embed all bond types: [B, N, N] -> [B, N, N, H]
        bond_embeds = self.bond_type_embedding(bond_matrix)
        
        # 2. Sum over neighbors where bond exists
        bond_exists = (bond_matrix > 0).float().unsqueeze(-1)
        neighbor_bonds = (bond_embeds * bond_exists).sum(dim=2)
        
        # 3. Project and add to atom embeddings (residual)
        bond_context = self.bond_proj(neighbor_bonds)
        enriched = self.layer_norm(atom_embeddings + bond_context)
        
        return enriched
```

#### BondMatrixPredictionHead (decoder side)
```python
# From _bond_prediction.py
class BondMatrixPredictionHead(nn.Module):
    """Predict bond matrix from atom features."""
    # Uses outer product of projected features to predict bond types
```

#### Usage in ProteinLigandEncoderModule
```python
# From _gen_ume_protein_ligand_encoder.py (line 274)
def _embed_ligand(self, ligand_atom_input_ids, ligand_structure_input_ids, bond_matrix, ligand_mask):
    # ... embed atoms and structure ...
    
    # Add bond information
    ligand_emb = self.bond_embedding(ligand_emb, bond_matrix, ligand_mask)
```

---

### 3. Data Pipeline Status

**Bond matrix IS already supported in collation:**

```python
# From datamodules/_utils.py (lines 477-542)
def collate_fn_ligand(batch):
    # ...
    has_bond_matrix = "bond_matrix" in batch[0]
    
    if has_bond_matrix:
        bond_matrix = atom_dict["bond_matrix"]
        # Pad bond_matrix from [n_atoms, n_atoms] to [max_length, max_length]
        padded_bond = torch.zeros(max_length, max_length, dtype=bond_matrix.dtype)
        padded_bond[:n_atoms, :n_atoms] = bond_matrix
        padded_bond_matrices.append(padded_bond)
    
    if padded_bond_matrices:
        out["bond_matrix"] = torch.stack(padded_bond_matrices, dim=0)
```

**Bond matrix extraction from SDF needs enhancement:**

The current `load_ligand()` function in `io/_load_pdb.py` does **not** extract bond matrix from RDKit molecules. It only extracts:
- `atom_coords`
- `atom_names`
- `atom_indices`
- `mask`

---

### 4. Bond Type Constants

**From:** `utils/residue_constants.py`

```python
# Bond types for bond matrix representation
BOND_TYPES = {
    "NONE": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
    "OTHER": 5,
}
NUM_BOND_TYPES = len(BOND_TYPES)  # 6
```

---

## Plan for Integration

### Phase 1: Data Loading Enhancement ✅ COMPLETE

**Status:** Implemented on 2026-01-24

**File:** `io/_load_pdb.py` → `load_ligand()`

**Changes implemented:**

1. Added `extract_bond_matrix(mol)` function - extracts bond types from RDKit molecule into symmetric [N,N] tensor
2. Added `extract_element_indices(mol)` function - maps atom symbols to ELEMENT_VOCAB_EXTENDED indices
3. Updated `load_ligand()` to include `bond_matrix` and `element_indices` in output for mol2/sdf files
4. Updated exports in `io/__init__.py`

**New return fields for mol2/sdf files:**
- `element_indices`: Tensor [N] with element type indices
- `bond_matrix`: Tensor [N, N] with bond types (0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5=other)

**New parameter for `load_ligand()`:**
- `use_extended_element_vocab: bool = False`
  - If `False` (default): Uses `ELEMENT_VOCAB` (14 tokens) for latent generator compatibility
  - If `True`: Uses `ELEMENT_VOCAB_EXTENDED` (25 tokens) to match Gen-UME

**Vocabulary comparison:**
| Vocabulary | Tokens | Use Case |
|-----------|--------|----------|
| `ELEMENT_VOCAB` | 14 | Latent generator (default) |
| `ELEMENT_VOCAB_EXTENDED` | 25 | Gen-UME protein-ligand encoder |

**Note:** PDB ligand files don't have bond information, so these fields are only available for mol2/sdf formats.

---

### Phase 2: Encoder Enhancement (Backwards Compatible) ✅ COMPLETE

**Status:** Implemented on 2026-01-24

**File:** `models/vit/_vit_utils.py` → `TimeCondUViTEncoder`

**Changes implemented:**

1. Added `use_ligand_bond_embedding: bool = False` parameter to `__init__`
2. Import and use `BondMatrixEmbedding` from `gen_ume._bond_embedding` when enabled
3. Added `ligand_bond_matrix=None` parameter to `forward()` method
4. Bond embedding is applied after atom type embeddings when both are enabled

Add optional bond matrix embedding:

```python
class TimeCondUViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        # ... existing params ...
        encode_ligand: bool = False,
        ligand_atom_embedding: bool = False,
        use_ligand_bond_embedding: bool = False,  # NEW
    ):
        # ... existing code ...
        
        if encode_ligand:
            # ... existing ligand embedding ...
            
            # NEW: Bond matrix embedding (optional)
            if use_ligand_bond_embedding:
                from lobster.model.gen_ume._bond_embedding import BondMatrixEmbedding
                self.ligand_bond_embedding = BondMatrixEmbedding(
                    hidden_size=embed_dim_hidden,
                    num_bond_types=NUM_BOND_TYPES,
                )
            else:
                self.ligand_bond_embedding = None

    def forward(
        self,
        coords,
        # ... existing params ...
        ligand_coords=None,
        ligand_mask=None,
        ligand_atom_types=None,
        ligand_bond_matrix=None,  # NEW: Optional bond matrix
        **kwargs,
    ):
        # ... existing code ...
        
        if self.encode_ligand and ligand_coords is not None:
            ligand_embedding = self.ligand_to_embedding(ligand_coords)
            
            if self.ligand_atom_embedding and ligand_atom_types is not None:
                ligand_type_embedding = self.ligand_atom_type_embedding(ligand_atom_types)
                ligand_embedding = ligand_embedding + ligand_type_embedding
            
            # NEW: Add bond information if available
            if self.ligand_bond_embedding is not None and ligand_bond_matrix is not None:
                ligand_embedding = self.ligand_bond_embedding(
                    ligand_embedding, ligand_bond_matrix, ligand_mask
                )
```

**Key point:** When `use_ligand_bond_embedding=False` (default) or `ligand_bond_matrix=None`, the encoder behaves identically to before → **Backwards Compatible**.

---

### Phase 3: ViTEncoder Wrapper Update (Simplified) ✅ COMPLETE

**Status:** Implemented on 2026-01-24

**File:** `structure_encoder/_vit_encoder.py`

**Key insight:** The `batch` dict is already passed to `ViTEncoder.forward()` via kwargs from `TokenizerMulti`. We can extract `bond_matrix` directly from the batch without changing the feature tuple.

**Changes implemented:**

1. Added `use_ligand_bond_embedding: bool = False` parameter to `__init__`
2. Pass `use_ligand_bond_embedding` to `TimeCondUViTEncoder`
3. Added `ligand_bond_matrix=None` parameter to `forward()`
4. Extract `bond_matrix` from batch kwargs if not passed explicitly
5. Pass `ligand_bond_matrix` to `self.net()`

**Changes needed (forward() only):**

```python
class ViTEncoder(BaseEncoder):
    def __init__(self, ..., use_ligand_bond_embedding: bool = False, ...):
        # ... existing code ...
        self.use_ligand_bond_embedding = use_ligand_bond_embedding
        
        # Pass to TimeCondUViTEncoder
        self.net = TimeCondUViTEncoder(
            ...,
            use_ligand_bond_embedding=use_ligand_bond_embedding,
        )
    
    def forward(
        self,
        coords, seq_mask, residue_index, sequence,
        ligand_coords=None, ligand_mask=None,
        ligand_residue_index=None, ligand_atom_types=None,
        ligand_bond_matrix=None,  # NEW: Can be passed explicitly
        **kwargs,
    ):
        # Extract bond_matrix from batch if not passed explicitly
        if ligand_bond_matrix is None and "batch" in kwargs:
            batch = kwargs["batch"]
            if batch is not None and "bond_matrix" in batch:
                ligand_bond_matrix = batch["bond_matrix"]
        
        emb = self.net(
            coords,
            # ... existing params ...
            ligand_bond_matrix=ligand_bond_matrix,  # NEW
        )
```

**Why this is simpler:**
- No changes to `featurize()` - tuple stays the same (4, 7, or 8 elements)
- No changes to `TokenizerMulti` - batch already flows through kwargs
- Bond matrix is extracted from batch dict in `forward()`

---

### ~~Phase 4: TokenizerMulti Update~~ - NOT NEEDED

~~**File:** `tokenizer/_tokenizer_multi.py`~~

**No changes required!** The `batch` dict is already passed to `encoder.forward()` via kwargs:

```python
# In TokenizerMulti.single_step():
x_emb = self.encoder(
    coords=x_feat[0],
    ...
    batch=batch,  # <-- batch already passed!
)
```

The `ViTEncoder.forward()` can extract `bond_matrix` from `kwargs["batch"]`.

---

### Phase 4: Hydra Config Updates ✅ COMPLETE

**Status:** Implemented on 2026-01-24

**File:** `hydra_config/tokenizer/structure_encoder/vit_encoder.yaml`

Added new config options:
```yaml
_target_: lobster.model.latent_generator.structure_encoder.ViTEncoder
# ... existing config ...
encode_ligand: true
ligand_atom_embedding: true
use_ligand_bond_embedding: false  # Set to true to enable bond matrix embedding
use_extended_element_vocab: false  # Set to true for 25-token vocab (matches Gen-UME)
```

**Vocabulary options:**
| Parameter | Tokens | Description |
|-----------|--------|-------------|
| `use_extended_element_vocab: false` | 14 | Standard vocab (PAD, B, Bi, Br, C, Cl, F, H, I, N, O, P, S, Si) |
| `use_extended_element_vocab: true` | 25 | Extended vocab (PAD, MASK, UNK, C, N, O, S, P, F, Cl, Br, I, B, Si, Se, As, Zn, Fe, Cu, Mg, Ca, Na, K, Bi, H) |

---

## Backwards Compatibility Checklist

| Component | Change | Backwards Compatible? |
|-----------|--------|----------------------|
| `load_ligand()` | Add `bond_matrix` to output | ✅ Yes - new optional key |
| `collate_fn_ligand()` | Already handles `bond_matrix` | ✅ Already done |
| `TimeCondUViTEncoder` | Add `use_ligand_bond_embedding` param | ✅ Yes - defaults to False |
| `TimeCondUViTEncoder.forward()` | Add `ligand_bond_matrix` param | ✅ Yes - defaults to None |
| `ViTEncoder.__init__()` | Add `use_ligand_bond_embedding` param | ✅ Yes - defaults to False |
| `ViTEncoder.forward()` | Extract `bond_matrix` from batch kwargs | ✅ Yes - gracefully handles missing |
| `TokenizerMulti` | No changes needed | ✅ N/A - batch already passed |
| Hydra config | New `use_ligand_bond_embedding` | ✅ Yes - defaults to false |
| **Existing checkpoints** | Load without bond embedding weights | ✅ Yes - new weights init randomly |

---

## Testing Strategy

1. **Unit Tests:**
   - Test `extract_bond_matrix()` on sample molecules
   - Test `BondMatrixEmbedding` shapes and gradients

2. **Integration Tests:**
   - Load existing checkpoint without bond embedding → should work
   - Train new model with bond embedding → should converge
   - Compare reconstruction quality with/without bond embedding

3. **Backwards Compatibility Tests:**
   - Load old ligand .pt files (without bond_matrix) → should work
   - Run inference with old checkpoint → should work

---

## Files to Modify

| File | Changes | Status |
|------|---------|--------|
| `io/_load_pdb.py` | Add `extract_bond_matrix()`, `extract_element_indices()`, update `load_ligand()` | ✅ Done |
| `io/__init__.py` | Export new functions | ✅ Done |
| `models/vit/_vit_utils.py` | Add `use_ligand_bond_embedding`, `ligand_bond_embedding` module, update `forward()` | ✅ Done |
| `structure_encoder/_vit_encoder.py` | Add `use_ligand_bond_embedding` param, extract `bond_matrix` from batch in `forward()` | ✅ Done |
| `tokenizer/_tokenizer_multi.py` | ~~Handle 9-element feature tuples~~ | ✅ Not needed |
| `hydra_config/tokenizer/structure_encoder/vit_encoder.yaml` | Add `use_ligand_bond_embedding` option | ✅ Done |

---

## Expected Benefits

1. **Better Structure Tokens:** Ligand structure tokens will capture molecular topology, not just 3D positions
2. **Improved Reconstruction:** Should see better ligand coordinate reconstruction
3. **Better Generalization:** Bond information is invariant to conformer - helps with different poses
4. **Consistency with Gen-UME:** Latent generator now matches the downstream Gen-UME protein-ligand encoder

---

## Open Questions

1. **Should we also add bond prediction loss to the Latent Generator?**
   - The Gen-UME encoder has `BondMatrixPredictionHead` for predicting bonds
   - Could add as auxiliary loss during tokenizer training

2. **Pre-computed bond matrices in dataset?**
   - If datasets already have `bond_matrix` in .pt files, no need to modify `load_ligand()`
   - Only need encoder changes

3. **Element indices consistency?**
   - Need to ensure `element_indices` in data matches `ELEMENT_VOCAB_EXTENDED` ordering

---

*Last Updated: January 2026*

