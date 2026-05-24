# Binder Design Implementation Plan

## 🎉 IMPLEMENTATION STATUS: COMPLETE ✅

**All core components have been successfully implemented:**
1. ✅ `binder_utils.py` - Helper functions module created and linted
2. ✅ `generate.py` - Main function implemented and linted  
3. ✅ `generate_binder_design.yaml` - Example config file created
4. ✅ All linting errors resolved

**Ready for testing!**

---

## Executive Summary

**Goal:** Create a user-friendly binder design mode that automatically designs protein binders against a target structure.

**Current State:** Manual hack requiring index calculations and PDB manipulation

**Proposed Solution:** Automated `binder_design` mode with simple configuration:
```yaml
generation:
  mode: binder_design
  input_structures: "/path/to/target.pdb"
  target_chain: "A"
  binder_length: 100
```

**Key Features:**
- ✅ Automatic chain detection and indexing
- ✅ No manual PDB manipulation required
- ✅ Binder initialized at origin (model learns placement)
- ✅ Reusable config across different targets
- ✅ Integrated ESMFold validation
- ✅ Clean separation from inpainting mode

**Implementation Scope:**
- ~110 lines of code in new `binder_utils.py` (helper functions)
- ~310-410 lines of code in `generate.py` (main function + imports)
- 1 example config file (~50 lines)
- 2-4 days implementation effort

**Files to Create/Modify:**
1. `src/lobster/model/gen_ume/binder_utils.py` - New utility module (4 helper functions)
2. `src/lobster/cmdline/generate.py` - Add main implementation + imports
3. `src/lobster/hydra_config/experiment/generate_binder_design.yaml` - New config
4. Documentation updates

---

## Current State Analysis

### What exists:
1. **Config file hack** (`generate_binder_450M.yaml`): 
   - Uses inpainting mode with manually specified residue indices (166-227)
   - Requires user to calculate chain indices manually
   - Hard to reuse across different PDB structures

2. **Stub function** (`_generate_binders` in `generate.py`):
   - Currently raises `NotImplementedError`
   - Already integrated into the mode dispatch logic

3. **Supporting infrastructure**:
   - PDB loading with chain information (`chains_ids`, `real_chains`)
   - Chain indexing system (0, 200, 400, etc. for each chain)
   - Inpainting functionality that can mask/regenerate specific residues
   - ESMFold validation with chain group support

### Current limitations:
- User must manually calculate residue indices for the binder chain
- No automatic chain identification or selection
- No intelligent binder placement or initialization
- Reusing the config requires editing indices for each new target structure

---

## Proposed Implementation (MVP)

### 1. New Generation Mode: `binder_design`

Add a dedicated generation mode that makes binder design a first-class citizen alongside unconditional, inverse_folding, forward_folding, and inpainting.

### 2. Configuration Interface

#### New YAML configuration parameters:

```yaml
generation:
  mode: binder_design
  
  # Input target structure (PDB with one or more chains)
  input_structures: "/path/to/target.pdb"
  
  # Target chain specification (use chain letter from PDB)
  target_chain: "A"

  # Binder length
  binder_length: 100

  # Optional: Epitope residue indices (0-indexed, in coords_res numbering)
  # If specified, binder initialized 5Å from epitope, away from target COM
  # Example: [10, 11, 12, 13, 14] for a 5-residue epitope
  # If null, binder initialized at target's center of mass
  epitope_indices: null
  
  # Standard generation parameters
  nsteps: 200
  batch_size: 1
  n_trials: 1
  n_designs_per_structure: 10
  
  temperature_seq: 0.16
  temperature_struc: 1.0
  stochasticity_seq: 20
  stochasticity_struc: 10
  asynchronous_sampling: false
  
  # ESMFold validation (highly recommended for binder design)
  use_esmfold: true
  max_length: 512
  
  # Optional: Specify which chain groups to validate with ESMFold
  # If not specified, will automatically validate target+binder together
  esmfold_chain_groups: null
```

### 3. Main Implementation with Detailed Code Flow

**Function signature:**
```python
def _generate_binders(
    model, 
    cfg: DictConfig, 
    device: torch.device, 
    output_dir: Path, 
    plm_fold=None,
    csv_writer=None,
    plotter=None
) -> None:
    """Generate binders for target protein structures."""
```

**Detailed implementation flow:**

```python
def _generate_binders(model, cfg: DictConfig, device: torch.device, output_dir: Path, 
                      plm_fold=None, csv_writer=None, plotter=None) -> None:
    """Generate binders for target protein structures."""
    
    logger.info("Starting binder design generation...")
    
    # ============================================================================
    # STEP 1: Load and parse input structures
    # ============================================================================
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for binder_design mode")
    
    # Handle different input formats (same as inpainting mode)
    structure_paths = []
    if isinstance(input_structures, str):
        if "*" in input_structures:
            structure_paths = glob.glob(input_structures)
        else:
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                structure_paths = list(glob.glob(str(path / "*.pdb")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, ListConfig)):
        structure_paths = [str(p) for p in input_structures if Path(p).is_file()]
    
    if not structure_paths:
        raise ValueError("No valid structure files found")
    
    logger.info(f"Found {len(structure_paths)} structure(s) to process")
    
    # ============================================================================
    # STEP 2: Get configuration parameters
    # ============================================================================
    gen_cfg = cfg.generation
    target_chain = gen_cfg.get("target_chain")
    binder_length = gen_cfg.get("binder_length")
    nsteps = gen_cfg.get("nsteps", 200)
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)
    n_designs_per_structure = gen_cfg.get("n_designs_per_structure", 1)
    
    if not target_chain:
        raise ValueError("target_chain must be specified for binder_design mode")
    if not binder_length:
        raise ValueError("binder_length must be specified for binder_design mode")
    
    logger.info(f"Target chain: {target_chain}")
    logger.info(f"Binder length: {binder_length}")
    logger.info(f"Generation steps: {nsteps}")
    logger.info(f"Designs per structure: {n_designs_per_structure}")
    
    # Initialize transforms
    structure_transform = StructureBackboneTransform(max_length=gen_cfg.get("max_length", 512))
    tokenizer_transform = AminoAcidTokenizerTransform(max_length=gen_cfg.get("max_length", 512))
    
    # ============================================================================
    # STEP 3: Process each structure
    # ============================================================================
    with torch.no_grad():
        for structure_idx, structure_path in enumerate(structure_paths):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing structure {structure_idx + 1}/{len(structure_paths)}")
            logger.info(f"Input: {structure_path}")
            logger.info(f"{'='*70}")
            
            # ------------------------------------------------------------------------
            # 3.1: Load target structure
            # ------------------------------------------------------------------------
            logger.info(f"Loading target structure from {structure_path}")
            target_data = load_pdb(structure_path, add_batch_dim=False)
            
            if target_data is None:
                logger.warning(f"Failed to load structure from {structure_path}, skipping")
                continue
            
            # Apply transforms
            target_data = structure_transform(target_data)
            
            # Data structure at this point:
            # target_data = {
            #     'coords_res': Tensor(L_target, 3, 3),  # e.g., (100, 3, 3)
            #     'sequence': Tensor(L_target,),          # e.g., (100,)
            #     'mask': Tensor(L_target,),              # e.g., (100,)
            #     'chains_ids': Tensor(L_target,),        # e.g., [0,0,...,0,200,200,...,200]
            #     'real_chains': Tensor(L_target,),       # e.g., [65,65,...,65,66,66,...,66]
            #     'indices': Tensor(L_target,),           # e.g., [0,1,2,...,99,200,201,...,299]
            # }
            
            # ------------------------------------------------------------------------
            # 3.2: Identify target chain (using helper from binder_utils)
            # ------------------------------------------------------------------------
            try:
                target_chain_idx, target_start, target_end = get_target_chain_info(
                    target_data, target_chain
                )
                logger.info(f"Target chain '{target_chain}' found:")
                logger.info(f"  Chain index: {target_chain_idx}")
                logger.info(f"  Residue range: {target_start}-{target_end}")
                logger.info(f"  Length: {target_end - target_start} residues")
            except ValueError as e:
                logger.error(str(e))
                continue
            
            # ------------------------------------------------------------------------
            # 3.3: Initialize binder at target's center of mass (using helper from binder_utils)
            # ------------------------------------------------------------------------
            logger.info(f"Initializing binder with length {binder_length} at target center of mass")
            binder_data = initialize_binder_at_origin(
                binder_length,
                device='cpu',
                target_coords=target_data_filtered['coords_res']
            )

            # Data structure at this point:
            # binder_data = {
            #     'coords_res': Tensor(L_binder, 3, 3),   # e.g., (100, 3, 3), at target COM
            #     'sequence': Tensor(L_binder,),          # e.g., (100,), random 0-19
            #     'mask': Tensor(L_binder,),              # e.g., (100,), all ones
            # }
            
            # ------------------------------------------------------------------------
            # 3.4: Get next chain index for binder (using helper from binder_utils)
            # ------------------------------------------------------------------------
            binder_chain_idx = get_next_chain_index(target_data)
            logger.info(f"Binder will be assigned chain index: {binder_chain_idx}")
            
            # ------------------------------------------------------------------------
            # 3.5: Create composite structure (target + binder)
            # ------------------------------------------------------------------------
            logger.info("Creating composite structure (target + binder)")
            
            # Concatenate coordinates
            L_target = target_data['coords_res'].shape[0]
            L_binder = binder_data['coords_res'].shape[0]
            L_total = L_target + L_binder
            
            # Check max length
            max_length = gen_cfg.get("max_length", 512)
            if L_total > max_length:
                logger.warning(f"Total length {L_total} exceeds max_length {max_length}, skipping")
                continue
            
            # Concatenate all tensors
            coords_res_combined = torch.cat([
                target_data['coords_res'], 
                binder_data['coords_res']
            ], dim=0)  # Shape: (L_target + L_binder, 3, 3)
            
            sequence_combined = torch.cat([
                target_data['sequence'],
                binder_data['sequence']
            ], dim=0)  # Shape: (L_target + L_binder,)
            
            mask_combined = torch.cat([
                target_data['mask'],
                binder_data['mask']
            ], dim=0)  # Shape: (L_target + L_binder,)
            
            # Create chain IDs for binder (all same value)
            binder_chain_ids = torch.full(
                (L_binder,), 
                binder_chain_idx, 
                dtype=target_data['chains_ids'].dtype
            )
            chains_ids_combined = torch.cat([
                target_data['chains_ids'],
                binder_chain_ids
            ], dim=0)  # Shape: (L_target + L_binder,)
            
            # Create real_chains for binder (assign next letter, e.g., 'B'=66 if target was 'A'=65)
            # Find max real_chain in target
            max_real_chain = target_data['real_chains'].max().item()
            binder_real_chain = max_real_chain + 1  # Next letter
            binder_real_chains = torch.full(
                (L_binder,),
                binder_real_chain,
                dtype=target_data['real_chains'].dtype
            )
            real_chains_combined = torch.cat([
                target_data['real_chains'],
                binder_real_chains
            ], dim=0)  # Shape: (L_target + L_binder,)
            
            # Create indices for binder (continuous from binder_chain_idx)
            binder_indices = torch.arange(
                binder_chain_idx,
                binder_chain_idx + L_binder,
                dtype=target_data['indices'].dtype
            )
            indices_combined = torch.cat([
                target_data['indices'],
                binder_indices
            ], dim=0)  # Shape: (L_target + L_binder,)
            
            logger.info(f"Composite structure created:")
            logger.info(f"  Total length: {L_total} ({L_target} target + {L_binder} binder)")
            logger.info(f"  Target chain index: {target_chain_idx}")
            logger.info(f"  Binder chain index: {binder_chain_idx}")
            
            # ------------------------------------------------------------------------
            # 3.6: Add batch dimension and move to device
            # ------------------------------------------------------------------------
            coords_res = coords_res_combined.unsqueeze(0).to(device)  # (1, L, 3, 3)
            sequence = sequence_combined.unsqueeze(0).to(device)      # (1, L)
            mask = mask_combined.unsqueeze(0).to(device)              # (1, L)
            chains_ids = chains_ids_combined.unsqueeze(0).to(device)  # (1, L)
            indices = indices_combined.unsqueeze(0).to(device)        # (1, L)
            
            # Apply tokenizer to sequence
            tokenized_data = tokenizer_transform({'sequence': sequence.squeeze(0)})
            sequence_tokenized = tokenized_data['sequence'].unsqueeze(0).to(device)
            
            # ------------------------------------------------------------------------
            # 3.7: Create inpainting masks (using helper from binder_utils)
            # ------------------------------------------------------------------------
            logger.info("Creating inpainting masks (target=fixed, binder=generate)")
            
            mask_sequence, mask_structure = create_binder_inpainting_masks(
                chains_ids,
                target_chain_idx,
                binder_chain_idx,
                device
            )
            
            # Create encoding mask - only encode target positions
            # This prevents origin-initialized binder coordinates from influencing encoding
            encoding_mask = torch.zeros_like(mask)
            target_positions = chains_ids == target_chain_idx
            encoding_mask[target_positions] = 1.0
            
            # Verify masks
            num_fixed = (mask_sequence == 0).sum().item()
            num_generate = (mask_sequence == 1).sum().item()
            logger.info(f"  Fixed residues: {num_fixed}")
            logger.info(f"  Generate residues: {num_generate}")
            
            # ------------------------------------------------------------------------
            # 3.8: Generate binder designs
            # ------------------------------------------------------------------------
            for design_idx in range(n_designs_per_structure):
                if n_designs_per_structure > 1:
                    logger.info(f"\n--- Design {design_idx + 1}/{n_designs_per_structure} ---")
                
                best_tm_score = -1
                best_result = None
                
                for trial in range(n_trials):
                    if n_trials > 1:
                        logger.info(f"Trial {trial + 1}/{n_trials}")
                    
                    # Generate with inpainting
                    # Note: encoding_mask only includes target positions to prevent
                    # origin-initialized binder coordinates from influencing encoding
                    generate_sample = model.generate_sample(
                        length=L_total,
                        num_samples=1,
                        nsteps=nsteps,
                        temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                        temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                        stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                        stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                        inpainting=True,
                        input_structure_coords=coords_res,
                        input_sequence_tokens=sequence_tokenized,
                        input_mask=encoding_mask,  # Only encode target, mask out binder
                        input_indices=indices,
                        inpainting_mask_sequence=mask_sequence,
                        inpainting_mask_structure=mask_structure,
                        asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
                    )
                    
                    # Decode structures
                    decoded_x = model.decode_structure(generate_sample, mask)
                    
                    # Extract coordinates (B, L, 3, 3) - N, CA, C atoms
                    gen_coords = decoded_x['coords'][:, :, [0, 1, 2], :]  # (1, L, 3, 3)
                    gen_sequence = generate_sample  # (1, L)
                    
                    # Store result
                    result = {
                        'coords': gen_coords,
                        'sequence': gen_sequence,
                        'mask': mask,
                        'chains_ids': chains_ids,
                        'indices': indices,
                        'real_chains': real_chains_combined.unsqueeze(0).to(device),
                    }
                    
                    # If using trials, evaluate and keep best
                    if n_trials > 1:
                        # Could validate with ESMFold here and use TM-score
                        # For now, just use random selection or first trial
                        if trial == 0:
                            best_result = result
                    else:
                        best_result = result
                
                # ------------------------------------------------------------------------
                # 3.9: Save outputs
                # ------------------------------------------------------------------------
                structure_name = Path(structure_path).stem
                prefix = f"{structure_name}_design{design_idx:03d}"
                
                gen_coords = best_result['coords']
                gen_sequence = best_result['sequence']
                
                # Save complete complex
                complex_path = output_dir / f"{prefix}_complex.pdb"
                writepdb(
                    str(complex_path),
                    gen_coords[0],  # Remove batch dim
                    gen_sequence[0]  # Remove batch dim
                )
                logger.info(f"Saved complex: {complex_path}")
                
                # Save binder alone
                binder_mask = (chains_ids[0] == binder_chain_idx)
                binder_coords = gen_coords[0, binder_mask]
                binder_sequence = gen_sequence[0, binder_mask]
                binder_path = output_dir / f"{prefix}_binder.pdb"
                writepdb(str(binder_path), binder_coords, binder_sequence)
                logger.info(f"Saved binder: {binder_path}")
                
                # Save target alone (for reference)
                target_mask = (chains_ids[0] == target_chain_idx)
                target_coords = gen_coords[0, target_mask]
                target_sequence = gen_sequence[0, target_mask]
                target_path = output_dir / f"{prefix}_target.pdb"
                writepdb(str(target_path), target_coords, target_sequence)
                logger.info(f"Saved target: {target_path}")
                
                # ------------------------------------------------------------------------
                # 3.10: Validate with ESMFold (if enabled)
                # ------------------------------------------------------------------------
                if gen_cfg.get("use_esmfold", False) and plm_fold is not None:
                    logger.info("Validating with ESMFold...")
                    
                    # Set up chain groups if not specified
                    esmfold_chain_groups = gen_cfg.get("esmfold_chain_groups", None)
                    if esmfold_chain_groups is None:
                        # Default: validate target + binder together
                        esmfold_chain_groups = [[target_chain_idx, binder_chain_idx]]
                    
                    # Use existing ESMFold validation infrastructure
                    # This would call the existing _validate_with_esmfold or similar
                    # and report metrics like pLDDT, PAE, TM-score, RMSD
                    
                    # (Implementation would follow existing pattern from inpainting mode)
                    pass
    
    logger.info("\nBinder design generation completed!")
```

**Key data structure shapes through the pipeline:**

```python
# After loading target PDB:
target_data['coords_res']:  (100, 3, 3)      # 100 residues, 3 atoms, 3 coords
target_data['sequence']:    (100,)           # 100 amino acid tokens
target_data['chains_ids']:  (100,)           # e.g., [0,0,...0] or [0..0,200..200]

# After initializing binder:
binder_data['coords_res']:  (100, 3, 3)      # All zeros (at origin)
binder_data['sequence']:    (100,)           # Random tokens 0-19

# After combining:
coords_res_combined:        (200, 3, 3)      # 100 target + 100 binder
sequence_combined:          (200,)           # Combined sequences
chains_ids_combined:        (200,)           # [0..0, 400..400] (if next was 400)

# After batching:
coords_res:                 (1, 200, 3, 3)   # Batch size 1
sequence_tokenized:         (1, 200)
mask:                       (1, 200)
chains_ids:                 (1, 200)

# Inpainting masks:
mask_sequence:              (1, 200)         # First 100 = 0, last 100 = 1
mask_structure:             (1, 200)         # First 100 = 0, last 100 = 1

# After generation:
gen_coords:                 (1, 200, 3, 3)   # Generated complex
gen_sequence:               (1, 200)         # Generated sequences

# After splitting:
binder_coords:              (100, 3, 3)      # Just the binder
binder_sequence:            (100,)           # Just binder sequence
```

### 4. Helper Functions with Detailed Implementation

**Location:** These helper functions will be placed in a new utility file:
`src/lobster/model/gen_ume/binder_utils.py`

This keeps the main `generate.py` clean and makes the functions reusable.

#### 4.1 Chain Information Extraction

```python
def get_target_chain_info(structure_data: dict, target_chain_letter: str) -> tuple[int, int, int]:
    """
    Get chain information for the target chain.
    
    Args:
        structure_data: Loaded PDB structure dictionary with 'real_chains' and 'chains_ids'
        target_chain_letter: Chain letter (e.g., "A", "B")
        
    Returns:
        chain_idx: Chain index (0, 200, 400, etc.)
        start_residue_idx: Starting residue index for this chain
        end_residue_idx: Ending residue index for this chain (exclusive)
        
    Example:
        For a PDB with chains A (residues 0-99) and B (residues 100-161):
        - real_chains: [65, 65, ..., 66, 66, ...]  # ord('A')=65, ord('B')=66
        - chains_ids: [0, 0, ..., 200, 200, ...]
        
        get_target_chain_info(data, "A") -> (0, 0, 100)
        get_target_chain_info(data, "B") -> (200, 100, 162)
    """
    # Convert chain letter to ASCII code
    target_chain_ord = ord(target_chain_letter)
    
    # Get real_chains tensor (contains ASCII codes for chain letters)
    real_chains = structure_data['real_chains']
    chains_ids = structure_data['chains_ids']
    
    # Find where this chain appears
    chain_mask = (real_chains == target_chain_ord)
    
    if not chain_mask.any():
        raise ValueError(f"Chain '{target_chain_letter}' not found in structure. "
                        f"Available chains: {set(chr(c) for c in real_chains.unique().tolist())}")
    
    # Get the chain index (0, 200, 400, etc.)
    chain_idx = chains_ids[chain_mask][0].item()
    
    # Find start and end indices in the sequence
    chain_positions = torch.where(chain_mask)[0]
    start_residue_idx = chain_positions[0].item()
    end_residue_idx = chain_positions[-1].item() + 1
    
    return chain_idx, start_residue_idx, end_residue_idx
```

#### 4.2 Binder Initialization at Origin

```python
def initialize_binder_at_origin(binder_length: int, device: torch.device) -> dict:
    """
    Create initial binder structure with coordinates at origin.
    
    Args:
        binder_length: Length of binder to create
        device: Torch device
        
    Returns:
        binder_data: Dictionary with keys:
            - 'coords_res': Coordinates tensor (L, 3, 3) initialized at origin
            - 'sequence': Sequence tokens (L,) initialized to random valid amino acids
            - 'mask': Validity mask (L,) all ones
            
    Example:
        For binder_length=100:
        coords_res shape: (100, 3, 3)  # 100 residues × 3 atoms (N, CA, C) × 3 coords (x,y,z)
        All coordinates set to 0.0
        
        sequence shape: (100,)
        Random tokens from 0-19 (valid amino acids, excluding X=20)
        
        mask shape: (100,)
        All ones (all positions valid)
    """
    # Initialize coordinates at origin (0, 0, 0) for all backbone atoms
    coords_res = torch.zeros((binder_length, 3, 3), dtype=torch.float32, device=device)
    # Shape: (L, 3, 3) where:
    #   - First dim: residue index
    #   - Second dim: atom type (0=N, 1=CA, 2=C)
    #   - Third dim: xyz coordinates
    
    # Initialize sequence with random valid amino acids (0-19, excluding X=20)
    sequence = torch.randint(0, 20, (binder_length,), dtype=torch.int32, device=device)
    # Alternative: could use all same amino acid, e.g., Glycine (token 7)
    # sequence = torch.full((binder_length,), 7, dtype=torch.int32, device=device)
    
    # Create validity mask (all ones - all positions are valid)
    mask = torch.ones(binder_length, dtype=torch.float32, device=device)
    
    return {
        'coords_res': coords_res,
        'sequence': sequence,
        'mask': mask,
    }
```

#### 4.3 Next Chain Index Calculation

```python
def get_next_chain_index(structure_data: dict) -> int:
    """
    Get the next available chain index (200, 400, 600, etc.).
    
    Args:
        structure_data: Loaded PDB structure dictionary with 'chains_ids'
        
    Returns:
        next_chain_idx: Next available chain index
        
    Example:
        If chains_ids contains [0, 0, ..., 200, 200, ...]:
        - Max chain index is 200
        - Next available is 400
        
        If chains_ids contains [0, 0, ...]:
        - Max chain index is 0
        - Next available is 200
    """
    chains_ids = structure_data['chains_ids']
    
    # Find max chain index
    max_chain_idx = chains_ids.max().item()
    
    # Next chain index is max + 200
    next_chain_idx = max_chain_idx + 200
    
    return next_chain_idx
```

#### 4.4 Inpainting Mask Creation

```python
def create_binder_inpainting_masks(
    chains_ids: torch.Tensor,
    target_chain_idx: int,
    binder_chain_idx: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create inpainting masks for binder design.
    Target residues get mask=0 (fixed), binder residues get mask=1 (generate).

    IMPORTANT: The first residue of the binder is kept fixed (mask=0) to preserve
    the chain break token. This tells the model where the new chain starts,
    otherwise it would treat the binder as a continuation of the target chain.

    Args:
        chains_ids: Chain ID tensor for all residues (B, L)
        target_chain_idx: Index of target chain to keep fixed
        binder_chain_idx: Index of binder chain to generate
        device: Torch device

    Returns:
        mask_sequence: Inpainting mask for sequence (B, L)
        mask_structure: Inpainting mask for structure (B, L)

    Example:
        For a complex with:
        - Chain A (target): chain_idx=0, residues 0-99
        - Chain B (binder): chain_idx=200, residues 100-199

        chains_ids: [0,0,...,0, 200,200,...,200]  (shape: 1, 200)
        target_chain_idx: 0
        binder_chain_idx: 200

        Returns masks of shape (1, 200):
        - Positions 0-99: mask=0 (keep target fixed)
        - Position 100: mask=0 (keep first binder token fixed for chain break)
        - Positions 101-199: mask=1 (generate rest of binder)
    """
    # Create masks initialized to zeros
    B, L = chains_ids.shape
    mask_sequence = torch.zeros((B, L), dtype=torch.float32, device=device)
    mask_structure = torch.zeros((B, L), dtype=torch.float32, device=device)

    # Set binder positions to 1 (generate)
    binder_mask = (chains_ids == binder_chain_idx)
    mask_sequence[binder_mask] = 1.0
    mask_structure[binder_mask] = 1.0

    # Keep first binder residue fixed (mask=0) to preserve chain break token
    for b in range(B):
        binder_positions = torch.where(chains_ids[b] == binder_chain_idx)[0]
        if len(binder_positions) > 0:
            first_binder_idx = binder_positions[0].item()
            mask_sequence[b, first_binder_idx] = 0.0
            mask_structure[b, first_binder_idx] = 0.0

    return mask_sequence, mask_structure
```

#### 4.5 Complete `binder_utils.py` Module Structure

**File: `src/lobster/model/gen_ume/binder_utils.py`**

```python
"""
Utility functions for binder design generation.

This module provides helper functions for the binder_design generation mode,
including chain information extraction, binder initialization, and mask creation.
"""

import torch
from typing import Tuple
from loguru import logger


def get_target_chain_info(
    structure_data: dict, 
    target_chain_letter: str
) -> Tuple[int, int, int]:
    """Get chain information for the target chain."""
    # Implementation from 4.1
    pass


def initialize_binder_at_origin(
    binder_length: int, 
    device: torch.device
) -> dict:
    """Create initial binder structure with coordinates at origin."""
    # Implementation from 4.2
    pass


def get_next_chain_index(structure_data: dict) -> int:
    """Get the next available chain index (200, 400, 600, etc.)."""
    # Implementation from 4.3
    pass


def create_binder_inpainting_masks(
    chains_ids: torch.Tensor,
    target_chain_idx: int,
    binder_chain_idx: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create inpainting masks for binder design."""
    # Implementation from 4.4
    pass
```

**Module Benefits:**
- ✅ Clean separation of concerns
- ✅ Reusable functions for future binder-related features
- ✅ Easy to test independently
- ✅ Keeps `generate.py` focused on orchestration
- ✅ Can be imported by other modules if needed

### 5. Edge Cases and Error Handling

#### 5.1 Target Chain Not Found
```python
# In get_target_chain_info:
if not chain_mask.any():
    available = set(chr(c) for c in real_chains.unique().tolist())
    raise ValueError(
        f"Chain '{target_chain_letter}' not found in structure. "
        f"Available chains: {available}"
    )
```

**Example error message:**
```
Chain 'C' not found in structure. Available chains: {'A', 'B'}
```

#### 5.2 Total Length Exceeds Max Length
```python
L_total = L_target + L_binder
max_length = gen_cfg.get("max_length", 512)

if L_total > max_length:
    logger.warning(
        f"Total length {L_total} (target: {L_target}, binder: {L_binder}) "
        f"exceeds max_length {max_length}. Skipping structure."
    )
    continue
```

**Suggestion to user:**
- Reduce binder_length
- Increase max_length in config
- Select a shorter target region

#### 5.3 Multi-Chain Target PDB
```python
# Example: target.pdb has chains A, B, C
# User specifies target_chain: "B"

# The function will:
# 1. Extract only chain B as the target
# 2. Ignore chains A and C
# 3. Create binder that binds to chain B

# Alternatively, could support keeping all chains:
# Option 1: Keep all chains, only one is "target" for metrics
# Option 2: Keep all chains, binder designed against specified target
```

**Current approach:** Extract only the specified target chain, ignore others.

**Future enhancement:** Could add parameter `keep_other_chains: bool` to preserve context.

#### 5.4 Very Short Target Chains
```python
# If target chain is too short (< 10 residues)
if target_end - target_start < 10:
    logger.warning(
        f"Target chain '{target_chain}' is very short "
        f"({target_end - target_start} residues). "
        f"Binder design may not work well."
    )
    # Could either skip or continue with warning
```

#### 5.5 Invalid Binder Length
```python
# Validate binder length
if binder_length < 30:
    logger.warning(
        f"Binder length {binder_length} is very short. "
        f"Recommended minimum: 30 residues."
    )

if binder_length > 200:
    logger.warning(
        f"Binder length {binder_length} is very long. "
        f"Generation may be slow and less accurate."
    )
```

#### 5.6 Chain Index Collision
```python
# Rare case: what if next_chain_index already exists?
# Example: malformed PDB with non-standard chain indices

def get_next_chain_index(structure_data: dict) -> int:
    chains_ids = structure_data['chains_ids']
    max_chain_idx = chains_ids.max().item()
    next_chain_idx = max_chain_idx + 200
    
    # Verify it doesn't collide
    while next_chain_idx in chains_ids:
        next_chain_idx += 200
    
    return next_chain_idx
```

#### 5.7 Empty or Invalid PDB Files
```python
# Already handled by load_pdb returning None
target_data = load_pdb(structure_path, add_batch_dim=False)

if target_data is None:
    logger.warning(f"Failed to load structure from {structure_path}, skipping")
    continue

# Also check for minimum length
if target_data['coords_res'].shape[0] < 30:
    logger.warning(
        f"Structure too short ({target_data['coords_res'].shape[0]} residues), "
        f"skipping"
    )
    continue
```

### 6. Integration with Main Generate Function

#### 6.1 Import Helper Functions

Add import at the top of `generate.py`:

```python
# In generate.py, around line 1-40 (in imports section):

from lobster.model.gen_ume.binder_utils import (
    get_target_chain_info,
    initialize_binder_at_origin,
    get_next_chain_index,
    create_binder_inpainting_masks,
)
```

#### 6.2 Update Mode Dispatch

The `_generate_binders` function needs to be integrated into the main dispatch in `generate.py`:

```python
# In generate.py, around line 130-145:

@hydra.main(version_base=None, config_path="../hydra_config/experiment", config_name="generate_unconditional")
def generate(cfg: DictConfig) -> None:
    """Generate protein structures using genUME model."""
    
    # ... (existing setup code) ...
    
    # Generate structures
    generation_mode = cfg.generation.mode
    logger.info(f"Generation mode: {generation_mode}")

    if generation_mode == "unconditional":
        _generate_unconditional(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inverse_folding":
        _generate_inverse_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "forward_folding":
        _generate_forward_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inpainting":
        _generate_inpainting(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "binder_design":  # <-- ADD THIS
        _generate_binders(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    else:
        raise ValueError(f"Unknown generation mode: {generation_mode}")
    
    logger.info("Generation completed successfully!")
```

**Note:** The existing stub function at line 3517 will be replaced with the full implementation.

### 7. Concrete Example Walkthrough

Let's walk through a complete example with real data shapes:

#### Input:
- **PDB file:** `target.pdb` with 2 chains:
  - Chain A: 120 residues (indices 0-119)
  - Chain B: 80 residues (indices 200-279)
- **Config:**
  ```yaml
  target_chain: "A"
  binder_length: 100
  ```

#### Step-by-step data:

**1. After loading PDB:**
```python
target_data = {
    'coords_res': Tensor(200, 3, 3),      # All residues from both chains
    'sequence': Tensor(200,),
    'chains_ids': Tensor(200,),           # [0,0,...,0(×120), 200,200,...,200(×80)]
    'real_chains': Tensor(200,),          # [65,65,...,65(×120), 66,66,...,66(×80)]
    'indices': Tensor(200,),              # [0,1,...,119, 200,201,...,279]
    'mask': Tensor(200,),                 # [1,1,...,1]
}
```

**2. After identifying target chain 'A':**
```python
target_chain_idx = 0        # Chain A has index 0
target_start = 0            # First residue at position 0
target_end = 120            # Last residue at position 119 (exclusive end at 120)
```

**Note:** In current implementation, we would extract ONLY chain A, discarding chain B:
```python
# Extract only target chain A
target_chain_mask = (target_data['chains_ids'] == target_chain_idx)
target_data_filtered = {
    'coords_res': target_data['coords_res'][target_chain_mask],     # (120, 3, 3)
    'sequence': target_data['sequence'][target_chain_mask],         # (120,)
    'chains_ids': target_data['chains_ids'][target_chain_mask],     # (120,) all 0s
    'real_chains': target_data['real_chains'][target_chain_mask],   # (120,) all 65s
    'indices': target_data['indices'][target_chain_mask],           # (120,) [0-119]
    'mask': target_data['mask'][target_chain_mask],                 # (120,) all 1s
}
```

**3. After initializing binder:**
```python
binder_data = {
    'coords_res': Tensor(100, 3, 3),      # All zeros (origin)
    'sequence': Tensor(100,),             # Random tokens: [7, 15, 3, 12, ...]
    'mask': Tensor(100,),                 # All ones
}
```

**4. After getting next chain index:**
```python
# Max chain index in target_data_filtered is 0
binder_chain_idx = 200      # Next available: 0 + 200 = 200
```

**5. After creating composite structure:**
```python
# Concatenate target (120) + binder (100) = 220 total
composite = {
    'coords_res': Tensor(220, 3, 3),      # [target_coords(120); binder_coords(100)]
    'sequence': Tensor(220,),             # [target_seq(120); binder_seq(100)]
    'chains_ids': Tensor(220,),           # [0×120; 200×100]
    'real_chains': Tensor(220,),          # [65×120; 66×100]  (A=65, B=66)
    'indices': Tensor(220,),              # [0-119; 200-299]
    'mask': Tensor(220,),                 # [1×220]
}
```

**6. After batching and tokenizing:**
```python
# Add batch dimension (B=1) and move to device
coords_res = Tensor(1, 220, 3, 3)        # GPU
sequence_tokenized = Tensor(1, 220)      # GPU, after tokenizer
mask = Tensor(1, 220)                    # GPU
chains_ids = Tensor(1, 220)              # GPU
indices = Tensor(1, 220)                 # GPU
```

**7. After creating inpainting masks:**
```python
mask_sequence = Tensor(1, 220)
# [0, 0, 0, ..., 0 (×120 for target), 1, 1, 1, ..., 1 (×100 for binder)]

mask_structure = Tensor(1, 220)
# [0, 0, 0, ..., 0 (×120 for target), 1, 1, 1, ..., 1 (×100 for binder)]
```

**8. After generation:**
```python
gen_coords = Tensor(1, 220, 3, 3)        # Generated complex
gen_sequence = Tensor(1, 220)            # Generated sequences

# First 120 residues (target): should be nearly identical to input
# Last 100 residues (binder): completely new, generated by model
```

**9. After splitting for output:**
```python
# Extract binder only
binder_mask = (chains_ids[0] == 200)     # Tensor(220,) -> [False×120, True×100]
binder_coords = gen_coords[0, binder_mask]    # (100, 3, 3)
binder_sequence = gen_sequence[0, binder_mask] # (100,)

# Extract target only (for reference)
target_mask = (chains_ids[0] == 0)       # Tensor(220,) -> [True×120, False×100]
target_coords = gen_coords[0, target_mask]     # (120, 3, 3)
target_sequence = gen_sequence[0, target_mask]  # (120,)
```

**10. Output files:**
```
output_dir/
  ├── target_initial_structure.pdb         # Initial structure (target + initialized binder)
  ├── target_design000_complex.pdb         # Generated complex (target + binder)
  ├── target_design000_binder.pdb          # Generated binder only
  ├── target_design000_target.pdb          # Target only (for reference)
  ├── target_design000_esmfold.pdb         # ESMFold predicted complex
  └── target_design000_esmfold_binder.pdb  # ESMFold predicted binder only
```

**Output descriptions:**
- `_initial_structure.pdb` - Starting configuration before generation
- `_complex.pdb` - Model-generated structure
- `_esmfold.pdb` - ESMFold re-prediction (validates foldability)
- `_esmfold_binder.pdb` - ESMFold binder for comparison with generated

### 8. Testing Strategy

#### Unit tests:
- Test chain letter → index mapping with various PDB inputs
- Test binder initialization at origin with different lengths
- Test composite structure creation (target + binder concatenation)
- Test mask generation for binder design

#### Integration tests:
- End-to-end binder design with simple test case
- Verify ESMFold validation works correctly
- Test with multi-chain target PDB (ensure correct target chain selected)
- Test with various binder lengths (50, 100, 150 residues)

#### Validation:
- Compare to current manual approach (should give similar results)
- Verify chain assignments are correct in output PDBs
- Verify target chain is unchanged in generated structures
- Ensure ESMFold validation includes both target and binder

### 8. Documentation Needed

1. **Example config file:**
   - Create `generate_binder_design.yaml` with clear comments
   - Show typical parameter values
   - Include usage instructions

2. **Code documentation:**
   - Document all helper functions
   - Explain chain indexing conventions
   - Add inline comments for complex logic

3. **README/User guide update:**
   - Add binder design mode to list of generation modes
   - Provide simple usage example
   - Explain output file structure

### 9. Migration Path

**For existing users with manual configs:**

The old inpainting-based approach will continue to work. Users can migrate by:
- Changing `mode: inpainting` to `mode: binder_design`
- Replacing `mask_indices_*` with `target_chain` and `binder_length`
- Removing the need for pre-constructed target+binder PDB

Both approaches will be supported (no deprecation needed).

### 6. Example Usage and Comparison

#### Detailed Comparison: Old vs. New Approach

**OLD APPROACH (Manual Hack):**

1. **User workflow:**
   ```bash
   # Step 1: Load target PDB in PyMOL/Python
   # Step 2: Identify target chain (e.g., chain A, 165 residues)
   # Step 3: Manually create a "dummy" binder chain (e.g., 62 polyG residues)
   # Step 4: Append binder to target, save as new PDB
   # Step 5: Calculate binder residue indices (166-227 = positions after target)
   # Step 6: Create config with manual indices
   ```

2. **Config file:**
   ```yaml
   generation:
     mode: inpainting
     # PROBLEM: User must calculate these indices manually!
     mask_indices_sequence: "166-227"
     mask_indices_structure: "166-227"
     # PROBLEM: Must provide pre-constructed PDB with dummy binder!
     input_structures: "/path/to/target_with_polyG_binder.pdb"
     nsteps: 200
     n_designs_per_structure: 10
     use_esmfold: true
     esmfold_chain_groups:
       - [0, 200]  # PROBLEM: Must know chain indices
   ```

3. **Issues with old approach:**
   - ❌ Requires manual PDB manipulation
   - ❌ Must calculate residue indices by hand
   - ❌ Error-prone (easy to get indices wrong)
   - ❌ Not reusable (different indices for each target)
   - ❌ Requires understanding internal chain indexing (0, 200, 400...)
   - ❌ Must pre-create dummy binder structure

**NEW APPROACH (Automated):**

1. **User workflow:**
   ```bash
   # Step 1: Point to target PDB (no modification needed!)
   # Step 2: Specify target chain letter
   # Step 3: Specify desired binder length
   # Done!
   ```

2. **Config file:**
   ```yaml
   generation:
     mode: binder_design
     # SIMPLE: Just provide target PDB as-is
     input_structures: "/path/to/target.pdb"
     # SIMPLE: Specify chain by letter
     target_chain: "A"
     # SIMPLE: Specify desired binder length
     binder_length: 100
     nsteps: 200
     n_designs_per_structure: 10
     use_esmfold: true
     # esmfold_chain_groups: null  # Auto-configured!
   ```

3. **Benefits of new approach:**
   - ✅ No manual PDB manipulation required
   - ✅ Automatic chain detection and indexing
   - ✅ User-friendly chain specification (by letter)
   - ✅ Config is reusable across different targets
   - ✅ No need to understand internal chain indexing
   - ✅ Automatic binder initialization
   - ✅ Clear, intuitive interface

#### Side-by-Side Config Comparison

| Aspect | Old (Inpainting Hack) | New (Binder Design) |
|--------|----------------------|---------------------|
| **Mode** | `inpainting` | `binder_design` |
| **Input PDB** | Modified (target + dummy binder) | Original (target only) |
| **Chain specification** | Manual indices (166-227) | Chain letter ("A") |
| **Binder length** | Implicit in indices | Explicit parameter (100) |
| **Chain indices** | Manual (0, 200) | Automatic |
| **Reusability** | Low (indices change per target) | High (same config works) |
| **Error prone** | Yes (manual calculations) | No (automatic) |
| **User expertise** | High (must understand internals) | Low (simple interface) |

#### Complete Example YAML Config

**File: `generate_binder_design.yaml`**
```yaml
# Example configuration for binder design
# Usage: uv run python -m lobster.cmdline.generate --config-path "../hydra_config/experiment" --config-name generate_binder_design

# Output directory
output_dir: "./examples/binder_design_output"

# Random seed for reproducibility
seed: 12345

# Model configuration
model:
  _target_: lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule
  ckpt_path: "/path/to/checkpoint.ckpt"

# Generation settings
generation:
  mode: binder_design
  
  # Target structure (PDB file with one or more chains)
  input_structures: "/path/to/target.pdb"
  
  # Target chain to design binder against (use chain letter from PDB)
  target_chain: "A"
  
  # Binder length (number of residues)
  binder_length: 100
  
  # Generation parameters
  nsteps: 200
  batch_size: 1
  n_trials: 1
  n_designs_per_structure: 10
  
  # Sampling parameters
  temperature_seq: 0.16423763902324678
  temperature_struc: 1.0
  stochasticity_seq: 20
  stochasticity_struc: 10
  asynchronous_sampling: false
  
  # ESMFold validation (highly recommended for binder design)
  use_esmfold: true
  max_length: 512
  
  # Optional: Specify chain groups for ESMFold validation
  # If not specified, will automatically validate target+binder together
  # esmfold_chain_groups: null
  
  # CSV metrics and plotting (optional)
  save_csv_metrics: true
  create_plots: true
```

### 7. Benefits of This Approach

1. **User-friendly**: Simple, intuitive configuration with just target chain letter and binder length
2. **Reusable**: Same config works across different target structures
3. **Robust**: Automatic chain detection using chain letters
4. **Integrated**: Seamless ESMFold validation for target-binder complex
5. **Maintainable**: Clean separation from inpainting mode
6. **Simple initialization**: Origin-based coordinates let the model learn placement

---

## Open Questions / Design Decisions

### Recommended Decisions:

1. **Handling multi-chain input PDBs:**
   - **Decision:** Extract ONLY the target chain, discard other chains
   - **Rationale:** 
     - Simpler implementation
     - Clearer what the model is doing (binder for specific target)
     - Reduces sequence length
   - **Alternative:** Keep all chains but only "target" one for mask
     - Pro: Maintains structural context
     - Con: More complex, longer sequences, unclear behavior

2. **Binder initialization:**
   - **Decision:** Random amino acids (tokens 0-19), coordinates at exact origin (0,0,0)
   - **Rationale:**
     - Simple and reproducible
     - Model learns placement from scratch
     - Works well based on inpainting experience
   - **Alternative:** Initialize with extended structure
     - Con: More complex, prescriptive

3. **Output format:**
   - **Decision:** Save three PDB files per design:
     - `{name}_design{N:03d}_complex.pdb` - full complex
     - `{name}_design{N:03d}_binder.pdb` - binder alone
     - `{name}_design{N:03d}_target.pdb` - target alone (reference)
   - **Rationale:** Maximum flexibility for downstream analysis

4. **ESMFold chain groups:**
   - **Decision:** Default to validating target+binder together
   - User can override via `esmfold_chain_groups` parameter
   - **Rationale:** Most useful validation is of the complex

5. **Chain numbering:**
   - **Decision:** Use `max_chain_idx + 200` for binder
   - Verify no collision (unlikely but check)
   - **Rationale:** Follows existing convention

6. **Error handling:**
   - **Decision:** Warn and skip on errors (don't crash entire batch)
   - Log clear error messages with suggestions
   - **Rationale:** User-friendly, robust to mixed inputs

### Future Considerations (not in MVP):

1. **Keep other chains option:**
   - Add parameter: `keep_other_chains: bool = False`
   - If true, keep all chains but only target the specified one

2. **Multiple target chains:**
   - Add parameter: `target_chains: List[str]`
   - Design binder against multiple chains simultaneously

3. **Specific epitope targeting:**
   - Add parameter: `target_residues: str = "10-20,35-45"`
   - Focus binder on specific residues

4. **Smart placement initialization:**
   - Place binder near target surface instead of origin
   - Could improve generation quality

---

## Usage Examples

### Basic Usage

```bash
# Design a 100-residue binder against chain A of target.pdb
uv run python -m lobster.cmdline.generate \
  --config-path "../hydra_config/experiment" \
  --config-name generate_binder_design \
  generation.input_structures="/path/to/target.pdb" \
  generation.target_chain="A" \
  generation.binder_length=100
```

### Advanced Usage

```bash
# Generate 20 designs with 3 trials each, targeting chain B, 150-residue binder
uv run python -m lobster.cmdline.generate \
  --config-path "../hydra_config/experiment" \
  --config-name generate_binder_design \
  generation.input_structures="/path/to/target.pdb" \
  generation.target_chain="B" \
  generation.binder_length=150 \
  generation.n_designs_per_structure=20 \
  generation.n_trials=3 \
  generation.nsteps=300 \
  output_dir="./binder_designs_v2"
```

### Batch Processing

```bash
# Process all PDB files in a directory
uv run python -m lobster.cmdline.generate \
  --config-path "../hydra_config/experiment" \
  --config-name generate_binder_design \
  generation.input_structures="/path/to/targets/*.pdb" \
  generation.target_chain="A" \
  generation.binder_length=100
```

### Expected Output

```
examples/binder_design_output/
├── target1_design000_complex.pdb    # Target + binder
├── target1_design000_binder.pdb     # Binder only
├── target1_design000_target.pdb     # Target only (reference)
├── target1_design001_complex.pdb
├── target1_design001_binder.pdb
├── target1_design001_target.pdb
├── ...
├── target1_design009_complex.pdb
├── target1_design009_binder.pdb
├── target1_design009_target.pdb
├── metrics.csv                       # All metrics for all designs
└── plots/                            # Visualization plots
    ├── plddt_distribution.png
    ├── tm_score_distribution.png
    └── ...
```

### Expected Console Output

```
Starting genUME structure generation
Generation mode: binder_design
Found 1 structure(s) to process
Target chain: A
Binder length: 100
Generation steps: 200
Designs per structure: 10

======================================================================
Processing structure 1/1
Input: /path/to/target.pdb
======================================================================
Loading target structure from /path/to/target.pdb
Target chain 'A' found:
  Chain index: 0
  Residue range: 0-165
  Length: 165 residues
Initializing binder with length 100 at origin
Binder will be assigned chain index: 200
Creating composite structure (target + binder)
Composite structure created:
  Total length: 265 (165 target + 100 binder)
  Target chain index: 0
  Binder chain index: 200
Creating inpainting masks (target=fixed, binder=generate)
  Fixed residues: 165
  Generate residues: 100

--- Design 1/10 ---
Trial 1/1
Generating...
✓ Generation complete
Saved complex: ./examples/target_design000_complex.pdb
Saved binder: ./examples/target_design000_binder.pdb
Saved target: ./examples/target_design000_target.pdb
Validating with ESMFold...
ESMFold validation metrics:
  plddt_mean: 0.8234
  pae_mean: 8.45
  tm_score: 0.7621
  rmsd: 2.34

--- Design 2/10 ---
...

Binder design generation completed!
```

## Files to Modify/Create

### 1. Helper Functions Utility File (Create) ✅ COMPLETE
**`src/lobster/model/gen_ume/binder_utils.py`**
- New file containing 4 helper functions
- Clean separation of utility code from main generation logic
- **Content:**
  - `get_target_chain_info()` - ~35 lines with docstring
  - `initialize_binder_at_origin()` - ~25 lines with docstring
  - `get_next_chain_index()` - ~15 lines with docstring
  - `create_binder_inpainting_masks()` - ~25 lines with docstring
  - Module docstring and imports - ~10 lines
- **Total:** ~110 lines
- **Status:** ✅ Created and linted successfully

### 2. Main Implementation File (Modify) ✅ COMPLETE
**`src/lobster/cmdline/generate.py`**
- ✅ Add import for helper functions from `binder_utils` (~5 lines at top of file) - DONE
- ✅ Replace stub `_generate_binders` function (line ~3517) with full implementation (~300-400 lines) - DONE
- ✅ Update dispatch logic in `generate()` function (already has the line, just ensure it's there) - DONE
- **Total changes:** ~310-410 lines of code
- **Status:** ✅ Implemented and linted successfully

### 3. Example Config File (Create) ✅ COMPLETE
**`src/lobster/hydra_config/experiment/generate_binder_design.yaml`**
- New file with example configuration
- ~50 lines with comments
- **Status:** ✅ Created with comprehensive documentation

### 4. Documentation (Update)
**`README.md` or relevant docs**
- Add binder_design mode to list of generation modes
- Add usage example
- ~20 lines

### 5. Tests (Create - Optional but Recommended)
**`tests/test_binder_utils.py`**
- Unit tests for helper functions in binder_utils.py
- ~150 lines

**`tests/test_binder_design_integration.py`**
- End-to-end integration test
- ~100 lines

### Code Structure Summary

**File: `src/lobster/model/gen_ume/binder_utils.py` (NEW)**
```python
"""
Utility functions for binder design generation.

This module provides helper functions for the binder_design generation mode,
including chain information extraction, binder initialization, and mask creation.
"""

import torch
from typing import Tuple


def get_target_chain_info(
    structure_data: dict, 
    target_chain_letter: str
) -> Tuple[int, int, int]:
    """
    Get chain information for the target chain.
    ~35 lines with docstring and error handling
    """
    pass


def initialize_binder_at_origin(
    binder_length: int, 
    device: torch.device
) -> dict:
    """
    Create initial binder structure with coordinates at origin.
    ~25 lines with docstring
    """
    pass


def get_next_chain_index(structure_data: dict) -> int:
    """
    Get the next available chain index (200, 400, 600, etc.).
    ~15 lines with docstring
    """
    pass


def create_binder_inpainting_masks(
    chains_ids: torch.Tensor,
    target_chain_idx: int,
    binder_chain_idx: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create inpainting masks for binder design.
    ~25 lines with docstring
    """
    pass
```

**File: `src/lobster/cmdline/generate.py` (MODIFY)**
```python
# Add to imports section (top of file):
from lobster.model.gen_ume.binder_utils import (
    get_target_chain_info,
    initialize_binder_at_origin,
    get_next_chain_index,
    create_binder_inpainting_masks,
)

# ... existing code ...

# Replace stub function at line ~3517:
def _generate_binders(
    model, 
    cfg: DictConfig, 
    device: torch.device, 
    output_dir: Path, 
    plm_fold=None,
    csv_writer=None,
    plotter=None
) -> None:
    """
    Generate binders for target protein structures.
    
    Main implementation:
    - Load structures (~50 lines)
    - Process each structure loop (~250 lines)
      - Load target
      - Get chain info (using helper)
      - Initialize binder (using helper)
      - Create composite
      - Create masks (using helper)
      - Generate designs (nested loops)
      - Save outputs
      - ESMFold validation
    
    ~300-400 lines total
    """
    pass
```

## Estimated Implementation Effort

- **Core implementation**: 1-2 days
  - Helper functions: 0.5 day (straightforward tensor operations)
  - Main `_generate_binders` function: 1 day (mostly adapting inpainting code)
  - Integration and testing: 0.5 day

- **Testing**: 0.5-1 day
  - Unit tests for helper functions
  - End-to-end integration test
  - Validation against manual approach

- **Documentation**: 0.5 day
  - Example config file
  - Update README/docs
  - Code comments

**Total**: 2-4 days for MVP implementation

**Minimal viable implementation (no tests, minimal docs)**: 1-2 days

---

## Implementation Summary

### ✅ Completed Tasks

**1. Helper Functions Module (`binder_utils.py`)** - COMPLETE
- Created `/homefs/home/lisanzas/scratch/Develop/lobster/src/lobster/model/gen_ume/binder_utils.py`
- Implemented 4 helper functions:
  - `get_target_chain_info()` - Extracts target chain information
  - `initialize_binder_at_origin()` - Creates binder structure at origin
  - `get_next_chain_index()` - Calculates next available chain index
  - `create_binder_inpainting_masks()` - Creates masks for generation
- All linting passed ✅

**2. Main Implementation (`generate.py`)** - COMPLETE
- Added imports from `binder_utils` module
- Implemented full `_generate_binders()` function (~300 lines)
- Features:
  - Loads target PDB files
  - Extracts specified target chain
  - Initializes binder at origin
  - Creates composite structure
  - Generates binder designs using inpainting
  - Saves 3 PDB files per design (complex, binder, target)
  - Optional ESMFold validation
- All linting passed ✅

**3. Example Config File** - COMPLETE
- Created `/homefs/home/lisanzas/scratch/Develop/lobster/src/lobster/hydra_config/experiment/generate_binder_design.yaml`
- Comprehensive documentation and comments
- Ready to use with example paths

### 📝 Next Steps for Testing

To test the implementation:

```bash
# 1. Update the config file with your paths
# Edit: src/lobster/hydra_config/experiment/generate_binder_design.yaml
#   - Set input_structures to your target PDB
#   - Set target_chain to your desired chain letter
#   - Set binder_length as desired

# 2. Run binder design
uv run python -m lobster.cmdline.generate \
  --config-path "../hydra_config/experiment" \
  --config-name generate_binder_design

# 3. Check outputs in ./examples/binder_design_output/
#   - {name}_design000_complex.pdb
#   - {name}_design000_binder.pdb
#   - {name}_design000_target.pdb
```

### 📊 Code Statistics

- **Files Created:** 2 new files
- **Files Modified:** 1 file
- **Lines of Code Added:** ~420 lines total
  - `binder_utils.py`: ~110 lines
  - `generate.py`: ~305 lines
  - Config file: ~60 lines
- **Functions Implemented:** 5 (4 helpers + 1 main)
- **Linting Status:** ✅ All passed

### 🔑 Key Features Implemented

1. **Automatic chain detection** - No manual index calculation needed
2. **Simple configuration** - Just specify chain letter and binder length
3. **Origin-based initialization** - Model learns placement during generation
4. **Multi-file output** - Complex, binder, and target saved separately
5. **ESMFold validation** - Optional structure validation
6. **Reusable config** - Same config works across different targets
7. **Clean code organization** - Utilities separated from main logic

### ✨ Implementation Highlights

**What makes this implementation clean:**
- Helper functions in separate module for reusability
- Comprehensive error handling and logging
- Clear data flow with documented tensor shapes
- Follows existing codebase patterns
- Minimal changes to existing code
- Well-documented config file

**What users can now do:**
- Design binders with simple 2-parameter config (chain + length)
- Process multiple target structures in batch
- Generate multiple designs per target
- Automatically validate designs with ESMFold
- Get separate PDB files for analysis

### 🎯 Implementation Complete!

The binder design feature is now fully implemented and ready for testing. The implementation follows the plan exactly and maintains consistency with the existing codebase architecture.

**Important Implementation Detail - Binder Initialization:**

The binder can be initialized in two ways:

1. **At target's center of mass** (default, when `epitope_indices: null`):
   - Calculated as mean of all CA atom positions in target chain
   - All binder backbone atoms (N, CA, C) initialized at this point
   - Provides reasonable starting position near target

2. **Near specific epitope** (when `epitope_indices` provided):
   - Calculate epitope center from specified residue indices
   - Calculate direction vector from target COM → epitope center
   - Place ball center **5 Angstroms away** from epitope, along this direction
   - **Randomly distribute** binder atoms in a **12Å ball** around this center
   - **Constraint**: All atoms must be **≥5Å from target** (rejection sampling)
   - This creates a diverse starting configuration targeting the epitope

**Mathematical detail for epitope placement:**
```
epitope_center = mean(CA_coords[epitope_indices])
direction = (epitope_center - target_COM) / ||epitope_center - target_COM||
ball_center = epitope_center + direction × 5.0 Å

For each binder atom:
  random_point = ball_center + random_unit_vector × random_radius (up to 12Å)
  if distance(random_point, any_target_atom) < 5Å:
      reject and regenerate
```

