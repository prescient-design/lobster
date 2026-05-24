# Token Space Rules: Gen-UME Structure Tokenizer (FSQ 4375)

## Model: `LG Protein Ligand fsq 4375`
- Vocabulary: 4,375 tokens (FSQ quantizer, levels produce this codebook size)
- Separate protein and ligand quantizers (both 4,375 tokens)
- Encoder: 6-layer ViT with pairwise distance features (20 Å cutoff)
- **No frame normalization** (`frame_type=None`) — tokens are orientation-dependent
- Typical vocabulary usage: ~179 unique tokens per protein (4.1% of codebook)

---

## Rule 1: Tokens encode ABSOLUTE coordinates, not local structure

**Finding**: Both rotations and translations change nearly **100% of all tokens**.

| Perturbation | Token change rate |
|---|---|
| 90° rotation (any axis) | 96–100% |
| 180° rotation | 98–100% |
| Translation by 10 Å | 98–100% |
| Translation by 100 Å | 97–100% |
| 0.1 Å perturbation at 1 residue | 98–99% |

**Implication**: The tokenizer operates on raw XYZ coordinates without canonical frame normalization. The same structure in different orientations produces completely different tokens. There is no SE(3) invariance.

**Practical consequence**: You cannot meaningfully manipulate individual tokens to achieve local structural changes. Changing one token changes the global context, which requires all other tokens to be re-decoded consistently.

---

## Rule 2: Token sensitivity is GLOBAL, not local

**Finding**: Perturbing a single residue by just 0.1 Å changes ~99% of all tokens across the entire protein.

| Perturbation magnitude | Avg tokens changed |
|---|---|
| 0.1 Å (1 residue) | 99.3% |
| 0.5 Å (1 residue) | 98.7% |
| 1.0 Å (1 residue) | 99.2% |
| 2.0 Å (1 residue) | 99.1% |
| 5.0 Å (1 residue) | 99.0% |

**Implication**: The encoder uses global self-attention with pairwise distance features. Even a tiny coordinate change propagates through all attention layers, affecting every position's token. The perturbation magnitude doesn't matter — any change, no matter how small, causes nearly complete token reassignment.

**Practical consequence**: Token space is NOT a space where "nearby tokens = similar structures." Two structures differing by 0.1 Å at one residue may have completely different token sequences, but decode to nearly identical structures. The decoder is much more robust than the encoder is discriminative.

---

## Rule 3: Tokens do NOT correspond to secondary structure elements

**Finding**: No strong association between specific tokens and secondary structure types (helix/sheet/coil). The most enriched token for any SS type appears at only ~0.5% frequency within that SS type.

- 10 proteins analyzed, ~2,700 residues
- 1,108 helix residues spread across ~800+ unique tokens
- 327 sheet residues spread across ~250+ unique tokens
- No token appears more than 5 times for any SS type

**Implication**: Because tokens encode absolute coordinates (Rule 1), the same helical turn at two different positions in 3D space gets completely different tokens. Secondary structure is implicitly encoded in the pairwise relationships between consecutive tokens, not in individual token identities.

**Practical consequence**: You cannot "insert a helix" by finding "helix tokens" and placing them. There is no such thing as a "helix token" — there are only "helix-at-this-specific-3D-position" tokens.

---

## Rule 4: The decoder is robust despite encoder sensitivity

**Finding**: Encode → decode roundtrip produces ~0.86 Å RMSD, even though the token representation is orientation-dependent.

- Reconstruction RMSD: 0.858 Å (tmbar protein, 192 residues)
- Prior studies: FSQ models achieve 0.8–1.5 Å reconstruction RMSD across diverse proteins

**Implication**: The decoder has learned to produce valid structures from any token configuration that arose from a real structure during training. The encoder-decoder pair is consistent even though the token space is not rotation-invariant.

**Practical consequence**: Token manipulation experiments (mutation, interpolation, swapping) work because the decoder is forgiving. Even though individual token meanings are position-dependent, the decoder has learned the global consistency constraints.

---

## Rule 5: Token interpolation produces smooth structural transitions

**Experiment** (FSQ 4375, efhand → tmbar, both 192 residues):

Two interpolation methods tested:

**Method 1 — Token replacement** (progressive swap from A→B):
| Fraction replaced | RMSD to A | RMSD to B |
|---|---|---|
| 0% | 0.00 | 16.03 |
| 25% | ~6.3 | ~16.7 |
| 50% | ~12.6 | ~13.3 |
| 75% | ~14.2 | ~8.0 |
| 100% | 16.03 | 0.00 |

**Method 2 — Embedding interpolation** (continuous blend α·B + (1-α)·A):
| α | RMSD to A | RMSD to B |
|---|---|---|
| 0.0 | 0.00 | 16.03 |
| 0.2 | 3.35 | 16.00 |
| 0.4 | 8.40 | 14.69 |
| 0.6 | 14.12 | 8.93 |
| 0.8 | 15.84 | 3.50 |
| 1.0 | 16.03 | 0.00 |

**Key observation**: Embedding interpolation is **smoother** — RMSD to A and B change monotonically and cross at α≈0.5. Token replacement is **choppier** — the RMSD trajectory is less predictable because swapping discrete tokens creates boundary artifacts between replaced and original regions.

**Structures saved**: `token_space_experiments/rule5_interpolation/structures/` — originals + 6 embedding interpolation snapshots + token replacement steps. Load in PyMOL/ChimeraX to visualize the structural morphing.

---

## Rule 6: Position sensitivity is non-uniform (FSQ 4375)

**Experiment**: Single-token mutations on tmbar (192 residues), 50 random token substitutions per position.

| Metric | Value |
|---|---|
| Mean RMSD across all positions | 0.56–1.37 Å |
| Most sensitive position (pos 191) | mean=1.37, max=3.31 Å |
| Second most sensitive (pos 116) | mean=1.26, max=2.76 Å |
| Least sensitive positions | mean=0.38–0.45 Å |

**Top 5 most sensitive positions**: 191, 116, 22, 176, 169
**Top 5 least sensitive positions**: positions in the 30–50 range (mean RMSD < 0.5 Å)

**Implication**: Terminal positions (191 = C-terminus) and certain interior positions are structural anchors. Mutating a token at these positions causes 3–4 Å global RMSD. Positions in the middle of the protein are more tolerant, possibly because the decoder's attention can compensate using surrounding context.

**Structures saved**: `token_space_experiments/rule6_position_sensitivity/structures/` — original + worst-case mutation for every 20th position + the 3 most sensitive positions. Compare original vs mutated in a viewer to see where the structure distorts.

---

## Rule 7: Ligand presence changes ~98% of protein tokens but decoded structures match within ~1.3 Å

**Experiment** (FSQ 4375, 10 PoseBusters protein-ligand complexes, with proper complex PDBs):

| Metric | Value |
|---|---|
| Mean protein token change rate | **98.2%** |
| Mean decoded protein RMSD (prot vs prot+lig) | **1.28 Å** |
| Mean decoded ligand RMSD vs GT | **48.1 Å** |

**Key findings**:
1. Adding a ligand changes nearly all protein tokens, but the decoded protein is nearly identical (~1.3 Å).
2. The decoded **ligand is placed far from the ground truth** (48 Å average). The encoder-decoder roundtrip does not preserve ligand placement.
3. Token changes are global (not binding site enriched) — this is a coordinate frame effect, not structural.

**Structures saved**: `token_space_experiments/rule7_ligand_fixed/structures/` — for each complex:
- `{name}_gt_complex.pdb` — ground truth protein+ligand
- `{name}_complex_decoded.pdb` — decoded protein+ligand from encoding
- `{name}_protein_only.pdb` — decoded protein without ligand

Open GT vs decoded complex side-by-side to see: protein backbone matches, ligand placement doesn't.

---

## Rule 8: Translation rules — the decoder outputs structures in a learned reference frame

**Experiment** (FSQ 4375, 5 PoseBusters complexes, translations of 5/10/50/100 Å):

### 8a. Translate ENTIRE complex (protein + ligand together)

| Translation | Token change | Aligned protein RMSD | Centroid shift |
|---|---|---|---|
| 5 Å | 99% | 1.33 Å | 7.3 Å |
| 10 Å | 100% | 1.35 Å | 6.7 Å |
| 50 Å | 100% | 1.37 Å | 6.3 Å |
| 100 Å | 100% | 1.33 Å | 4.9 Å |

**Rule**: Translating the input by ANY amount changes ALL tokens, but the decoded structure is identical after alignment (~1.3 Å RMSD). The decoded centroid shifts ~5-7 Å regardless of input translation magnitude — **the decoder outputs structures in its own learned reference frame**, not at the translated position.

### 8b. Translate ONLY the ligand (protein stays fixed)

| Translation | Token change | Protein RMSD | Decoded ligand shift |
|---|---|---|---|
| 5 Å | 100% | 1.30 Å | 11.7 Å |
| 10 Å | 99% | 1.37 Å | 11.5 Å |
| 20 Å | 100% | 1.30 Å | 13.4 Å |
| 50 Å | 99% | 1.43 Å | 37.3 Å |

**Rule**: Moving the ligand changes all tokens but the decoded protein is preserved (~1.3 Å). The decoded ligand shifts but NOT proportionally to the input translation — at small translations (5-20 Å) the decoded ligand moves ~12 Å regardless. Only at 50 Å does the decoded ligand displacement approach the input (37 Å). The decoder has a **weak memory of ligand position** relative to the protein.

### 8c. Translate ONLY the protein (ligand stays fixed)

| Translation | Token change | Protein RMSD | Decoded P-L distance |
|---|---|---|---|
| 5 Å | 100% | 1.28 Å | 15.2 Å |
| 10 Å | 99% | 1.34 Å | 19.3 Å |
| 20 Å | 100% | 1.32 Å | 28.0 Å |
| 50 Å | 100% | 1.38 Å | 56.4 Å |

**Rule**: Translating the protein away from the ligand increases the decoded protein-ligand distance approximately linearly. The decoder **preserves the relative distance** between protein and ligand — if you encode them far apart, they decode far apart. The protein structure itself is unchanged (~1.3 Å RMSD).

### Translation Summary

1. **Tokens are NOT translation-invariant** — any translation changes ~100% of tokens
2. **Decoded structures ARE translation-invariant for internal structure** — the protein backbone reconstructs at ~1.3 Å regardless of where it was in input space
3. **Protein-ligand relative position IS preserved** — translating the protein away from the ligand increases decoded P-L distance proportionally
4. **The decoder has a learned reference frame** — output centroids are ~5-7 Å regardless of input centroid, but relative positions within the complex are maintained
5. **Practical consequence**: Token space encodes absolute positions, but the decoder normalizes away global position while preserving relative geometry. This is the mechanism by which the model handles SE(3) augmentation during training.

**Structures saved**: `token_space_experiments/rule_translation/structures/` — 35 PDBs showing decoded complexes at various translations. Compare `{name}_original.pdb` vs `{name}_translate_prot_50A.pdb` to see protein-ligand distance increase.

---

## Rule 9: No predictable linear mapping from Cartesian space to embedding space

**Experiment** (FSQ 4375, 10 proteins, 5 directions × 6 magnitudes = 300 translation experiments):

Attempted to find a linear mapping f: Δx_cartesian → Δz_embedding by fitting W such that Δz ≈ Δx @ W.

| Question | Finding |
|---|---|
| Is emb shift proportional to Cartesian magnitude? | **NO** — emb norm ~17-22 for ALL magnitudes (1 Å to 50 Å) |
| Is emb shift consistent across positions? | **NO** — consistency ~0.11 (0 = random, 1 = uniform). Each position shifts differently |
| Does decoded centroid track input direction? | **NO** — cosine similarity is mixed/negative (-0.88 to +0.64) |
| Linear fit R²? | **-0.0017** — no linear relationship whatsoever |
| Does predicted emb shift improve decoding? | **NO** — RMSD unchanged (1.16 Å with or without predicted shift) |

**Implication**: The encoder's response to Cartesian translations is:
1. **Non-proportional** — 1 Å and 50 Å shifts cause similar embedding changes
2. **Non-directional** — shifting +x in Cartesian doesn't produce a consistent direction in embedding space
3. **Position-dependent** — each residue's embedding shifts differently for the same global translation
4. **Non-linear** — no linear model captures the relationship

**Root cause**: The transformer's self-attention with pairwise distance features creates a highly non-linear mapping from coordinates to embeddings. Global translations change the absolute position of every atom, which propagates through 6 attention layers in unpredictable ways.

**Conclusion**: Trying to control the encoder (Cartesian → tokens) is futile. The productive direction is to study the **decoder** (tokens → Cartesian): given a token manipulation, what structural change does the decoder produce? The decoder is deterministic and may have learnable patterns.

**Data saved**: `token_space_experiments/translation_mapping/` — `translation_mapping.csv` (300 experiments), `cartesian_to_embedding_matrix_W.npy` (failed linear fit), verification structures.

---

## Rule 10: Encode→decode is NOT a fixed point — structures drift ~0.15 Å per cycle

**Experiment**: Encode tmbar (192 residues), decode, re-encode the decoded structure, decode again, repeat 20 times.

| Iteration | Tokens changed vs prev | RMSD vs prev | RMSD vs original |
|---|---|---|---|
| 1 | 100% | 0.78 Å | 1.07 Å |
| 5 | 100% | 0.89 Å | 1.85 Å |
| 10 | 100% | 0.87 Å | 2.51 Å |
| 15 | 100% | 0.83 Å | 2.69 Å |
| 20 | 100% | 0.81 Å | 3.50 Å |

**Key findings:**
1. **Tokens change ~100% at every iteration** — the encoder never produces the same tokens twice for structures that differ by even 0.8 Å
2. **Per-step reconstruction error is constant** (~0.8 Å) — the decoder is consistently ~0.8 Å off from its input
3. **Cumulative drift is roughly linear** — ~0.15 Å/iteration, reaching 3.5 Å after 20 cycles
4. **The cycle never converges** — there is no fixed-point structure that encodes→decodes to itself

**Why this happens:**
- The decoder has ~0.8 Å reconstruction error (this is the quantization + decoding noise floor)
- The encoder is hypersensitive to input changes (Rule 2: 0.1 Å → 99% token change)
- Each cycle: decode adds 0.8 Å noise → encoder maps to completely different tokens → decoder produces another slightly-different structure → repeat
- The errors don't cancel out; they accumulate as a random walk in structure space

**Implication:** You cannot iteratively refine a structure by encode→decode cycling. Each cycle adds noise rather than converging. For structure refinement, use external tools (energy minimization, ESMFold) rather than re-encoding.

---

# Part II: Decoder Rules — What Token Changes Do to Decoded Structures

The encoder is chaotic (Rules 1-9). But the **decoder is deterministic** — same tokens always produce the same structure. This makes the decoder amenable to systematic study. The key question becomes: given a token manipulation, WHERE and HOW MUCH does the decoded structure change?

---

## Decoder Rule D1: Token mutations are LOCAL — the decoder has a ~38x locality ratio

**Experiment**: Single-token mutations at every position of tmbar (192 residues, 10 random substitutions per position), measuring per-residue CA RMSD.

| Metric | Value |
|---|---|
| Mean local RMSD (at mutated position) | **10.3 Å** |
| Mean neighbor RMSD (±5 residues) | **1.9 Å** |
| Mean far RMSD (>10 residues away) | **0.16 Å** |
| **Mean locality ratio** | **38x** |

**Rule**: Changing one token primarily distorts the **local** decoded structure at that position (~10 Å displacement), with a decay to neighbors (~2 Å within ±5 residues), and negligible effect on distant residues (~0.16 Å). The locality ratio of ~38x means the local effect is 38 times larger than the global effect.

**This is the opposite of what the encoder does.** The encoder changes ALL tokens for any local perturbation (Rule 2), but the decoder only changes the LOCAL structure for any single-token change. This asymmetry is key: the decoder has learned to "isolate" the structural effect of each token to its local neighborhood.

**Structures saved**: `token_space_experiments/decoder_d1_local_effect/structures/` — original + mutated structures at every 20th position. Also per-residue RMSD profiles as `.npy` files.

---

## Decoder Rule D2: Token index has NO relationship to structural similarity

**Experiment**: At 5 positions, substitute ~200 tokens each and measure RMSD. Check correlation between |token_index_A - token_index_B| and structural RMSD.

| Metric | Value |
|---|---|
| Correlation (token index distance vs RMSD) | **0.045** (≈ zero) |
| RMSD range across tokens (same position) | 0.45 – 2.3 Å |

**Rule**: Token index (0-4374) is arbitrary — nearby indices do NOT produce similar structures. Token 4053 and token 2478 may produce nearly identical local geometry (0.5 Å), while token 4052 (index neighbor) could produce 2 Å RMSD. The FSQ codebook ordering is not meaningful for structural similarity.

**However**: Each position has "compatible" tokens that produce low RMSD (<0.5 Å) and "incompatible" tokens that produce high RMSD (>2 Å). This compatibility is position-specific — a token that is compatible at position 32 may be incompatible at position 96.

**Structures saved**: `token_space_experiments/decoder_d2_token_neighborhood/structures/` — most similar and most different token substitutions at 5 positions.

---

## Decoder Rule D3: Block operations reveal the decoder's positional awareness

**Experiment**: Circular shift, reverse, duplicate, swap, and mask operations on tmbar token sequence.

### Circular shift
| Shift | RMSD |
|---|---|
| 1 position | 3.9 Å |
| 5 positions | 11.3 Å |
| 10 positions | 16.5 Å |
| 96 positions (half) | 17.5 Å |

**Rule**: The decoder is sensitive to positional alignment — shifting tokens by even 1 position causes 3.9 Å RMSD. By 10 positions, the structure is essentially destroyed (16 Å). The decoder expects specific tokens at specific positions, confirming that tokens encode position-dependent information.

### Reverse
| Operation | RMSD |
|---|---|
| Reverse token sequence | 9.4 Å |

**Rule**: Reversing the token order produces a moderately disrupted structure (9.4 Å). This is lower than half-swapping (17.5 Å), suggesting some degree of palindromic structural encoding.

### Region masking (replace with constant token)
| Region size | Global RMSD | Local RMSD | Far RMSD |
|---|---|---|---|
| 5 residues | 1.8 Å | 9.3 Å | 0.5 Å |
| 10 residues | 2.8 Å | 11.0 Å | 0.9 Å |
| 20 residues | 4.9 Å | 13.9 Å | 1.6 Å |
| 50 residues | 6.9 Å | 12.2 Å | 2.3 Å |

**Rule**: Masking a region with a constant token destroys the LOCAL structure (~10-14 Å at the masked region) but leaves FAR regions mostly intact (0.5-2.3 Å). The effect scales sub-linearly with region size — masking 50 residues (26% of the protein) only causes 6.9 Å global RMSD because the far regions compensate.

**This confirms D1**: the decoder's structural effect is fundamentally LOCAL. Token manipulations primarily affect the ±5-10 residue neighborhood, with exponential decay beyond that.

**Structures saved**: `token_space_experiments/decoder_d3_block_ops/structures/` — original, shifted, reversed, duplicated, swapped, and masked structures.

---

## Summary: Practical Rules for Token Manipulation

### What WORKS (decoder side):
1. **Local editing**: Changing 1-5 tokens at a target position modifies local structure (~10 Å displacement) without affecting the rest of the protein (<0.2 Å)
2. **Region replacement**: Masking a region and replacing with new tokens produces a "designed" local region while preserving the global fold
3. **Compatible token substitution**: Each position has a set of ~compatible tokens (RMSD <0.5 Å) that produce nearly identical local geometry — these are effectively synonymous codebook entries

### What DOESN'T WORK:
1. **Encoder-side prediction**: Cannot predict what tokens correspond to a desired Cartesian change (Rules 1-9)
2. **Token index arithmetic**: Token indices are arbitrary — adding/subtracting from indices has no structural meaning (D2)
3. **Global sequence operations**: Shifting, reversing, or swapping token blocks destroys the structure (D3)

### Practical workflow for structure editing:
1. **Encode** the structure to get tokens + embeddings
2. **Identify** the target region (residue range)
3. **Sample** new tokens for that region (from a generative model or vocabulary search)
4. **Keep** all other tokens + embeddings unchanged
5. **Decode** → the target region will be modified, the rest preserved

This is exactly what Gen-UME's flow matching does: it learns to sample tokens that produce valid structures, position by position, conditioned on the surrounding context.

---

# Part III: SVD Analysis of Embedding Space

## The embedding space is low-rank: 256 dims → 5 effective dimensions

**Experiment**: Encoded 200 proteins (32,664 residues total), computed SVD on the 256-dim embedding matrix.

### Explained variance

| Top PCs | Cumulative variance |
|---|---|
| 1 | 34.9% |
| 2 | 58.3% |
| 3 | 77.8% |
| 5 | **91.8%** |
| 10 | 95.3% |
| 20 | 98.6% |

**Rule**: The 256-dimensional embedding space is actually a ~5-dimensional manifold. 92% of all embedding variance is captured by just 5 components. The remaining 251 dimensions carry only noise-level information.

### Singular value spectrum

Top 10 singular values: `1871, 1532, 1398, 1133, 345, 320, 258, 252, 246, 240`

There's a **sharp drop after PC4** (1133 → 345). The first 4 components carry the dominant signal; PCs 5-10 are roughly equal in magnitude and carry residual structure.

### What the top PCs encode

#### PCs 1-3: Global 3D coordinate axes

| PC | Correlates with | Spearman ρ |
|---|---|---|
| PC1 | Y coordinate | 0.107 |
| PC2 | X coordinate | 0.117 |
| PC2 | Z coordinate | 0.101 |
| PC3 | Y coordinate | 0.107 |
| PC3 | Z coordinate | 0.103 |

Correlations are weak but highly significant (p < 1e-75). **PCs 1-3 encode a mixture of the XYZ coordinate axes** — not a clean rotation, but a nonlinear function of absolute position.

#### PC4: Distance from centroid

| PC | Correlates with | Spearman ρ |
|---|---|---|
| PC4 | Distance from centroid | **0.572** |
| PC5 | Distance from centroid | -0.220 |
| PC2 | Distance from centroid | 0.148 |
| PC3 | Distance from centroid | -0.149 |

**PC4 strongly encodes how far a residue is from the protein's center of mass** (ρ = 0.572). This is the most interpretable component: it distinguishes surface residues (high PC4) from core residues (low PC4). PC5 also contributes to this separation.

#### PCs do NOT encode sequence position

No PC correlates with relative sequence position (all ρ < 0.02). The embedding encodes **spatial** information, not **sequence** information.

### Within-protein vs between-protein variance

| PC | Within-protein variance | Between-protein variance | Ratio |
|---|---|---|---|
| PC1 | 97.7 | 0.10 | 0.001x |
| PC2 | 64.2 | 0.13 | 0.002x |
| PC3 | 53.8 | 0.53 | 0.01x |
| PC4 | 32.0 | 0.99 | 0.03x |
| PC5 | 3.0 | 0.05 | 0.02x |

**All variance is WITHIN proteins, not between proteins.** PCs 1-4 capture the spatial arrangement of residues *within* each protein — they're essentially encoding relative 3D position. Different proteins have nearly identical PC distributions (the means are close to zero with small spread).

### Implications

1. **The encoder compresses 3D coordinates into a ~5-dim manifold** embedded in 256 dimensions. The top 3 PCs are coordinate-like, PC4 is centroid-distance, PCs 5+ are residual.

2. **This explains why translations change all tokens** (Rule 1): the top PCs encode absolute XYZ position. Translating the protein shifts all residues along these PCs, changing their quantization boundary crossings.

3. **This explains the decoder's locality** (D1): the decoder only needs ~5 effective dimensions per residue. Changing one token perturbs the local embedding slightly along these 5 dimensions, which the decoder interprets as a local coordinate shift.

4. **Practical use**: To predict the decoder's response to a token change, project the embedding change onto the top 5 PCs. The structural effect will be approximately proportional to the change in these 5 components, weighted by their singular values.

5. **The FSQ quantizer operates on the full 256 dims** but only ~5 carry signal. Most of the codebook's 4,375 tokens differentiate along these 5 dimensions, which is why 1,585/4,375 tokens are used (the rest fall in the noise dimensions).

**Data saved**: `token_space_experiments/svd_*.npy` — embeddings, tokens, singular values, components, projections.

---

## Rule 8 (from prior studies): Segmented encoding breaks global consistency

**From prior studies** (segmented_encoding_study_fsq):

| Encoding strategy | RMSD |
|---|---|
| Full protein | 0.87 Å |
| 2 segments (halves) | 13.31 Å |
| 3 segments (thirds) | 11.62 Å |
| 4 segments (quarters) | 12.28 Å |

**Implication**: The encoder's self-attention creates token representations that are interdependent. Encoding a protein in segments destroys the global context, producing inconsistent tokens at segment boundaries. The full protein must be encoded as a single unit.

**Practical consequence**: You cannot "stitch together" token sequences from separately encoded fragments. The token representation is holistic.

---

## Rule 8: ~60% of the codebook is used

**From prior studies** (logit_distribution_analysis_slq_fa for SLQ; our experiments for FSQ):

- FSQ 4375: ~179 unique tokens per protein (4.1% of codebook)
- SLQ 256: ~64–68 unique tokens per protein (25–27% of codebook)
- Mean logit entropy: 5.54 bits (SLQ model)

**Implication**: The codebook is not fully utilized. Many tokens are never or rarely assigned, suggesting the effective structural vocabulary is much smaller than the nominal codebook size. The FSQ model uses an even smaller fraction because its codebook is larger (4375 vs 256).

---

## Summary: Can we manipulate structures in token space?

**Short answer**: Not directly with this encoder (no frame normalization). The token representation is orientation-dependent and globally coupled, making it unsuitable for local structural editing.

**What works**:
- **Interpolation** between two encoded structures (in embedding space)
- **Vocabulary-constrained sampling** (random tokens from a reference protein's vocabulary)
- **Full re-encoding** after structural modifications (modify coordinates → re-encode → decode)
- **Flow matching generation** (Gen-UME's actual approach: learn the token distribution, sample new tokens)

**What doesn't work**:
- Swapping individual tokens to change local structure
- Finding "helix tokens" or "sheet tokens" to insert
- Combining token sequences from different proteins
- Local token edits for local structural changes

**To enable local token manipulation**, the encoder would need:
1. Frame normalization (`frame_type="pca_frame"` or `"mol_frame"`) — makes tokens rotation/translation-invariant
2. Local attention (not global self-attention) — makes tokens position-independent
3. Or: a VQ-VAE with learned codebook entries that correspond to structural fragments

---

## Experimental Details

All experiments run on `LG Protein Ligand fsq 4375` model with PoseBusters benchmark proteins. Results stored in `token_space_experiments/`.

| Experiment | N proteins | Finding |
|---|---|---|
| 1: SE(3) invariance | 5 | 96–100% tokens change under rotation/translation |
| 2: Local perturbation | 3 | 0.1 Å perturbation at 1 residue → 99% global token change |
| 3: SS ↔ token mapping | 10 | No strong token-SS association (max 0.5% enrichment) |
| 5: Interpolation (FSQ 4375) | 2 (efhand, tmbar) | Smooth embedding interp, choppy token replacement |
| 6: Position sensitivity (FSQ 4375) | 1 (tmbar, 192 pos × 50 mutations) | Terminal/anchor positions most sensitive (1.4 Å mean) |
| 7: Protein-ligand interaction (FSQ 4375) | 20 | 99.5% tokens change, 1.29 Å decoded RMSD, no binding site enrichment |
