
# Lobster Metrics

## AlphaFold2 Scoring

Lobster includes tools for scoring protein binders using AlphaFold2 predictions via ColabDesign. This is useful for evaluating binder-target interactions and binder stability.

This scoring is used by BindCraft:

> Pacesa, Martin, Lennart Nickel, Christian Schellhaas, Joseph Schmidt, Ekaterina Pyatova, Lucas Kissling, Patrick Barendse et al. "BindCraft: one-shot design of functional protein binders." bioRxiv (2024): 2024-09.

AlphaFold2 Reference:
> Jumper, John, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool et al. "Highly accurate protein structure prediction with AlphaFold." nature 596, no. 7873 (2021): 583-589.

### Installation

Make sure you install the right extra dependencies for AlphaFold2 eval:
```bash
uv sync --extra struct-gpu 
```
or
```bash
uv sync --extra struct-cpu
```

### Usage Example

AlphaFold2 weights will be automatically downloaded on first use (~5GB) into `alphafold_weights_dir` provided to the function.
If you provide output_dir, PDB predictions will be saved there. Otherwise, only scores are returned.


```python
from lobster.metrics import alphafold2_binder_scores, alphafold2_complex_scores

# Score binder alone 
peptide_sequence = "LTFEYWAQLSAA"
binder_scores = alphafold2_binder_scores(
    binder_sequence=peptide_sequence,
    output_dir="./my-structures/",
    alphafold_weights_dir="data/alphafold2/weights"
)
print(f"Binder scores: {binder_scores}")

>>> {'pLDDT': 0.84, 'pTM': 0.05, 'pAE': 0.11}

# Score binder-target complex 
pdb_path = "test_data/4N5T.pdb"
target_chain = "A"
binder_sequence = "LTFEYWAQLSAA"
complex_scores = alphafold2_complex_scores(
    target_pdb=pdb_path,
    target_chain=target_chain,
    binder_sequence=binder_sequence,
    output_dir="./my-structures",
    alphafold_weights_dir="data/alphafold2/weights"
)
print(f"Complex scores: {complex_scores}")

>>> {'pLDDT': 0.41, 'pTM': 0.77, 'i_pTM': 0.32, 'pAE': 0.46, 'i_pAE': 0.51}
```

**Key Metrics:**
- `pLDDT`: Overall confidence (0-100, higher is better)
- `pTM`: Predicted TM-score (0-1, measures global structure quality)
- `i_pTM`: Interface TM-score (complex only, measures binding interface quality)
- `pAE`: Predicted aligned error (lower is better)
- `i_pAE`: Interface PAE (complex only, lower indicates better interface prediction)

