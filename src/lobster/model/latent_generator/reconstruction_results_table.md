# Reconstruction Evaluation Results

**Evaluation Set**: CASP15 proteins ≤ 512 residues (26 successful reconstructions out of 30 total structures)

| Model | Average RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|------------------|--------------|--------------|--------------|
| LG full attention | 1.707 | 0.643 | 0.839 | 3.434 |
| LG 10A | 3.698 | 1.756 | 1.952 | 7.664 |
| LG 20A c6d Aux | 4.395 | 2.671 | 1.678 | 11.306 |
| LG 20A seq 3di c6d Aux | 4.428 | 1.723 | 2.757 | 8.556 |
| LG 20A 3di c6d Aux | 4.484 | 2.458 | 2.390 | 11.696 |
| LG 20A | 4.470 | 3.540 | 1.630 | 12.864 |
| LG 20A seq 3di c6d 512 Aux | 5.761 | 4.349 | 1.188 | 17.442 |
| LG 20A seq Aux | 5.449 | 2.862 | 3.063 | 13.342 |
| LG 20A seq 3di Aux | 6.112 | 3.723 | 2.973 | 17.839 |
| LG 20A 3di Aux | 7.844 | 4.289 | 3.119 | 16.500 |

## Summary

- **Best performing model**: LG full attention (1.707 ± 0.643 Å)
- **Second best**: LG 10A (3.698 ± 1.756 Å)
- **Third best**: LG 20A seq 3di c6d Aux (4.428 ± 1.723 Å)

All models successfully reconstructed 26 out of 30 structures (86.7% success rate). 