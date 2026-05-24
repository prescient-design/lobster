# Best-of-N (N=10) forward folding — TM / RMSD / pass rate

CAMEO benchmark, 127 targets per checkpoint. Per-target picker selects 1 of 10 candidates;
metrics are computed on the selected set (n=127 per picker).

- **PASS**: RMSD < 2.0 Å


## GenUME-denovo

_source: `gen_ume_denovo_cameo_bestofN_pll/bestofN_ff_candidates_20260430T220423.csv`_

| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 127 | 0.575 | 0.584 | 13.59 | 9.90 | 4.7 |
| seq_pll | 127 | 0.620 | 0.676 | 10.12 | 5.71 | 7.1 |
| struc_pll | 127 | 0.598 | 0.679 | 17.12 | 7.89 | 7.1 |
| joint_pll | 127 | 0.623 | 0.691 | 10.08 | 5.75 | 7.1 |
| oracle | 127 | 0.663 | 0.722 | 9.60 | 5.04 | 8.7 |

## GenUME-base

_source: `gen_ume_base_cameo_bestofN_pll/bestofN_ff_candidates_20260501T002139.csv`_

| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 127 | 0.675 | 0.767 | 11.45 | 4.50 | 15.0 |
| seq_pll | 127 | 0.657 | 0.740 | 14.69 | 4.87 | 14.2 |
| struc_pll | 127 | 0.684 | 0.779 | 14.59 | 4.12 | 15.7 |
| joint_pll | 127 | 0.656 | 0.740 | 15.90 | 4.35 | 14.2 |
| oracle | 127 | 0.723 | 0.809 | 7.47 | 3.95 | 20.5 |

## GenUME-TED

_source: `gen_ume_ted_cameo_bestofN_pll/bestofN_ff_candidates_20260501T002150.csv`_

| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 127 | 0.653 | 0.746 | 11.53 | 4.86 | 13.4 |
| seq_pll | 127 | 0.663 | 0.763 | 9.37 | 4.58 | 13.4 |
| struc_pll | 127 | 0.694 | 0.792 | 10.31 | 3.89 | 16.5 |
| joint_pll | 127 | 0.668 | 0.773 | 10.77 | 4.35 | 14.2 |
| oracle | 127 | 0.731 | 0.821 | 7.21 | 3.50 | 25.2 |

## Headline: struc_pll picker vs single-shot random (Δ over random)

| ckpt | random TM | struc_pll TM | Δ TM | random RMSD | struc_pll RMSD | Δ RMSD | random PASS | struc_pll PASS | Δ PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GenUME-denovo | 0.575 | 0.598 | +0.023 | 13.59 | 17.12 | +3.54 | 4.7 | 7.1 | +2.4 |
| GenUME-base | 0.675 | 0.684 | +0.009 | 11.45 | 14.59 | +3.14 | 15.0 | 15.7 | +0.8 |
| GenUME-TED | 0.653 | 0.694 | +0.042 | 11.53 | 10.31 | -1.22 | 13.4 | 16.5 | +3.1 |

## Best-PLL-picker per checkpoint vs random (Δ over random)

| ckpt | best PLL picker | random TM | best TM | Δ TM | random RMSD | best RMSD | Δ RMSD | random PASS | best PASS | Δ PASS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GenUME-denovo | joint_pll | 0.575 | 0.623 | +0.048 | 13.59 | 10.08 | -3.51 | 4.7 | 7.1 | +2.4 |
| GenUME-base | struc_pll | 0.675 | 0.684 | +0.009 | 11.45 | 14.59 | +3.14 | 15.0 | 15.7 | +0.8 |
| GenUME-TED | struc_pll | 0.653 | 0.694 | +0.042 | 11.53 | 10.31 | -1.22 | 13.4 | 16.5 | +3.1 |