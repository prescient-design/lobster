# Best-of-N (N=30) forward folding — TM / RMSD / pass rate

CAMEO benchmark, 127 targets per checkpoint. Per-target picker selects 1 of 30 candidates;
metrics are computed on the selected set (n=127 per picker).

- **PASS**: RMSD < 2.0 Å


## GenUME-denovo

_source: `gen_ume_denovo_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025342.csv`_

| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 127 | 0.575 | 0.584 | 13.59 | 9.90 | 4.7 |
| seq_pll | 127 | 0.619 | 0.692 | 9.85 | 6.57 | 5.5 |
| struc_pll | 127 | 0.598 | 0.648 | 19.01 | 8.04 | 7.9 |
| joint_pll | 127 | 0.625 | 0.695 | 11.32 | 5.31 | 6.3 |
| oracle | 127 | 0.684 | 0.740 | 8.40 | 4.87 | 11.0 |

## GenUME-base

_source: `gen_ume_base_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025348.csv`_

| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 127 | 0.675 | 0.767 | 11.45 | 4.50 | 15.0 |
| seq_pll | 127 | 0.642 | 0.718 | 16.02 | 5.48 | 15.0 |
| struc_pll | 127 | 0.685 | 0.790 | 12.79 | 4.08 | 16.5 |
| joint_pll | 127 | 0.642 | 0.726 | 14.40 | 4.88 | 15.0 |
| oracle | 127 | 0.739 | 0.822 | 6.83 | 3.90 | 23.6 |

## GenUME-TED

_source: `gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025401.csv`_

| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |
|---|---:|---:|---:|---:|---:|---:|
| random | 127 | 0.653 | 0.746 | 11.53 | 4.86 | 13.4 |
| seq_pll | 127 | 0.660 | 0.731 | 10.53 | 5.20 | 13.4 |
| struc_pll | 127 | 0.693 | 0.811 | 12.34 | 3.85 | 17.3 |
| joint_pll | 127 | 0.673 | 0.786 | 13.45 | 4.35 | 15.7 |
| oracle | 127 | 0.751 | 0.846 | 6.73 | 3.03 | 26.8 |

## Headline: struc_pll picker vs single-shot random (Δ over random)

| ckpt | random TM | struc_pll TM | Δ TM | random RMSD | struc_pll RMSD | Δ RMSD | random PASS | struc_pll PASS | Δ PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GenUME-denovo | 0.575 | 0.598 | +0.024 | 13.59 | 19.01 | +5.43 | 4.7 | 7.9 | +3.1 |
| GenUME-base | 0.675 | 0.685 | +0.010 | 11.45 | 12.79 | +1.34 | 15.0 | 16.5 | +1.6 |
| GenUME-TED | 0.653 | 0.693 | +0.040 | 11.53 | 12.34 | +0.80 | 13.4 | 17.3 | +3.9 |

## Best-PLL-picker per checkpoint vs random (Δ over random)

| ckpt | best PLL picker | random TM | best TM | Δ TM | random RMSD | best RMSD | Δ RMSD | random PASS | best PASS | Δ PASS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GenUME-denovo | joint_pll | 0.575 | 0.625 | +0.051 | 13.59 | 11.32 | -2.27 | 4.7 | 6.3 | +1.6 |
| GenUME-base | struc_pll | 0.675 | 0.685 | +0.010 | 11.45 | 12.79 | +1.34 | 15.0 | 16.5 | +1.6 |
| GenUME-TED | struc_pll | 0.653 | 0.693 | +0.040 | 11.53 | 12.34 | +0.80 | 13.4 | 17.3 | +3.9 |