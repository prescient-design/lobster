# Best-of-N PLL scaling — empirical (N≤10) + extrapolation (N≤50)

Per-target: enumerate C(10,N) subsets exactly for N≤10; fit `TM(N) = a − b·exp(−N/τ)`
to the empirical curve and extrapolate. Extrapolation assumes additional candidates
remain i.i.d. samples from the same per-target distribution we already observed.


## denovo

### Empirical curves (mean TM across 127 targets)

| picker | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | N=9 | N=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 |
| struc_pll | 0.576 | 0.589 | 0.593 | 0.595 | 0.596 | 0.597 | 0.597 | 0.598 | 0.598 | 0.598 |
| joint_pll | 0.576 | 0.594 | 0.601 | 0.607 | 0.611 | 0.614 | 0.616 | 0.619 | 0.621 | 0.623 |
| oracle | 0.576 | 0.611 | 0.626 | 0.636 | 0.643 | 0.649 | 0.653 | 0.657 | 0.660 | 0.663 |

### Extrapolation (fits both `a − b·exp(−N/τ)` and `a − b·N^(−α)`; reports better fit)

| picker | model | params | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=8 | N=10 | N=16 | N=20 | N=30 | N=50 | ΔN=10→30 | ΔN=10→50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | constant | const=0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | 0.576 | +0.000 | +0.000 |
| struc_pll | power | a=0.600, b=0.024, α=1.05 | 0.576 | 0.589 | 0.593 | 0.595 | 0.596 | 0.597 | 0.598 | 0.598 | 0.599 | 0.599 | 0.600 | 0.600 | +0.001 | +0.002 |
| joint_pll | power | a=0.693, b=0.117, α=0.22 | 0.577 | 0.593 | 0.601 | 0.607 | 0.611 | 0.614 | 0.619 | 0.622 | 0.629 | 0.632 | 0.637 | 0.643 | +0.015 | +0.021 |
| oracle | power | a=0.729, b=0.153, α=0.36 | 0.576 | 0.610 | 0.626 | 0.636 | 0.644 | 0.649 | 0.657 | 0.663 | 0.673 | 0.677 | 0.684 | 0.692 | +0.022 | +0.029 |

## base

### Empirical curves (mean TM across 127 targets)

| picker | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | N=9 | N=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 |
| struc_pll | 0.671 | 0.678 | 0.680 | 0.681 | 0.682 | 0.683 | 0.683 | 0.684 | 0.684 | 0.684 |
| joint_pll | 0.671 | 0.669 | 0.666 | 0.664 | 0.662 | 0.661 | 0.659 | 0.658 | 0.657 | 0.656 |
| oracle | 0.671 | 0.692 | 0.702 | 0.708 | 0.712 | 0.715 | 0.718 | 0.720 | 0.722 | 0.723 |

### Extrapolation (fits both `a − b·exp(−N/τ)` and `a − b·N^(−α)`; reports better fit)

| picker | model | params | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=8 | N=10 | N=16 | N=20 | N=30 | N=50 | ΔN=10→30 | ΔN=10→50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | constant | const=0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | +0.000 | +0.000 |
| struc_pll | power | a=0.687, b=0.016, α=0.69 | 0.671 | 0.677 | 0.680 | 0.681 | 0.682 | 0.683 | 0.684 | 0.684 | 0.685 | 0.685 | 0.686 | 0.686 | +0.002 | +0.002 |
| joint_pll | exp | a=0.662, b=0.000, τ=192.89 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | +0.000 | +0.000 |
| oracle | power | a=0.755, b=0.084, α=0.42 | 0.671 | 0.692 | 0.702 | 0.708 | 0.712 | 0.715 | 0.720 | 0.723 | 0.729 | 0.731 | 0.735 | 0.739 | +0.012 | +0.016 |

## ted

### Empirical curves (mean TM across 127 targets)

| picker | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | N=9 | N=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 |
| struc_pll | 0.661 | 0.676 | 0.681 | 0.684 | 0.686 | 0.688 | 0.690 | 0.691 | 0.693 | 0.694 |
| joint_pll | 0.661 | 0.670 | 0.672 | 0.672 | 0.672 | 0.671 | 0.670 | 0.670 | 0.669 | 0.668 |
| oracle | 0.661 | 0.692 | 0.705 | 0.712 | 0.718 | 0.722 | 0.725 | 0.727 | 0.729 | 0.731 |

### Extrapolation (fits both `a − b·exp(−N/τ)` and `a − b·N^(−α)`; reports better fit)

| picker | model | params | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=8 | N=10 | N=16 | N=20 | N=30 | N=50 | ΔN=10→30 | ΔN=10→50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | constant | const=0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | +0.000 | +0.000 |
| struc_pll | power | a=0.708, b=0.047, α=0.51 | 0.661 | 0.675 | 0.681 | 0.685 | 0.687 | 0.689 | 0.692 | 0.693 | 0.696 | 0.698 | 0.700 | 0.701 | +0.006 | +0.008 |
| joint_pll | exp | a=0.671, b=0.973, τ=0.22 | 0.661 | 0.670 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | 0.671 | +0.000 | +0.000 |
| oracle | power | a=0.760, b=0.100, α=0.53 | 0.661 | 0.691 | 0.705 | 0.713 | 0.718 | 0.722 | 0.727 | 0.731 | 0.737 | 0.740 | 0.744 | 0.748 | +0.013 | +0.017 |

## Headline: struc_pll_pick scaling

| ckpt | empirical N=10 | fit N=10 | fit N=20 | fit N=30 | fit N=50 | asymptote | Δ(10→30) | Δ(10→50) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| denovo | 0.598 | 0.598 | 0.599 | 0.600 | 0.600 | 0.600 | +0.001 | +0.002 |
| base | 0.684 | 0.684 | 0.685 | 0.686 | 0.686 | 0.687 | +0.002 | +0.002 |
| ted | 0.694 | 0.693 | 0.698 | 0.700 | 0.701 | 0.708 | +0.006 | +0.008 |

## Headline: oracle_pick scaling (upper bound)

| ckpt | empirical N=10 | fit N=10 | fit N=20 | fit N=30 | fit N=50 | asymptote | Δ(10→30) | Δ(10→50) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| denovo | 0.663 | 0.663 | 0.677 | 0.684 | 0.692 | 0.729 | +0.022 | +0.029 |
| base | 0.723 | 0.723 | 0.731 | 0.735 | 0.739 | 0.755 | +0.012 | +0.016 |
| ted | 0.731 | 0.731 | 0.740 | 0.744 | 0.748 | 0.760 | +0.013 | +0.017 |

## How to read this

- Two saturation models are fit to the empirical curve and the better-fitting one is reported.
  - `exp`: `TM(N) = a − b·exp(−N/τ)` saturates fast (Gumbel-like tails).
  - `power`: `TM(N) = a − b·N^(−α)` saturates slowly (heavy-tailed). Smaller α = slower.
- **Δ(10→30)** is the projected additional mean TM from tripling N. Compare to the
  3× compute cost: a gain ≲ 0.005 mean TM is generally not worth 3× compute.
- The PLL-pick asymptote sits below the oracle asymptote because PLL-vs-TM rank
  correlation is finite (ρ ≈ −0.82 on base/TED). This irreducible gap cannot be
  closed by adding more candidates — only by a better selector.
- *Caveat*: extrapolation assumes new candidates are i.i.d. samples from the same
  per-target distribution. Real generation can drift (mode-collapse → less benefit;
  diverse modes → more benefit). The estimate is a best-case under the i.i.d. assumption.
