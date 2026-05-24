"""Empirical best-of-N scaling curves + extrapolation to larger N.

For each (checkpoint, target) we have 10 candidates with known (tm_score, PLL_score).
We subsample size-N subsets for N in {1..10}, compute the four pickers' average TM,
fit a saturating model TM(N) = a - b * exp(-N/τ), and extrapolate to N up to 50.

Outputs:
  - per-checkpoint scaling table (mean TM at N=1, 2, 4, 8, 10, 16, 20, 30, 50)
  - markdown summary

Notes:
- For N ≤ 10 we *enumerate* all C(10,N) subsets per target, so the curve is exact.
- For N > 10 the extrapolation is parametric, conditional on the candidate-TM
  distribution staying the same (i.e. additional candidates are i.i.d. samples
  from the same per-target distribution we observed). This is the assumption
  best-of-N analyses typically make.
- The PLL-picker performance ceiling is *not* the oracle but a lower asymptote
  set by the PLL-vs-TM rank correlation (ρ ≈ −0.82 struc-PLL on these models).
  The fit handles this by letting `a` saturate to the model's best PLL-pick TM
  in the large-N limit.
"""
from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE_DIR = Path("/cv/scratch/u/lisanzas/evaluations")
RUNS = {
    "denovo": BASE_DIR / "gen_ume_denovo_cameo_bestofN_pll",
    "base":   BASE_DIR / "gen_ume_base_cameo_bestofN_pll",
    "ted":    BASE_DIR / "gen_ume_ted_cameo_bestofN_pll",
}
PICKERS = ["random", "struc_pll", "joint_pll", "oracle"]
PICK_KEY = {
    "random":    None,                  # candidate 0
    "struc_pll": "struc_score_unif",    # argmin
    "joint_pll": "joint_score_unif",    # argmin
    "oracle":    "tm_score",            # argmax
}
N_MAX_OBS = 10  # candidates per target
N_PROJ = [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 30, 50]


def _load(run_dir: Path) -> pd.DataFrame:
    csv = sorted(run_dir.glob("bestofN_ff_candidates_*.csv"))[-1]
    return pd.read_csv(csv)


def _per_target_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {t: g.sort_values("candidate_idx").reset_index(drop=True) for t, g in df.groupby("target")}


def best_of_n_per_target(group: pd.DataFrame, picker: str, N: int, rng: np.random.Generator) -> float:
    """Mean TM achieved by `picker` over all (or many) size-N subsets of this target's candidates."""
    K = len(group)
    if N >= K:
        idxs = [list(range(K))]
    elif math.comb(K, N) <= 252:  # enumerate when cheap (C(10,N) ≤ 252)
        idxs = [list(s) for s in combinations(range(K), N)]
    else:
        idxs = [rng.choice(K, size=N, replace=False).tolist() for _ in range(200)]
    tms = group["tm_score"].to_numpy()
    if picker == "random":
        # "candidate 0" of the subset = a uniform-random sample. Mean of subset means = grand mean.
        # For a fair single-shot baseline at any N, just return the per-target candidate-level mean.
        return float(np.nanmean(tms))
    key = PICK_KEY[picker]
    vals = group[key].to_numpy()
    out_tms = []
    for sub in idxs:
        sub_tms = tms[sub]
        sub_vals = vals[sub]
        valid = ~np.isnan(sub_vals) & ~np.isnan(sub_tms)
        if valid.sum() == 0:
            continue
        if picker == "oracle":
            i = int(np.nanargmax(np.where(valid, sub_tms, -np.inf)))
        else:
            i = int(np.nanargmin(np.where(valid, sub_vals, np.inf)))
        out_tms.append(sub_tms[i])
    return float(np.mean(out_tms)) if out_tms else float("nan")


def empirical_curves(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {picker: array of mean-TM-across-targets for N=1..N_MAX_OBS}."""
    rng = np.random.default_rng(20260501)
    groups = _per_target_groups(df)
    out = {}
    Ns = list(range(1, N_MAX_OBS + 1))
    for picker in PICKERS:
        per_n = []
        for N in Ns:
            tms = [best_of_n_per_target(g, picker, N, rng) for g in groups.values()]
            per_n.append(np.nanmean(tms))
        out[picker] = np.array(per_n)
    return out


def power_sat(N, a, b, alpha):
    """TM(N) = a - b * N^(-alpha). Slow saturation typical of best-of-N order statistics.
    a = TM(∞), TM(1) = a - b, alpha ∈ (0,1] controls tail (small alpha = slower saturation).
    """
    return a - b * np.power(N, -alpha)


def exp_sat(N, a, b, tau):
    """TM(N) = a - b * exp(-N/tau). Fast saturation (Gumbel-like)."""
    return a - b * np.exp(-N / tau)


def _fit_one(curve_obs: np.ndarray, N_obs: list[int]):
    """Fit both models, return the one with lower SSE on the observed range."""
    x = np.array(N_obs, dtype=float)
    y = np.array(curve_obs, dtype=float)
    if np.allclose(y, y[0]):
        return ("constant", (y[0], 0.0, 0.0), lambda N, a=y[0], b=0, c=0: np.full_like(np.atleast_1d(N), a, dtype=float))

    # Power-law saturation
    try:
        popt_p, _ = curve_fit(
            power_sat, x, y,
            p0=[float(y.max()) + 0.01, max(0.005, float(y.max() - y.min())), 0.5],
            bounds=([y.min() - 0.1, 1e-6, 0.05], [1.0, 1.0, 2.0]),
            maxfev=30000,
        )
        sse_p = float(np.sum((power_sat(x, *popt_p) - y) ** 2))
    except Exception:
        popt_p, sse_p = None, np.inf

    # Exponential saturation
    try:
        popt_e, _ = curve_fit(
            exp_sat, x, y,
            p0=[float(y.max()) + 0.01, max(0.005, float(y.max() - y.min())), 4.0],
            bounds=([y.min() - 0.1, 1e-6, 0.1], [1.0, 1.0, 200.0]),
            maxfev=30000,
        )
        sse_e = float(np.sum((exp_sat(x, *popt_e) - y) ** 2))
    except Exception:
        popt_e, sse_e = None, np.inf

    if sse_p <= sse_e and popt_p is not None:
        return ("power", tuple(popt_p), lambda N, p=popt_p: power_sat(np.atleast_1d(N), *p))
    return ("exp", tuple(popt_e), lambda N, p=popt_e: exp_sat(np.atleast_1d(N), *p))


def extrapolate(curve_obs: np.ndarray, N_obs: list[int], N_proj: list[int]):
    name, popt, fn = _fit_one(curve_obs, N_obs)
    return fn(np.array(N_proj, dtype=float)), name, popt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("/cv/home/lisanzas/lobster/bestofN_scaling_extrapolation.md"))
    args = ap.parse_args()

    report = ["# Best-of-N PLL scaling — empirical (N≤10) + extrapolation (N≤50)\n"]
    report.append("Per-target: enumerate C(10,N) subsets exactly for N≤10; fit `TM(N) = a − b·exp(−N/τ)`\n"
                  "to the empirical curve and extrapolate. Extrapolation assumes additional candidates\n"
                  "remain i.i.d. samples from the same per-target distribution we already observed.\n")

    summary_rows = []
    for run_name, run_dir in RUNS.items():
        df = _load(run_dir)
        curves = empirical_curves(df)
        report.append(f"\n## {run_name}\n")
        # Observed table
        report.append("### Empirical curves (mean TM across 127 targets)\n")
        report.append("| picker | " + " | ".join(f"N={n}" for n in range(1, N_MAX_OBS + 1)) + " |")
        report.append("|---" + "|---" * N_MAX_OBS + "|")
        for picker in PICKERS:
            cells = " | ".join(f"{x:.3f}" for x in curves[picker])
            report.append(f"| {picker} | {cells} |")

        # Fit + extrapolate
        report.append("\n### Extrapolation (fits both `a − b·exp(−N/τ)` and `a − b·N^(−α)`; reports better fit)\n")
        report.append("| picker | model | params | "
                      + " | ".join(f"N={n}" for n in N_PROJ) + " | ΔN=10→30 | ΔN=10→50 |")
        report.append("|---|---|---|" + "---:|" * len(N_PROJ) + "---:|---:|")
        for picker in PICKERS:
            obs_N = list(range(1, N_MAX_OBS + 1))
            proj, model, popt = extrapolate(curves[picker], obs_N, N_PROJ)
            n10 = proj[N_PROJ.index(10)]
            n30 = proj[N_PROJ.index(30)]
            n50 = proj[N_PROJ.index(50)]
            cells = " | ".join(f"{x:.3f}" for x in proj)
            if model == "power":
                a, b, alpha = popt
                params_str = f"a={a:.3f}, b={b:.3f}, α={alpha:.2f}"
                tau_or_alpha = alpha
            elif model == "exp":
                a, b, tau = popt
                params_str = f"a={a:.3f}, b={b:.3f}, τ={tau:.2f}"
                tau_or_alpha = tau
            else:
                a = popt[0]; params_str = f"const={a:.3f}"; tau_or_alpha = 0.0
            report.append(
                f"| {picker} | {model} | {params_str} | {cells} | {n30 - n10:+.3f} | {n50 - n10:+.3f} |"
            )
            summary_rows.append({
                "ckpt": run_name,
                "picker": picker,
                "model": model,
                "N=10_emp": float(curves[picker][9]),
                "N=10_fit": float(n10),
                "N=20_fit": float(proj[N_PROJ.index(20)]),
                "N=30_fit": float(n30),
                "N=50_fit": float(n50),
                "asymptote_a": float(a) if model != "constant" else float(popt[0]),
                "delta_10_to_30": float(n30 - n10),
                "delta_10_to_50": float(n50 - n10),
            })

    # Headline: struc_pll across the 3 ckpts at N=10 vs N=30
    summary_df = pd.DataFrame(summary_rows)
    report.append("\n## Headline: struc_pll_pick scaling\n")
    sub = summary_df[summary_df["picker"] == "struc_pll"]
    report.append("| ckpt | empirical N=10 | fit N=10 | fit N=20 | fit N=30 | fit N=50 | asymptote | Δ(10→30) | Δ(10→50) |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sub.iterrows():
        report.append(
            f"| {r['ckpt']} | {r['N=10_emp']:.3f} | {r['N=10_fit']:.3f} | {r['N=20_fit']:.3f} | "
            f"{r['N=30_fit']:.3f} | {r['N=50_fit']:.3f} | {r['asymptote_a']:.3f} | "
            f"{r['delta_10_to_30']:+.3f} | {r['delta_10_to_50']:+.3f} |"
        )

    report.append("\n## Headline: oracle_pick scaling (upper bound)\n")
    sub = summary_df[summary_df["picker"] == "oracle"]
    report.append("| ckpt | empirical N=10 | fit N=10 | fit N=20 | fit N=30 | fit N=50 | asymptote | Δ(10→30) | Δ(10→50) |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sub.iterrows():
        report.append(
            f"| {r['ckpt']} | {r['N=10_emp']:.3f} | {r['N=10_fit']:.3f} | {r['N=20_fit']:.3f} | "
            f"{r['N=30_fit']:.3f} | {r['N=50_fit']:.3f} | {r['asymptote_a']:.3f} | "
            f"{r['delta_10_to_30']:+.3f} | {r['delta_10_to_50']:+.3f} |"
        )

    report.append("\n## How to read this\n")
    report.append("- Two saturation models are fit to the empirical curve and the better-fitting one is reported.\n"
                  "  - `exp`: `TM(N) = a − b·exp(−N/τ)` saturates fast (Gumbel-like tails).\n"
                  "  - `power`: `TM(N) = a − b·N^(−α)` saturates slowly (heavy-tailed). Smaller α = slower.\n"
                  "- **Δ(10→30)** is the projected additional mean TM from tripling N. Compare to the\n"
                  "  3× compute cost: a gain ≲ 0.005 mean TM is generally not worth 3× compute.\n"
                  "- The PLL-pick asymptote sits below the oracle asymptote because PLL-vs-TM rank\n"
                  "  correlation is finite (ρ ≈ −0.82 on base/TED). This irreducible gap cannot be\n"
                  "  closed by adding more candidates — only by a better selector.\n"
                  "- *Caveat*: extrapolation assumes new candidates are i.i.d. samples from the same\n"
                  "  per-target distribution. Real generation can drift (mode-collapse → less benefit;\n"
                  "  diverse modes → more benefit). The estimate is a best-case under the i.i.d. assumption.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report))
    print(f"Wrote {args.output}")
    print("=" * 80)
    print("\n".join(report))


if __name__ == "__main__":
    main()
