"""Length-matched, balanced version of the SR forward-fold-TM threshold sweep.

For every SR-rejected sample of length L we pair it with one randomly-selected
SR-accepted sample of the same length L. This removes the length-distribution
imbalance between accepted (capped at 100 / length) and rejected (variable;
heavily skewed toward L=500) attempts that the original sweep had.

Produces three figures:

  1) sr_qc_tm_threshold_sweep_balanced.png
     Pooled-across-lengths sweep on the balanced dataset, one panel per
     checkpoint. Mirrors the original sr_qc_tm_threshold_sweep.png layout
     (ESM-pass rate of retained, fraction retained, designable / total).

  2) sr_qc_tm_threshold_sweep_per_length.png
     Per-length sweep: one panel per checkpoint, one coloured line per length.
     Y-axis is the ESM-pass rate of the retained set after applying lobster_tm
     >= T inside that length bucket only. Pooled balanced curve is overlaid in
     bold black for reference.

  3) sr_qc_tm_threshold_sweep_designable_per_length.png
     Same shape as (2) but Y-axis is "designable / total balanced attempts at
     that length", i.e. the absolute yield curve which is what tells you
     whether tightening or loosening the gate would gain or lose designable
     samples for that length bucket.

Numerical summary printed to stdout:
  - per-length balanced N
  - ESM-pass rate of retained at T = 0, current SR (0.833), and 0.9
  - argmax T for designable count, and the designable count there

Usage:
  uv run python scripts/plot_sr_qc_tm_sweep_balanced.py \
      --ted-dir <ted_concordance_dir> \
      --base-dir <base_concordance_dir> \
      --stoch-dir <ted_stoch_concordance_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_sr_qc_threshold_sweep import (  # noqa: E402
    ESM_RMSD_PASS,
    ESM_TM_PASS,
    SR_FORWARD_TM,
    esm_pass,
    load_run,
)


SEED = 0
LENGTHS = [100, 200, 300, 400, 500]
LENGTH_COLOURS = dict(zip(LENGTHS, plt.cm.viridis(np.linspace(0.05, 0.85, len(LENGTHS)))))


def balance_per_length(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """For every length L take min(n_rej[L], n_acc[L]) of each class.

    Down-samples the larger class without replacement using a fixed seed.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for L, sub in df.groupby("sequence_length"):
        rej = sub[~sub.sr_pass]
        acc = sub[sub.sr_pass]
        k = min(len(rej), len(acc))
        if k == 0:
            continue
        rej_idx = rng.choice(rej.index.values, size=k, replace=False)
        acc_idx = rng.choice(acc.index.values, size=k, replace=False)
        pieces.append(df.loc[rej_idx])
        pieces.append(df.loc[acc_idx])
    out = pd.concat(pieces).reset_index(drop=True)
    return out


def _sweep(df: pd.DataFrame, thresholds: np.ndarray, criterion: str):
    """Returns (keep_frac, esm_pass_rate_of_retained, n_design) over thresholds."""
    n_total = len(df)
    ep = esm_pass(df, criterion)
    keep_frac = np.empty_like(thresholds, dtype=float)
    pass_rate = np.empty_like(thresholds, dtype=float)
    n_design = np.empty_like(thresholds, dtype=int)
    for i, t in enumerate(thresholds):
        mask = df.lobster_tm.values >= t
        keep_frac[i] = mask.sum() / max(n_total, 1)
        if mask.sum() == 0:
            pass_rate[i] = np.nan
            n_design[i] = 0
        else:
            ep_kept = ep.values[mask]
            pass_rate[i] = ep_kept.mean()
            n_design[i] = int(ep_kept.sum())
    return keep_frac, pass_rate, n_design


# ------------------- plot 1: balanced pooled sweep ----------------------------


def plot_balanced_pooled(runs: dict[str, pd.DataFrame], out_path: Path,
                         criterion: str = "joint") -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.0),
                             squeeze=False)

    for ax, (label, df) in zip(axes[0], runs.items()):
        bal = balance_per_length(df)
        n_total = len(bal)
        keep_frac, pass_rate, n_design = _sweep(bal, thresholds, criterion)

        ax2 = ax.twinx()
        ax.plot(thresholds, pass_rate, color="tab:blue", lw=2,
                label="ESM-pass rate of retained set")
        ax2.plot(thresholds, keep_frac, color="tab:orange", lw=2,
                 label="Fraction of attempts retained")
        ax2.plot(thresholds, n_design / max(n_total, 1),
                 color="tab:green", lw=2, ls="--",
                 label="Designable / total attempts (balanced)")

        idx = int(np.argmin(np.abs(thresholds - SR_FORWARD_TM)))
        ax.axvline(SR_FORWARD_TM, color="red", ls=":", lw=1, alpha=0.7)
        ax.scatter([SR_FORWARD_TM], [pass_rate[idx]], color="red", zorder=5, s=40)
        ax.annotate(
            f"current SR\nT={SR_FORWARD_TM}\n  retain={keep_frac[idx]*100:.1f}%\n"
            f"  ESM-pass={pass_rate[idx]*100:.1f}%\n  designable={n_design[idx]}/{n_total}",
            xy=(SR_FORWARD_TM, pass_rate[idx]),
            xytext=(0.05, 0.92), textcoords="axes fraction",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        best = int(np.nanargmax(n_design))
        if n_design[best] > n_design[idx]:
            ax.axvline(thresholds[best], color="green", ls=":", lw=1, alpha=0.6)
            ax.annotate(
                f"max designable\nT={thresholds[best]:.2f}\n"
                f"  retain={keep_frac[best]*100:.1f}%\n"
                f"  ESM-pass={pass_rate[best]*100:.1f}%\n"
                f"  designable={n_design[best]}/{n_total}",
                xy=(thresholds[best], pass_rate[best]),
                xytext=(0.55, 0.18), textcoords="axes fraction",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="green", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forward-fold-TM threshold (SR QC)")
        ax.set_ylabel("Designable / retained\n(= ESM pass-rate of kept set)",
                      color="tab:blue")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax2.set_ylabel("Fraction retained / Designable / total",
                       color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax2.set_ylim(0, 1)
        ax.set_title(f"{label}  (balanced n={n_total} = {n_total // 2} rej + {n_total // 2} acc)")
        ax.grid(True, alpha=0.2)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="lower left", fontsize=8, framealpha=0.85)

    crit_name = {"tm": "ESM TM ≥ 0.9", "rmsd": "ESM RMSD < 2 Å",
                 "joint": "ESM TM ≥ 0.9 AND RMSD < 2 Å"}[criterion]
    fig.suptitle(
        f"Forward-fold-TM threshold sweep — length-matched balanced set "
        f"(criterion: {crit_name})",
        fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ------------------- plot 2/3: per-length sweeps -----------------------------


def plot_per_length(runs: dict[str, pd.DataFrame], out_path_passrate: Path,
                    out_path_yield: Path, criterion: str = "joint") -> None:
    thresholds = np.linspace(0.0, 1.0, 101)

    fig_p, axes_p = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.0),
                                 squeeze=False)
    fig_y, axes_y = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.0),
                                 squeeze=False)

    for (ax_p, ax_y), (label, df) in zip(zip(axes_p[0], axes_y[0]), runs.items()):
        bal = balance_per_length(df)
        keep_frac_pool, pass_rate_pool, n_design_pool = _sweep(
            bal, thresholds, criterion)
        n_total_pool = len(bal)

        # per-length curves
        for L in LENGTHS:
            sub = bal[bal.sequence_length == L]
            if len(sub) == 0:
                continue
            keep_frac, pass_rate, n_design = _sweep(sub, thresholds, criterion)
            n_total = len(sub)
            colour = LENGTH_COLOURS[L]
            ax_p.plot(thresholds, pass_rate, color=colour, lw=1.6,
                      label=f"L={L}  (n={n_total})")
            ax_y.plot(thresholds, n_design / max(n_total, 1),
                      color=colour, lw=1.6,
                      label=f"L={L}  (n={n_total})")

        # pooled balanced overlay
        ax_p.plot(thresholds, pass_rate_pool, color="black", lw=2.4,
                  label=f"pooled balanced (n={n_total_pool})")
        ax_y.plot(thresholds, n_design_pool / max(n_total_pool, 1),
                  color="black", lw=2.4,
                  label=f"pooled balanced (n={n_total_pool})")

        for ax in (ax_p, ax_y):
            ax.axvline(SR_FORWARD_TM, color="red", ls=":", lw=1, alpha=0.7,
                       label=f"SR T={SR_FORWARD_TM}")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Forward-fold-TM threshold")
            ax.set_title(label)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7, framealpha=0.85)
        ax_p.set_ylim(0, 1)
        ax_p.set_ylabel("ESM-pass rate of retained set")
        ax_y.set_ylim(0, 1)
        ax_y.set_ylabel("Designable / total balanced attempts at that length")

    crit_name = {"tm": "ESM TM ≥ 0.9", "rmsd": "ESM RMSD < 2 Å",
                 "joint": "ESM TM ≥ 0.9 AND RMSD < 2 Å"}[criterion]
    fig_p.suptitle(
        f"Per-length ESM-pass rate of retained — balanced set (criterion: {crit_name})",
        fontsize=12, y=1.04)
    fig_y.suptitle(
        f"Per-length designable yield — balanced set (criterion: {crit_name})",
        fontsize=12, y=1.04)
    fig_p.tight_layout()
    fig_y.tight_layout()
    fig_p.savefig(out_path_passrate, dpi=150, bbox_inches="tight")
    fig_y.savefig(out_path_yield, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path_passrate}")
    print(f"Saved: {out_path_yield}")


# ------------------- text summary --------------------------------------------


def print_summary(runs: dict[str, pd.DataFrame], criterion: str = "joint") -> None:
    print(f"\n=== Balanced-set summary  (ESM criterion: {criterion}) ===")
    snap_thresholds = [0.0, SR_FORWARD_TM, 0.9]
    for label, df in runs.items():
        bal = balance_per_length(df)
        print(f"\n--- {label}  (balanced pooled n={len(bal)}) ---")
        print(f"{'L':>4} | {'n_bal':>5} | "
              + " | ".join([f"T={t:<5.3f}: pass% (design/n)" for t in snap_thresholds])
              + " | argmax T  designable")

        for L in LENGTHS + ["pooled"]:
            sub = bal if L == "pooled" else bal[bal.sequence_length == L]
            n = len(sub)
            if n == 0:
                continue
            ep = esm_pass(sub, criterion)
            row = f"{str(L):>4} | {n:>5} | "
            cells = []
            for t in snap_thresholds:
                mask = sub.lobster_tm.values >= t
                kept = mask.sum()
                pr = ep.values[mask].mean() * 100 if kept > 0 else float("nan")
                ds = int(ep.values[mask].sum())
                cells.append(f"{pr:5.1f}% ({ds:3d}/{kept:3d})")
            row += " | ".join(cells)

            ts = np.linspace(0.0, 1.0, 101)
            best_d, best_t, best_n_kept, best_pr = -1, 0.0, 0, 0.0
            for t in ts:
                mask = sub.lobster_tm.values >= t
                if mask.sum() == 0:
                    continue
                d = int(ep.values[mask].sum())
                if d > best_d:
                    best_d = d
                    best_t = t
                    best_n_kept = int(mask.sum())
                    best_pr = ep.values[mask].mean() * 100
            row += f" | T={best_t:.2f} -> {best_d}/{best_n_kept} ({best_pr:4.1f}%)"
            print(row)


# ------------------- main ----------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ted-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--stoch-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/cv/home/lisanzas/lobster"))
    ap.add_argument("--criterion", choices=["tm", "rmsd", "joint"],
                    default="joint")
    args = ap.parse_args()

    runs: dict[str, pd.DataFrame] = {}
    for label, d in [("TED-val25-base", args.ted_dir),
                     ("val25-base", args.base_dir),
                     ("TED-stoch", args.stoch_dir)]:
        if d is None:
            continue
        try:
            runs[label] = load_run(d)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}", file=sys.stderr)

    if not runs:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_balanced_pooled(
        runs, args.out_dir / "sr_qc_tm_threshold_sweep_balanced.png",
        criterion=args.criterion)
    plot_per_length(
        runs,
        args.out_dir / "sr_qc_tm_threshold_sweep_per_length_passrate.png",
        args.out_dir / "sr_qc_tm_threshold_sweep_per_length_yield.png",
        criterion=args.criterion)
    print_summary(runs, criterion=args.criterion)


if __name__ == "__main__":
    main()
