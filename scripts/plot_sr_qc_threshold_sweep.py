"""Threshold-sweep analysis for the SR QC gates.

Inputs (per concordance dir):
  - unconditional_metrics_*.csv    -> accepted samples (lobster forward TM in
                                       self_reflection_* row, ESMFold metrics in
                                       unconditional_* row, %id in self_reflection_*)
  - esmfold_failed_attempts_*.csv  -> rejected samples (forward_tm only;
                                       lobster_forward_tm + post-hoc ESMFold;
                                       %id is NaN since reject happens before
                                       inverse-folding)

Outputs:
  1) Per-checkpoint scatter (lobster TM x ESMFold TM) with %id as colour for
     accepted samples; rejected drawn in grey.
  2) %id histogram (accepted only) split by ESMFold pass / reject.
  3) Forward-fold-TM threshold sweep: for T in [0, 1], how many samples are
     retained (accept = lobster_TM >= T) and what fraction of those retained
     pass an ESMFold-based gate. Lets us see if a different cutoff would have
     been better.
  4) %id threshold sweep on accepted samples only (we only have %id for those):
     for thr in [50, 100], how many of the accepted samples remain and what's
     their ESMFold pass rate.

Usage:
  uv run python scripts/plot_sr_qc_threshold_sweep.py \
      --ted-dir <ted_concordance_dir> \
      --base-dir <base_concordance_dir> \
      --stoch-dir <ted_stoch_concordance_dir>
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SR_FORWARD_TM = 0.833
SR_PCT_ID_LO = 50.0
ESM_TM_PASS = 0.9
ESM_RMSD_PASS = 2.0
RUN_ID_RX = re.compile(r"length_(\d+)_iter_(\d+)")


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _parse_iter(s: str) -> tuple[int, int] | tuple[float, float]:
    m = RUN_ID_RX.search(str(s))
    if not m:
        return float("nan"), float("nan")
    return int(m.group(1)), int(m.group(2))


def load_run(conc_dir: Path) -> pd.DataFrame:
    """Returns a single dataframe with one row per attempt and columns:
       sequence_length, lobster_tm, esmfold_tm, esmfold_rmsd, percent_id, sr_pass.
    """
    csv = _latest(str(conc_dir / "unconditional_metrics_*.csv"))
    if csv is None:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {conc_dir}")
    df = pd.read_csv(csv)
    df = df[df["mode"] == "unconditional"].copy()
    df[["L", "iter"]] = df["run_id"].apply(lambda s: pd.Series(_parse_iter(s)))
    sr = df[df["run_id"].str.startswith("self_reflection_")][[
        "L", "iter",
        "tm_score_unconditional_to_forward",
        "percent_identity_self_reflection",
    ]].rename(columns={
        "tm_score_unconditional_to_forward": "lobster_tm",
        "percent_identity_self_reflection": "percent_id",
    })
    un = df[df["run_id"].str.startswith("unconditional_")][[
        "L", "iter", "tm_score", "rmsd", "sequence_length"]].rename(columns={
        "tm_score": "esmfold_tm", "rmsd": "esmfold_rmsd"})
    accepted = sr.merge(un, on=["L", "iter"], how="inner")
    accepted["sr_pass"] = True

    rcsv = _latest(str(conc_dir / "esmfold_failed_attempts_*.csv"))
    if rcsv is None:
        raise FileNotFoundError(f"No esmfold_failed_attempts_*.csv in {conc_dir}")
    r = pd.read_csv(rcsv)
    r = r[r["failure_reason"] == "forward_tm"].rename(columns={
        "tm_score_unconditional_to_forward": "lobster_tm",
        "percent_identity_self_reflection": "percent_id",
        "sequence_length": "sequence_length",
    })
    rejected = r[["sequence_length", "lobster_tm", "percent_id",
                  "esmfold_tm", "esmfold_rmsd"]].copy()
    rejected["sr_pass"] = False
    rejected["L"] = rejected["sequence_length"]
    rejected["iter"] = np.nan

    keep = ["L", "iter", "sequence_length", "lobster_tm", "percent_id",
            "esmfold_tm", "esmfold_rmsd", "sr_pass"]
    return pd.concat([accepted[keep], rejected[keep]], ignore_index=True)


def esm_pass(df: pd.DataFrame, criterion: str) -> pd.Series:
    if criterion == "tm":
        return df["esmfold_tm"] >= ESM_TM_PASS
    if criterion == "rmsd":
        return df["esmfold_rmsd"] < ESM_RMSD_PASS
    if criterion == "joint":
        return (df["esmfold_tm"] >= ESM_TM_PASS) & (df["esmfold_rmsd"] < ESM_RMSD_PASS)
    raise ValueError(criterion)


# ----------------------------- plot 1: scatter w/ %id colour ---------

def plot_scatter_pid(runs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 6.2),
                             squeeze=False)
    for ax, (label, df) in zip(axes[0], runs.items()):
        rej = df[~df.sr_pass]
        acc = df[df.sr_pass]
        ax.scatter(rej.lobster_tm, rej.esmfold_tm, c="lightgrey", s=18,
                   alpha=0.55, edgecolor="black", linewidths=0.3,
                   label=f"SR-rejected (n={len(rej)}, no %id)")
        sc = ax.scatter(acc.lobster_tm, acc.esmfold_tm, c=acc.percent_id,
                        cmap="viridis", vmin=50, vmax=100, s=18, alpha=0.85,
                        edgecolor="none",
                        label=f"SR-accepted (n={len(acc)})")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percent identity (initial vs refined seq)")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="y = x")
        ax.axvline(SR_FORWARD_TM, color="red", ls=":", lw=1, alpha=0.6,
                   label=f"SR fwd-fold TM = {SR_FORWARD_TM}")
        ax.axhline(ESM_TM_PASS, color="purple", ls=":", lw=1, alpha=0.6,
                   label=f"ESMFold TM = {ESM_TM_PASS}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Leflur forward-fold TM\n(forward-fold(initial seq) vs initial backbone)")
        ax.set_ylabel("ESMFold TM\n(ESMFold(initial seq) vs initial backbone)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.suptitle(
        "SR-QC: forward-fold TM vs ESMFold TM, coloured by %id (accepted only)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ----------------------------- plot 2: %id histograms ----------------

def plot_pid_histograms(runs: dict[str, pd.DataFrame], out_path: Path,
                        criterion: str = "joint") -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4.5),
                             squeeze=False)
    bins = np.arange(50, 102, 2)
    for ax, (label, df) in zip(axes[0], runs.items()):
        acc = df[df.sr_pass].copy()
        acc["esm_pass"] = esm_pass(acc, criterion)
        ax.hist(acc[acc.esm_pass]["percent_id"], bins=bins, alpha=0.7,
                color="tab:blue", label=f"ESM-pass ({(acc.esm_pass).sum()})")
        ax.hist(acc[~acc.esm_pass]["percent_id"], bins=bins, alpha=0.7,
                color="tab:red", label=f"ESM-reject ({(~acc.esm_pass).sum()})")
        ax.axvline(SR_PCT_ID_LO, color="red", ls=":", lw=1, alpha=0.7,
                   label=f"SR floor = {SR_PCT_ID_LO}")
        ax.set_xlabel("Percent identity (initial vs refined sequence)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} (accepted only)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25)
    crit_name = {"tm": "TM ≥ 0.9", "rmsd": "RMSD < 2 Å",
                 "joint": "TM ≥ 0.9 AND RMSD < 2 Å"}[criterion]
    fig.suptitle(f"%id distribution split by ESMFold pass ({crit_name})",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ----------------------------- plot 3: TM threshold sweep ------------

def plot_tm_sweep(runs: dict[str, pd.DataFrame], out_path: Path,
                  criterion: str = "joint") -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.0),
                             squeeze=False)

    for ax, (label, df) in zip(axes[0], runs.items()):
        df = df.copy()
        df["esm_pass"] = esm_pass(df, criterion)
        n_total = len(df)

        keep_frac = []
        designability = []
        n_design = []
        for t in thresholds:
            kept = df[df.lobster_tm >= t]
            keep_frac.append(len(kept) / n_total)
            if len(kept) > 0:
                designability.append(kept["esm_pass"].mean())
                n_design.append(int(kept["esm_pass"].sum()))
            else:
                designability.append(np.nan)
                n_design.append(0)

        ax2 = ax.twinx()
        ax.plot(thresholds, designability, color="tab:blue", lw=2,
                label="ESM-pass rate of retained set")
        ax2.plot(thresholds, keep_frac, color="tab:orange", lw=2,
                 label="Fraction of attempts retained")
        ax2.plot(thresholds, [n / n_total for n in n_design],
                 color="tab:green", lw=2, ls="--",
                 label="Designable / total attempts")

        # annotate current SR threshold
        idx = np.argmin(np.abs(thresholds - SR_FORWARD_TM))
        ax.axvline(SR_FORWARD_TM, color="red", ls=":", lw=1, alpha=0.7)
        ax.scatter([SR_FORWARD_TM], [designability[idx]], color="red",
                   zorder=5, s=40)
        ax.annotate(
            f"current SR\nT={SR_FORWARD_TM}\n  retain={keep_frac[idx]*100:.1f}%\n"
            f"  ESM-pass={designability[idx]*100:.1f}%\n  designable={n_design[idx]}/{n_total}",
            xy=(SR_FORWARD_TM, designability[idx]),
            xytext=(0.05, 0.92), textcoords="axes fraction",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        # find max designable count + its threshold
        best = int(np.nanargmax(n_design))
        if n_design[best] > n_design[idx]:
            ax.axvline(thresholds[best], color="green", ls=":", lw=1, alpha=0.6)
            ax.annotate(
                f"max designable\nT={thresholds[best]:.2f}\n  retain={keep_frac[best]*100:.1f}%\n"
                f"  ESM-pass={designability[best]*100:.1f}%\n  designable={n_design[best]}/{n_total}",
                xy=(thresholds[best], designability[best]),
                xytext=(0.55, 0.18), textcoords="axes fraction",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="green", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Forward-fold-TM threshold (SR QC)")
        ax.set_ylabel("Designable / retained (= ESM pass-rate of kept set)",
                      color="tab:blue")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax2.set_ylabel("Fraction of attempts retained / Designable / total",
                       color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax2.set_ylim(0, 1)
        ax.set_title(f"{label} (n={n_total})")
        ax.grid(True, alpha=0.2)
        # combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="lower left", fontsize=8, framealpha=0.85)

    crit_name = {"tm": "ESM TM ≥ 0.9", "rmsd": "ESM RMSD < 2 Å",
                 "joint": "ESM TM ≥ 0.9 AND RMSD < 2 Å"}[criterion]
    fig.suptitle(
        f"Forward-fold-TM threshold sweep (criterion: {crit_name})",
        fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ----------------------------- plot 4: %id threshold sweep -----------

def plot_pid_sweep(runs: dict[str, pd.DataFrame], out_path: Path,
                   criterion: str = "joint") -> None:
    thresholds = np.linspace(50, 100, 51)
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.0),
                             squeeze=False)

    for ax, (label, df) in zip(axes[0], runs.items()):
        # Only accepted samples have %id; the gate is a *post-acceptance* tightener
        acc = df[df.sr_pass].copy()
        acc["esm_pass"] = esm_pass(acc, criterion)
        n_total = len(acc)

        keep_frac, designability, n_design = [], [], []
        for t in thresholds:
            kept = acc[acc.percent_id >= t]
            keep_frac.append(len(kept) / n_total)
            if len(kept) > 0:
                designability.append(kept["esm_pass"].mean())
                n_design.append(int(kept["esm_pass"].sum()))
            else:
                designability.append(np.nan)
                n_design.append(0)

        ax2 = ax.twinx()
        ax.plot(thresholds, designability, color="tab:blue", lw=2,
                label="ESM-pass rate of retained set")
        ax2.plot(thresholds, keep_frac, color="tab:orange", lw=2,
                 label="Fraction of accepted retained")
        ax2.plot(thresholds, [n / n_total for n in n_design],
                 color="tab:green", lw=2, ls="--",
                 label="Designable / total accepted")

        idx = np.argmin(np.abs(thresholds - SR_PCT_ID_LO))
        ax.axvline(SR_PCT_ID_LO, color="red", ls=":", lw=1, alpha=0.7)
        ax.scatter([SR_PCT_ID_LO], [designability[idx]], color="red",
                   zorder=5, s=40)
        ax.annotate(
            f"current SR\n%id≥{SR_PCT_ID_LO:.0f}\n  retain={keep_frac[idx]*100:.1f}%\n"
            f"  ESM-pass={designability[idx]*100:.1f}%\n  designable={n_design[idx]}/{n_total}",
            xy=(SR_PCT_ID_LO, designability[idx]),
            xytext=(0.55, 0.85), textcoords="axes fraction",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        best = int(np.nanargmax(n_design))
        if n_design[best] != n_design[idx]:
            ax.axvline(thresholds[best], color="green", ls=":", lw=1, alpha=0.6)
            ax.annotate(
                f"max designable\n%id≥{thresholds[best]:.0f}\n  retain={keep_frac[best]*100:.1f}%\n"
                f"  ESM-pass={designability[best]*100:.1f}%\n  designable={n_design[best]}/{n_total}",
                xy=(thresholds[best], designability[best]),
                xytext=(0.05, 0.10), textcoords="axes fraction",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="green", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8))

        ax.set_xlim(50, 100); ax.set_ylim(0, 1)
        ax.set_xlabel("Percent-identity threshold (post-acceptance tightener)")
        ax.set_ylabel("Designable / retained", color="tab:blue")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax2.set_ylabel("Fraction retained / Designable / total accepted",
                       color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax2.set_ylim(0, 1)
        ax.set_title(f"{label} (n_accepted={n_total})")
        ax.grid(True, alpha=0.2)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", fontsize=8, framealpha=0.85)

    crit_name = {"tm": "ESM TM ≥ 0.9", "rmsd": "ESM RMSD < 2 Å",
                 "joint": "ESM TM ≥ 0.9 AND RMSD < 2 Å"}[criterion]
    fig.suptitle(
        f"Percent-identity threshold sweep on accepted samples (criterion: {crit_name})",
        fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# --------------------------- main ------------------------------------

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
    plot_scatter_pid(runs, args.out_dir / "sr_qc_scatter_with_pid.png")
    plot_pid_histograms(runs, args.out_dir / "sr_qc_pid_hist.png",
                        criterion=args.criterion)
    plot_tm_sweep(runs, args.out_dir / "sr_qc_tm_threshold_sweep.png",
                  criterion=args.criterion)
    plot_pid_sweep(runs, args.out_dir / "sr_qc_pid_threshold_sweep.png",
                   criterion=args.criterion)


if __name__ == "__main__":
    main()
