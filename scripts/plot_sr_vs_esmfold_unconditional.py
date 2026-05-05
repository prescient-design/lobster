"""Per-sample scatter: lobster forward-fold TM vs ESMFold TM, both vs the
*generated* (initial) backbone, for ALL accepted + rejected SR attempts.

For each iteration:
  - x = lobster_forward_tm = TM(forward-fold(initial seq), initial backbone)
  - y = esmfold_tm         = TM(ESMFold(initial seq), initial backbone)

Accepted samples come from ``unconditional_metrics_*.csv``. The lobster forward
TM is stored on the *paired* ``self_reflection_*`` row's
``tm_score_unconditional_to_forward`` column; the ESMFold TM is on the
``unconditional_*`` row's ``tm_score`` column. We join the pair via
(sequence_length, iteration) parsed out of ``run_id``.

Rejected samples come from ``esmfold_failed_attempts_*.csv`` (produced by
``scripts/esmfold_failed_attempts.py``); both axes are present per row.

Usage:
    uv run python scripts/plot_sr_vs_esmfold_unconditional.py \\
        --ted-dir <ted_concordance_dir> \\
        [--base-dir <base_concordance_dir>] \\
        --out /cv/home/lisanzas/lobster/sr_vs_esmfold_unconditional.png
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
from scipy.stats import pearsonr, spearmanr


SR_FORWARD_TM = 0.833
ESM_TM_PASS = 0.9
ESM_RMSD_PASS = 2.0
RUN_ID_RX = re.compile(r"length_(\d+)_iter_(\d+)")


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _parse_iter(run_id: str) -> tuple[int, int] | tuple[None, None]:
    m = RUN_ID_RX.search(str(run_id))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_pairs(conc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv = _latest(str(conc_dir / "unconditional_metrics_*.csv"))
    if csv is None:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {conc_dir}")
    df = pd.read_csv(csv)
    df = df[df["mode"] == "unconditional"].copy()
    df[["L", "iter"]] = df["run_id"].apply(lambda s: pd.Series(_parse_iter(s)))
    sr = df[df["run_id"].str.startswith("self_reflection_")][[
        "L", "iter", "tm_score_unconditional_to_forward"]].rename(
        columns={"tm_score_unconditional_to_forward": "lobster_tm"})
    un = df[df["run_id"].str.startswith("unconditional_")][[
        "L", "iter", "tm_score", "rmsd"]].rename(
        columns={"tm_score": "esmfold_tm", "rmsd": "esmfold_rmsd"})
    accepted = sr.merge(un, on=["L", "iter"], how="inner")
    accepted["sr_pass"] = True
    accepted = accepted.dropna(subset=["lobster_tm", "esmfold_tm"])

    rejected_csv = _latest(str(conc_dir / "esmfold_failed_attempts_*.csv"))
    if rejected_csv is None:
        raise FileNotFoundError(f"No esmfold_failed_attempts_*.csv in {conc_dir} -- "
                                "run scripts/esmfold_failed_attempts.py first")
    rdf = pd.read_csv(rejected_csv)
    rdf = rdf[rdf["failure_reason"] == "forward_tm"].copy()
    rdf = rdf.rename(columns={
        "tm_score_unconditional_to_forward": "lobster_tm",
        "esmfold_tm": "esmfold_tm",
        "esmfold_rmsd": "esmfold_rmsd",
        "sequence_length": "L",
        "iteration": "iter",
    })
    rejected = rdf[["L", "iter", "lobster_tm", "esmfold_tm", "esmfold_rmsd"]].copy()
    rejected["sr_pass"] = False
    rejected = rejected.dropna(subset=["lobster_tm", "esmfold_tm"])

    return accepted, rejected


def report_stats(accepted: pd.DataFrame, rejected: pd.DataFrame, label: str) -> None:
    full = pd.concat([accepted, rejected], ignore_index=True)
    print(f"\n=== {label} ===")
    print(f"  accepted: n={len(accepted)}   rejected: n={len(rejected)}   combined: n={len(full)}")
    if len(full) >= 3:
        pr, ppv = pearsonr(full.lobster_tm, full.esmfold_tm)
        sr, spv = spearmanr(full.lobster_tm, full.esmfold_tm)
        print(f"  combined  Pearson r = {pr:.3f} (p={ppv:.2e})")
        print(f"  combined  Spearman = {sr:.3f} (p={spv:.2e})")
    if len(accepted) >= 3:
        pr, _ = pearsonr(accepted.lobster_tm, accepted.esmfold_tm)
        sr, _ = spearmanr(accepted.lobster_tm, accepted.esmfold_tm)
        print(f"  accepted  Pearson r = {pr:.3f}   Spearman = {sr:.3f}")
    if len(rejected) >= 3:
        pr, _ = pearsonr(rejected.lobster_tm, rejected.esmfold_tm)
        sr, _ = spearmanr(rejected.lobster_tm, rejected.esmfold_tm)
        print(f"  rejected  Pearson r = {pr:.3f}   Spearman = {sr:.3f}")


def plot(runs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], out_path: Path,
         y_metric: str = "tm") -> None:
    """``y_metric`` chooses the y-axis quantity / ESMFold pass criterion:
        - ``"tm"``    : y = ESMFold TM, pass when y >= 0.9 (default)
        - ``"rmsd"`` : y = ESMFold RMSD (Å), pass when y < 2.0
    The x-axis (lobster forward-fold TM, SR threshold = 0.833) is unchanged in
    both cases, so the SR pass/reject coloring still uses the existing QC.
    """
    if y_metric == "rmsd":
        y_col = "esmfold_rmsd"
        y_label = "ESMFold RMSD (Å)\n(ESMFold(initial seq) vs initial backbone)"
        y_threshold = ESM_RMSD_PASS
        y_threshold_label = f"ESMFold RMSD = {ESM_RMSD_PASS} Å"
        y_top = 10.0  # cap visible range; outliers clipped to top edge
        esm_pass_op = lambda v: v < y_threshold  # noqa: E731
        threshold_dir = "<"
    else:
        y_col = "esmfold_tm"
        y_label = "ESMFold TM\n(ESMFold(initial seq) vs initial backbone)"
        y_threshold = ESM_TM_PASS
        y_threshold_label = f"ESMFold TM = {ESM_TM_PASS}"
        y_top = 1.0
        esm_pass_op = lambda v: v >= y_threshold  # noqa: E731
        threshold_dir = ">="

    fig, axes = plt.subplots(1, len(runs), figsize=(6.6 * len(runs), 6.4),
                             squeeze=False)

    for ax, (label, (accepted, rejected)) in zip(axes[0], runs.items()):
        # clip y for display only
        acc_y = accepted[y_col].clip(upper=y_top) if y_metric == "rmsd" else accepted[y_col]
        rej_y = rejected[y_col].clip(upper=y_top) if y_metric == "rmsd" else rejected[y_col]

        ax.scatter(accepted.lobster_tm, acc_y,
                   c="tab:blue", s=14, alpha=0.45, edgecolor="none",
                   label=f"SR-accepted (n={len(accepted)})")
        ax.scatter(rejected.lobster_tm, rej_y,
                   c="tab:red", s=22, alpha=0.7,
                   edgecolor="black", linewidths=0.3,
                   label=f"SR-rejected (n={len(rejected)})")

        if y_metric == "tm":
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="y = x")
        ax.axvline(SR_FORWARD_TM, color="red", ls=":", lw=1, alpha=0.6,
                   label=f"SR threshold = {SR_FORWARD_TM}")
        ax.axhline(y_threshold, color="purple", ls=":", lw=1, alpha=0.6,
                   label=y_threshold_label)

        full = pd.concat([accepted, rejected], ignore_index=True)
        if len(full) >= 3 and full[y_col].notna().sum() >= 3:
            pr, _ = pearsonr(full.lobster_tm, full[y_col])
            sr, _ = spearmanr(full.lobster_tm, full[y_col])
        else:
            pr = sr = float("nan")

        # quadrant counts
        df = full.copy()
        df["esm_pass"] = esm_pass_op(df[y_col])
        df["sr_pass_thr"] = df["lobster_tm"] >= SR_FORWARD_TM
        aa = ((df.sr_pass_thr) & (df.esm_pass)).sum()
        ar = ((df.sr_pass_thr) & (~df.esm_pass)).sum()
        ra = ((~df.sr_pass_thr) & (df.esm_pass)).sum()
        rr = ((~df.sr_pass_thr) & (~df.esm_pass)).sum()
        text = (
            f"n total       = {len(full)}\n"
            f"Pearson r     = {pr:.3f}\n"
            f"Spearman      = {sr:.3f}\n"
            f"ESM pass: y {threshold_dir} {y_threshold}\n"
            f"SR-pass + ESM-pass:    {aa}\n"
            f"SR-pass + ESM-reject:  {ar}\n"
            f"SR-reject + ESM-pass:  {ra}\n"
            f"SR-reject + ESM-reject:{rr}"
        )
        ax.text(0.04, 0.97, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="black", alpha=0.85))

        ax.set_xlim(0.0, 1.0)
        if y_metric == "rmsd":
            ax.set_ylim(0.0, y_top)
        else:
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Leflur forward-fold TM\n(forward-fold(initial seq) vs initial backbone)")
        ax.set_ylabel(y_label)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.25)
        loc = "upper right" if y_metric == "rmsd" else "lower right"
        ax.legend(loc=loc, fontsize=8, framealpha=0.85)

    yax = "TM" if y_metric == "tm" else f"RMSD (clipped at {y_top:.0f} Å for display)"
    fig.suptitle(
        f"Leflur forward-fold TM (SR QC) vs ESMFold {yax}, both vs initial backbone\n"
        "(all unconditional SR attempts: accepted + rejected)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ted-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("/cv/home/lisanzas/lobster/sr_vs_esmfold_unconditional.png"))
    ap.add_argument("--y-metric", choices=["tm", "rmsd"], default="tm",
                    help="Y-axis quantity / ESMFold pass criterion")
    args = ap.parse_args()

    runs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for label, d in [("TED-val25-base", args.ted_dir), ("val25-base", args.base_dir)]:
        if d is None:
            continue
        try:
            acc, rej = load_pairs(d)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}", file=sys.stderr)
            continue
        runs[label] = (acc, rej)
        report_stats(acc, rej, label)

    if runs:
        plot(runs, args.out, y_metric=args.y_metric)


if __name__ == "__main__":
    main()
