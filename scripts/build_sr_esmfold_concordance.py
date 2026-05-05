"""Build the SR-QC vs ESMFold-QC concordance matrix.

For each checkpoint:
  - Accepted samples come from ``unconditional_metrics_*.csv`` (rows where
    ``mode=='unconditional'``). On these, ESMFold metrics were already computed
    inline by the SR run (use_esmfold_validation=true). Only forward-fold-TM
    accept matters here -- by construction every accepted row passed the SR QC,
    so SR-QC = PASS for all of them.
  - Rejected samples come from ``esmfold_failed_attempts_*.csv`` (this script's
    sibling ``esmfold_failed_attempts.py`` produced these). For each row we
    have lobster forward-fold TM (which was < 0.833) and the post-hoc
    ESMFold TM/RMSD/pLDDT vs the saved initial backbone.

The 2x2 matrix uses:
  - SR-QC pass    <-> sample is in the accepted set (forward-fold TM >= 0.833)
  - SR-QC reject  <-> sample is in the failed set (forward-fold TM < 0.833)
  - ESMFold pass  <-> ESMFold TM >= 0.9 AND RMSD < 2.0
  - ESMFold reject <-> NOT ESMFold pass

Usage:
    uv run python scripts/build_sr_esmfold_concordance.py \\
        --ted-dir <ted_concordance_dir> \\
        --base-dir <base_concordance_dir>
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SR_FORWARD_TM = 0.833
ESM_TM_PASS = 0.9
ESM_RMSD_PASS = 2.0


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_accepted(conc_dir: Path) -> pd.DataFrame:
    """Read unconditional_metrics_*.csv -- the accepted samples."""
    csv = _latest(str(conc_dir / "unconditional_metrics_*.csv"))
    if csv is None:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {conc_dir}")
    df = pd.read_csv(csv)
    # IMPORTANT: every accepted iteration writes TWO rows with mode=='unconditional':
    #   run_id=self_reflection_*  (diagnostic, tm_score/rmsd are NaN)
    #   run_id=unconditional_*    (the real ESMFold-vs-initial metrics)
    # Keep only the real ones.
    df = df[(df["mode"] == "unconditional") & df["run_id"].str.startswith("unconditional_")].copy()
    df["sr_pass"] = True
    df["esmfold_tm"] = pd.to_numeric(df["tm_score"], errors="coerce")
    df["esmfold_rmsd"] = pd.to_numeric(df["rmsd"], errors="coerce")
    df["esmfold_plddt"] = pd.to_numeric(df.get("plddt", pd.Series([np.nan] * len(df))), errors="coerce")
    df["sequence_length"] = pd.to_numeric(df["sequence_length"], errors="coerce")
    return df[["sequence_length", "sr_pass", "esmfold_tm", "esmfold_rmsd", "esmfold_plddt"]]


def load_rejected(conc_dir: Path) -> pd.DataFrame:
    """Read esmfold_failed_attempts_*.csv -- the saved+ESMFold-d rejected samples."""
    csv = _latest(str(conc_dir / "esmfold_failed_attempts_*.csv"))
    if csv is None:
        raise FileNotFoundError(f"No esmfold_failed_attempts_*.csv in {conc_dir} -- "
                                "run scripts/esmfold_failed_attempts.py first")
    df = pd.read_csv(csv)
    df = df[df["failure_reason"] == "forward_tm"].copy()
    df["sr_pass"] = False
    df["esmfold_tm"] = pd.to_numeric(df["esmfold_tm"], errors="coerce")
    df["esmfold_rmsd"] = pd.to_numeric(df["esmfold_rmsd"], errors="coerce")
    df["esmfold_plddt"] = pd.to_numeric(df["esmfold_plddt"], errors="coerce")
    df["sequence_length"] = pd.to_numeric(df["sequence_length"], errors="coerce")
    df["lobster_tm_failed"] = pd.to_numeric(df["tm_score_unconditional_to_forward"], errors="coerce")
    return df[["sequence_length", "sr_pass", "esmfold_tm", "esmfold_rmsd",
               "esmfold_plddt", "lobster_tm_failed"]]


def esmfold_pass(df: pd.DataFrame) -> pd.Series:
    return (df["esmfold_tm"] >= ESM_TM_PASS) & (df["esmfold_rmsd"] < ESM_RMSD_PASS)


def quadrant_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Build the 2x2 SR-QC × ESMFold-QC matrix."""
    df = df.copy()
    df["esm_pass"] = esmfold_pass(df)
    aa = ((df["sr_pass"]) & (df["esm_pass"])).sum()
    ar = ((df["sr_pass"]) & (~df["esm_pass"])).sum()
    ra = ((~df["sr_pass"]) & (df["esm_pass"])).sum()
    rr = ((~df["sr_pass"]) & (~df["esm_pass"])).sum()
    n = len(df)
    print(f"\n=== {label} (n={n}) ===")
    print(f"                ESMFold pass   ESMFold reject   total")
    print(f"  SR pass     :   {aa:>5}        {ar:>5}            {aa+ar:>5}")
    print(f"  SR reject   :   {ra:>5}        {rr:>5}            {ra+rr:>5}")
    print(f"  total       :   {aa+ra:>5}        {ar+rr:>5}            {n:>5}")
    if (aa + ar) > 0:
        print(f"  P(ESMpass | SRpass)   = {aa / (aa + ar):.3f}")
    if (ra + rr) > 0:
        print(f"  P(ESMpass | SRreject) = {ra / (ra + rr):.3f}")
    if (aa + ra) > 0:
        print(f"  P(SRpass | ESMpass)   = {aa / (aa + ra):.3f}")
    if (ar + rr) > 0:
        print(f"  P(SRpass | ESMreject) = {ar / (ar + rr):.3f}")
    if n > 0:
        print(f"  Cohen's kappa         = {cohens_kappa(aa, ar, ra, rr):.3f}")
    return pd.DataFrame(
        {"ESMpass": [aa, ra], "ESMreject": [ar, rr]},
        index=["SRpass", "SRreject"],
    )


def cohens_kappa(aa: int, ar: int, ra: int, rr: int) -> float:
    n = aa + ar + ra + rr
    if n == 0:
        return float("nan")
    po = (aa + rr) / n
    p_sr = (aa + ar) / n
    p_esm = (aa + ra) / n
    pe = p_sr * p_esm + (1 - p_sr) * (1 - p_esm)
    if abs(1 - pe) < 1e-9:
        return float("nan")
    return (po - pe) / (1 - pe)


def plot_concordance(combined: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(combined), figsize=(6.5 * len(combined), 6),
                             squeeze=False)
    for ax, (label, df) in zip(axes[0], combined.items()):
        accepted = df[df["sr_pass"]]
        rejected = df[~df["sr_pass"]]
        if "lobster_tm_failed" in rejected.columns:
            x_rej = rejected["lobster_tm_failed"].values
        else:
            x_rej = np.full(len(rejected), np.nan)
        # accepted lobster_tm: by definition >= 0.833 -- we don't have the exact value
        # so we only show ESMFold dist for them via boxplot at x=0.95 anchor
        ax.scatter(
            x_rej, rejected["esmfold_tm"],
            c="tab:red", s=24, alpha=0.6, edgecolor="black", linewidths=0.3,
            label=f"SR-rejected (n={len(rejected)})",
        )
        # show accepted as a band on the right -- they are all SR-pass (lobster TM >= 0.833)
        if len(accepted):
            ax.scatter(
                np.full(len(accepted), 0.95) + np.random.RandomState(0).uniform(-0.04, 0.04, size=len(accepted)),
                accepted["esmfold_tm"],
                c="tab:blue", s=10, alpha=0.25, edgecolor="none",
                label=f"SR-accepted (n={len(accepted)})",
            )

        ax.axvline(SR_FORWARD_TM, color="red", ls=":", lw=1, alpha=0.6,
                   label=f"SR threshold = {SR_FORWARD_TM}")
        ax.axhline(ESM_TM_PASS, color="purple", ls=":", lw=1, alpha=0.6,
                   label=f"ESMFold TM = {ESM_TM_PASS}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Lobster forward-fold TM\n(blue band = SR-accepted, exact TM not stored)")
        ax.set_ylabel("ESMFold TM (vs initial backbone)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

        # quadrant counts overlay
        df2 = df.copy()
        df2["esm_pass"] = esmfold_pass(df2)
        aa = ((df2["sr_pass"]) & (df2["esm_pass"])).sum()
        ar = ((df2["sr_pass"]) & (~df2["esm_pass"])).sum()
        ra = ((~df2["sr_pass"]) & (df2["esm_pass"])).sum()
        rr = ((~df2["sr_pass"]) & (~df2["esm_pass"])).sum()
        text = (
            f"SR-pass + ESM-pass:    {aa}\n"
            f"SR-pass + ESM-reject:  {ar}\n"
            f"SR-reject + ESM-pass:  {ra}\n"
            f"SR-reject + ESM-reject:{rr}\n"
            f"P(ESM-pass | SR-reject) = {ra/(ra+rr) if (ra+rr) else float('nan'):.3f}"
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.85))

    fig.suptitle(
        "SR-QC vs ESMFold-QC concordance (unconditional)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ted-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--out-plot", type=Path,
                    default=Path("/cv/home/lisanzas/lobster/sr_esmfold_concordance.png"))
    args = ap.parse_args()

    runs: dict[str, pd.DataFrame] = {}
    for label, d in [("TED-val25-base", args.ted_dir), ("val25-base", args.base_dir)]:
        if d is None:
            continue
        try:
            acc = load_accepted(d)
            rej = load_rejected(d)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}", file=sys.stderr)
            continue
        # align columns
        cols = ["sequence_length", "sr_pass", "esmfold_tm", "esmfold_rmsd", "esmfold_plddt"]
        acc = acc[cols].assign(lobster_tm_failed=np.nan)
        rej = rej[cols + ["lobster_tm_failed"]]
        df = pd.concat([acc, rej], ignore_index=True)
        runs[label] = df
        quadrant_table(df, label)

    if runs:
        plot_concordance(runs, args.out_plot)


if __name__ == "__main__":
    main()
