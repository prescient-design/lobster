"""Per-target before/after jump plot for IF self-reflection on CAMEO.

For each target we have:
  - "before"  = single-shot ESMFold metrics (random_pick from the
                best-of-30 IF run; same model/checkpoint/hyperparameters)
  - "after_<thr>" = ESMFold metrics on the SR-accepted (or fallback)
                    candidate at internal-FF TM threshold = <thr>

Produces a multi-panel plot:
  Top:    sc-RMSD before vs after, one line per target sorted by Δ
          (designability threshold at 2 Å overlaid).
  Middle: sc-TM before vs after.
  Bottom: per-target Δ-RMSD distribution per cutoff (paired bars).

Usage:
    uv run python scripts/plot_if_sr_jump.py \\
        --before-bestofN /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse/bestofN_if_summary_20260503T010309.csv \\
        --sr 0.833:/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_if_self_reflection/if_sr_summary_20260503T174324.csv \\
        --sr 0.9:/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_if_self_reflection_tm0_9/if_sr_summary_*.csv \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_if_self_reflection
"""

from __future__ import annotations

import argparse
import glob
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_if_sr_jump")


def _resolve_glob(spec: str) -> Path | None:
    matches = sorted(glob.glob(spec))
    return Path(matches[-1]) if matches else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--before-bestofN",
        type=Path,
        default=Path(
            "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_inverse/bestofN_if_summary_20260503T010309.csv"
        ),
    )
    p.add_argument(
        "--sr",
        action="append",
        required=True,
        help="One or more SR runs as <threshold>:<path-or-glob>",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    bn = pd.read_csv(args.before_bestofN)
    before = bn[
        ["target", "length", "random_pick_esmfold_tm_score", "random_pick_esmfold_rmsd",
         "random_pick_esmfold_plddt", "random_pick_aar"]
    ].rename(
        columns={
            "random_pick_esmfold_tm_score": "before_tm",
            "random_pick_esmfold_rmsd": "before_rmsd",
            "random_pick_esmfold_plddt": "before_plddt",
            "random_pick_aar": "before_aar",
        }
    )
    logger.info("Loaded before-baseline (single-shot from best-of-30 random_pick): %d targets", len(before))

    sr_runs = []
    for spec in args.sr:
        thr_str, fp_str = spec.split(":", 1)
        thr = float(thr_str)
        fp = _resolve_glob(fp_str)
        if fp is None or not fp.exists():
            logger.warning("Skipping cutoff %.3f — no SR summary at %s", thr, fp_str)
            continue
        d = pd.read_csv(fp)
        d = d.rename(
            columns={
                "accepted_esmfold_tm": f"after_tm_{thr}",
                "accepted_esmfold_rmsd": f"after_rmsd_{thr}",
                "accepted_esmfold_plddt": f"after_plddt_{thr}",
                "accepted_aar": f"after_aar_{thr}",
                "accepted": f"accepted_{thr}",
                "attempts_used": f"attempts_used_{thr}",
                "fallback_used": f"fallback_used_{thr}",
            }
        )[
            [
                "target", "length",
                f"after_tm_{thr}", f"after_rmsd_{thr}", f"after_plddt_{thr}", f"after_aar_{thr}",
                f"accepted_{thr}", f"attempts_used_{thr}", f"fallback_used_{thr}",
            ]
        ]
        sr_runs.append((thr, d))
        logger.info("Loaded SR @ %.3f from %s (%d targets)", thr, fp.name, len(d))

    if not sr_runs:
        raise RuntimeError("No SR runs loaded — check --sr arguments")

    df = before
    for _, d in sr_runs:
        df = df.merge(d, on=["target", "length"], how="inner")
    logger.info("Joined targets: %d", len(df))

    # ---------------- Print summary ----------------
    base_pass = (df.before_rmsd < 2.0)
    base_tm = df.before_tm
    print(f"\n=== IF self-reflection: before/after summary (n={len(df)}) ===")
    print(f"  Single-shot baseline:  designable {base_pass.mean()*100:.1f}%   sc-TM {base_tm.mean():.3f}   sc-RMSD {df.before_rmsd.mean():.2f}")
    for thr, _ in sr_runs:
        rmsd = df[f"after_rmsd_{thr}"]
        tm   = df[f"after_tm_{thr}"]
        att  = df[f"attempts_used_{thr}"]
        acc  = df[f"accepted_{thr}"]
        ppass = (rmsd < 2.0)
        # paired tests
        n_better_pass = int(((~base_pass) & ppass).sum())
        n_worse_pass  = int((base_pass & (~ppass)).sum())
        try:
            mp = stats.binomtest(n_better_pass, n_better_pass + n_worse_pass, 0.5,
                                 alternative="two-sided").pvalue if n_better_pass + n_worse_pass else float("nan")
        except Exception:
            mp = float("nan")
        try:
            wp = stats.wilcoxon(rmsd, df.before_rmsd, alternative="less", zero_method="wilcox").pvalue
        except Exception:
            wp = float("nan")
        d_pass = (ppass.mean() - base_pass.mean()) * 100
        print(
            f"  SR @ TM≥{thr:.3f}:  designable {ppass.mean()*100:5.1f}%  Δ {d_pass:+.1f} pp (McN p={mp:.4f}, sign {n_worse_pass}/{n_better_pass})   "
            f"sc-TM {tm.mean():.3f}  sc-RMSD {rmsd.mean():.2f} (Wilcox p={wp:.4f})   "
            f"accept-rate {acc.mean()*100:.1f}%  mean attempts {att.mean():.1f}"
        )

    # ---------------- Plot ----------------
    n_thr = len(sr_runs)
    fig, axes = plt.subplots(2 + n_thr, 1, figsize=(13, 4 + 3 * n_thr), sharex=False)
    if n_thr == 0:
        axes = [axes]

    cmap_thrs = plt.get_cmap("plasma", max(n_thr, 2))
    thr_colors = {thr: cmap_thrs(i) for i, (thr, _) in enumerate(sr_runs)}

    # ---- Panel 0: paired sc-RMSD per target (sorted by max Δ-rmsd across cutoffs) ----
    ax = axes[0]
    # For sort key use the deepest cutoff if available (largest threshold = strongest)
    best_thr = max(thr for thr, _ in sr_runs)
    delta_rmsd_best = df[f"after_rmsd_{best_thr}"] - df.before_rmsd
    order = np.argsort(delta_rmsd_best.values)
    x = np.arange(len(df))
    for thr, _ in sr_runs:
        ax.scatter(
            x, df[f"after_rmsd_{thr}"].values[order],
            color=thr_colors[thr], s=9, alpha=0.85, label=f"after SR @ TM≥{thr:.3f}",
        )
    ax.scatter(x, df.before_rmsd.values[order], color="black", s=9, alpha=0.5, label="single-shot (before)")
    ax.axhline(2.0, color="red", lw=0.8, ls="--", label="designable (RMSD<2)")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_ylabel("sc-RMSD (Å)")
    ax.set_xlabel("target index (sorted by Δ-RMSD vs single-shot, deepest cutoff)")
    ax.set_title(f"IF self-reflection: per-target sc-RMSD before vs after (n={len(df)})")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)

    # ---- Panel 1: paired sc-TM per target ----
    ax = axes[1]
    delta_tm_best = df[f"after_tm_{best_thr}"] - df.before_tm
    order_tm = np.argsort(-delta_tm_best.values)
    for thr, _ in sr_runs:
        ax.scatter(x, df[f"after_tm_{thr}"].values[order_tm], color=thr_colors[thr], s=9, alpha=0.85,
                   label=f"after SR @ TM≥{thr:.3f}")
    ax.scatter(x, df.before_tm.values[order_tm], color="black", s=9, alpha=0.5, label="single-shot (before)")
    ax.set_ylabel("sc-TM")
    ax.set_xlabel("target index (sorted by Δ-TM, deepest cutoff)")
    ax.set_title("Per-target sc-TM before vs after")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=8)

    # ---- Per-cutoff Δ panels ----
    for i, (thr, _) in enumerate(sr_runs):
        ax = axes[2 + i]
        d_rmsd = df[f"after_rmsd_{thr}"] - df.before_rmsd
        ord_d = np.argsort(d_rmsd.values)
        colors = ["forestgreen" if v < 0 else ("crimson" if v > 0 else "grey") for v in d_rmsd.values[ord_d]]
        ax.bar(x, d_rmsd.values[ord_d], color=colors, edgecolor="none")
        ax.axhline(0, color="black", lw=0.5)
        n_better = int((d_rmsd < 0).sum()); n_worse = int((d_rmsd > 0).sum())
        ax.set_xlabel(f"target (sorted by Δ-RMSD, n_better={n_better} / n_worse={n_worse})")
        ax.set_ylabel(f"Δ-RMSD (after@{thr:.3f} − before)")
        ax.set_title(
            f"Per-target Δ-RMSD, SR @ TM≥{thr:.3f}.  green = SR improved, red = SR worsened"
        )
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    out = args.output_dir / f"if_sr_before_after_{ts}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)

    # also save merged data CSV for reuse
    df.to_csv(args.output_dir / f"if_sr_before_after_{ts}.csv", index=False)
    logger.info("Wrote %s", args.output_dir / f"if_sr_before_after_{ts}.csv")


if __name__ == "__main__":
    main()
