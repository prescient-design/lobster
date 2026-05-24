"""Conference-style table for the existing N=10 best-of-N forward-folding runs.

Reports mean TM-score, mean RMSD (Å), and pass rate per (checkpoint × picker):
    PASS = RMSD < 2.0 Å

Pickers (per CAMEO target):
    random      = candidate index 0  (= existing single-shot baseline)
    seq_pll     = argmin seq_score_unif
    struc_pll   = argmin struc_score_unif
    joint_pll   = argmin joint_score_unif
    oracle      = argmax tm_score
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import argparse

BASE_DIR = Path("/cv/scratch/u/lisanzas/evaluations")
RUNS_BY_N = {
    10: {
        "GenUME-denovo": BASE_DIR / "gen_ume_denovo_cameo_bestofN_pll",
        "GenUME-base":   BASE_DIR / "gen_ume_base_cameo_bestofN_pll",
        "GenUME-TED":    BASE_DIR / "gen_ume_ted_cameo_bestofN_pll",
    },
    30: {
        "GenUME-denovo": BASE_DIR / "gen_ume_denovo_cameo_bestofN_pll_N30",
        "GenUME-base":   BASE_DIR / "gen_ume_base_cameo_bestofN_pll_N30",
        "GenUME-TED":    BASE_DIR / "gen_ume_ted_cameo_bestofN_pll_N30",
    },
}

PICKERS = [
    ("random",    None,                "candidate_idx==0"),
    ("seq_pll",   "seq_score_unif",    "argmin"),
    ("struc_pll", "struc_score_unif",  "argmin"),
    ("joint_pll", "joint_score_unif",  "argmin"),
    ("oracle",    "tm_score",          "argmax"),
]


def _select_per_target(df: pd.DataFrame, key: str | None, mode: str) -> pd.DataFrame:
    if key is None:  # random = candidate 0
        sel = df[df["candidate_idx"] == 0].copy()
        return sel
    rows = []
    for _, g in df.groupby("target", sort=False):
        if mode == "argmin":
            i = g[key].idxmin()
        elif mode == "argmax":
            i = g[key].idxmax()
        else:
            raise ValueError(mode)
        rows.append(g.loc[i])
    return pd.DataFrame(rows)


def _metrics(sel: pd.DataFrame) -> dict[str, float]:
    tm = sel["tm_score"].to_numpy(dtype=float)
    rm = sel["rmsd"].to_numpy(dtype=float)
    valid = ~np.isnan(tm) & ~np.isnan(rm)
    tm, rm = tm[valid], rm[valid]
    return {
        "n": int(valid.sum()),
        "mean_TM": float(np.mean(tm)),
        "median_TM": float(np.median(tm)),
        "mean_RMSD": float(np.mean(rm)),
        "median_RMSD": float(np.median(rm)),
        "pass_RMSDlt2.0_pct": float((rm < 2.0).mean() * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10, choices=[10, 30])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    RUNS = RUNS_BY_N[args.N]
    out_md = [f"# Best-of-N (N={args.N}) forward folding — TM / RMSD / pass rate\n"]
    out_md.append(f"CAMEO benchmark, 127 targets per checkpoint. Per-target picker selects 1 of {args.N} candidates;\n"
                  "metrics are computed on the selected set (n=127 per picker).\n")
    out_md.append("- **PASS**: RMSD < 2.0 Å\n")

    rows = []
    for ckpt_name, run_dir in RUNS.items():
        csv = sorted(run_dir.glob("bestofN_ff_candidates_*.csv"))[-1]
        df = pd.read_csv(csv)
        out_md.append(f"\n## {ckpt_name}\n")
        out_md.append(f"_source: `{csv.relative_to(BASE_DIR)}`_\n")
        out_md.append(
            "| picker | n | mean TM | median TM | mean RMSD (Å) | median RMSD (Å) | PASS RMSD<2.0 Å (%) |"
        )
        out_md.append("|---|---:|---:|---:|---:|---:|---:|")
        for picker_name, key, mode in PICKERS:
            sel = _select_per_target(df, key, mode)
            m = _metrics(sel)
            out_md.append(
                f"| {picker_name} | {m['n']} | {m['mean_TM']:.3f} | {m['median_TM']:.3f} | "
                f"{m['mean_RMSD']:.2f} | {m['median_RMSD']:.2f} | "
                f"{m['pass_RMSDlt2.0_pct']:.1f} |"
            )
            rows.append({"ckpt": ckpt_name, "picker": picker_name, **m})

    # Compact cross-checkpoint headline (struc_pll vs random)
    df_all = pd.DataFrame(rows)
    out_md.append("\n## Headline: struc_pll picker vs single-shot random (Δ over random)\n")
    out_md.append("| ckpt | random TM | struc_pll TM | Δ TM | random RMSD | struc_pll RMSD | Δ RMSD | "
                  "random PASS | struc_pll PASS | Δ PASS |")
    out_md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ckpt in RUNS:
        r = df_all[(df_all["ckpt"] == ckpt) & (df_all["picker"] == "random")].iloc[0]
        s = df_all[(df_all["ckpt"] == ckpt) & (df_all["picker"] == "struc_pll")].iloc[0]
        out_md.append(
            f"| {ckpt} | {r['mean_TM']:.3f} | {s['mean_TM']:.3f} | {s['mean_TM']-r['mean_TM']:+.3f} | "
            f"{r['mean_RMSD']:.2f} | {s['mean_RMSD']:.2f} | {s['mean_RMSD']-r['mean_RMSD']:+.2f} | "
            f"{r['pass_RMSDlt2.0_pct']:.1f} | "
            f"{s['pass_RMSDlt2.0_pct']:.1f} | "
            f"{s['pass_RMSDlt2.0_pct']-r['pass_RMSDlt2.0_pct']:+.1f} |"
        )

    out_md.append("\n## Best-PLL-picker per checkpoint vs random (Δ over random)\n")
    BEST = {"GenUME-denovo": "joint_pll", "GenUME-base": "struc_pll", "GenUME-TED": "struc_pll"}
    out_md.append("| ckpt | best PLL picker | random TM | best TM | Δ TM | random RMSD | best RMSD | Δ RMSD | "
                  "random PASS | best PASS | Δ PASS |")
    out_md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ckpt, best_picker in BEST.items():
        r = df_all[(df_all["ckpt"] == ckpt) & (df_all["picker"] == "random")].iloc[0]
        s = df_all[(df_all["ckpt"] == ckpt) & (df_all["picker"] == best_picker)].iloc[0]
        out_md.append(
            f"| {ckpt} | {best_picker} | {r['mean_TM']:.3f} | {s['mean_TM']:.3f} | {s['mean_TM']-r['mean_TM']:+.3f} | "
            f"{r['mean_RMSD']:.2f} | {s['mean_RMSD']:.2f} | {s['mean_RMSD']-r['mean_RMSD']:+.2f} | "
            f"{r['pass_RMSDlt2.0_pct']:.1f} | "
            f"{s['pass_RMSDlt2.0_pct']:.1f} | "
            f"{s['pass_RMSDlt2.0_pct']-r['pass_RMSDlt2.0_pct']:+.1f} |"
        )

    out_path = args.output or Path(f"/cv/home/lisanzas/lobster/bestofN_pass_rate_N{args.N}.md")
    out_path.write_text("\n".join(out_md))
    print(f"Wrote {out_path}\n")
    print("\n".join(out_md))


if __name__ == "__main__":
    main()
