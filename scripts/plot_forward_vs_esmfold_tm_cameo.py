"""Per-target correlation on CAMEO forward folding:
  x: lobster forward-fold TM   (vs CAMEO ground truth)
  y: ESMFold TM                (vs CAMEO ground truth)

Both quantities are computed on the SAME 127 CAMEO targets so each point is
truly paired. The lobster value is taken from the cached forward-folding
metrics CSV when ``input_file`` matches the target ID; otherwise it is
recomputed via tm_align between the saved generated and original PDBs (the
saved metrics CSVs use ``batch_NNN`` placeholders, so we recompute by default
to keep target identity unambiguous).

Usage:
  uv run python scripts/plot_forward_vs_esmfold_tm_cameo.py \\
      --ted-dir /cv/scratch/u/lisanzas/evaluations/<TED_cameo_forward_folding_dir> \\
      --base-dir /cv/scratch/u/lisanzas/evaluations/<base_cameo_forward_folding_dir> \\
      --esmfold-dir /cv/home/lisanzas/lobster/examples/esmfold_baseline_cameo \\
      --out /cv/home/lisanzas/lobster/cameo_forward_vs_esmfold_tm.png
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tmtools import tm_align


# ---- minimal PDB CA-coordinate parser (avoid heavy biotite/biopython deps) ----

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",  # treat selenomethionine as Met
}


def read_ca(pdb_path: Path) -> tuple[np.ndarray, str]:
    coords = []
    seq_chars = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resname = line[17:20].strip()
            aa = THREE_TO_ONE.get(resname, "X")
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
            seq_chars.append(aa)
    return np.asarray(coords, dtype=np.float64), "".join(seq_chars)


def compute_tm(generated_pdb: Path, original_pdb: Path) -> tuple[float, float, int]:
    """Return (TM-score normalised to chain1=generated, RMSD, length)."""
    gen, gen_seq = read_ca(generated_pdb)
    orig, orig_seq = read_ca(original_pdb)
    if len(gen) == 0 or len(orig) == 0:
        return float("nan"), float("nan"), 0
    out = tm_align(gen, orig, gen_seq, orig_seq)
    return out.tm_norm_chain2, out.rmsd, min(len(gen), len(orig))


def gather_lobster(eval_dir: Path) -> pd.DataFrame:
    """Compute lobster forward-fold TM per CAMEO target from saved PDBs."""
    rows = []
    for orig in sorted(eval_dir.glob("forward_folding_*_original.pdb")):
        target = orig.name[len("forward_folding_"):-len("_original.pdb")]
        gen = eval_dir / f"forward_folding_{target}_generated.pdb"
        if not gen.exists():
            continue
        tm, rmsd, L = compute_tm(gen, orig)
        rows.append(dict(target=target, lobster_tm=tm, lobster_rmsd=rmsd, length=L))
    return pd.DataFrame(rows)


def load_esmfold(esmfold_dir: Path) -> pd.DataFrame:
    csvs = sorted(glob.glob(str(esmfold_dir / "esmfold_baseline_metrics_*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No esmfold_baseline_metrics_*.csv in {esmfold_dir}")
    df = pd.read_csv(csvs[-1])
    df = df.rename(columns={"input_file": "target",
                            "tm_score": "esmfold_tm",
                            "rmsd": "esmfold_rmsd",
                            "plddt": "esmfold_plddt"})
    return df[["target", "esmfold_tm", "esmfold_rmsd", "esmfold_plddt", "sequence_length"]]


def report_correlation(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: lobster forward TM vs ESMFold TM (n={len(df)}) ===")
    pr, ppv = pearsonr(df.lobster_tm, df.esmfold_tm)
    sr, spv = spearmanr(df.lobster_tm, df.esmfold_tm)
    print(f"  Pearson  r = {pr:.3f} (p={ppv:.2e})")
    print(f"  Spearman r = {sr:.3f} (p={spv:.2e})")
    print(f"  lobster_tm: mean={df.lobster_tm.mean():.3f}  median={df.lobster_tm.median():.3f}")
    print(f"  esmfold_tm: mean={df.esmfold_tm.mean():.3f}  median={df.esmfold_tm.median():.3f}")


def plot(runs: list[tuple[str, pd.DataFrame]], out_path: Path,
         qc_threshold: float = 0.8334) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 6),
                             squeeze=False)

    for ax, (label, df) in zip(axes[0], runs):
        sc = ax.scatter(
            df.lobster_tm, df.esmfold_tm,
            c=df.length if "length" in df.columns else "tab:blue",
            cmap="viridis", s=24, alpha=0.7, edgecolor="black", linewidths=0.3,
        )
        if "length" in df.columns:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("sequence length (CA atoms)")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="y = x")

        pr, ppv = pearsonr(df.lobster_tm, df.esmfold_tm)
        sr, spv = spearmanr(df.lobster_tm, df.esmfold_tm)
        text = (
            f"n = {len(df)}\n"
            f"Pearson  r = {pr:.3f}\n"
            f"Spearman = {sr:.3f}"
        )
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="black", alpha=0.85))

        ax.set_xlabel("Leflur forward-fold TM\n(generated vs CAMEO ground truth)")
        ax.set_ylabel("ESMFold TM\n(predicted vs CAMEO ground truth)")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "CAMEO per-target TM agreement: Leflur forward-fold vs ESMFold",
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
    ap.add_argument("--esmfold-dir", type=Path,
                    default=Path("/cv/home/lisanzas/lobster/examples/esmfold_baseline_cameo"))
    ap.add_argument("--out", type=Path,
                    default=Path("/cv/home/lisanzas/lobster/cameo_forward_vs_esmfold_tm.png"))
    args = ap.parse_args()

    esmfold = load_esmfold(args.esmfold_dir)

    print("Loading TED-val25-base lobster forward-fold (this recomputes TM via tm_align per target)...")
    ted = gather_lobster(args.ted_dir).merge(esmfold, on="target", how="inner")
    if "sequence_length" in ted.columns:
        ted["length"] = ted["sequence_length"]
    report_correlation(ted, "TED-val25-base")

    runs = [("TED-val25-base CAMEO forward fold", ted)]
    if args.base_dir is not None:
        print("Loading val25-base lobster forward-fold...")
        base = gather_lobster(args.base_dir).merge(esmfold, on="target", how="inner")
        if "sequence_length" in base.columns:
            base["length"] = base["sequence_length"]
        report_correlation(base, "val25-base")
        runs.append(("val25-base CAMEO forward fold", base))

    plot(runs, args.out)


if __name__ == "__main__":
    main()
