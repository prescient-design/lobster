"""Per-length designability bar plot for TED-val25-base and TED-stoch.

Designability = ESMFold RMSD < 2 Å against the designed backbone (matching the
benchmark definition; n=100/length).

Counts are pulled directly from each run's `unconditional_metrics_*.csv`
(`mode == 'unconditional'` rows only), so they are deterministic from the SR-
paired CSVs that were already on disk.

Significance: per-length Fisher's exact test on a 2x2 contingency
([[design_no_SR, fail_no_SR], [design_SR, fail_SR]]).

Outputs:
  /cv/home/lisanzas/lobster/per_length_designability_bars.png
  Prints the underlying counts and p-values to stdout for the plan table.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

LENGTHS = [100, 200, 300, 400, 500]
RMSD_PASS = 2.0

SR_LABEL = "SR (Tm_score_min=0.833)"

DIRS: dict[tuple[str, str], str] = {
    ("LEFLUR-P-VAL", "no-SR"):
        "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25",
    ("LEFLUR-P-VAL", SR_LABEL):
        "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_selfreflect_paired",
    ("LEFLUR-P", "no-SR"):
        "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10",
    ("LEFLUR-P", SR_LABEL):
        "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10_selfreflect_paired",
}


def load_counts(eval_dir: str) -> dict[int, tuple[int, int]]:
    """Returns {length: (n_pass_rmsd, n_total)}."""
    files = sorted(glob.glob(f"{eval_dir}/unconditional_metrics_*.csv"))
    if not files:
        raise FileNotFoundError(f"No unconditional_metrics_*.csv in {eval_dir}")
    df = pd.read_csv(files[-1])
    df = df[df["mode"] == "unconditional"]
    df = df[df["run_id"].str.startswith("unconditional_")].copy()
    df["L"] = df["run_id"].str.extract(r"length_(\d+)").astype(int)
    out: dict[int, tuple[int, int]] = {}
    for L, sub in df.groupby("L"):
        out[int(L)] = (int((sub["rmsd"] < RMSD_PASS).sum()), int(len(sub)))
    return out


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot(out_path: Path) -> None:
    counts = {k: load_counts(v) for k, v in DIRS.items()}
    checkpoints = ["LEFLUR-P-VAL", "LEFLUR-P"]

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(7.0 * len(checkpoints), 5.4),
                             sharey=True)
    width = 0.4
    x = np.arange(len(LENGTHS))

    for ax, ckpt in zip(axes, checkpoints):
        no_sr = [counts[(ckpt, "no-SR")][L][0] for L in LENGTHS]
        sr = [counts[(ckpt, SR_LABEL)][L][0] for L in LENGTHS]
        n_no = [counts[(ckpt, "no-SR")][L][1] for L in LENGTHS]
        n_sr = [counts[(ckpt, SR_LABEL)][L][1] for L in LENGTHS]

        b1 = ax.bar(x - width / 2, no_sr, width, color="tab:gray",
                    edgecolor="black", linewidth=0.5, label="no-SR")
        b2 = ax.bar(x + width / 2, sr, width, color="tab:blue",
                    edgecolor="black", linewidth=0.5, label=SR_LABEL)
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                ax.annotate(f"{int(h)}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        # significance bracket
        for i, (a, b, na, nb) in enumerate(zip(no_sr, sr, n_no, n_sr)):
            d = b - a
            colour = "tab:green" if d > 0 else ("tab:red" if d < 0 else "black")
            _, p = fisher_exact([[a, na - a], [b, nb - b]])
            top = max(a, b) + 4
            ax.plot([x[i] - width / 2, x[i] - width / 2,
                     x[i] + width / 2, x[i] + width / 2],
                    [top, top + 1.5, top + 1.5, top],
                    lw=1, color="black")
            ax.annotate(f"Δ{d:+d}\n{stars(p)}\n(p={p:.2g})",
                        xy=(x[i], top + 2),
                        ha="center", va="bottom", fontsize=7,
                        color=colour, fontweight="bold")

        # checkpoint pooled aggregate Fisher
        a_pool = sum(no_sr); na_pool = sum(n_no)
        b_pool = sum(sr);    nb_pool = sum(n_sr)
        _, p_pool = fisher_exact([[a_pool, na_pool - a_pool],
                                   [b_pool, nb_pool - b_pool]])

        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in LENGTHS])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(f"Designable / 100 (ESMFold RMSD < {RMSD_PASS:.0f} Å)")
        ax.set_ylim(0, 120)
        ax.set_title(
            f"{ckpt}\nall lengths: {a_pool}/{na_pool} → {b_pool}/{nb_pool}  "
            f"({stars(p_pool)} p={p_pool:.2g})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="lower left", fontsize=9)

    sig_text = ("Fisher's exact test, two-sided.  *** p<0.001, ** p<0.01, * p<0.05, "
                "ns p≥0.05.  Δ = SR − no-SR (designable count).")
    fig.suptitle(
        f"Per-length designability — accepted designs (n=100/length)\n{sig_text}",
        fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}\n")

    # text summary for the plan
    print(f"{'ckpt':<16} {'L':>4} {'no-SR':>7} {'SR':>7} {'Δpp':>5}  {'p':>10}  sig")
    for ckpt in checkpoints:
        for L in LENGTHS:
            a, na = counts[(ckpt, "no-SR")][L]
            b, nb = counts[(ckpt, SR_LABEL)][L]
            _, p = fisher_exact([[a, na - a], [b, nb - b]])
            print(f"{ckpt:<16} {L:>4} {a:>3}/{na:<3} {b:>3}/{nb:<3} "
                  f"{(b - a):>+4}  {p:>10.3g}  {stars(p)}")
        a_pool = sum(counts[(ckpt, "no-SR")][L][0] for L in LENGTHS)
        na_pool = sum(counts[(ckpt, "no-SR")][L][1] for L in LENGTHS)
        b_pool = sum(counts[(ckpt, SR_LABEL)][L][0] for L in LENGTHS)
        nb_pool = sum(counts[(ckpt, SR_LABEL)][L][1] for L in LENGTHS)
        _, p_pool = fisher_exact([[a_pool, na_pool - a_pool],
                                   [b_pool, nb_pool - b_pool]])
        print(f"{ckpt:<16} {'pool':>4} {a_pool:>3}/{na_pool:<3} {b_pool:>3}/{nb_pool:<3} "
              f"{(b_pool - a_pool):>+4}  {p_pool:>10.3g}  {stars(p_pool)}")
        print()


def main() -> None:
    out_dir = Path("/cv/home/lisanzas/lobster")
    plot(out_dir / "per_length_designability_bars.png")


if __name__ == "__main__":
    main()
