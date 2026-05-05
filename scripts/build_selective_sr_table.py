"""Build a hybrid 'selective SR' policy table for LEFLUR-P-VAL (TED-val25-base).

Hybrid policy: use the no-SR run for L in {100, 200, 300} and the SR (T=0.833,
paired) run for L in {400, 500}. Reports per-length and aggregate values for:
  - Designability (RMSD < 2 Å, n=100/length)  + Fisher's exact vs no-SR baseline
  - Diversity (foldseek clusters / designable; weighted by designable count)
  - Novelty vs DeNovo (mean max TM, weighted by query count)
  - Novelty vs PDB   (mean max TM, weighted by query count)
  - Secondary structure: helix_pct, strand_pct, coil_pct, all_helical %, mixed %

Sources are the per-length CSVs already on disk under each eval dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

LENGTHS = [100, 200, 300, 400, 500]
SR_LENGTHS_DEFAULT = [400, 500]

NO_SR_DIR = Path(
    "/cv/scratch/u/lisanzas/evaluations/"
    "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_"
    "seq20_struc60_biasV1.0_steps25"
)
SR_DIR = Path(
    "/cv/scratch/u/lisanzas/evaluations/"
    "gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_"
    "seq20_struc60_biasV1.0_steps25_selfreflect_paired"
)


def _glob_one(eval_dir: Path, pattern: str) -> Path:
    matches = sorted(eval_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} not in {eval_dir}")
    return matches[-1]


def load_designability(eval_dir: Path) -> dict[int, tuple[int, int]]:
    """{L: (n_pass_rmsd, n_total)} from unconditional_pass_rates_length_*."""
    out: dict[int, tuple[int, int]] = {}
    for L in LENGTHS:
        f = _glob_one(eval_dir, f"unconditional_pass_rates_length_{L}_*.csv")
        df = pd.read_csv(f)
        rmsd_row = df[df["metric"] == "_rmsd"].iloc[0]
        out[L] = (int(rmsd_row["pass_count"]), int(rmsd_row["total_count"]))
    return out


def load_diversity(eval_dir: Path) -> dict[int, tuple[int, int, float]]:
    """{L: (num_clusters, total_designable, diversity_pct)}."""
    out: dict[int, tuple[int, int, float]] = {}
    for L in LENGTHS:
        f = _glob_one(eval_dir, f"unconditional_diversity_length_{L}_*.csv")
        row = pd.read_csv(f).iloc[0]
        out[L] = (int(row["num_clusters"]), int(row["total_structures"]),
                  float(row["diversity_percentage"]))
    return out


def load_esmfold_tm(eval_dir: Path) -> dict[int, tuple[float, int]]:
    """{L: (mean_tm_score, n_samples)} from unconditional_metrics_*.csv."""
    f = _glob_one(eval_dir, "unconditional_metrics_*.csv")
    df = pd.read_csv(f)
    df = df[df["mode"] == "unconditional"]
    df = df[df["run_id"].str.startswith("unconditional_")].copy()
    df["L"] = df["run_id"].str.extract(r"length_(\d+)").astype(int)
    out: dict[int, tuple[float, int]] = {}
    for L, sub in df.groupby("L"):
        out[int(L)] = (float(sub["tm_score"].mean()), int(len(sub)))
    return out


def load_novelty(eval_dir: Path, target: str) -> dict[int, tuple[float, int]]:
    """{L: (mean_max_tm, total_queries)}.

    target in {"denovo", "pdb"}.
    """
    f = eval_dir / f"novelty_vs_{target}_summary.csv"
    df = pd.read_csv(f)
    return {int(r["length"]): (float(r["mean_max_tmscore"]),
                                int(r["total_queries"]))
            for _, r in df.iterrows()}


def load_sse(eval_dir: Path) -> dict[int, dict[str, float]]:
    """{L: {helix_pct, strand_pct, coil_pct, all_helical, mixed, all_beta}}."""
    df = pd.read_csv(eval_dir / "uncond_sse_by_length.csv")
    out: dict[int, dict[str, float]] = {}
    for _, r in df.iterrows():
        out[int(r["length"])] = {
            "helix_pct": float(r["helix_pct"]),
            "strand_pct": float(r["strand_pct"]),
            "coil_pct": float(r["coil_pct"]),
            "all_helical": float(r["all_helical"]),
            "mixed": float(r["mixed"]),
            "all_beta": float(r["all_beta"]),
            "n": int(r["n"]),
        }
    return out


def hybrid_metric(no_sr: dict, sr: dict, sr_lengths: list[int]) -> dict:
    return {L: (sr[L] if L in sr_lengths else no_sr[L]) for L in LENGTHS}


def aggregate_designability(d: dict[int, tuple[int, int]]) -> tuple[int, int, float]:
    p = sum(v[0] for v in d.values())
    n = sum(v[1] for v in d.values())
    return p, n, 100.0 * p / n


def aggregate_diversity(d: dict[int, tuple[int, int, float]]) -> float:
    """Weighted by total_structures (designable count) per length."""
    num = sum(div_pct * n_struct for (_, n_struct, div_pct) in d.values())
    den = sum(n_struct for (_, n_struct, _) in d.values())
    return num / den if den else float("nan")


def aggregate_novelty(d: dict[int, tuple[float, int]]) -> float:
    num = sum(tm * n for (tm, n) in d.values())
    den = sum(n for (_, n) in d.values())
    return num / den if den else float("nan")


def aggregate_sse(d: dict[int, dict[str, float]]) -> dict[str, float]:
    n_total = sum(d[L]["n"] for L in LENGTHS)
    out: dict[str, float] = {}
    for k in ("helix_pct", "strand_pct", "coil_pct", "all_helical", "mixed", "all_beta"):
        out[k] = sum(d[L][k] * d[L]["n"] for L in LENGTHS) / n_total
    return out


def fisher(a: int, na: int, b: int, nb: int) -> float:
    return fisher_exact([[a, na - a], [b, nb - b]])[1]


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


def render_table(no_sr: dict, sr: dict, hyb: dict, label: str, kind: str,
                 baseline_design: dict[int, tuple[int, int]] | None = None) -> str:
    """kind in {'design', 'diversity', 'novelty', 'sse_pct'}."""
    rows = ["| L | no-SR | SR | Hybrid (no-SR @ 100/200/300, SR @ 400/500) |"]
    rows.append("|---|---|---|---|")
    for L in LENGTHS:
        if kind == "design":
            a, na = no_sr[L]; b, nb = sr[L]; h, nh = hyb[L]
            rows.append(f"| {L} | {a}/{na} | {b}/{nb} | {h}/{nh} |")
        elif kind == "diversity":
            _, n0, p0 = no_sr[L]; _, n1, p1 = sr[L]; _, nh, ph = hyb[L]
            rows.append(f"| {L} | {p0:.1f}% ({_:d}/{n0}) "
                        f"| {p1:.1f}% ({sr[L][0]:d}/{n1}) "
                        f"| {ph:.1f}% ({hyb[L][0]:d}/{nh}) |")
        elif kind == "novelty":
            v0, _ = no_sr[L]; v1, _ = sr[L]; vh, _ = hyb[L]
            rows.append(f"| {L} | {v0:.3f} | {v1:.3f} | {vh:.3f} |")
        elif kind == "sse_pct":
            rows.append(f"| {L} | {no_sr[L]:.2f} | {sr[L]:.2f} | {hyb[L]:.2f} |")
    return f"**{label}:**\n\n" + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr-lengths", type=int, nargs="+",
                    default=SR_LENGTHS_DEFAULT,
                    help="lengths at which to use SR (rest use no-SR)")
    args = ap.parse_args()
    sr_lengths = sorted(set(args.sr_lengths))

    # ---- load all metrics
    d_no = load_designability(NO_SR_DIR)
    d_sr = load_designability(SR_DIR)
    d_h = hybrid_metric(d_no, d_sr, sr_lengths)

    div_no = load_diversity(NO_SR_DIR)
    div_sr = load_diversity(SR_DIR)
    div_h = hybrid_metric(div_no, div_sr, sr_lengths)

    nov_de_no = load_novelty(NO_SR_DIR, "denovo")
    nov_de_sr = load_novelty(SR_DIR, "denovo")
    nov_de_h = hybrid_metric(nov_de_no, nov_de_sr, sr_lengths)

    nov_pdb_no = load_novelty(NO_SR_DIR, "pdb")
    nov_pdb_sr = load_novelty(SR_DIR, "pdb")
    nov_pdb_h = hybrid_metric(nov_pdb_no, nov_pdb_sr, sr_lengths)

    sse_no = load_sse(NO_SR_DIR)
    sse_sr = load_sse(SR_DIR)
    sse_h = hybrid_metric(sse_no, sse_sr, sr_lengths)

    tm_no = load_esmfold_tm(NO_SR_DIR)
    tm_sr = load_esmfold_tm(SR_DIR)
    tm_h = hybrid_metric(tm_no, tm_sr, sr_lengths)

    # ---- per-length print
    print(f"=== LEFLUR-P-VAL: hybrid SR policy (SR applied at L = {sr_lengths}) ===\n")
    print("Per-length (no-SR / SR / Hybrid):\n")
    print(f"{'L':>4} | {'design':<20} | {'esm-TM':<22} | {'#clust':<14} | "
          f"{'div %':<20} | {'nov-DeNovo':<22} | {'nov-PDB':<22} | {'helix %':<22}")
    print("-" * 170)
    for L in LENGTHS:
        a, na = d_no[L]; b, nb = d_sr[L]; h, nh = d_h[L]
        marker = "(SR)" if L in sr_lengths else "(no-SR)"
        print(f"{L:>4} | {f'{a}/{b}/{h} {marker}':<20}"
              f" | {tm_no[L][0]:.3f} / {tm_sr[L][0]:.3f} / {tm_h[L][0]:.3f} "
              f"| {div_no[L][0]:>2d} / {div_sr[L][0]:>2d} / {div_h[L][0]:>2d}      "
              f"| {div_no[L][2]:>5.1f} / {div_sr[L][2]:>5.1f} / {div_h[L][2]:>5.1f} "
              f"| {nov_de_no[L][0]:>5.3f} / {nov_de_sr[L][0]:>5.3f} / {nov_de_h[L][0]:>5.3f} "
              f"| {nov_pdb_no[L][0]:>5.3f} / {nov_pdb_sr[L][0]:>5.3f} / {nov_pdb_h[L][0]:>5.3f} "
              f"| {sse_no[L]['helix_pct']:>5.2f} / {sse_sr[L]['helix_pct']:>5.2f} / {sse_h[L]['helix_pct']:>5.2f}")

    # ---- aggregates + significance
    a0, n0, pct0 = aggregate_designability(d_no)
    a1, n1, pct1 = aggregate_designability(d_sr)
    ah, nh, pcth = aggregate_designability(d_h)
    p_no_h = fisher(a0, n0, ah, nh)
    p_sr_h = fisher(a1, n1, ah, nh)
    p_no_sr = fisher(a0, n0, a1, n1)

    print(f"\nAggregate (n=500 per condition):")
    print(f"  Designability  : no-SR={a0}/{n0} ({pct0:.1f}%)  "
          f"SR={a1}/{n1} ({pct1:.1f}%)  HYBRID={ah}/{nh} ({pcth:.1f}%)")
    print(f"    Fisher hybrid vs no-SR: p={p_no_h:.4g} {stars(p_no_h)}")
    print(f"    Fisher hybrid vs SR   : p={p_sr_h:.4g} {stars(p_sr_h)}")
    print(f"    Fisher SR     vs no-SR: p={p_no_sr:.4g} {stars(p_no_sr)}")

    # ESMFold TM mean (weighted by sample count = 100/length, equiv. simple mean)
    def _avg_tm(d: dict[int, tuple[float, int]]) -> float:
        num = sum(tm * n for (tm, n) in d.values())
        den = sum(n for (_, n) in d.values())
        return num / den if den else float("nan")

    print(f"  ESMFold TM     : no-SR={_avg_tm(tm_no):.3f}  "
          f"SR={_avg_tm(tm_sr):.3f}  HYBRID={_avg_tm(tm_h):.3f}  "
          f"(higher = better fold-fidelity)")

    n_clu_no = sum(div_no[L][0] for L in LENGTHS)
    n_clu_sr = sum(div_sr[L][0] for L in LENGTHS)
    n_clu_h  = sum(div_h[L][0] for L in LENGTHS)
    print(f"  # clusters     : no-SR={n_clu_no}  SR={n_clu_sr}  HYBRID={n_clu_h}  "
          f"(sum across lengths; foldseek TM<0.5)")

    print(f"  Diversity (wgt): no-SR={aggregate_diversity(div_no):.2f}%  "
          f"SR={aggregate_diversity(div_sr):.2f}%  "
          f"HYBRID={aggregate_diversity(div_h):.2f}%")

    print(f"  Novelty DeNovo : no-SR={aggregate_novelty(nov_de_no):.3f}  "
          f"SR={aggregate_novelty(nov_de_sr):.3f}  "
          f"HYBRID={aggregate_novelty(nov_de_h):.3f}  (lower = more novel)")
    print(f"  Novelty PDB    : no-SR={aggregate_novelty(nov_pdb_no):.3f}  "
          f"SR={aggregate_novelty(nov_pdb_sr):.3f}  "
          f"HYBRID={aggregate_novelty(nov_pdb_h):.3f}")

    sse_no_agg = aggregate_sse(sse_no)
    sse_sr_agg = aggregate_sse(sse_sr)
    sse_h_agg = aggregate_sse(sse_h)
    print(f"  SSE helix %    : no-SR={sse_no_agg['helix_pct']:.2f}  "
          f"SR={sse_sr_agg['helix_pct']:.2f}  HYBRID={sse_h_agg['helix_pct']:.2f}")
    print(f"  SSE strand %   : no-SR={sse_no_agg['strand_pct']:.2f}  "
          f"SR={sse_sr_agg['strand_pct']:.2f}  HYBRID={sse_h_agg['strand_pct']:.2f}")
    print(f"  SSE coil %     : no-SR={sse_no_agg['coil_pct']:.2f}  "
          f"SR={sse_sr_agg['coil_pct']:.2f}  HYBRID={sse_h_agg['coil_pct']:.2f}")
    print(f"  SSE all-helical: no-SR={sse_no_agg['all_helical']:.1f}%  "
          f"SR={sse_sr_agg['all_helical']:.1f}%  HYBRID={sse_h_agg['all_helical']:.1f}%")
    print(f"  SSE mixed      : no-SR={sse_no_agg['mixed']:.1f}%  "
          f"SR={sse_sr_agg['mixed']:.1f}%  HYBRID={sse_h_agg['mixed']:.1f}%")
    print(f"  SSE all-beta   : no-SR={sse_no_agg['all_beta']:.1f}%  "
          f"SR={sse_sr_agg['all_beta']:.1f}%  HYBRID={sse_h_agg['all_beta']:.1f}%")


if __name__ == "__main__":
    main()
