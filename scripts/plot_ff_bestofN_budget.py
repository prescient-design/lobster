"""Best-of-N budget curve for forward-folding CAMEO best-of-30 PLL run.

For each sample budget N (candidates 0 .. N-1 per target), compute the TM of
the design selected by each ranker, then average over targets.

Rankers:
  - random_pick   (candidate 0 = single-shot baseline)
  - seq_pll_pick  (argmin seq_score_unif in pool)
  - struc_pll_pick
  - joint_pll_pick (argmin joint_score_unif; additive joint)
  - oracle_pick   (argmax tm_score in pool)

Outputs CSV + PNG to --output-dir (default: candidates CSV parent).

Usage:
    python scripts/plot_ff_bestofN_budget.py \\
        --candidates /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025401.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_ff_bestofN_budget")

_PICKERS = [
    ("random_pick", "random"),
    ("seq_pll_pick", "seq"),
    ("struc_pll_pick", "struc"),
    ("joint_pll_pick", "joint"),
    ("joint_true_pll_pick", "joint_true"),
    ("oracle_pick", "oracle"),
]

_LABELS = {
    "random_pick": "random (cand 0)",
    "seq_pll_pick": "seq_pll",
    "struc_pll_pick": "struc_pll",
    "joint_pll_pick": "joint_pll (sum)",
    "joint_true_pll_pick": "joint_true_pll",
    "oracle_pick": "oracle",
}

_COLORS = {
    "random_pick": "#888888",
    "seq_pll_pick": "#4C72B0",
    "struc_pll_pick": "#C44E52",
    "joint_pll_pick": "#55A868",
    "joint_true_pll_pick": "#CCB974",
    "oracle_pick": "#8172B2",
}


def _load_by_target(path: Path) -> dict[str, list[dict]]:
    by_target: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            by_target[r["target"]].append(r)
    for target, rows in by_target.items():
        rows.sort(key=lambda x: int(x["candidate_idx"]))
    return by_target


def _pick_tm(pool: list[dict], method: str) -> float:
    if method == "random":
        return float(pool[0]["tm_score"])
    if method == "oracle":
        return max(float(r["tm_score"]) for r in pool)
    key = {
        "seq": "seq_score_unif",
        "struc": "struc_score_unif",
        "joint": "joint_score_unif",
        "joint_true": "joint_true_score_unif",
    }[method]
    pool_valid = [
        r for r in pool
        if r.get(key) not in (None, "", "nan")
        and float(r[key]) == float(r[key])  # not NaN
    ]
    if not pool_valid:
        return float(pool[0]["tm_score"])
    best = min(pool_valid, key=lambda r: float(r[key]))
    return float(best["tm_score"])


def compute_budget_curve(
    by_target: dict[str, list[dict]],
    budgets: list[int],
) -> dict[str, list[float]]:
    n_max = min(len(rows) for rows in by_target.values())
    curves = {name: [] for name, _ in _PICKERS}
    for n in budgets:
        if n < 1 or n > n_max:
            raise ValueError(f"Budget N={n} out of range [1, {n_max}]")
        for name, method in _PICKERS:
            tms = [_pick_tm(rows[:n], method) for rows in by_target.values()]
            curves[name].append(float(np.mean(tms)))
    return curves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/"
            "bestofN_ff_candidates_20260501T025401.csv"
        ),
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--budgets",
        type=str,
        default="1-30",
        help="Comma-separated N values or range like 1-30 or 1,2,5,10,20,30",
    )
    ap.add_argument(
        "--ref-lines",
        type=str,
        default="ESMFold=0.849,DPLM-2=0.704",
        help=(
            "Comma-separated label=value pairs to draw as horizontal reference lines. "
            "Defaults to CAMEO FF mean-TM baselines from Table 4 of the conference plan."
        ),
    )
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = args.candidates.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_target = _load_by_target(args.candidates)
    n_targets = len(by_target)
    n_max = min(len(rows) for rows in by_target.values())
    logger.info("Loaded %d targets, max candidates = %d", n_targets, n_max)

    if "-" in args.budgets and "," not in args.budgets:
        lo, hi = args.budgets.split("-", 1)
        budgets = list(range(int(lo), int(hi) + 1))
    else:
        budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    curves = compute_budget_curve(by_target, budgets)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    csv_path = args.output_dir / f"ff_bestofN_budget_curve_{ts}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_designs"] + [name for name, _ in _PICKERS])
        for i, n in enumerate(budgets):
            w.writerow([n] + [curves[name][i] for name, _ in _PICKERS])
    logger.info("Wrote %s", csv_path)

    ref_lines: list[tuple[str, float]] = []
    for chunk in args.ref_lines.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, _, val = chunk.partition("=")
        try:
            ref_lines.append((label.strip(), float(val.strip())))
        except ValueError:
            logger.warning("Skipping malformed --ref-lines entry: %r", chunk)

    _REF_COLORS = ["#000000", "#444444", "#888888"]
    _REF_STYLES = ["--", ":", "-."]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, _ in _PICKERS:
        ax.plot(
            budgets,
            curves[name],
            marker="o",
            markersize=3 if len(budgets) > 15 else 5,
            linewidth=2,
            label=_LABELS[name],
            color=_COLORS[name],
        )

    for i, (label, val) in enumerate(ref_lines):
        color = _REF_COLORS[i % len(_REF_COLORS)]
        style = _REF_STYLES[i % len(_REF_STYLES)]
        ax.axhline(val, color=color, linestyle=style, linewidth=1.4,
                   label=f"{label} ({val:.3f})", alpha=0.85)

    ax.set_xlabel("Designs sampled per target (N)")
    ax.set_ylabel("Mean TM-score (CAMEO, n=127)")
    ax.set_title("Forward folding best-of-N — PLL rankers (GenUME-TED)")
    ax.set_xlim(min(budgets) - 0.5, max(budgets) + 0.5)
    series_min = min(min(curves[name]) for name, _ in _PICKERS)
    series_max = max(max(curves[name]) for name, _ in _PICKERS)
    if ref_lines:
        series_min = min(series_min, min(v for _, v in ref_lines))
        series_max = max(series_max, max(v for _, v in ref_lines))
    ax.set_ylim(series_min - 0.01, series_max + 0.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    png_path = args.output_dir / f"ff_bestofN_budget_curve_{ts}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", png_path)

    print(f"\nCAMEO forward folding — budget curve (n={n_targets} targets)")
    print(f"Source: {args.candidates}\n")
    header = f"{'N':>4}" + "".join(f"{_LABELS[n]:>18}" for n, _ in _PICKERS)
    print(header)
    for i, n in enumerate(budgets):
        row = f"{n:4d}" + "".join(f"{curves[name][i]:18.3f}" for name, _ in _PICKERS)
        print(row)


if __name__ == "__main__":
    main()
