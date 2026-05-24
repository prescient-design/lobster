"""Analyze the forward-folding best-of-N PLL ablation.

Reads the candidate-level CSV produced by `scripts/forward_fold_bestofN_pll.py` and
emits a markdown report comparing five rankers on the held-out CAMEO target set:

  - random_pick       (= candidate 0; existing single-shot baseline)
  - seq_pll_pick      (argmin seq_score_unif)
  - struc_pll_pick    (argmin struc_score_unif)
  - joint_pll_pick    (argmin joint_score_unif)
  - oracle_pick       (argmax tm_score; upper bound)

Reports per-target TM at each pick and aggregate stats:
  - mean / median TM
  - pass-rate at TM > {0.5, 0.7, 0.8}
  - per-length-bucket breakdown (<=150, 150-300, 300-450, >450)
  - PLL-vs-oracle gap stats

Usage:
    uv run python scripts/analyze_bestofN_ff.py \\
        --candidates /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_cameo_bestofN_pll/bestofN_ff_candidates_<ts>.csv \\
        --output     /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_cameo_bestofN_pll/bestofN_ff_report.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


_PICKERS = ["random_pick", "seq_pll_pick", "struc_pll_pick", "joint_pll_pick", "oracle_pick"]
_PASS_THRESHOLDS = (0.5, 0.7, 0.8)
_LEN_BUCKETS = (("≤150", 0, 150), ("150-300", 150, 300), ("300-450", 300, 450), (">450", 450, 10**6))


def _load_candidates(path: Path):
    by_target: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            by_target[r["target"]].append(r)
    out = []
    for target, rows in by_target.items():
        rows.sort(key=lambda x: int(x["candidate_idx"]))
        try:
            L = int(rows[0]["length"])
        except (KeyError, ValueError):
            continue
        ok = []
        for r in rows:
            try:
                ok.append(
                    {
                        "candidate_idx": int(r["candidate_idx"]),
                        "tm_score": float(r["tm_score"]),
                        "rmsd": float(r.get("rmsd") or "nan"),
                        "seq_score_unif": float(r.get("seq_score_unif") or "nan"),
                        "struc_score_unif": float(r.get("struc_score_unif") or "nan"),
                        "joint_score_unif": float(r.get("joint_score_unif") or "nan"),
                    }
                )
            except ValueError:
                continue
        if not ok:
            continue
        out.append({"target": target, "length": L, "rows": ok})
    return out


def _picks_for_target(target_rows: list[dict]) -> dict[str, dict]:
    """Return {picker: {idx, tm}} for one target."""
    out = {}
    out["random_pick"] = {"idx": 0, "tm": target_rows[0]["tm_score"]}

    def argmin(key):
        vals = [(i, r[key]) for i, r in enumerate(target_rows) if not _isnan(r[key])]
        if not vals:
            return 0
        return min(vals, key=lambda x: x[1])[0]

    def argmax(key):
        vals = [(i, r[key]) for i, r in enumerate(target_rows) if not _isnan(r[key])]
        if not vals:
            return 0
        return max(vals, key=lambda x: x[1])[0]

    out["seq_pll_pick"] = {"idx": argmin("seq_score_unif")}
    out["struc_pll_pick"] = {"idx": argmin("struc_score_unif")}
    out["joint_pll_pick"] = {"idx": argmin("joint_score_unif")}
    out["oracle_pick"] = {"idx": argmax("tm_score")}
    for k, v in out.items():
        v["tm"] = target_rows[v["idx"]]["tm_score"]
    return out


def _isnan(x):
    return x != x


def _aggregate(targets: list[dict], length_filter=None):
    filt = targets if length_filter is None else [t for t in targets if length_filter(t["length"])]
    n = len(filt)
    if n == 0:
        return None
    per_picker_tms = {p: [] for p in _PICKERS}
    candidate_tms_all = []
    oracle_gap_by_picker = defaultdict(list)
    for t in filt:
        picks = _picks_for_target(t["rows"])
        oracle_tm = picks["oracle_pick"]["tm"]
        for p in _PICKERS:
            per_picker_tms[p].append(picks[p]["tm"])
            if p != "oracle_pick":
                oracle_gap_by_picker[p].append(oracle_tm - picks[p]["tm"])
        candidate_tms_all.extend([r["tm_score"] for r in t["rows"]])

    summary = {
        "n_targets": n,
        "n_candidates_total": len(candidate_tms_all),
        "candidate_tm_mean": statistics.mean(candidate_tms_all),
        "candidate_tm_median": statistics.median(candidate_tms_all),
        "picker_stats": {},
        "oracle_gap": {},
    }
    for p in _PICKERS:
        tms = per_picker_tms[p]
        passes = {f"pass@{thr}": sum(1 for x in tms if x > thr) / n for thr in _PASS_THRESHOLDS}
        summary["picker_stats"][p] = {
            "mean_tm": statistics.mean(tms),
            "median_tm": statistics.median(tms),
            "std_tm": statistics.pstdev(tms),
            **passes,
        }
    for p, gaps in oracle_gap_by_picker.items():
        summary["oracle_gap"][p] = {
            "mean": statistics.mean(gaps),
            "median": statistics.median(gaps),
            "max": max(gaps),
        }
    return summary


def _format_picker_table(summary: dict, title: str) -> str:
    lines = [f"### {title} (n={summary['n_targets']} targets, {summary['n_candidates_total']} candidates)\n"]
    lines.append(
        "| picker | mean TM | median TM | std TM | "
        + " | ".join(f"pass>{thr}" for thr in _PASS_THRESHOLDS)
        + " |"
    )
    lines.append("|---|---:|---:|---:|" + "---:|" * len(_PASS_THRESHOLDS))
    for p in _PICKERS:
        s = summary["picker_stats"][p]
        cells = [
            p,
            f"{s['mean_tm']:.3f}",
            f"{s['median_tm']:.3f}",
            f"{s['std_tm']:.3f}",
            *[f"{s[f'pass@{thr}']:.1%}" for thr in _PASS_THRESHOLDS],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(
        f"\nCandidate-level (no selection) mean TM = {summary['candidate_tm_mean']:.3f}, "
        f"median = {summary['candidate_tm_median']:.3f}.\n"
    )
    return "\n".join(lines)


def _format_gap_table(summary: dict) -> str:
    lines = ["### Gap to oracle (lower is better)\n"]
    lines.append("| picker | mean gap | median gap | max gap |")
    lines.append("|---|---:|---:|---:|")
    for p in _PICKERS:
        if p == "oracle_pick":
            continue
        g = summary["oracle_gap"][p]
        lines.append(f"| {p} | {g['mean']:.3f} | {g['median']:.3f} | {g['max']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    targets = _load_candidates(args.candidates)
    if not targets:
        raise SystemExit(f"No usable rows in {args.candidates}")

    overall = _aggregate(targets)
    out_lines = [
        "# Forward-folding best-of-N PLL selection — CAMEO\n",
        f"Source: `{args.candidates}`\n",
        _format_picker_table(overall, "Overall"),
        _format_gap_table(overall),
        "\n## Per-length-bucket breakdown\n",
    ]
    for label, lo, hi in _LEN_BUCKETS:
        bucket = _aggregate(targets, length_filter=lambda L, lo=lo, hi=hi: lo < L <= hi)
        if bucket is None:
            out_lines.append(f"### Length {label}: no targets.\n")
            continue
        out_lines.append(_format_picker_table(bucket, f"Length {label}"))
    report = "\n".join(out_lines)

    if args.output is None:
        out = args.candidates.with_name("bestofN_ff_report.md")
    else:
        out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Wrote {out}")
    print()
    print(_format_picker_table(overall, "Overall"))


if __name__ == "__main__":
    main()
