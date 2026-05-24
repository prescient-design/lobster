"""Concatenate per-target shards from a SLURM-array FF best-of-N run.

Each array task writes into:
  <run_dir>/shards/<target>/bestofN_ff_candidates_<ts>.csv
  <run_dir>/shards/<target>/bestofN_ff_summary_<ts>.csv

This script merges them into single files at <run_dir>/.

Usage:
    uv run python scripts/concat_bestofN_ff_array.py \\
        --run-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N100
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def _latest_csv(shard_dir: Path, prefix: str) -> Path | None:
    matches = sorted(shard_dir.glob(f"{prefix}_*.csv"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--shards-subdir", default="shards")
    args = ap.parse_args()

    shards_root = args.run_dir / args.shards_subdir
    if not shards_root.is_dir():
        raise FileNotFoundError(f"No shards dir: {shards_root}")

    cand_parts: list[Path] = []
    summ_parts: list[Path] = []
    for target_dir in sorted(p for p in shards_root.iterdir() if p.is_dir()):
        c = _latest_csv(target_dir, "bestofN_ff_candidates")
        s = _latest_csv(target_dir, "bestofN_ff_summary")
        if c is None or s is None:
            print(f"SKIP incomplete shard: {target_dir.name}")
            continue
        cand_parts.append(c)
        summ_parts.append(s)

    if not cand_parts:
        raise SystemExit(f"No candidate shards under {shards_root}")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_cand = args.run_dir / f"bestofN_ff_candidates_{ts}.csv"
    out_summ = args.run_dir / f"bestofN_ff_summary_{ts}.csv"

    with out_cand.open("w", newline="") as out_fh:
        writer = None
        for part in cand_parts:
            with part.open(newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                if writer is None:
                    writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    with out_summ.open("w", newline="") as out_fh:
        writer = None
        for part in summ_parts:
            with part.open(newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                if writer is None:
                    writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    print(f"Merged {len(cand_parts)} targets")
    print(f"  {out_cand}")
    print(f"  {out_summ}")


if __name__ == "__main__":
    main()
