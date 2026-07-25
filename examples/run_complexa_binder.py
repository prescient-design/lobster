#!/usr/bin/env python
"""Portable Complexa binder-design benchmark runner.

Loops the 38 Complexa targets and runs LeFlur de-novo binder design against
each one, using the ``experiment/generate_binder_3di`` (default) or
``experiment/generate_binder_disto`` Hydra config. Per target it overrides
``input_structures`` / ``target_chain`` / ``epitope_indices`` /
``binder_length`` from the benchmark's ``complexa_gen_targets.csv`` manifest.

This is the non-cluster replacement for the slurm-array driver
``slurm/scripts/gen_complexa_minibinders.sh``: it runs the targets serially
(or a chosen subset) on the local GPU.

The benchmark itself is fetched from HuggingFace on first use::

    lobster_leflur_benchmarks fetch complexa-binder

The manifest paths (``pdb_path``) are relative to the benchmark directory, so
the whole thing is portable — no ``/cv/...`` paths required.

Scoring
-------
Design generation is only the first half of the benchmark. PASS is defined by
Protenix co-folding (``pTM > 0.80 AND ipTM > 0.70``), which runs in a separate,
heavy environment — see ``scripts/_score_sabdab_minibinders.py`` and
``docs/leflur/binder_design.md``. This runner does NOT score; it emits the
generated sequences + structures that the scoring step consumes.

Examples
--------
Smoke test — one target, one design::

    uv run python examples/run_complexa_binder.py --limit 1 --n-designs 1

Full 38-target 3Di run, 100 designs/target::

    uv run python examples/run_complexa_binder.py --n-designs 100 \
        --out-dir ~/complexa_out/leflur_3di

Non-3Di (disto) arm on two named targets::

    uv run python examples/run_complexa_binder.py \
        --config experiment/generate_binder_disto \
        --targets 01_PD1 02_PDL1
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _read_manifest(bench_dir: Path) -> list[dict[str, str]]:
    """Read ``complexa_gen_targets.csv`` from the benchmark directory."""
    manifest = bench_dir / "complexa_gen_targets.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"manifest not found: {manifest}\n"
            "Fetch the benchmark first: `lobster_leflur_benchmarks fetch complexa-binder`, "
            "or pass --benchmark-dir pointing at a local copy."
        )
    with open(manifest) as f:
        return list(csv.DictReader(f))


def _resolve_benchmark_dir(explicit: str | None) -> Path:
    """Resolve the benchmark dir: explicit path, else fetch from HuggingFace."""
    if explicit:
        d = Path(explicit).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"--benchmark-dir does not exist: {d}")
        return d
    # Lazy import so `--benchmark-dir` runs never pay the lobster import cost.
    from lobster.model.leflur.benchmarks import fetch_benchmark

    return Path(fetch_benchmark("complexa-binder"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        default="experiment/generate_binder_3di",
        help="Hydra generate config (default: the 3Di binder config).",
    )
    p.add_argument(
        "--benchmark-dir",
        default=None,
        help="Local Complexa benchmark dir (with complexa_gen_targets.csv). "
        "If omitted, fetched from HuggingFace via lobster_leflur_benchmarks.",
    )
    p.add_argument(
        "--out-dir",
        default="./complexa_binder_out",
        help="Root output dir; each target writes to <out-dir>/<target_id>.",
    )
    p.add_argument("--n-designs", type=int, default=100, help="Designs per target (default 100).")
    p.add_argument("--targets", nargs="*", default=None, help="Subset of target_ids to run (default: all 38).")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N manifest rows (smoke tests).")
    p.add_argument("--seed", type=int, default=101, help="Generation seed.")
    p.add_argument("--paths", default="public", help="Hydra paths overlay (default: public).")
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra Hydra overrides passed through verbatim (must come last).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = p.parse_args(argv)

    bench_dir = _resolve_benchmark_dir(args.benchmark_dir)
    rows = _read_manifest(bench_dir)
    if args.targets:
        wanted = set(args.targets)
        rows = [r for r in rows if r["target_id"] in wanted]
        missing = wanted - {r["target_id"] for r in rows}
        if missing:
            print(f"WARNING: requested target_ids not in manifest: {sorted(missing)}", file=sys.stderr)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No targets to run.", file=sys.stderr)
        return 1

    out_root = Path(args.out_dir).expanduser().resolve()
    print(f"benchmark dir : {bench_dir}")
    print(f"config        : {args.config}")
    print(f"targets       : {len(rows)}  (n_designs={args.n_designs})")
    print(f"output root   : {out_root}\n")

    failures: list[str] = []
    for i, r in enumerate(rows, 1):
        target_id = r["target_id"]
        pdb = (bench_dir / r["pdb_path"]).resolve()
        epi = r.get("epitope_indices", "").strip()
        epi_arg = f"[{epi}]" if epi else "[]"
        use_epi = "true" if epi else "false"
        length_arg = f"[{r['binder_len_min']},{r['binder_len_max']}]"

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lobster.cmdline.generate",
            f"--config-name={args.config}",
            f"paths={args.paths}",
            f"generation.input_structures={pdb}",
            f"generation.target_chain={r.get('target_chain', 'A')}",
            f"generation.epitope_indices={epi_arg}",
            f"generation.use_epitope_conditioning={use_epi}",
            f"generation.binder_length={length_arg}",
            f"generation.n_designs_per_structure={args.n_designs}",
            f"seed={args.seed}",
            f"output_dir={out_root / target_id}",
            *args.extra,
        ]
        print(f"[{i}/{len(rows)}] {target_id}  len={length_arg}  epi={epi_arg}  use_epi={use_epi}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"  FAILED: {target_id} (exit {proc.returncode})", file=sys.stderr)
            failures.append(target_id)

    if failures:
        print(f"\n{len(failures)} target(s) failed: {failures}", file=sys.stderr)
        return 1
    print(f"\nDone. {len(rows)} target(s) generated under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
