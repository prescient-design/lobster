#!/usr/bin/env python3
"""CLI for listing, inspecting, fetching, uploading, and clearing LeFlur benchmarks.

Companion to :mod:`lobster.model.leflur.benchmarks` (analogous to
``lobster_leflur_checkpoints`` for model weights). Registered as the
``lobster_leflur_benchmarks`` console entry point.

.. code:: bash

    # List every registered benchmark (id, hf path, description).
    lobster_leflur_benchmarks list

    # Print full metadata for one benchmark.
    lobster_leflur_benchmarks inspect cameo

    # Download into the LeFlur cache (no-op if already cached).
    lobster_leflur_benchmarks fetch cameo
    lobster_leflur_benchmarks fetch multiflow_test

    # Show or wipe the local cache.
    lobster_leflur_benchmarks cache
    lobster_leflur_benchmarks cache --clear --dry-run
    lobster_leflur_benchmarks cache --clear

    # Maintainers only: populate the HF dataset repo from internal /cv/...
    lobster_leflur_benchmarks upload --all --dry-run
    lobster_leflur_benchmarks upload cameo --token "$HF_TOKEN"

By design this CLI does **not** add/update/delete registry entries — the
publication scope freezes the registry in source. To add a new benchmark
edit :mod:`lobster.model.leflur.benchmarks`.
"""

from __future__ import annotations

import argparse
import sys

from lobster.model.leflur import (
    KNOWN_BENCHMARKS,
    cache_benchmark_dir,
    cached_benchmark_files,
    clear_benchmark_cache,
    fetch_benchmark,
    generate_dataset_card_md,
    list_benchmarks,
    upload_benchmark,
    upload_dataset_card,
)
from lobster.model.leflur.benchmarks import BenchmarkInfo


# --- Helpers ---------------------------------------------------------------


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


def _filter_entries(tag: str | None) -> list[BenchmarkInfo]:
    return list_benchmarks(tag=tag)


# --- Subcommand handlers ---------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    entries = _filter_entries(args.tag)
    if not entries:
        sys.stdout.write("No benchmarks match the supplied filters.\n")
        return 1

    rows: list[tuple[str, str, str, str]] = [
        ("short_name", "hf_subdir", "tags", "description"),
    ]
    rows.extend(
        (
            info.short_name,
            info.hf_subdir,
            ",".join(info.tags) or "-",
            info.description.split(". ")[0] + ("." if "." in info.description else ""),
        )
        for info in entries
    )
    col_widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for i, row in enumerate(rows):
        sys.stdout.write("  ".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row)) + "\n")
        if i == 0:
            sys.stdout.write("  ".join("-" * w for w in col_widths) + "\n")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    info = KNOWN_BENCHMARKS.get(args.short_name)
    if info is None:
        sys.stderr.write(f"Unknown short name: {args.short_name!r}. Known names: {sorted(KNOWN_BENCHMARKS)}\n")
        return 2
    sys.stdout.write(f"short_name      : {info.short_name}\n")
    sys.stdout.write(f"tags            : {', '.join(info.tags) or '-'}\n")
    sys.stdout.write(f"hf_uri          : {info.hf_uri}\n")
    sys.stdout.write(f"https_url       : {info.https_url}\n")
    sys.stdout.write(f"cache_subdir    : {info.cache_subdir}\n")
    sys.stdout.write(f"pattern         : {info.pattern}\n")
    sys.stdout.write(f"schema_keys     : {', '.join(info.schema_keys) or '-'}\n")
    sys.stdout.write(f"license         : {info.license or '-'}\n")
    sys.stdout.write(f"citation        : {info.citation or '-'}\n")
    sys.stdout.write(f"description     : {info.description}\n")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        local_path = fetch_benchmark(args.short_name)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"fetch failed: {exc}\n")
        return 2
    info = KNOWN_BENCHMARKS[args.short_name]
    files = sorted(local_path.glob(info.pattern))
    total = sum(p.stat().st_size for p in files)
    sys.stdout.write(f"{args.short_name!r} -> {local_path}\n")
    sys.stdout.write(f"  {len(files)} files matching {info.pattern!r}, total {_format_size(total)}\n")
    return 0


def cmd_dataset_card(args: argparse.Namespace) -> int:
    if args.print:
        sys.stdout.write(generate_dataset_card_md())
        return 0
    summary = upload_dataset_card(
        repo_id=args.repo,
        token=args.token,
        dry_run=args.dry_run,
        commit_message=args.commit_message,
    )
    verb = "would upload" if args.dry_run else "uploaded"
    sys.stdout.write(f"  {verb} dataset card ({int(summary['total_bytes'])} bytes) -> {summary['repo_id']}/README.md")
    if summary["commit_url"]:
        sys.stdout.write(f"\n      commit: {summary['commit_url']}")
    sys.stdout.write("\n")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    targets = list(KNOWN_BENCHMARKS) if args.all else list(args.short_names)
    if not targets:
        sys.stderr.write("upload: pass at least one short name, or use --all\n")
        return 2

    if args.with_card:
        try:
            card_summary = upload_dataset_card(
                repo_id=args.repo,
                token=args.token,
                dry_run=args.dry_run,
                commit_message=args.commit_message,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"  ! <dataset-card>: {exc}\n")
            return 1
        verb = "would upload" if args.dry_run else "uploaded"
        sys.stdout.write(
            f"  {verb} dataset card ({int(card_summary['total_bytes'])} bytes) -> {card_summary['repo_id']}/README.md\n"
        )

    errors: list[tuple[str, Exception]] = []
    for name in targets:
        try:
            summary = upload_benchmark(
                name,
                source_dir=args.source,
                repo_id=args.repo,
                token=args.token,
                dry_run=args.dry_run,
                commit_message=args.commit_message,
                sanitize=not args.no_sanitize,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"  ! {name}: {exc}\n")
            errors.append((name, exc))
            continue

        size_mib = int(summary["total_bytes"]) / (1024 * 1024)
        verb = "would upload" if args.dry_run else "uploaded"
        sys.stdout.write(
            f"  {verb} {name} ({summary['num_files']} files, "
            f"{size_mib:.1f} MiB) -> {summary['repo_id']}/{summary['hf_subdir']}"
        )
        if summary["commit_url"]:
            sys.stdout.write(f"\n      commit: {summary['commit_url']}")
        sys.stdout.write("\n")

    if errors:
        sys.stderr.write(f"\nupload finished with {len(errors)} error(s).\n")
        return 1
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        deleted = list(clear_benchmark_cache(dry_run=args.dry_run))
        verb = "would remove" if args.dry_run else "removed"
        if not deleted:
            sys.stdout.write("LeFlur benchmark cache is empty — nothing to clear.\n")
            return 0
        sys.stdout.write(f"{verb} {len(deleted)} entries from {cache_benchmark_dir()}:\n")
        for path in deleted:
            sys.stdout.write(f"  - {path}\n")
        return 0

    root = cache_benchmark_dir()
    files = cached_benchmark_files()
    sys.stdout.write(f"Cache root : {root}\n")
    if not files:
        sys.stdout.write("(cache is empty)\n")
        return 0
    total = sum(p.stat().st_size for p in files)
    sys.stdout.write(f"Total size : {_format_size(total)} across {len(files)} files\n\n")
    for path in files[:50]:
        sys.stdout.write(f"  {_format_size(path.stat().st_size):>10}  {path.relative_to(root)}\n")
    if len(files) > 50:
        sys.stdout.write(f"  ... ({len(files) - 50} more)\n")
    return 0


# --- argparse wiring -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lobster_leflur_benchmarks",
        description=(
            "Manage LeFlur benchmark datasets (Sidney-Lisanza/leflur HF "
            "dataset repo). All actions read from "
            "lobster.model.leflur.KNOWN_BENCHMARKS."
        ),
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_list = subs.add_parser("list", help="List registered benchmarks.")
    p_list.add_argument(
        "--tag",
        help="Filter to a single tag (e.g. 'canonical', 'protein', 'publication').",
    )
    p_list.set_defaults(func=cmd_list)

    p_inspect = subs.add_parser("inspect", help="Print full metadata for one short name.")
    p_inspect.add_argument("short_name")
    p_inspect.set_defaults(func=cmd_inspect)

    p_fetch = subs.add_parser(
        "fetch",
        help=(
            "Snapshot-download a benchmark from HuggingFace into the LeFlur "
            "cache. The resolved local path matches "
            "${paths.benchmarks.<short_name>} from paths/public.yaml, so "
            "generate configs work unchanged after fetch."
        ),
    )
    p_fetch.add_argument(
        "short_name",
        help="Short name (e.g. cameo, multiflow_test).",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    p_upload = subs.add_parser(
        "upload",
        help=(
            "Upload one/many registered benchmarks to HuggingFace. Use "
            "--all to push every benchmark or pass short names explicitly. "
            "By default the CAMEO 'pdb_path' field is rewritten to its "
            "basename to strip internal paths; pass --no-sanitize to "
            "upload bit-identical files."
        ),
    )
    p_upload.add_argument(
        "short_names",
        nargs="*",
        help="Short names to upload (e.g. cameo multiflow_test).",
    )
    p_upload.add_argument(
        "--all",
        action="store_true",
        help="Upload every entry in KNOWN_BENCHMARKS.",
    )
    p_upload.add_argument(
        "--source",
        help=("Override the registered local source directory (only meaningful with a single short name)."),
    )
    p_upload.add_argument(
        "--repo",
        help="Override the target HF repo id (defaults to entry.hf_repo_id).",
    )
    p_upload.add_argument(
        "--token",
        help="HF token. Falls back to $HF_TOKEN / `huggingface-cli login`.",
    )
    p_upload.add_argument(
        "--commit-message",
        help="Override the auto-generated commit message.",
    )
    p_upload.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without contacting HF.",
    )
    p_upload.add_argument(
        "--no-sanitize",
        action="store_true",
        help=(
            "Disable the default CAMEO 'pdb_path' rewrite. Use when you "
            "want bit-identical uploads (e.g. for reproducibility diffs)."
        ),
    )
    p_upload.add_argument(
        "--with-card",
        action="store_true",
        help=(
            "Also (re)upload the dataset-card README.md generated from "
            "KNOWN_BENCHMARKS. Convenient for first-time repo creation."
        ),
    )
    p_upload.set_defaults(func=cmd_upload)

    p_card = subs.add_parser(
        "dataset-card",
        help=(
            "Build (and optionally upload) the LeFlur benchmarks dataset "
            "card README.md from KNOWN_BENCHMARKS. Use --print to dump to "
            "stdout for review without contacting HF."
        ),
    )
    p_card.add_argument(
        "--print",
        action="store_true",
        help="Print the generated README.md to stdout (no upload).",
    )
    p_card.add_argument(
        "--repo",
        help="Override the target HF dataset repo id.",
    )
    p_card.add_argument(
        "--token",
        help="HF token. Falls back to $HF_TOKEN / `huggingface-cli login`.",
    )
    p_card.add_argument(
        "--commit-message",
        help="Override the auto-generated commit message.",
    )
    p_card.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without contacting HF.",
    )
    p_card.set_defaults(func=cmd_dataset_card)

    p_cache = subs.add_parser("cache", help="Show or clear the local LeFlur benchmark cache.")
    p_cache.add_argument(
        "--clear",
        action="store_true",
        help="Delete cached files (use with --dry-run to preview).",
    )
    p_cache.add_argument(
        "--dry-run",
        action="store_true",
        help="Combine with --clear to preview what would be removed.",
    )
    p_cache.set_defaults(func=cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    """argparse entry point for the ``lobster_leflur_benchmarks`` console script.

    See the module docstring for the available subcommands
    (``list / inspect / fetch / upload / cache``). Returns the exit code
    from the selected subcommand: ``0`` on success, non-zero on failure.

    Parameters
    ----------
    argv
        Optional argument list (defaults to ``sys.argv[1:]``). Useful in
        tests for invoking the CLI without process spawning.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
