#!/usr/bin/env python3
"""CLI for listing, inspecting, fetching, and clearing LeFlur checkpoints.

The companion to :mod:`lobster.model.leflur.checkpoints`. Registered as the
``lobster_leflur_checkpoints`` console entry point so external users can:

.. code:: bash

    # List every registered checkpoint (id, family, description, hf URI).
    lobster_leflur_checkpoints list

    # Filter to publication-ready protein-ligand checkpoints.
    lobster_leflur_checkpoints list --family protein_ligand --tag canonical

    # Print full metadata for one checkpoint.
    lobster_leflur_checkpoints inspect leflur-ted

    # Download into the LeFlur cache (no-op if already cached).
    lobster_leflur_checkpoints fetch leflur-ted

    # Show or wipe the local cache.
    lobster_leflur_checkpoints cache
    lobster_leflur_checkpoints cache --clear --dry-run
    lobster_leflur_checkpoints cache --clear

By design this CLI does **not** add/update/delete registry entries — the
publication scope freezes the registry in source. Use a follow-up edit to
:mod:`lobster.model.leflur.checkpoints` instead.
"""

from __future__ import annotations

import argparse
import sys

from lobster.model.leflur import (
    KNOWN_CHECKPOINTS,
    PAIRED_LG_CHECKPOINTS,
    cached_files,
    cache_dir,
    clear_cache,
    list_checkpoints,
    resolve_checkpoint,
    upload_checkpoint,
)
from lobster.model.leflur.checkpoints import CheckpointInfo


# --- Helpers ---------------------------------------------------------------


def _format_size(num_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PiB"


def _filter_entries(
    family: str | None, tag: str | None
) -> list[CheckpointInfo]:
    entries = list_checkpoints(family=family)
    if tag is not None:
        entries = [e for e in entries if tag in e.tags]
    return entries


# --- Subcommand handlers ---------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    entries = _filter_entries(args.family, args.tag)
    if not entries:
        sys.stdout.write("No checkpoints match the supplied filters.\n")
        return 1

    rows = [
        ("short_name", "family", "tags", "description"),
    ]
    rows.extend(
        (
            info.short_name,
            info.family,
            ",".join(info.tags) or "-",
            info.description.split(". ")[0] + ("." if "." in info.description else ""),
        )
        for info in entries
    )
    col_widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for i, row in enumerate(rows):
        sys.stdout.write(
            "  ".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row))
            + "\n"
        )
        if i == 0:
            sys.stdout.write("  ".join("-" * w for w in col_widths) + "\n")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    info = KNOWN_CHECKPOINTS.get(args.short_name)
    if info is None:
        sys.stderr.write(
            f"Unknown short name: {args.short_name!r}. Known names: "
            f"{sorted(KNOWN_CHECKPOINTS)}\n"
        )
        return 2
    sys.stdout.write(f"short_name      : {info.short_name}\n")
    sys.stdout.write(f"family          : {info.family}\n")
    sys.stdout.write(f"tags            : {', '.join(info.tags) or '-'}\n")
    sys.stdout.write(f"hf_uri          : {info.hf_uri}\n")
    sys.stdout.write(f"https_url       : {info.https_url}\n")
    sys.stdout.write(
        f"paired_lg_codec : {info.paired_lg_codec or '-'}\n"
    )
    sys.stdout.write(
        f"recommended_cfg : {info.recommended_generation_config or '-'}\n"
    )
    sys.stdout.write(f"description     : {info.description}\n")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        local_path = resolve_checkpoint(args.target)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"fetch failed: {exc}\n")
        return 2
    sys.stdout.write(f"{args.target!r} -> {local_path}\n")
    if local_path.exists():
        sys.stdout.write(f"  size: {_format_size(local_path.stat().st_size)}\n")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload one or many registered checkpoints to HuggingFace."""
    if args.all:
        targets = list(KNOWN_CHECKPOINTS) + (
            list(PAIRED_LG_CHECKPOINTS) if args.include_lg_codecs else []
        )
    elif args.lg_codecs:
        targets = list(PAIRED_LG_CHECKPOINTS)
    else:
        if not args.short_names:
            sys.stderr.write(
                "upload: pass at least one short name, or use --all / --lg-codecs\n"
            )
            return 2
        targets = list(args.short_names)

    errors: list[tuple[str, Exception]] = []
    for name in targets:
        try:
            summary = upload_checkpoint(
                name,
                source_path=args.source,
                repo_id=args.repo,
                token=args.token,
                dry_run=args.dry_run,
                commit_message=args.commit_message,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"  ! {name}: {exc}\n")
            errors.append((name, exc))
            continue

        size_mib = int(summary["size_bytes"]) / (1024 * 1024)
        verb = "would upload" if args.dry_run else "uploaded"
        sys.stdout.write(
            f"  {verb} {name} ({size_mib:.0f} MiB) "
            f"-> {summary['repo_id']}/{summary['hf_path']}"
        )
        if summary["commit_url"]:
            sys.stdout.write(f"\n      commit: {summary['commit_url']}")
        sys.stdout.write("\n")

    if errors:
        sys.stderr.write(
            f"\nupload finished with {len(errors)} error(s).\n"
        )
        return 1
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        deleted = list(clear_cache(dry_run=args.dry_run))
        verb = "would remove" if args.dry_run else "removed"
        if not deleted:
            sys.stdout.write("LeFlur cache is empty — nothing to clear.\n")
            return 0
        sys.stdout.write(f"{verb} {len(deleted)} entries from {cache_dir()}:\n")
        for path in deleted:
            sys.stdout.write(f"  - {path}\n")
        return 0

    root = cache_dir()
    files = cached_files()
    sys.stdout.write(f"Cache root : {root}\n")
    if not files:
        sys.stdout.write("(cache is empty)\n")
        return 0
    total = sum(p.stat().st_size for p in files)
    sys.stdout.write(f"Total size : {_format_size(total)} across {len(files)} files\n\n")
    for path in files:
        sys.stdout.write(
            f"  {_format_size(path.stat().st_size):>10}  "
            f"{path.relative_to(root)}\n"
        )
    return 0


# --- argparse wiring -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lobster_leflur_checkpoints",
        description=(
            "Manage LeFlur publication checkpoints (Sidney-Lisanza/leflur HF "
            "repo). All actions read from lobster.model.leflur.KNOWN_CHECKPOINTS."
        ),
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_list = subs.add_parser("list", help="List registered checkpoints.")
    p_list.add_argument("--family", choices=("protein", "protein_ligand"))
    p_list.add_argument(
        "--tag",
        help=(
            "Filter to a single tag (e.g. 'canonical', 'research', 'legacy')."
        ),
    )
    p_list.set_defaults(func=cmd_list)

    p_inspect = subs.add_parser(
        "inspect", help="Print full metadata for one short name."
    )
    p_inspect.add_argument("short_name")
    p_inspect.set_defaults(func=cmd_inspect)

    p_fetch = subs.add_parser(
        "fetch",
        help=(
            "Materialise a short name / hf:// URI / local path to a local "
            "file via resolve_checkpoint() and print the result."
        ),
    )
    p_fetch.add_argument(
        "target",
        help=(
            "Short name (e.g. leflur-ted), hf://... URI, or local path."
        ),
    )
    p_fetch.set_defaults(func=cmd_fetch)

    p_upload = subs.add_parser(
        "upload",
        help=(
            "Upload one/many registered checkpoints to HuggingFace. Use "
            "--all to push every canonical LeFlur ckpt, --lg-codecs for the "
            "two paired LG codecs in Sidney-Lisanza/latent_generator, or "
            "pass short names explicitly."
        ),
    )
    p_upload.add_argument(
        "short_names",
        nargs="*",
        help="Short names to upload (e.g. leflur-ted leflur-base).",
    )
    p_upload.add_argument(
        "--all",
        action="store_true",
        help="Upload every entry in KNOWN_CHECKPOINTS.",
    )
    p_upload.add_argument(
        "--include-lg-codecs",
        action="store_true",
        help="When used with --all, also upload PAIRED_LG_CHECKPOINTS.",
    )
    p_upload.add_argument(
        "--lg-codecs",
        action="store_true",
        help="Upload only the paired LG codec entries.",
    )
    p_upload.add_argument(
        "--source",
        help=(
            "Override the registered local source path (only meaningful "
            "with a single short name)."
        ),
    )
    p_upload.add_argument(
        "--repo",
        help="Override the target HF repo id (defaults to entry.hf_repo_id).",
    )
    p_upload.add_argument(
        "--token",
        help=(
            "HF token. Falls back to $HF_TOKEN / `huggingface-cli login`."
        ),
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
    p_upload.set_defaults(func=cmd_upload)

    p_cache = subs.add_parser(
        "cache", help="Show or clear the local LeFlur checkpoint cache."
    )
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
    """argparse entry point for the ``lobster_leflur_checkpoints`` console script.

    See the module docstring for the available subcommands
    (``list / inspect / fetch / cache``). Returns the exit code from the
    selected subcommand: ``0`` on success, non-zero on failure.

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
