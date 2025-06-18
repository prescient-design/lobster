#!/usr/bin/env python3
"""
Manage UME model checkpoints in S3.
Supports adding, updating, deleting, and listing checkpoints.

Example usage:
uv run python src/lobster/cmdline/manage_ume_checkpoints.py action=list
uv run python src/lobster/cmdline/manage_ume_checkpoints.py action=add model_name=ume-small-base-12M-test checkpoint_path=s3://prescient-lobster/ume/runs/... dry_run=true
uv run python src/lobster/cmdline/manage_ume_checkpoints.py action=update model_name=ume-mini-base-12M checkpoint_path=s3://prescient-lobster/ume/runs/... dry_run=true
uv run python src/lobster/cmdline/manage_ume_checkpoints.py action=delete model_name=ume-small-base-12M-test dry_run=true
"""

import json
import pprint
import tempfile

import hydra
from omegaconf import DictConfig

from lobster.constants import UME_CHECKPOINT_DICT_S3_URI
from lobster.data._utils import download_from_s3, upload_to_s3


def get_checkpoints(s3_uri: str) -> dict[str, str]:
    """Get checkpoints from S3."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        temp_file_path = temp_file.name

        download_from_s3(s3_uri, temp_file_path)

        with open(temp_file_path) as f:
            checkpoints = json.load(f)

        return checkpoints


def _save_checkpoints_to_s3(s3_uri: str, checkpoints: dict[str, str]) -> None:
    """Save checkpoints to S3."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(checkpoints, temp_file, indent=2)
        temp_file_path = temp_file.name

    upload_to_s3(s3_uri, temp_file_path)


def _check_checkpoint_path(filepath: str | None) -> None:
    """Check if the S3 URI is valid."""
    if filepath is None:
        raise ValueError("Checkpoint path cannot be None")

    if not str(filepath).startswith("s3://"):
        raise ValueError("Checkpoint path must be an S3 path")


def list_checkpoints(s3_uri: str) -> None:
    """List all checkpoints with structured output."""
    checkpoints = get_checkpoints(s3_uri)

    if not checkpoints:
        print("❌ No checkpoints found.")
        return

    print(f"\n📊 Checkpoints ({len(checkpoints)} models):")
    pprint.pprint(checkpoints, width=120)


def add_checkpoint(s3_uri: str, model_name: str, checkpoint_path: str, dry_run: bool = False) -> None:
    """Add a new checkpoint."""
    _check_checkpoint_path(checkpoint_path)

    checkpoints = get_checkpoints(s3_uri)

    if model_name in checkpoints:
        print(f"⚠️  Warning: Model '{model_name}' already exists. Use 'update' action instead.")
        return

    checkpoints[model_name] = checkpoint_path

    if dry_run:
        print("🔍 [DRY RUN] Would add checkpoint:")
        print(f"   {model_name}: {checkpoint_path}")
        print(f"📊 Total checkpoints after addition: {len(checkpoints)}")
    else:
        _save_checkpoints_to_s3(s3_uri, checkpoints)
        print("✅ Successfully added checkpoint:")
        print(f"   {model_name}: {checkpoint_path}")


def update_checkpoint(s3_uri: str, model_name: str, checkpoint_path: str, dry_run: bool = False) -> None:
    """Update an existing checkpoint."""
    _check_checkpoint_path(checkpoint_path)

    checkpoints = get_checkpoints(s3_uri)

    if model_name not in checkpoints:
        print(f"❌ Error: Model '{model_name}' not found. Use 'add' action instead.")
        return

    old_path = checkpoints[model_name]
    checkpoints[model_name] = checkpoint_path

    if dry_run:
        print("🔍 [DRY RUN] Would update checkpoint:")
        print(f"   📝 Model: {model_name}")
        print(f"   🔄 Old: {old_path}")
        print(f"   🔄 New: {checkpoint_path}")
    else:
        _save_checkpoints_to_s3(s3_uri, checkpoints)
        print("✅ Successfully updated checkpoint:")
        print(f"   📝 Model: {model_name}")
        print(f"   🔄 Old: {old_path}")
        print(f"   🔄 New: {checkpoint_path}")


def delete_checkpoint(s3_uri: str, model_name: str, dry_run: bool = False) -> None:
    """Delete a checkpoint."""
    checkpoints = get_checkpoints(s3_uri)

    if model_name not in checkpoints:
        print(f"❌ Error: Model '{model_name}' not found.")
        return

    checkpoint_path = checkpoints[model_name]

    # Check if the checkpoint is already not available
    if not checkpoint_path:
        print(f"⚠️  Model '{model_name}' exists but has no checkpoint path (already empty).")
        if dry_run:
            print(f"🔍 [DRY RUN] Would remove the empty entry for: {model_name}")
        else:
            del checkpoints[model_name]
            _save_checkpoints_to_s3(s3_uri, checkpoints)
            print(f"🗑️  Removed empty entry for: {model_name}")
        return

    del checkpoints[model_name]

    if dry_run:
        print("🔍 [DRY RUN] Would delete checkpoint:")
        print(f"   {model_name}: {checkpoint_path}")
        print(f"📊 Total checkpoints after deletion: {len(checkpoints)}")
    else:
        _save_checkpoints_to_s3(s3_uri, checkpoints)
        print("🗑️  Successfully deleted checkpoint:")
        print(f"   {model_name}: {checkpoint_path}")


@hydra.main(version_base=None, config_path="../hydra_config", config_name="manage_ume_checkpoints")
def main(cfg: DictConfig) -> None:
    """Main function for managing UME models."""
    action = cfg.action.lower()
    s3_uri = UME_CHECKPOINT_DICT_S3_URI
    model_name = cfg.get("model_name")
    checkpoint_path = cfg.get("checkpoint_path")

    # Fix dry_run parameter handling
    dry_run = cfg.get("dry_run", False)

    print("🚀 UME Checkpoint Manager")
    print(f"S3 URI: {s3_uri}")
    print(f"Action: {action}")

    if dry_run:
        print("Mode: DRY RUN")
    print()

    if action == "list":
        list_checkpoints(s3_uri)
    elif action == "add":
        add_checkpoint(s3_uri, model_name, checkpoint_path, dry_run)
    elif action == "update":
        update_checkpoint(s3_uri, model_name, checkpoint_path, dry_run)
    elif action == "delete":
        delete_checkpoint(s3_uri, model_name, dry_run)
    else:
        print(f"❌ Error: Unknown action '{action}'. Supported actions: list, add, update, delete")


if __name__ == "__main__":
    main()
