"""S3 Checkpoint Backup Callback for PyTorch Lightning.

This callback automatically backs up checkpoints to S3 after they are saved,
providing disaster recovery capability for training runs.

Usage:
    Add to your training config:
    ```yaml
    callbacks:
      s3_backup:
        _target_: lobster.callbacks._s3_checkpoint_callback.S3CheckpointBackupCallback
        s3_bucket: "prescient-lobster"
        s3_prefix: "checkpoints"
        project_name: "latent_generator"
    ```
"""

import logging
import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

py_logger = logging.getLogger(__name__)


class S3CheckpointBackupCallback(Callback):
    """Automatically backup checkpoints to S3 after saving.

    This callback uploads checkpoints to S3 whenever PyTorch Lightning saves
    a checkpoint, providing a backup in case of local storage failures.

    Args:
        s3_bucket: S3 bucket name for backup storage.
        s3_prefix: Prefix path within the bucket (e.g., "checkpoints/latent_generator").
        project_name: Project name for organizing checkpoints.
        upload_every_n_epochs: Upload periodic checkpoints every N epochs.
        upload_best_only: If True, only upload the best checkpoint.
        upload_last: If True, also upload the last checkpoint.
        dry_run: If True, log uploads without actually uploading.
    """

    def __init__(
        self,
        s3_bucket: str = "prescient-pcluster-data",
        s3_prefix: str = "gen_ume/checkpoints",
        project_name: str | None = None,
        upload_every_n_epochs: int = 10,
        upload_best_only: bool = False,
        upload_last: bool = True,
        dry_run: bool = False,
    ):
        super().__init__()
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.project_name = project_name
        self.upload_every_n_epochs = upload_every_n_epochs
        self.upload_best_only = upload_best_only
        self.upload_last = upload_last
        self.dry_run = dry_run
        self._s3_client = None

    @property
    def s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            try:
                import boto3

                self._s3_client = boto3.client("s3")
            except ImportError:
                py_logger.error("boto3 not installed. Run: pip install boto3")
                raise
        return self._s3_client

    def _get_s3_key(self, local_path: str, checkpoint_type: str = "periodic") -> str:
        """Generate S3 key for a checkpoint file.

        Args:
            local_path: Local path to the checkpoint file.
            checkpoint_type: Type of checkpoint ("best", "last", or "periodic").

        Returns:
            S3 key string.
        """
        filename = Path(local_path).name
        project = self.project_name or "unknown"
        return f"{self.s3_prefix}/{project}/{checkpoint_type}/{filename}"

    def _upload_to_s3(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3.

        Args:
            local_path: Local path to the file.
            s3_key: S3 key (path within bucket).

        Returns:
            True if upload succeeded, False otherwise.
        """
        if self.dry_run:
            py_logger.info(f"[DRY RUN] Would upload {local_path} to s3://{self.s3_bucket}/{s3_key}")
            return True

        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            py_logger.info(f"✅ Uploaded checkpoint to s3://{self.s3_bucket}/{s3_key}")
            return True
        except Exception as e:
            py_logger.error(f"❌ Failed to upload {local_path} to S3: {e}")
            return False

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Called when a checkpoint is saved.

        Uploads the checkpoint to S3 based on configuration.
        """
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return

        current_epoch = trainer.current_epoch

        # Upload best checkpoint
        if ckpt_callback.best_model_path and os.path.exists(ckpt_callback.best_model_path):
            s3_key = self._get_s3_key(ckpt_callback.best_model_path, "best")
            self._upload_to_s3(ckpt_callback.best_model_path, s3_key)

        # Skip if upload_best_only is set
        if self.upload_best_only:
            return

        # Upload last checkpoint
        if self.upload_last and ckpt_callback.last_model_path:
            if os.path.exists(ckpt_callback.last_model_path):
                s3_key = self._get_s3_key(ckpt_callback.last_model_path, "last")
                self._upload_to_s3(ckpt_callback.last_model_path, s3_key)

        # Upload periodic checkpoints
        if self.upload_every_n_epochs > 0 and current_epoch % self.upload_every_n_epochs == 0:
            # Upload any checkpoint saved at this epoch
            if ckpt_callback.last_model_path and os.path.exists(ckpt_callback.last_model_path):
                s3_key = self._get_s3_key(ckpt_callback.last_model_path, f"epoch_{current_epoch}")
                self._upload_to_s3(ckpt_callback.last_model_path, s3_key)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends. Upload final checkpoints."""
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return

        # Final upload of best checkpoint
        if ckpt_callback.best_model_path and os.path.exists(ckpt_callback.best_model_path):
            s3_key = self._get_s3_key(ckpt_callback.best_model_path, "final_best")
            self._upload_to_s3(ckpt_callback.best_model_path, s3_key)

        # Final upload of last checkpoint
        if ckpt_callback.last_model_path and os.path.exists(ckpt_callback.last_model_path):
            s3_key = self._get_s3_key(ckpt_callback.last_model_path, "final_last")
            self._upload_to_s3(ckpt_callback.last_model_path, s3_key)

        py_logger.info(f"📦 Training complete. Checkpoints backed up to s3://{self.s3_bucket}/{self.s3_prefix}/")
