import logging
import os
import tempfile

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader
from upath import UPath

from lobster.data import upload_to_s3

logger = logging.getLogger(__name__)


class DataLoaderCheckpointCallback(Callback):
    """
    Lightning callback that saves dataloader state during training.

    This callback periodically saves the state of dataloaders attached to the trainer
    during the training process. It specifically works with dataloaders that implement
    the `state_dict()` and `load_state_dict()` methods, such as litdata.StreamingDataLoader.

    Parameters
    ----------
    dirpath : str
        Directory path where checkpoint files will be saved. Can be a local path
        or an S3 URI (starting with "s3://").
    every_n_steps : int, default=1000
        Frequency (in steps) at which to save dataloader checkpoints.

    Notes
    -----
    This callback only works with dataloader classes that implement state_dict() and
    load_state_dict() methods, such as StreamingDataLoader. Standard PyTorch DataLoaders
    do not support this functionality and will cause errors if used with this callback.
    """

    def __init__(
        self,
        dirpath: str,
        every_n_steps: int = 1000,
    ):
        super().__init__()
        self.dirpath = UPath(dirpath)
        self.every_n_steps = every_n_steps

        self._is_s3_uri = str(self.dirpath).startswith("s3://")

    def _should_skip_saving_checkpoint(self, trainer: L.Trainer) -> bool:
        """
        Determine if checkpoint saving should be skipped.

        Parameters
        ----------
        trainer : L.Trainer
            The Lightning trainer instance.

        Returns
        -------
        bool
            True if checkpoint saving should be skipped, False otherwise.
        """
        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
        )

    def _save_dataloader(self, dataloader: DataLoader, filename: str) -> None:
        """
        Save the dataloader state to a file.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to save. Must implement state_dict() method.
        filename : str
            Name of the checkpoint file.

        Raises
        ------
        AttributeError
            If the dataloader does not implement state_dict().
        """
        save_filepath = self.dirpath / filename

        if self._is_s3_uri:
            with tempfile.NamedTemporaryFile() as tmp_file:
                temp_path = tmp_file.name
                torch.save(dataloader.state_dict(), temp_path)
                upload_to_s3(save_filepath, temp_path)
        else:
            os.makedirs(self.dirpath, exist_ok=True)
            logger.info(f"Saving dataloader checkpoint to {save_filepath}")
            torch.save(dataloader.state_dict(), save_filepath)

        logger.info(f"Saved dataloader checkpoint to {save_filepath}")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Lightning hook called at the end of validation.

        Saves dataloader states if appropriate conditions are met.

        Parameters
        ----------
        trainer : L.Trainer
            The Lightning trainer instance.
        pl_module : L.LightningModule
            The Lightning module being trained.
        """
        if self._should_skip_saving_checkpoint(trainer):
            return

        if trainer.global_step % self.every_n_steps != 0:
            return

        for dataloader_name in ["train_dataloader", "val_dataloaders", "test_dataloaders"]:
            if not hasattr(trainer, dataloader_name):
                continue

            dataloader = getattr(trainer, dataloader_name)
            if dataloader is None:
                continue

            filename = f"epoch={trainer.current_epoch}-step={trainer.global_step}-{dataloader_name}.pt"
            self._save_dataloader(dataloader, filename)
