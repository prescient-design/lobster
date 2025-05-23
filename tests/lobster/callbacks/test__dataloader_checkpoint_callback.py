from unittest.mock import Mock, patch

from lightning.pytorch.trainer.states import TrainerFn

from lobster.callbacks._dataloader_checkpoint_callback import DataLoaderCheckpointCallback


class TestDataLoaderCheckpointCallback:
    def test__init__(self):
        # Local
        local_path = "local/path"
        callback = DataLoaderCheckpointCallback(local_path, every_n_steps=100)

        assert str(callback.dirpath) == local_path
        assert callback.every_n_steps == 100
        assert callback._is_s3_uri is False

        # S3
        s3_path = "s3://bucket/path"
        callback = DataLoaderCheckpointCallback(s3_path)

        assert str(callback.dirpath) == s3_path
        assert callback.every_n_steps == 1000
        assert callback._is_s3_uri is True

    def test__should_skip_saving_checkpoint(self):
        callback = DataLoaderCheckpointCallback("local/path")

        trainer = Mock()
        trainer.fast_dev_run = True
        assert callback._should_skip_saving_checkpoint(trainer) is True

        trainer.fast_dev_run = False
        trainer.state.fn = "not_fitting"
        assert callback._should_skip_saving_checkpoint(trainer) is True

        trainer.state.fn = TrainerFn.FITTING
        trainer.sanity_checking = True
        assert callback._should_skip_saving_checkpoint(trainer) is True

        trainer.sanity_checking = False
        assert callback._should_skip_saving_checkpoint(trainer) is False

    @patch("lobster.callbacks._dataloader_checkpoint_callback.upload_to_s3")
    @patch("lobster.callbacks._dataloader_checkpoint_callback.torch.save")
    def test__save_dataloader_s3(self, mock_torch_save, mock_upload_to_s3):
        callback = DataLoaderCheckpointCallback("s3://bucket/path")

        dataloader = Mock()
        mock_state_dict = {"key": "value"}
        dataloader.state_dict.return_value = mock_state_dict

        with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
            mock_tmp_file = Mock()
            mock_tmp_file.name = "/tmp/tempfile"
            mock_tempfile.return_value.__enter__.return_value = mock_tmp_file

            callback._save_dataloader(dataloader, "test_file.pt")

            dataloader.state_dict.assert_called_once()
            mock_torch_save.assert_called_once_with(mock_state_dict, "/tmp/tempfile")
            mock_upload_to_s3.assert_called_once_with(callback.dirpath / "test_file.pt", "/tmp/tempfile")

    @patch("lobster.callbacks._dataloader_checkpoint_callback.os.makedirs")
    @patch("lobster.callbacks._dataloader_checkpoint_callback.torch.save")
    def test__save_dataloader_local(self, mock_torch_save, mock_makedirs):
        # Test saving to local path
        callback = DataLoaderCheckpointCallback("local/path")

        dataloader = Mock()
        mock_state_dict = {"key": "value"}
        dataloader.state_dict.return_value = mock_state_dict

        callback._save_dataloader(dataloader, "test_file.pt")

        dataloader.state_dict.assert_called_once()
