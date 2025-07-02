from unittest.mock import MagicMock, Mock, patch

import lightning as L
import pytest
import torch

from lobster.callbacks import PEEREvaluationCallback
from lobster.constants import PEERTask


class MockTransform:
    def __init__(self, max_length=32):
        self.max_length = max_length
        self.tokenizer = Mock()
        self.tokenizer.cls_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.tokenizer.pad_token_id = 2
        self.tokenizer.sep_token_id = 3
        self.tokenizer.mask_token_id = 4
        self.tokenizer.unk_token_id = 5

    def __call__(self, sequence):
        return {
            "input_ids": torch.tensor([0, 10, 11, 12, 1, 2, 2, 2], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
        }


class MockPEERDataset:
    def __init__(self, task, split, transform_fn=None, n_samples=10):
        self.task = task
        self.split = split
        self.transform_fn = transform_fn
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.task == PEERTask.STABILITY:
            x = "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGM"
            if self.transform_fn:
                x = self.transform_fn(x)
            y = torch.tensor([0.75])

        elif self.task == PEERTask.HUMANPPI:
            protein1 = "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGM"
            protein2 = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHL"
            if self.transform_fn:
                x = [self.transform_fn(protein1), self.transform_fn(protein2)]
            else:
                x = [protein1, protein2]
            y = torch.tensor(1)

        elif self.task == PEERTask.SECONDARY_STRUCTURE:
            x = "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGM"
            if self.transform_fn:
                x = self.transform_fn(x)
            y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, -100, -100])

        elif self.task == PEERTask.PROTEINNET:
            x = "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGM"
            if self.transform_fn:
                x = self.transform_fn(x)
            tertiary = torch.randn(10, 3)
            valid_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
            y = (tertiary, valid_mask)

        else:
            x = "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGM"
            if self.transform_fn:
                x = self.transform_fn(x)
            y = torch.rand(1)

        return x, y


class MockLightningModule(L.LightningModule):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim

    def get_embeddings(self, inputs, modality="protein", per_residue=False):
        batch_size = len(inputs)

        if per_residue:
            if isinstance(inputs[0], dict) and "attention_mask" in inputs[0]:
                seq_len = int(inputs[0]["attention_mask"].sum().item())
                return torch.randn(batch_size, seq_len, self.embedding_dim)
            else:
                return torch.randn(batch_size, 10, self.embedding_dim)
        else:
            if isinstance(inputs[0], list) and len(inputs[0]) == 2:
                embeddings1 = torch.randn(batch_size, self.embedding_dim)
                embeddings2 = torch.randn(batch_size, self.embedding_dim)
                return torch.cat([embeddings1, embeddings2], dim=1)
            else:
                return torch.randn(batch_size, self.embedding_dim)


@pytest.fixture
def mock_transform():
    return MockTransform(max_length=32)


@pytest.fixture
def callback(monkeypatch, mock_transform):
    monkeypatch.setattr(
        "lobster.callbacks._peer_evaluation_callback.UMETokenizerTransform",
        lambda **kwargs: mock_transform,
    )

    return PEEREvaluationCallback(
        max_length=32,
        tasks=[PEERTask.STABILITY, PEERTask.SECONDARY_STRUCTURE, PEERTask.HUMANPPI],
        batch_size=4,
    )


@pytest.fixture
def mock_model():
    return MockLightningModule()


@pytest.fixture
def mock_trainer():
    mock_trainer = Mock(spec=L.Trainer)
    mock_trainer.current_epoch = 0
    mock_trainer.logger = Mock()
    mock_trainer.logger.log_metrics = Mock()
    return mock_trainer


class TestPEEREvaluationCallback:
    def test_initialization(self, callback, mock_transform):
        assert PEERTask.STABILITY in callback.selected_tasks
        assert PEERTask.SECONDARY_STRUCTURE in callback.selected_tasks
        assert PEERTask.HUMANPPI in callback.selected_tasks
        assert len(callback.selected_tasks) == 3

        assert callback.transform_fn is mock_transform
        assert callback.transform_fn.max_length == 32

        assert callback.batch_size == 4
        assert callback.embedders == {}
        assert callback.datasets == {}

    def test_get_task_test_splits(self, callback):
        stability_splits = callback._get_task_test_splits(PEERTask.STABILITY)
        assert len(stability_splits) == 1
        assert stability_splits[0] == "test"

        ss_splits = callback._get_task_test_splits(PEERTask.SECONDARY_STRUCTURE)
        assert len(ss_splits) == 3
        assert all(split in ss_splits for split in ["casp12", "cb513", "ts115"])

        ppi_splits = callback._get_task_test_splits(PEERTask.HUMANPPI)
        assert len(ppi_splits) == 2
        assert all(split in ppi_splits for split in ["test", "cross_species_test"])

    @patch("lobster.callbacks._peer_evaluation_callback.PEERDataset", MockPEERDataset)
    def test_get_task_datasets(self, callback):
        train_dataset, test_datasets = callback._get_task_datasets(PEERTask.STABILITY)

        assert isinstance(train_dataset, MockPEERDataset)
        assert train_dataset.split == "train"
        assert isinstance(test_datasets, dict)
        assert "test" in test_datasets

        cache_key = str(PEERTask.STABILITY)
        assert cache_key in callback.datasets

        _, ss_test_datasets = callback._get_task_datasets(PEERTask.SECONDARY_STRUCTURE)
        assert len(ss_test_datasets) == 3
        assert all(split in ss_test_datasets for split in ["casp12", "cb513", "ts115"])

    @patch("lobster.callbacks._peer_evaluation_callback.DataLoader")
    def test_get_embeddings_standard_task(self, mock_dataloader, callback, mock_model):
        mock_loader = MagicMock()
        mock_dataloader.return_value = mock_loader

        test_batch = (
            {"input_ids": torch.ones(2, 10, dtype=torch.long), "attention_mask": torch.ones(2, 10)},
            torch.tensor([0.75, 0.5]),
        )
        mock_loader.__iter__.return_value = [test_batch]

        with patch.object(callback, "_process_and_embed", return_value=torch.randn(2, 32)) as mock_proc:
            embeddings, targets = callback._get_embeddings(mock_model, mock_loader, PEERTask.STABILITY)

            assert embeddings.shape == (2, 32)
            assert targets.shape == (2,)
            mock_proc.assert_called_once_with(
                mock_model,
                test_batch[0],
                modality="amino_acid",
                aggregate=True,
            )

    @patch("lobster.callbacks._peer_evaluation_callback.DataLoader")
    def test_get_paired_embeddings(self, mock_dataloader, callback, mock_model):
        mock_loader = MagicMock()
        mock_dataloader.return_value = mock_loader

        test_batch = (
            [
                {"input_ids": torch.ones(2, 10, dtype=torch.long), "attention_mask": torch.ones(2, 10)},
                {"input_ids": torch.ones(2, 10, dtype=torch.long), "attention_mask": torch.ones(2, 10)},
            ],
            torch.tensor([1, 0]),
        )
        mock_loader.__iter__.return_value = [test_batch]

        # likewise patch _process_and_embed for each leg of the pair
        with patch.object(callback, "_process_and_embed") as mock_proc:
            mock_proc.side_effect = [
                torch.randn(2, 32),  # embedding for sequence1
                torch.randn(2, 32),  # embedding for sequence2
            ]

            embeddings, targets = callback._get_paired_embeddings(mock_model, mock_loader, PEERTask.HUMANPPI)

            assert embeddings.shape == (2, 64)
            assert targets.shape == (2,)
            assert mock_proc.call_count == 2

    def test_flatten_and_filter_token_embeddings(self, callback):
        batch_embeddings = torch.randn(2, 5, 10)
        targets = torch.tensor(
            [
                [0, 1, 2, -100, -100],
                [2, 1, 0, 2, -100],
            ]
        )
        input_ids = torch.tensor(
            [
                [0, 10, 11, 12, 1],
                [0, 13, 14, 15, 16],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        filtered_embeddings, filtered_targets = callback._flatten_and_filter_token_embeddings(
            batch_embeddings=batch_embeddings,
            targets=targets,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # we now filter out both the -100 targets and all special tokens
        # for this toy example you end up with 5 remaining residues
        assert filtered_embeddings.shape[0] == 5
        assert filtered_targets.shape[0] == 5

        filtered_embeddings2, filtered_targets2 = callback._flatten_and_filter_token_embeddings(
            batch_embeddings=batch_embeddings,
            targets=targets,
        )

        assert filtered_embeddings2.shape[0] == filtered_targets2.shape[0]

    @patch("lobster.callbacks._peer_evaluation_callback.LinearRegression")
    def test_train_probe_regression(self, mock_linear_regression, callback):
        embeddings = torch.randn(10, 32)
        targets = torch.rand(10, 1)

        mock_model = MagicMock()
        mock_linear_regression.return_value = mock_model

        callback._train_probe(embeddings, targets, PEERTask.STABILITY)

        assert mock_linear_regression.call_count > 0
        mock_model.fit.assert_called_once()
        args, _ = mock_model.fit.call_args
        assert args[0].shape == (10, 32)
        assert args[1].shape == (10, 1)

    @patch("lobster.callbacks._peer_evaluation_callback.LogisticRegression")
    def test_train_probe_classification(self, mock_logistic_regression, callback):
        embeddings = torch.randn(10, 32)
        targets = torch.randint(0, 3, (10,))

        mock_model = MagicMock()
        mock_logistic_regression.return_value = mock_model

        with patch.object(callback, "_set_metrics"):
            callback._train_probe(embeddings, targets, PEERTask.SECONDARY_STRUCTURE)

            assert mock_logistic_regression.call_count > 0
            assert mock_logistic_regression.call_args[1].get("multi_class") == "multinomial"
            mock_model.fit.assert_called_once()

    @patch("lobster.callbacks._peer_evaluation_callback.PEERDataset", MockPEERDataset)
    @patch("lobster.callbacks._peer_evaluation_callback.DataLoader")
    def test_evaluate_task(self, mock_dataloader, callback, mock_model, mock_trainer):
        mock_train_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_dataloader.side_effect = [mock_train_loader, mock_test_loader]

        with patch.object(callback, "_get_embeddings") as mock_get_embeddings:
            mock_get_embeddings.side_effect = [
                (torch.randn(10, 32), torch.rand(10, 1)),
                (torch.randn(5, 32), torch.rand(5, 1)),
            ]

            with patch.object(callback, "_train_probe") as mock_train_probe:
                mock_probe = MagicMock()
                mock_train_probe.return_value = mock_probe

                with patch.object(callback, "_evaluate_probe") as mock_evaluate_probe:
                    mock_evaluate_probe.return_value = {"mse": 0.1, "r2": 0.85}

                    results = callback._evaluate_task(PEERTask.STABILITY, mock_trainer, mock_model)

                    assert "test" in results
                    assert "mse" in results["test"]
                    assert "r2" in results["test"]
                    assert str(PEERTask.STABILITY) in callback.probes

    @patch("lobster.callbacks._peer_evaluation_callback.PEEREvaluationCallback._evaluate_task")
    @patch("lobster.callbacks._peer_evaluation_callback.tqdm")
    def test_on_validation_epoch_end(self, mock_tqdm, mock_evaluate_task, callback, mock_trainer, mock_model):
        mock_tqdm.return_value = callback.selected_tasks

        mock_evaluate_task.side_effect = [
            {"test": {"mse": 0.1, "r2": 0.85}},
            {
                "casp12": {"accuracy": 0.75, "precision": 0.7},
                "cb513": {"accuracy": 0.72, "precision": 0.68},
                "ts115": {"accuracy": 0.78, "precision": 0.73},
            },
            {"test": {"accuracy": 0.8, "precision": 0.75}},
        ]

        callback.on_validation_epoch_end(mock_trainer, mock_model)

        assert mock_evaluate_task.call_count == 3
        assert mock_trainer.logger.log_metrics.call_count > 0

    def test_skip_mechanism(self, callback, mock_trainer):
        mock_trainer.global_rank = 0

        assert callback._skip(mock_trainer) is False

        callback_with_skip = PEEREvaluationCallback(
            max_length=32, tasks=[PEERTask.STABILITY], batch_size=4, run_every_n_epochs=2
        )

        mock_trainer.current_epoch = 0
        assert callback_with_skip._skip(mock_trainer) is False

        mock_trainer.current_epoch = 1
        assert callback_with_skip._skip(mock_trainer) is True

        mock_trainer.current_epoch = 2
        assert callback_with_skip._skip(mock_trainer) is False
