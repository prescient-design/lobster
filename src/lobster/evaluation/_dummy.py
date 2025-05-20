import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DummyModel(nn.Module):
    """Dummy model that returns random embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the embeddings to generate
    """

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, **inputs):
        """Return random embeddings.

        Returns
        -------
        torch.Tensor
            Random embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]
        return torch.randn(batch_size, seq_len, self.embedding_dim)

    def embed(self, inputs, aggregate: bool = True):
        """Generate embeddings with option to aggregate over sequence length.

        Parameters
        ----------
        inputs : dict
            Dictionary containing input_ids and attention_mask
        aggregate : bool
            Whether to average pool over sequence length

        Returns
        -------
        torch.Tensor
            Embeddings of shape [batch_size, embedding_dim] if aggregate=True
            or [batch_size, seq_len, embedding_dim] if aggregate=False
        """
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]
        embeddings = torch.randn(batch_size, seq_len, self.embedding_dim)

        if aggregate:
            # Mean pool over sequence length
            return embeddings.mean(dim=1)
        return embeddings

    def tokens_to_latents(self, **inputs):
        """Return random token-level embeddings.

        Returns
        -------
        torch.Tensor
            Random embeddings of shape [batch_size * seq_len, embedding_dim]
        """
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]
        return torch.randn(batch_size * seq_len, self.embedding_dim)

    def embed_single_batch(self, batch):
        """Embed a single batch.

        Parameters
        ----------
        batch : dict | tuple
            Batch containing input data

        Returns
        -------
        torch.Tensor
            Embeddings of shape [batch_size, embedding_dim]
        """
        if isinstance(batch, tuple) and len(batch) == 2:
            x, _ = batch
            batch_size = 8 if not hasattr(x, "shape") else x.shape[0]

            # Make sure batch dictionary has the expected format for UmapVisualizationCallback
            if isinstance(x, dict) and "dataset" in x:
                # The standard callback expects 'batch' itself to have the group_by key
                batch = {**x}
        else:
            batch_size = 8  # Arbitrary batch size

        return torch.randn(batch_size, self.embedding_dim)


class DummyLightningModule(L.LightningModule):
    """Dummy Lightning module for evaluation.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the embeddings to generate
    """

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.model = DummyModel(embedding_dim)
        self.tokenizer_transforms = {
            "amino_acid": lambda x: {"input_ids": torch.randint(0, 20, (8, 128)), "attention_mask": torch.ones(8, 128)},
            "nucleotide": lambda x: {"input_ids": torch.randint(0, 5, (8, 128)), "attention_mask": torch.ones(8, 128)},
            "SMILES": lambda x: {"input_ids": torch.randint(0, 50, (8, 128)), "attention_mask": torch.ones(8, 128)},
        }

    def embed_single_batch(self, batch):
        """Embed a single batch.

        Parameters
        ----------
        batch : dict | tuple
            Batch containing input data

        Returns
        -------
        torch.Tensor
            Embeddings of shape [batch_size, embedding_dim]
        """
        return self.model.embed_single_batch(batch)

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device


class DummyDataset(Dataset):
    """Dummy dataset that returns random data with dataset grouping information.

    Returns tuples of (input_dict, label) that match the expected format for callbacks.
    """

    def __init__(self, num_samples=32, seq_len=128):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.groups = ["group1", "group2"] * (num_samples // 2)
        if num_samples % 2 == 1:
            self.groups.append("group1")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create input tensor
        input_ids = torch.randint(0, 100, (self.seq_len,))
        attention_mask = torch.ones_like(input_ids)

        # Return dict for x and tensor for y
        x = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dataset": self.groups[idx],  # Add dataset field for UMAP grouping
        }

        y = torch.randint(0, 2, (1,)).item()

        return x, y


class DummyLightningDataModule(L.LightningDataModule):
    """Dummy Lightning data module for evaluation."""

    def __init__(self, num_samples=32, seq_len=128):
        super().__init__()
        self.dataset = DummyDataset(num_samples, seq_len)

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, num_workers=4)
