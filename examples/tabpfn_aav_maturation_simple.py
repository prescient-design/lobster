"""Train TabPFN on AAV maturation data using DataFrameDatasetInMemory.

This is a cleaner version that uses the existing generic parquet dataset class.

Requirements
------------
Install TabPFN optional dependency:
    uv sync --extra tabpfn

S3 Data:
    s3://prescient-spark-collaboration-dev/data/pre-processed/cap0270_maturation/filtered/
"""

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lobster.data import DataFrameDatasetInMemory
from lobster.model import TabPFNProteinModel
from lobster.transforms import ESMTokenizerTransform


def collate_fn(batch):
    """Collate function for batching.
    
    Parameters
    ----------
    batch : list of tuples
        List of (tokenized_dict, label) tuples
        
    Returns
    -------
    dict
        Batched dictionary with input_ids, attention_mask, and labels
    """
    inputs, labels = zip(*batch)
    
    # ESMTokenizerTransform returns BatchEncoding, extract tensors
    input_ids = torch.stack([item["input_ids"].squeeze(0) for item in inputs])
    attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in inputs])
    labels = torch.stack([label for label in labels]).float()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main(
    embedding_model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 32,
    max_length: int = 512,
    tabpfn_n_ensemble: int = 4,
    sequence_column: str = "sequence",
    label_column: str = "AAVLibrary_Log2FC_vs_PlasmidLibrary",
    accelerator: str = "auto",
    devices: int = 1,
):
    """Train TabPFN on AAV maturation data.
    
    Parameters
    ----------
    embedding_model_name : str, default="esm2_t33_650M_UR50D"
        Name of the embedding model to use
    batch_size : int, default=32
        Batch size for training
    max_length : int, default=512
        Maximum sequence length
    tabpfn_n_ensemble : int, default=4
        Number of TabPFN ensemble models
    sequence_column : str, default="sequence"
        Column name containing protein sequences
    label_column : str, default="AAVLibrary_Log2FC_vs_PlasmidLibrary"
        Column name containing labels
    accelerator : str, default="auto"
        Accelerator type (auto, cpu, gpu, cuda, etc.)
    devices : int, default=1
        Number of devices to use
    """
    
    print("=" * 80)
    print("TabPFN + ESM2/PMLM on AAV Maturation Data")
    print("=" * 80)
    
    s3_train_path = "s3://prescient-spark-collaboration-dev/data/pre-processed/cap0270_maturation/filtered/train/"
    s3_val_path = "s3://prescient-spark-collaboration-dev/data/pre-processed/cap0270_maturation/filtered/val/"
    s3_test_path = "s3://prescient-spark-collaboration-dev/data/pre-processed/cap0270_maturation/filtered/test/"
    
    print(f"\nInitializing ESM tokenizer transform for: {embedding_model_name}")
    transform = ESMTokenizerTransform(
        pretrained_model_name_or_path=f"facebook/{embedding_model_name}",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    
    print("\n" + "=" * 80)
    print("Loading data from S3...")
    print("=" * 80)
    sequence_column = "Peptide"
    
    print(f"Train: {s3_train_path}")
    train_df = pd.read_parquet(s3_train_path)
    train_df = train_df.dropna(subset=[sequence_column, label_column])
    print(f"  Loaded {len(train_df):,} training samples")
    
    print(f"Val: {s3_val_path}")
    val_df = pd.read_parquet(s3_val_path)
    val_df = val_df.dropna(subset=[sequence_column, label_column])
    print(f"  Loaded {len(val_df):,} validation samples")
    
    print(f"Test: {s3_test_path}")
    test_df = pd.read_parquet(s3_test_path)
    test_df = test_df.dropna(subset=[sequence_column, label_column])
    print(f"  Loaded {len(test_df):,} test samples")
    
    print(f"\nLabel statistics:")
    print(f"  Train - mean: {train_df[label_column].mean():.3f}, std: {train_df[label_column].std():.3f}")
    print(f"  Val   - mean: {val_df[label_column].mean():.3f}, std: {val_df[label_column].std():.3f}")
    print(f"  Test  - mean: {test_df[label_column].mean():.3f}, std: {test_df[label_column].std():.3f}")
    
    print("\n" + "=" * 80)
    print("Creating datasets with transforms...")
    print("=" * 80)
    
    train_dataset = DataFrameDatasetInMemory(
        data=train_df,
        columns=[sequence_column],
        target_columns=[label_column],
        transform_fn=transform,
    )
    
    val_dataset = DataFrameDatasetInMemory(
        data=val_df,
        columns=[sequence_column],
        target_columns=[label_column],
        transform_fn=transform,
    )
    
    test_dataset = DataFrameDatasetInMemory(
        data=test_df,
        columns=[sequence_column],
        target_columns=[label_column],
        transform_fn=transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    print("\n" + "=" * 80)
    print("Initializing TabPFN model...")
    print("=" * 80)
    
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_type="LobsterPMLM",
        embedding_model_name=embedding_model_name,
        freeze_embeddings=True,
        pooling="mean",
        max_length=max_length,
        tabpfn_n_ensemble=tabpfn_n_ensemble,
    )
    
    print(f"\nModel configuration:")
    print(f"  - Embedding model: {embedding_model_name}")
    print(f"  - Embedding dimension: {model._embedding_dim}")
    print(f"  - TabPFN ensemble size: {tabpfn_n_ensemble}")
    print(f"  - Frozen embeddings: {model._freeze_embeddings}")
    
    print("\n" + "=" * 80)
    print("Training...")
    print("=" * 80)
    
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=True,
        logger=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("Testing...")
    print("=" * 80)
    
    trainer.test(model, test_loader)
    
    print("\n" + "=" * 80)
    print("Making predictions on test set...")
    print("=" * 80)
    
    predictions = trainer.predict(model, test_loader)
    
    all_preds = []
    for pred_dict in predictions:
        all_preds.extend(pred_dict["predictions"])
    
    print(f"\nPredictions shape: {len(all_preds)}")
    print(f"Predictions range: [{min(all_preds):.3f}, {max(all_preds):.3f}]")
    
    results_df = pd.DataFrame({
        "true_values": test_df[label_column].values,
        "predictions": all_preds,
    })
    
    output_path = "aav_maturation_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TabPFN on AAV maturation data")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="Embedding model name (default: esm2_t33_650M_UR50D)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length (default: 512)")
    parser.add_argument("--ensemble-size", type=int, default=4, help="TabPFN ensemble size (default: 4)")
    parser.add_argument(
        "--sequence-column",
        type=str,
        default="sequence",
        help="Sequence column name (default: sequence)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="AAVLibrary_Log2FC_vs_PlasmidLibrary",
        help="Label column name (default: AAVLibrary_Log2FC_vs_PlasmidLibrary)",
    )
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (default: auto)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (default: 1)")
    
    args = parser.parse_args()
    
    main(
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tabpfn_n_ensemble=args.ensemble_size,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        accelerator=args.accelerator,
        devices=args.devices,
    )

