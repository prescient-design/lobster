import logging

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lobster.constants import Modality, Split
from lobster.datasets import UmeStreamingDataset
from lobster.model import Ume
from lobster.transforms import AminoAcidToNucleotideAndSmilesTransform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyProteinDataset(UmeStreamingDataset):
    """
    Dummy protein dataset for dummy training.
    """

    SPLITS = {Split.TRAIN: "/Users/zadorozk/Desktop/lobster/examples/ume/data"}
    MODALITY = Modality.AMINO_ACID
    SEQUENCE_KEY = "sequence"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def collate_with_modality(
    batch: list[dict[str, Tensor | Modality]], return_sequence: bool = True, return_dataset: bool = True
) -> dict[str, Tensor | list[Modality] | list[str]]:
    modalities = [item.get("modality") for item in batch]
    sequences = [item.get("sequence", "") for item in batch]
    dataset = [item.get("dataset", "") for item in batch]

    # Only collate input_ids and attention_mask
    tensor_batch = [{key: item[key] for key in item if key in ["input_ids", "attention_mask"]} for item in batch]
    tensor_batch = torch.utils.data.default_collate(tensor_batch)

    output = {**tensor_batch, "modality": modalities}

    if return_sequence:
        output = {**output, "sequence": sequences}
    if return_dataset:
        output = {**output, "dataset": dataset}

    return output


def main():
    # Initialize the model
    model = Ume(
        model_name="UME_mini",
        max_length=32,
        lr=1e-4,
        contrastive_loss_weight=0.5,  # Use both MLM and contrastive loss
        use_disco_clip=False,  # Don't use distributed training for this example
        use_flash_attn=False,
    )

    transform = AminoAcidToNucleotideAndSmilesTransform(
        max_input_length=32,
        add_stop_codon=True,
        randomize_smiles=True,
    )

    # Create dataset and dataloader
    dataset = DummyProteinDataset(
        split=Split.TRAIN,
        max_length=32,
        transform_fn=transform,
    )

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_with_modality)
    logger.info(f"Dataset size: {len(dataset)}")

    example_batch = next(iter(dataloader))
    logger.info(f"""
                Example batch:
                
                   input_ids.shape: {example_batch["input_ids"].shape}
                   attention_mask.shape: {example_batch["attention_mask"].shape}
                   modality: {example_batch["modality"]}
                   sequence (single item): {example_batch["sequence"][0]}
                   dataset: {example_batch["dataset"]}
                """)

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",  # Use CPU for this example
        devices=1,
        logger=True,
        max_steps=2,
    )

    # Train for one epoch
    trainer.fit(model, dataloader)

    # Get some embeddings
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batches = model._split_combined_batch(batch)
            embeddings = model.embed(batches[0])
            print(f"Embedding shape: {embeddings.shape}")  # [5, 384]
            print(f"Embedding mean: {embeddings.mean().item():.4f}")  # 0.4664
            print(f"Embedding std: {embeddings.std().item():.4f}")  # 0.8177
            break


if __name__ == "__main__":
    main()
