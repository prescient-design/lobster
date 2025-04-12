from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer

from lobster.data import FastaLightningDataModule

# ds = load_dataset("bloyal/uniref100", streaming=False, cache_dir="/data/lux70/data/uniref100")

class BuildParquetFile:
    """From a FASTA file and a CLM HuggingFace Model, get the
    per-token loss and save it as Parquet shards."""

    def __init__(
        self,
        fasta_file: str,
        output_dir: str,
        model_name: str = "lightonai/RITA_xl",
        batch_size: int = 128,
        max_length: int = 512,
        max_num_per_shard: int = 100_000,
        device: str | torch.device = "cuda",
        cur_num_in_shard: int = 0,  # for checkpointing
        cur_shard_num: int = 0, # for checkpointing
    ):
        self.fasta_file = fasta_file
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_num_per_shard = max_num_per_shard
        self.device = device

        self.load_model()
        self.load_dataloader()

        self.model.eval()
        self.model.to(self.device)

        self.cur_shard_num = cur_shard_num 
        self.cur_num_in_shard = cur_num_in_shard

        if os.path.exists(self.output_dir / "curshard.txt"):
            self._load_cur_spot_in_iter()
        
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)
    
    def _save_cur_spot_in_iter(self):
        with open(self.output_dir / "curshard.txt","w") as f:
            f.write(f"{self.cur_shard_num},{self.cur_num_in_shard}")

    def _load_cur_spot_in_iter(self):
        with open(self.output_dir / "curshard.txt","r") as f:
            self.cur_shard_num, self.cur_num_in_shard = [int(x) for x in f.read().split(",")]

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.model = AutoModelForCausalLM.from_pretrained("lightonai/RITA_xl", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")
        self.tokenizer.pad_token = "<PAD>"
        self.tokenizer.pad_token_id = self.tokenizer.vocab['<PAD>']


    def load_dataloader(self):
        """Load the dataset and dataloader."""
        self.dm = FastaLightningDataModule(
            path_to_fasta=self.fasta_file,
            transform_fn=lambda x: x,
            mlm=False,
            shuffle=False,
            batch_size=self.batch_size
        )
        self.dm.setup("fit")
        self.train_dl = self.dm.train_dataloader()
        self.val_dl = self.dm.val_dataloader()

    def compute_loss(self, batch):
        sequences, headers = batch
        sequences = [s[:self.max_length] for s in sequences]
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        attn_mask = torch.tensor(inputs['attention_mask']).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )

        targets = input_ids[:, 1:].reshape(-1)
        logits = output['logits']
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        per_token_loss = cross_entropy(logits, targets, reduction="none")
        return sequences, headers, per_token_loss.cpu()
   
    def _save_to_parquet(self):
        """Save the sequences and per-token loss to a Parquet file."""
        result = {
            'sequence': [],
            'header': [],
            'per_token_loss': []
        }

        def _save(result):
            output_file = f"{self.output_dir}/shard_{self.cur_shard_num:06}.parquet"
            result = pd.DataFrame(result)
            result.to_parquet(output_file, engine='pyarrow', index=False)
            print("Save result to", output_file)
            self.cur_shard_num += 1
            self.cur_num_in_shard = 0

        # dataset should only have the 'train' split
        for batch in tqdm(self.dm.train_dataloader()):
            try:
                if self.cur_num_in_shard <= self.max_num_per_shard:
                    sequences, headers, per_token_loss = self.compute_loss(batch)
                    result['sequence'].extend(sequences)
                    result['header'].extend(headers)
                    result['per_token_loss'].extend(per_token_loss.tolist())
                    self.cur_num_in_shard += len(sequences)
                else:
                    _save(result)
                
            except Exception as e:
                print(e)
                self._save_cur_spot_in_iter()
                _save(result)
                break

    
    def save_to_parquet(self):
        try:
            self._save_to_parquet()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: saving current shard")
            self._save_cur_spot_in_iter()
            print("Saved current shard")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fasta_file",
        type=str,
        default="/data/lux70/data/uniref90/partial.fasta",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/lux70/data/uniref90/token_losses",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/RITA_xl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_num_per_shard",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--cur_num_in_shard",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cur_shard_num",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    build_cls = BuildParquetFile(
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_num_per_shard=args.max_num_per_shard,
        device=args.device,
        cur_num_in_shard=args.cur_num_in_shard,
        cur_shard_num=args.cur_shard_num,
    )
    build_cls.save_to_parquet()