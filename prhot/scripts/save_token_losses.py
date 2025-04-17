from typing import Dict, List, Any
from pathlib import Path
from glob import glob
import os

from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer

from lobster.data import FastaLightningDataModule


class BuildParquetFile:
    """From a FASTA file and a CLM HuggingFace Model, get the
    per-token loss and save it as Parquet shards."""

    def __init__(
        self,
        fasta_file: str,
        output_dir: str,
        *,
        fasta_offset_file: Optional[str] = None,
        filename_prefix: str = "",  # for identifying multiple shards
        model_name: str = "lightonai/RITA_xl",
        batch_size: int = 128,
        max_length: int = 512,
        max_num_per_shard: int = 100_000,
        device: str | torch.device = "cuda",
        cur_num_in_shard: int = 0,  # for checkpointing
        cur_shard_num: int = 0, # for checkpointing
    ):
        self.fasta_file = fasta_file
        self.fasta_offset_file = fasta_offset_file
        self.filename_prefix = filename_prefix
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
        self.model.compile()

        self.cur_shard_num = cur_shard_num 
        self.cur_num_in_shard = cur_num_in_shard
        self.results_tmp_list = []  # will store rows here before saving to parquet

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
        self.tokenizer.max_length = self.max_length

    def load_dataloader(self):
        """Load the dataset and dataloader."""
        self.dm = FastaLightningDataModule(
            path_to_fasta=self.fasta_file,
            transform_fn=lambda x: x,
            mlm=False,
            shuffle=False,
            batch_size=self.batch_size
        )

        self.dm.setup("predict")
        self.dataloader = self.dm.predict_dataloader()

    def compute_loss(self, batch) -> List[Dict[str, Any]]:
        sequences, headers = batch
        sequences = [s[:self.max_length] for s in sequences]
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=False)
        input_ids = inputs['input_ids'].to(self.device)
        attn_mask = inputs['attention_mask'].to(self.device)
        N, L = input_ids.shape[0], input_ids.shape[1] - 1   # remove EOS token

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )

        targets = input_ids[:, 1:].reshape(-1)
        logits = output['logits']
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        per_token_loss = cross_entropy(logits, targets, reduction="none")
        per_token_loss = per_token_loss.reshape(-1, L).half()  # store as float16.
        print(per_token_loss.shape, logits.shape, input_ids.shape)

        processed = []

        for i in range(len(sequences)):
            seqlen = min(len(sequences[i]), self.max_length)
            row = {
                "sequence": sequences[i],
                "header": headers[i],
                "per_token_loss": per_token_loss[i, :seqlen].cpu().tolist(),
            }
            processed.append(row)

        return processed
   
    def save_to_parquet(self):
        """Save the sequences and per-token loss to a Parquet file."""

        def _save_and_reset(result: pd.DataFrame):
            output_file = f"{self.output_dir}/shard_{self.cur_shard_num:06}.parquet"
            result.to_parquet(output_file, engine='pyarrow', index=False)
            print("Save result to", output_file)
            self.cur_shard_num += 1
            self.cur_num_in_shard = 0
            self.results_tmp_list = []

        # dataset should only have the 'train' split
        try:
            for batch in tqdm(self.dm.dataloader()):
                batch_results = self.compute_loss(batch)
                self.results_tmp_list.extend(batch_results) 
                if self.cur_num_in_shard <= self.max_num_per_shard:
                    self.cur_num_in_shard += len(batch_results)
                else:
                    result = pd.DataFrame(self.results_tmp_list)
                    _save_and_reset(result)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: saving current shard")
            self._save_cur_spot_in_iter()
            _save_and_reset(pd.DataFrame(self.results_tmp_list))
            print(f"Saved current shard to {self.output_dir}/shard_{self.cur_shard_num:06}.parquet")


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

    if args.offset_file_grep_pattern is None:
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
    
    else:
        offset_files_paths = glob.glob(args.offset_file_grep_pattern)
        if len(offset_files_paths) == 0:
            raise ValueError(f"No offset files found with pattern {args.offset_file_grep_pattern}")

        offset_files_paths.sort()
        builders = []

        for i, offset_path in enumerate(offset_files_paths):
            # adhoc, only for the uniref50.fasta.split_offsets_00.npy filename pattern
            partition = int(offset_files_paths.split("/")[-1].split(".")[-2].split("_")[-1])
            builder = BuildParquetFile(
                fasta_file=args.fasta_file,
                output_dir=args.output_dir,
                fasta_offset_file=offset_path,
                model_name=args.model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_num_per_shard=args.max_num_per_shard,
                device=args.device,
                cur_num_in_shard=args.cur_num_in_shard,
                cur_shard_num=args.cur_shard_num,
                filename_prefix=f"partition{partition:02}_",
            )
            builders.append(builder)


