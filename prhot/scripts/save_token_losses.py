from typing import Dict, List, Any
import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.functional import cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer

from lobster.datasets import FASTADataset 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fasta_file",
        type=str,
        default="/data/lux70/data/uniref90/partial.fasta",
    )
    parser.add_argument(
        "--offset_array_path",
        type=str,
        default="/data/lux70/data/uniref90/partial.fasta.offsets.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/lux70/data/uniref90/token_losses",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/RITA_s",
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
    return parser.parse_args()


def setup(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_model(model_name: str = "lightonai/RITA_xl", max_length: int = 512) -> torch.nn.Module:
    """Load the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("model_name")
    tokenizer.pad_token = "<PAD>"
    tokenizer.pad_token_id = tokenizer.vocab['<PAD>']
    tokenizer.max_length = max_length

    model = model.eval()
    model = model.compile()
    return model, tokenizer


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def compute_loss(batch, model, tokenizer, max_length) -> List[Dict[str, Any]]:
    sequences, headers = batch
    device = get_model_device(model)

    sequences = [s[:max_length] for s in sequences]
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=False)
    input_ids = inputs['input_ids'].to(device)
    attn_mask = inputs['attention_mask'].to(device)
    N, L = input_ids.shape[0], input_ids.shape[1] - 1   # remove EOS token

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )

    targets = input_ids[:, 1:].reshape(-1)
    logits = output['logits']
    logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    per_token_loss = cross_entropy(logits, targets, reduction="none")
    per_token_loss = per_token_loss.reshape(-1, L).half()  # store as float16.

    processed = [
        {
            "sequence": sequences[i],
            "header": headers[i],
            "per_token_loss": per_token_loss[i, :min(len(sequences[i], max_length))].cpu().tolist(),
        }
        for i in range(len(sequences))
    ]
    return processed


def main(args, rank, world_size):
    setup(rank, world_size)

    offset_array = np.load(args.offset_array_path)
    assert len(offset_array.shape) == 2
    assert offset_array.shape[0] == 2
    
    # Partition data for this GPU
    per_gpu_size = len(offset_array) // world_size
    start_idx = rank * per_gpu_size
    end_idx = start_idx + per_gpu_size if rank < world_size - 1 else len(offset_array)
    local_offsets = offset_array[:, start_idx:end_idx]
    
    # Create dataset and dataloader for this GPU
    dataset = FASTADataset(
        root=args.fasta_file,
        offset_file=local_offsets,
        use_text_descriptions=True
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler
    )
    
    # Create model and wrap with DDP
    model, tokenizer = load_model(args.model_name, args.max_length)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    
    # Inference loop
    results_tmp_list = []
    cur_shard_num = []
    cur_num_in_shard = []
    model_name = args.model_name.replace("/", "_")

    for batch in dataloader:
        with torch.no_grad():
            outputs = compute_loss(batch, ddp_model, tokenizer, args.max_length)
            results_tmp_list.extend(outputs)
            cur_num_in_shard += len(outputs)

            if cur_num_in_shard >= args.max_num_per_shard:
                output_file = f"{args.output_dir}/{model_name}/rank_{rank:02}_shard_{cur_shard_num:06}.parquet"
                pd.DataFrame(results_tmp_list).to_parquet(output_file, engine='pyarrow', index=False)
                print(f"Saved shard {cur_shard_num} to {output_file}")

                cur_shard_num += 1
                cur_num_in_shard = 0
                results_tmp_list = []
            
            else:
                print(f"Processed {cur_num_in_shard} sequences in shard {cur_shard_num}")


if __name__ == "__main__":
    args = get_args()
    world_size = torch.cuda.device_count()
    rank = int(os.environ["LOCAL_RANK"])

    main(args, rank, world_size)
