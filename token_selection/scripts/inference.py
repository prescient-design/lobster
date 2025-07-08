import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ShardedParquetDataset(Dataset):
    def __init__(self, 
                parquet_dir, 
                percentile_threshold=90,
                loss_threshold=None,
                stats_file=None,
                rank=None,
                world_size=None):
        """
        Distributed dataset for sharded parquet files.
        
        Args:
            parquet_dir: Directory containing parquet shards
            percentile_threshold: Only include tokens with loss below this percentile
            loss_threshold: Optional explicit loss threshold (if pre-computed)
            stats_file: Path to pre-computed statistics file
            rank: Process rank in distributed training
            world_size: Total number of processes
        """
        self.parquet_dir = parquet_dir
        self.percentile_threshold = percentile_threshold
        
        # Get list of all shard files
        self.shard_files = sorted(glob.glob(f"{parquet_dir}/partition_id=*/part-*.parquet"))
        
        # If running distributed, only use shards for this rank
        if rank is not None and world_size is not None:
            # Distribute shards across workers
            self.shard_files = [
                f for i, f in enumerate(self.shard_files) 
                if i % world_size == rank
            ]
        
        # Set loss threshold either from argument or by loading stats
        if loss_threshold is not None:
            self.loss_threshold = loss_threshold
        elif stats_file and os.path.exists(stats_file):
            # Load pre-computed statistics
            import json
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            self.loss_threshold = stats['percentiles'][str(percentile_threshold)]
        else:
            # Calculate threshold (ideally, this is pre-computed)
            self.loss_threshold = self._calculate_percentile()
            
        # Load sequence metadata from all assigned shards
        self.sequence_data = self._load_sequence_metadata()
        
    def _calculate_percentile(self):
        """Calculate percentile threshold from samples."""
        # Only calculate on rank 0 and broadcast if distributed
        if dist.is_initialized() and dist.get_rank() != 0:
            # Non-root processes wait for result
            threshold = torch.zeros(1, dtype=torch.float32).cuda()
            dist.broadcast(threshold, 0)
            return threshold.item()
            
        # Root process (or non-distributed) calculates
        print(f"Calculating {self.percentile_threshold}th percentile threshold...")
        samples = []
        
        # Sample from each shard
        for shard in self.shard_files[:10]:  # Limit to 10 shards for efficiency
            df = pd.read_parquet(shard, columns=['loss'])
            # Take a sample proportional to size
            sample_size = min(10000, len(df))
            if sample_size > 0:
                samples.append(df.sample(sample_size)['loss'].values)
                
        # Calculate threshold from samples
        if samples:
            all_samples = np.concatenate(samples)
            threshold = float(np.percentile(all_samples, self.percentile_threshold))
        else:
            threshold = float('inf')  # No samples available
            
        # Broadcast result if distributed
        if dist.is_initialized():
            threshold_tensor = torch.tensor([threshold], dtype=torch.float32).cuda()
            dist.broadcast(threshold_tensor, 0)
            threshold = threshold_tensor.item()
            
        print(f"Using loss threshold: {threshold}")
        return threshold
    
    def _load_sequence_metadata(self):
        """Load sequence metadata from assigned shards."""
        sequences = []
        
        for shard_file in self.shard_files:
            # Read just sequence metadata for efficiency
            try:
                # Group by sequence_id and get sizes
                df = pd.read_parquet(
                    shard_file, 
                    columns=['sequence_id', 'position']
                )
                seq_info = df.groupby('sequence_id').agg({'position': 'max'})
                
                for seq_id, max_pos in seq_info.itertuples():
                    sequences.append({
                        'sequence_id': seq_id,
                        'length': max_pos + 1,  # Convert to length
                        'shard_file': shard_file
                    })
            except Exception as e:
                print(f"Error loading metadata from {shard_file}: {e}")
                
        return sequences
        
    def __len__(self):
        return len(self.sequence_data)
        
    def __getitem__(self, idx):
        """Get a filtered sequence by index."""
        seq_info = self.sequence_data[idx]
        seq_id = seq_info['sequence_id']
        shard_file = seq_info['shard_file']
        
        # Read this sequence with filtering
        try:
            # Use PyArrow filter pushdown for efficiency
            df = pd.read_parquet(
                shard_file,
                filters=[
                    ('sequence_id', '=', seq_id),
                    ('loss', '<=', self.loss_threshold)
                ]
            )
            
            # Sort by position to maintain sequence order
            if not df.empty:
                df = df.sort_values('position')
                
                return {
                    'sequence_id': seq_id,
                    'tokens': df['token'].values,
                    'positions': df['position'].values,
                    'losses': df['loss'].values
                }
            else:
                # No tokens passed the filter
                return {
                    'sequence_id': seq_id,
                    'tokens': np.array([], dtype=np.int64),
                    'positions': np.array([], dtype=np.int64),
                    'losses': np.array([], dtype=np.float32)
                }
                
        except Exception as e:
            print(f"Error loading sequence {seq_id}: {e}")
            # Return empty sequence on error
            return {
                'sequence_id': seq_id,
                'tokens': np.array([], dtype=np.int64),
                'positions': np.array([], dtype=np.int64),
                'losses': np.array([], dtype=np.float32)
            }


def collate_variable_length_sequences(batch):
    """Custom collate function for variable-length sequences."""
    # Filter out empty sequences
    non_empty = [b for b in batch if len(b['tokens']) > 0]
    
    if not non_empty:
        # All sequences were empty after filtering
        return {
            'sequence_ids': [],
            'tokens': torch.zeros(0, dtype=torch.int64),
            'positions': torch.zeros(0, dtype=torch.int64),
            'losses': torch.zeros(0, dtype=torch.float32),
            'batch_indices': torch.zeros(0, dtype=torch.int64)
        }
    
    # Gather data
    sequence_ids = [b['sequence_id'] for b in non_empty]
    tokens_list = [torch.tensor(b['tokens'], dtype=torch.int64) for b in non_empty]
    positions_list = [torch.tensor(b['positions'], dtype=torch.int64) for b in non_empty]
    losses_list = [torch.tensor(b['losses'], dtype=torch.float32) for b in non_empty]
    
    # Create batch indices for reconstructing sequences later
    batch_sizes = [len(t) for t in tokens_list]
    batch_indices = torch.cat([
        torch.full((size,), i, dtype=torch.int64)
        for i, size in enumerate(batch_sizes)
    ])
    
    # Concatenate all tokens
    tokens = torch.cat(tokens_list)
    positions = torch.cat(positions_list)
    losses = torch.cat(losses_list)
    
    return {
        'sequence_ids': sequence_ids,
        'tokens': tokens,
        'positions': positions,
        'losses': losses,
        'batch_indices': batch_indices
    }


def setup_distributed():
    """Initialize distributed training environment."""
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU-only
        init_method='env://'
    )
    
    # Get global rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    return rank, world_size

def create_distributed_dataloader(parquet_dir, percentile_threshold=90, 
                                 batch_size=32, num_workers=4):
    """Create a distributed dataloader for sharded parquet files."""
    # Setup distributed environment
    rank, world_size = setup_distributed()
    
    # Create dataset with this rank's shards
    dataset = ShardedParquetDataset(
        parquet_dir=parquet_dir,
        percentile_threshold=percentile_threshold,
        stats_file=f"{parquet_dir}/stats.json",
        rank=rank,
        world_size=world_size
    )
    
    # Create distributed sampler to handle partitioning
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_variable_length_sequences,
        pin_memory=True
    )
    
    return dataloader, rank, world_size