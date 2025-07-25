import rdkit
from atomic_datasets import QM9
from atomic_datasets.utils.rdkit import is_molecule_sane
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, DataStructs, rdFingerprintGenerator, rdMolAlign, Descriptors, rdMolDescriptors, rdShapeHelpers
import selfies as sf
import numpy as np
import py3Dmol
import pickle
import os
from tqdm import tqdm
import itertools
import polars as pl
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.distance import euclidean, cdist
from utils_qm9 import human_readable_size
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign

data_dir = os.getenv('DATA_DIR') 
qm9_dir = os.path.join(data_dir, 'qm9') #"/data/lawrenh6/qm9"

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_data_lists(dataset):
    # get precomputed data
    smiles_list, selfies_list, fps, confs = [], [], [], []
    for i, graph in tqdm(enumerate(dataset)):
        smiles = dataset[i]['smiles']
        selfies = dataset[i]['selfies'] 
        fingerprint = dataset[i]['fingerprint']
        smiles_list.append(smiles)
        selfies_list.append(selfies)
        fps.append(fingerprint)
        confs.append(dataset[0]['nodes']['positions'])
    return smiles_list, selfies_list, fps, confs


def len_of_parquet(path):
    table = pq.read_table(path, columns=[])
    length = table.num_rows
    print(f"Length of file: {length}\n")

    file_size_bytes = os.path.getsize(path)
    print(f"File size: {human_readable_size(file_size_bytes)}")
    return length

class ParquetStreamWriter:
    def __init__(self, path, schema):
        self.path = path
        self.schema = schema
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.row_count = 0

    def write_chunk(self, chunk):
        """
        chunk: list of (item_i, item_j, distance, smiles_i, smiles_j) tuples
        """
        batch = pa.table({
            "item_i": [r[0] for r in chunk],
            "item_j": [r[1] for r in chunk],
            "distance": [r[2] for r in chunk],
            "smiles_i": [r[3] for r in chunk],
            "smiles_j": [r[4] for r in chunk],
        }, schema=self.schema)
        self.writer.write_table(batch)
        self.row_count += batch.num_rows

    def close(self):
        self.writer.close()
        print(f"✅ Finalized {self.row_count} rows written to {self.path}")


def compute_all_pairs_blockwise_to_parquet(
    items,
    pairwise_func,
    out_path,
    smiles_list=None,
    block_size=1000,
    write_every=1_000_000,
    max_blocks=-1,
    func_type="block"
):
    """
    General blockwise pairwise distance computation and Parquet writer.

    Parameters:
        items (List[Any]): List of arbitrary objects to compare
        pairwise_func (Callable): Function that takes either (item_i, block_j) or (block_i, block_j) and returns distances
        out_path (str): Output Parquet file path
        smiles_list (List[str]): List of SMILES strings corresponding to items
        block_size (int): Number of items per block
        write_every (int): Number of (i,j,distance) rows to write per flush
        max_blocks (int): max number of blocks to compute, or -1 for all
        func_type (str): "single_item" if pairwise_func expects (item_i, block_j), "block" if it expects (block_i, block_j)
    """
    n = len(items)
    blocks = int(np.ceil(n / block_size))
    chunk_results = []

    # Update schema to include SMILES columns
    schema = pa.schema([
        ("item_i", pa.int32()),
        ("item_j", pa.int32()),
        ("distance", pa.float32()),
        ("smiles_i", pa.string()),
        ("smiles_j", pa.string())
    ])

    stream_writer = ParquetStreamWriter(out_path, schema)
    block_total = blocks * (blocks + 1) // 2
    block_count = 0
    start_time = time.time()

    exit_all_loops = False
    for bi in range(blocks):
        start_i = bi * block_size
        end_i = min((bi + 1) * block_size, n)
        block_i = items[start_i:end_i]

        for bj in range(bi, blocks):
            start_j = bj * block_size
            end_j = min((bj + 1) * block_size, n)
            block_j = items[start_j:end_j]

            if func_type == "single_item":
                # Original behavior: compute distances for each item in block_i against block_j
                for ii, item_i in enumerate(block_i):
                    # Distance function returns distances from item_i to all items in block_j
                    sims = pairwise_func(item_i, block_j)

                    for jj, dist in enumerate(sims):
                        global_i = start_i + ii
                        global_j = start_j + jj
                        if global_i < global_j:
                            # Get SMILES strings if available
                            smiles_i = smiles_list[global_i] if smiles_list else ""
                            smiles_j = smiles_list[global_j] if smiles_list else ""
                            chunk_results.append((global_i, global_j, dist, smiles_i, smiles_j))                        
            elif func_type == "block":
                # New behavior: compute distances between entire blocks at once
                # pairwise_func should return a matrix of distances (block_i x block_j)
                print('about to call pairwise_func', pairwise_func)
                distance_matrix = pairwise_func(block_i, block_j)
                
                # Process the distance matrix
                for ii in range(len(block_i)):
                    for jj in range(len(block_j)):
                        global_i = start_i + ii
                        global_j = start_j + jj
                        if global_i < global_j:
                            dist = distance_matrix[ii, jj]
                            # Get SMILES strings if available
                            smiles_i = smiles_list[global_i] if smiles_list else ""
                            smiles_j = smiles_list[global_j] if smiles_list else ""
                            chunk_results.append((global_i, global_j, dist, smiles_i, smiles_j))
            else:
                raise ValueError(f"Invalid func_type: {func_type}. Must be 'single_item' or 'block'")
                
            # Check if we need to write chunks (for both function types)
            if len(chunk_results) >= write_every:
                print(f"Writing chunk of {len(chunk_results)} pairs")
                stream_writer.write_chunk(chunk_results)
                chunk_results = []
            

            # Tracking progress
            block_count += 1
            elapsed = time.time() - start_time
            avg_block_time = elapsed / block_count
            remaining = block_total - block_count
            eta = remaining * avg_block_time
            print(f"[Block {block_count}/{block_total}] "
                  f"Elapsed: {elapsed/60:.2f} min | "
                  f"Avg/block: {avg_block_time:.2f} s | "
                  f"ETA: {eta/60:.2f} min")

            if max_blocks > 0 and block_count > max_blocks:
                exit_all_loops = True
                break
        if exit_all_loops:
            break

    if chunk_results:
        print(f"Final write: {len(chunk_results)} remaining pairs flushed.")
        stream_writer.write_chunk(chunk_results)

    stream_writer.close()
    len_of_parquet(out_path)

def view_parquet(path_to_parquet):
    df = pl.read_parquet(path_to_parquet)

    # View first few rows
    print(df.head())

    # Check shape and schema
    print(f"Rows: {df.height}, Columns: {df.width}")
    print(df.schema)

    # Filter example
    #df.filter(pl.col("tanimoto_distance") < 0.1).limit(10)

    return

# row vs block functions

def tanimoto_row_vs_block(fp, block):
    sims = DataStructs.BulkTanimotoSimilarity(fp, block)
    return [1.0 - s for s in sims]  # Convert similarity → distance

# RMSD without alignment

def rmsd_no_align_row_vs_block(conf_i, block):
    """
    conf_i: (N_atoms, 3) numpy array
    block: list of (N_atoms, 3) numpy arrays
    Returns list of RMSD (no alignment) between conf_i and each in block
    """
    A = conf_i
    B = np.stack(block)  # (len(block), N_atoms, 3)
    rmsds = np.sqrt(np.mean(np.sum((A[None] - B)**2, axis=-1), axis=-1))  # (len(block),)
    return rmsds.tolist()

# RMSD with alignment

def rmsd_with_align_row_vs_block(mol_i, block):
    """
    mol_i: RDKit molecule with a conformer
    block: list of RDKit molecules with conformers
    Returns list of RMSDs after optimal alignment
    """
    rmsds = []
    for mol_j in block:
        rmsd = rdMolAlign.GetBestRMS(mol_i, mol_j)
        rmsds.append(rmsd)
    return rmsds

# descriptor distance

def descriptor_row_vs_block(mol_i, block, desc_funcs=None):
    if desc_funcs is None:
        desc_funcs = [f for _, f in Descriptors.descList]
    desc_i = np.array([f(mol_i) for f in desc_funcs])
    desc_i = np.nan_to_num(desc_i)
    distances = []
    for mol_j in block:
        desc_j = np.array([f(mol_j) for f in desc_funcs])
        desc_j = np.nan_to_num(desc_j)
        distances.append(euclidean(desc_i, desc_j))
    return distances

# USR distance

def usr_row_vs_block(mol_i, block):
    """
    Returns 1 - USRScore(mol_i, mol_j) for all j in block
    """
    return [1.0 - rdMolDescriptors.GetUSRScore(mol_i, mol_j) for mol_j in block]

# shape tanimoto distance

def shape_tanimoto_row_vs_block(mol_i, block):
    """
    Returns ShapeTanimotoDist(mol_i, mol_j) for each mol_j in block
    """
    return [rdShapeHelpers.ShapeTanimotoDist(mol_i, mol_j) for mol_j in block]

# Example block-wise functions for the new func_type="block" option

def tanimoto_block_vs_block(block_i, block_j):
    """
    Efficient tensorized Tanimoto distance computation between two blocks of fingerprints.
    Uses RDKit's BulkTanimotoSimilarity for optimal performance.
    
    Args:
        block_i: list of fingerprints
        block_j: list of fingerprints
    
    Returns:
        numpy array of shape (len(block_i), len(block_j)) with distances
    """
    # Use efficient bulk computation for each fingerprint in block_i against block_j
    similarities = np.zeros((len(block_i), len(block_j)))
    
    for i, fp_i in enumerate(block_i):
        # Use BulkTanimotoSimilarity for efficient computation of one vs many
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, block_j)
        similarities[i, :] = sims
    
    # Convert similarity to distance
    distances = 1.0 - similarities
    return distances

def rmsd_no_align_block_vs_block(block_i, block_j):
    """
    Compute RMSD distances between two blocks of conformations without alignment.
    
    Args:
        block_i: list of (N_atoms, 3) numpy arrays
        block_j: list of (N_atoms, 3) numpy arrays
    
    Returns:
        numpy array of shape (len(block_i), len(block_j)) with distances
    """
    # Stack all conformations
    confs_i = np.stack(block_i)  # (len(block_i), N_atoms, 3)
    confs_j = np.stack(block_j)  # (len(block_j), N_atoms, 3)
    
    # Compute pairwise distances efficiently using broadcasting
    # (len(block_i), 1, N_atoms, 3) - (1, len(block_j), N_atoms, 3)
    diff = confs_i[:, None, :, :] - confs_j[None, :, :, :]
    distances = np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=-1))  # (len(block_i), len(block_j))
    
    return distances

def euclidean_block_vs_block(block_i, block_j):
    """
    Compute Euclidean distances between two blocks of feature vectors.
    
    Args:
        block_i: list of feature vectors (numpy arrays or lists)
        block_j: list of feature vectors (numpy arrays or lists)
    
    Returns:
        numpy array of shape (len(block_i), len(block_j)) with distances
    """
    # Convert to numpy arrays
    features_i = np.array(block_i)
    features_j = np.array(block_j)
    
    # Compute pairwise Euclidean distances
    distances = cdist(features_i, features_j, metric='euclidean')
    
    return distances

# bad if no matches....if there's no match, just pick the closest atom?
def typed_chamfer_distance(p1, t1, p2, t2, penalty=100):
    """
    Atom-type–aware Chamfer distance.
    p1, p2: (N, 3) and (M, 3) tensors (coordinates)
    t1, t2: (N,) and (M,) tensors (atom types, same dtype and domain)
    """
    unique_types = np.unique(np.concatenate([t1, t2]))
    total = 0.0
    count = 0

    for atom_type in unique_types:
        mask1 = torch.tensor(t1 == atom_type)
        mask2 = torch.tensor(t2 == atom_type)

        if mask1.any() and mask2.any():
            coords1 = p1[mask1]
            coords2 = p2[mask2]

            dists = torch.cdist(coords1.unsqueeze(0).to(device), coords2.unsqueeze(0).to(device), p=2).squeeze(0)
            total += dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
            count += 1
        else: # pick the closest atom regardless of type    
            dists = torch.cdist(p1.unsqueeze(0).to(device), p2.unsqueeze(0).to(device), p=2).squeeze(0)
            total += penalty * (dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean())
            count += 1

    return total / count if count > 0 else torch.tensor(float('inf'))


def typed_chamfer_block_vs_block(block_i, block_j):
    # self-distances get computed, but not added to the parquet file
    """
    Atom-type–aware Chamfer distance between two blocks of conformations.
    block_i, block_j: list of (N, 3) tensors (coordinates) and (N,) tensors (atom types)
    
    Returns:
        numpy array of shape (len(block_i), len(block_j)) with distances
    """
    distances = np.zeros((len(block_i), len(block_j)))
    print('update')
    
    for i, (p1, t1) in enumerate(block_i):
        for j, (p2, t2) in enumerate(block_j):
            distances[i, j] = float(typed_chamfer_distance(p1, t1, p2, t2))
    
    return distances

def generic_to_block_to_block(single_distance_func, symmetric=True, fill_diagonal=np.nan):
    """
    Returns a function that computes a distance matrix between two blocks using the provided single-element distance function.
    Avoids redundant computation if symmetric=True and block_i is block_j (i.e., only computes upper triangle and mirrors).

    Args:
        single_distance_func: function taking (elem_i, elem_j) and returning a distance (float)
        symmetric (bool): If True, assumes d(i, j) == d(j, i) and avoids redundant computation when block_i is block_j
        fill_diagonal: Value to fill the diagonal with (default: np.nan)
    Returns:
        block_distance_func(block_i, block_j): returns a (len(block_i), len(block_j)) numpy array
    """
    def block_distance_func(block_i, block_j):
        n_i = len(block_i)
        n_j = len(block_j)
        distances = np.zeros((n_i, n_j), dtype=float)
        
        if symmetric and block_i is block_j:
            # Only compute upper triangle (including diagonal)
            for i in range(n_i):
                for j in range(i, n_j):
                    if i == j:
                        distances[i, j] = fill_diagonal
                    else:
                        d = single_distance_func(block_i[i], block_j[j])
                        distances[i, j] = d
                        distances[j, i] = d
        else:
            for i in range(n_i):
                for j in range(n_j):
                    if symmetric and i == j and block_i is block_j:
                        distances[i, j] = fill_diagonal
                    else:
                        distances[i, j] = single_distance_func(block_i[i], block_j[j])
        return distances
    return block_distance_func

def compute_filtered_pairs_blockwise_to_parquet(
    items,
    pairwise_func,
    out_path,
    distance_threshold,
    smiles_list=None,
    block_size=1000,
    write_every=1_000_000,
    max_blocks=-1,
    func_type="single_item",
    verbose=True
):
    """
    Compute all pairwise distances but only store pairs below a distance threshold.
    
    This function is similar to compute_all_pairs_blockwise_to_parquet but filters
    results to only include pairs where distance < distance_threshold, which is
    useful for large datasets where storing all pairs would be impractical.

    Parameters:
        items (List[Any]): List of arbitrary objects to compare
        pairwise_func (Callable): Function that takes either (item_i, block_j) or (block_i, block_j) and returns distances
        out_path (str): Output Parquet file path
        distance_threshold (float): Only store pairs with distance < this threshold
        smiles_list (List[str]): List of SMILES strings corresponding to items
        block_size (int): Number of items per block
        write_every (int): Number of (i,j,distance) rows to write per flush
        max_blocks (int): max number of blocks to compute, or -1 for all
        func_type (str): "single_item" if pairwise_func expects (item_i, block_j), "block" if it expects (block_i, block_j)
        verbose (bool): Whether to print progress information
    """
    n = len(items)
    blocks = int(np.ceil(n / block_size))
    chunk_results = []
    total_pairs_computed = 0
    total_pairs_stored = 0

    # Update schema to include SMILES columns
    schema = pa.schema([
        ("item_i", pa.int32()),
        ("item_j", pa.int32()),
        ("distance", pa.float32()),
        ("smiles_i", pa.string()),
        ("smiles_j", pa.string())
    ])

    stream_writer = ParquetStreamWriter(out_path, schema)
    block_total = blocks * (blocks + 1) // 2
    block_count = 0
    start_time = time.time()

    exit_all_loops = False
    for bi in range(blocks):
        start_i = bi * block_size
        end_i = min((bi + 1) * block_size, n)
        block_i = items[start_i:end_i]

        for bj in range(bi, blocks):
            start_j = bj * block_size
            end_j = min((bj + 1) * block_size, n)
            block_j = items[start_j:end_j]

            if func_type == "single_item":
                # Original behavior: compute distances for each item in block_i against block_j
                for ii, item_i in enumerate(block_i):
                    # Distance function returns distances from item_i to all items in block_j
                    sims = pairwise_func(item_i, block_j)

                    for jj, dist in enumerate(sims):
                        global_i = start_i + ii
                        global_j = start_j + jj
                        if global_i < global_j:
                            total_pairs_computed += 1
                            # Only store pairs below the threshold
                            if dist < distance_threshold:
                                # Get SMILES strings if available
                                smiles_i = smiles_list[global_i] if smiles_list else ""
                                smiles_j = smiles_list[global_j] if smiles_list else ""
                                chunk_results.append((global_i, global_j, dist, smiles_i, smiles_j))
                                total_pairs_stored += 1
                        
            elif func_type == "block":
                # New behavior: compute distances between entire blocks at once
                # pairwise_func should return a matrix of distances (block_i x block_j)
                if verbose:
                    print('about to call pairwise_func', pairwise_func)
                distance_matrix = pairwise_func(block_i, block_j)
                
                # Process the distance matrix
                for ii in range(len(block_i)):
                    for jj in range(len(block_j)):
                        global_i = start_i + ii
                        global_j = start_j + jj
                        if global_i < global_j:
                            dist = distance_matrix[ii, jj]
                            total_pairs_computed += 1
                            # Only store pairs below the threshold
                            if dist < distance_threshold:
                                # Get SMILES strings if available
                                smiles_i = smiles_list[global_i] if smiles_list else ""
                                smiles_j = smiles_list[global_j] if smiles_list else ""
                                chunk_results.append((global_i, global_j, dist, smiles_i, smiles_j))
                                total_pairs_stored += 1
            else:
                raise ValueError(f"Invalid func_type: {func_type}. Must be 'single_item' or 'block'")
                
            # Check if we need to write chunks (for both function types)
            if len(chunk_results) >= write_every:
                if verbose:
                    print(f"Writing chunk of {len(chunk_results)} pairs (threshold: {distance_threshold})")
                stream_writer.write_chunk(chunk_results)
                chunk_results = []
            

            # Tracking progress
            block_count += 1
            elapsed = time.time() - start_time
            avg_block_time = elapsed / block_count
            remaining = block_total - block_count
            eta = remaining * avg_block_time
            
            if verbose:
                print(f"[Block {block_count}/{block_total}] "
                      f"Elapsed: {elapsed/60:.2f} min | "
                      f"Avg/block: {avg_block_time:.2f} s | "
                      f"ETA: {eta/60:.2f} min | "
                      f"Computed: {total_pairs_computed} | "
                      f"Stored: {total_pairs_stored} | "
                      f"Filter rate: {total_pairs_stored/max(total_pairs_computed, 1)*100:.1f}%")

            if max_blocks > 0 and block_count > max_blocks:
                exit_all_loops = True
                break
        if exit_all_loops:
            break

    if chunk_results:
        if verbose:
            print(f"Final write: {len(chunk_results)} remaining pairs flushed.")
        stream_writer.write_chunk(chunk_results)

    stream_writer.close()
    
    if verbose:
        print(f"\n Summary:")
        print(f"   Total pairs computed: {total_pairs_computed:,}")
        print(f"   Total pairs stored: {total_pairs_stored:,}")
        print(f"   Filter rate: {total_pairs_stored/max(total_pairs_computed, 1)*100:.1f}%")
        print(f"   Distance threshold: {distance_threshold}")
    
    len_of_parquet(out_path)
    return total_pairs_computed, total_pairs_stored