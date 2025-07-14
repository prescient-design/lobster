import rdkit
from atomic_datasets import QM9

from rdkit import Chem
from rdkit.Chem import rdchem
import selfies as sf
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
import pickle
import os
from tqdm import tqdm
from atomic_datasets.utils.rdkit import is_molecule_sane
from distances import compute_filtered_pairs_blockwise_to_parquet

from distances import *

from distances import compute_all_pairs_blockwise_to_parquet, tanimoto_row_vs_block
from distances import view_parquet
from distances import *
from utils_qm9 import * #is_valency_ok, visualize_mol, visualize_mol_pairs_grid

import itertools
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import polars as pl
#from rdkit.Chem.rdmolfiles import SDMolSupplier

data_dir = os.getenv('DATA_DIR') 
qm9_dir = os.path.join(data_dir, 'qm9') #"/data/lawrenh6/qm9"

load_qm9 = True
overwrite_file = False
pickle_file = os.path.join(qm9_dir, 'qm9_fps_2048_4096.pkl') 

if load_qm9:
    
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    smiles_list, selfies_list, fps_2048, fps_4096 = [], [], [], []
    for i, graph in tqdm(enumerate(dataset)):
        smiles = dataset[i]['smiles']
        selfies = dataset[i]['selfies'] 
        fingerprint_2048 = dataset[i]['fingerprint_2048'] 
        fingerprint_4096 = dataset[i]['fingerprint_4096']
        smiles_list.append(smiles)
        selfies_list.append(selfies)
        #fps.append(fingerprint)

        if overwrite_file:
            with open(pickle_file, 'wb') as f:
                dct = {'dataset': dataset, 'smiles_list': smiles_list, 'selfies_list': selfies_list, 'fps_2048': fps_2048, 'fps_4096': fps_4096}
                pickle.dump(dataset, f)
else:

    dataset = QM9(
        root_dir=qm9_dir,
        check_with_rdkit=True,
    )

    # one time compute fingerprints 

    num_errs = 0
    smiles_list, selfies_list, fps_2048, fps_4096 = [], [], [], []
    mfpgen_4096 = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=4096)
    mfpgen_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    for i, graph in tqdm(enumerate(dataset)):
        mol = graph['properties']['rdkit_mol']
        smiles = Chem.MolToSmiles(mol)
        fingerprint_4096 = mfpgen_4096.GetFingerprint(mol)
        fingerprint_2048 = mfpgen_2048.GetFingerprint(mol)
        try:
            selfies = sf.encoder(smiles)
        except Exception as e:
            val_ok = is_valency_ok(mol)
            is_sane = is_molecule_sane(mol)
            print(f'Error occurred at index {i}, for which valency check is {val_ok} and sanity is {is_sane}: {e}. Saving as None')
            num_errs += 1
            selfies = None
        dataset[i]['smiles'] = smiles
        dataset[i]['selfies'] = selfies
        dataset[i]['fingerprint_2048'] = fingerprint_2048
        dataset[i]['fingerprint_4096'] = fingerprint_4096
        smiles_list.append(smiles)
        selfies_list.append(selfies)
        fps_2048.append(fingerprint_2048)
        fps_4096.append(fingerprint_4096)

    if overwrite_file:
        with open(pickle_file, 'wb') as f:
            dct = {'dataset': dataset, 'smiles_list': smiles_list, 'selfies_list': selfies_list, 'fps_2048': fps_2048, 'fps_4096': fps_4096}
            pickle.dump(dataset, f)

# get a random subset
num_samples=100
rng = np.random.default_rng(42)
inds_for_hist = np.array(range(0, num_samples)) #rng.choice(len(dataset), size=num_samples, replace=False)
parquet_name = f"subset_tanimoto_{}" # "subset_tanimoto"

# for example purposes, compute subset of dataset
sub_dataset = [dataset[i] for i in inds_for_hist]
sub_fps = [datapt['fingerprint_2048'] for datapt in sub_dataset]
sub_pos_and_types = [(torch.tensor(datapt['nodes']['positions']), datapt['nodes']['atom_types']) for datapt in sub_dataset]

# ultimately only save pairs below a certain threshold, as doing all pairs would take too much space

# out_path = f"{parquet_name}.parquet"
# compute_all_pairs_blockwise_to_parquet(
#     items=sub_fps,
#     pairwise_func=tanimoto_row_vs_block, #typed_chamfer_block_vs_block,
#     out_path=out_path, 
#     block_size=1000,
#     write_every=1_000_000,
#     max_blocks=-1,
#     func_type='single_item',
# )


# Then replace the existing compute_all_pairs_blockwise_to_parquet call with:

# Set a distance threshold - only store pairs with Tanimoto distance < 0.3
distance_threshold = 0.3

# Compute all pairs but only store those below the threshold
out_path = f"{parquet_name}_filtered_{distance_threshold}.parquet"

total_computed, total_stored = compute_filtered_pairs_blockwise_to_parquet(
    items=sub_fps,
    pairwise_func=tanimoto_row_vs_block,
    out_path=out_path, 
    distance_threshold=distance_threshold,
    smiles_list=[datapt['smiles'] for datapt in sub_dataset],  # Add SMILES for reference
    block_size=1000,
    write_every=1_000_000,
    max_blocks=-1,
    func_type='single_item',
    verbose=True
)

print(f"Computed {total_computed} total pairs, stored {total_stored} pairs below threshold {distance_threshold} in file {out_path}")