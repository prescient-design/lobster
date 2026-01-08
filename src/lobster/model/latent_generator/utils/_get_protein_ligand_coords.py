import numpy as np
import torch
import tqdm
import glob
import os
from loguru import logger
from latent_generator.io import load_pdb, load_ligand


def process_pdb(file_name_protein, file_name_ligand, save_path):
    # if file exists, return
    if os.path.exists(save_path + file_name_protein.split("/")[-1].split(".")[0] + ".pt") and os.path.exists(
        save_path + file_name_ligand.split("/")[-1].split(".")[0] + ".pt"
    ):
        return None
    try:
        structure_data_protein = load_pdb(file_name_protein, add_batch_dim=False)
        structure_data_ligand = load_ligand(file_name_ligand, add_batch_dim=False)
        # Save the processed data
        save_path_protein = save_path + file_name_protein.split("/")[-1].split(".")[0] + ".pt"
        save_path_ligand = save_path + file_name_ligand.split("/")[-1].split(".")[0] + ".pt"
        torch.save(structure_data_protein, save_path_protein)
        torch.save(structure_data_ligand, save_path_ligand)
        # Clear memory
        del structure_data_protein
        del structure_data_ligand
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return True
    except Exception as e:
        logger.error(f"Error processing {file_name_protein} and {file_name_ligand}: {str(e)}")
        return None


def process_pdb_parallel(file_name_protein, file_name_ligand, save_path):
    try:
        return process_pdb(file_name_protein, file_name_ligand, save_path)
    except Exception as e:
        logger.error(f"Error in parallel processing of {file_name_protein} and {file_name_ligand}: {str(e)}")
        return None


if __name__ == "__main__":
    import concurrent.futures
    import multiprocessing

    pdb_dir = "/data/bucket/lisanza/structures/pdb_bind/1981-2000/"
    # pdb_dir = "/data/bucket/lisanza/structures/pdb_bind/2001-2010/"
    # pdb_dir = "/data/bucket/lisanza/structures/pdb_bind/2011-2020/"
    # pdb_dir = "/data/bucket/lisanza/structures/pdb_bind/2021-2023/"
    # save_path = "/data/bucket/lisanza/structures/pdb_bind/processed/"
    save_path = "/data/bucket/lisanza/structures/pdb_bind/processed_2/"

    os.makedirs(save_path, exist_ok=True)
    # proteins have *_protein.pdb and ligands have *_ligand.sdf
    pdb_paths = glob.glob(pdb_dir + "*/" + "*protein.pdb")
    ligand_paths = glob.glob(pdb_dir + "*/" + "*ligand.sdf")
    # sort pdb_paths and ligand_paths
    pdb_paths.sort()
    ligand_paths.sort()
    # zip pdb_paths and ligand_paths
    pdb_paths = list(zip(pdb_paths, ligand_paths))
    # shuffle pdb_paths
    np.random.shuffle(pdb_paths)
    logger.info(f"Number of pdb_paths: {len(pdb_paths)}")

    # Calculate optimal number of workers based on available memory
    # Use 1/4 of available CPU cores to avoid memory issues
    num_workers = max(1, multiprocessing.cpu_count() // 4)
    logger.info(f"Using {num_workers} workers")

    logger.info(f"Processing {len(pdb_paths)} pdb files in parallel")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_pdb_parallel, file_name_protein, file_name_ligand, save_path)
            for file_name_protein, file_name_ligand in pdb_paths
        ]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
