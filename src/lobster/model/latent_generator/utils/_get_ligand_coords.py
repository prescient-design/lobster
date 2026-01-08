import torch
import tqdm
import os
import pandas as pd
import s3fs
from loguru import logger
from latent_generator.io import load_ligand
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
import hashlib
import random
from rdkit import Chem
from rdkit.Chem import rdFreeSASA, Descriptors3D
from upath import UPath


def load_mol_from_s3_sdf(s3_path: str):
    """Load RDKit molecule directly from S3 SDF without downloading"""
    path = UPath(s3_path)

    # Read file content directly into memory
    with path.open("r") as f:
        sdf_content = f.read()

    # Create molecule from SDF content
    mol = Chem.MolFromMolBlock(sdf_content)
    return mol


def calc_rg_sasa(conformer_data):
    """Calculate radius of gyration and SASA from conformer data"""
    s3_path = conformer_data["sdf_filepath"]

    mol = load_mol_from_s3_sdf(s3_path)

    # Calculate RG
    rg = Descriptors3D.RadiusOfGyration(mol)

    # Calculate SASA
    ptable = Chem.GetPeriodicTable()
    radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    sasa = rdFreeSASA.CalcSASA(mol, radii)

    return rg, sasa


def load_geom_parquet_from_s3(s3_path: str) -> pd.DataFrame:
    """
    Load a parquet file from S3 containing GEOM dataset data.

    Args:
        s3_path: S3 path to the parquet file

    Returns:
        pandas DataFrame containing the data
    """
    try:
        logger.info(f"Loading parquet file from: {s3_path}")

        # Use s3fs to read parquet file directly
        fs = s3fs.S3FileSystem()
        df = pd.read_parquet(s3_path, filesystem=fs)

        logger.info(f"Successfully loaded data with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")
        raise


def check_parquet_already_processed(parquet_file, save_path):
    """
    Check if all ligand PT files for a parquet file already exist.

    Args:
        parquet_file: S3 path to the parquet file
        save_path: Directory where PT files are saved

    Returns:
        bool: True if all PT files exist, False otherwise
    """
    try:
        # Load parquet file to get all unique SMILES
        df = load_geom_parquet_from_s3(parquet_file)
        rows = df.to_dict(orient="records")

        # Group by SMILES to get unique SMILES strings
        smiles_groups = defaultdict(list)
        for row in rows:
            smiles_groups[row["smiles"]].append(row)

        unique_smiles = list(smiles_groups.keys())

        # Check if all corresponding PT files exist
        missing_files = 0
        for smiles in unique_smiles:
            smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:8]
            filename = f"ligand_{smiles_hash}.pt"
            output_path = os.path.join(save_path, filename)

            if not os.path.exists(output_path):
                missing_files += 1

        if missing_files == 0:
            logger.info(
                f"All {len(unique_smiles)} PT files for {os.path.basename(parquet_file)} already exist, skipping"
            )
            return True
        else:
            logger.info(
                f"Missing {missing_files}/{len(unique_smiles)} PT files for {os.path.basename(parquet_file)}, processing"
            )
            return False

    except Exception as e:
        logger.error(f"Error checking if parquet file is already processed: {e}")
        return False


def process_single_ligand(row_data, fs):
    """
    Process a single ligand from the parquet data.

    Args:
        row_data: Dictionary containing 'smiles' and 'sdf_path'
        fs: S3 filesystem object

    Returns:
        tuple: (success: bool, ligand_data: dict or None, error: str or None)
    """
    try:
        smiles = row_data["smiles"]
        sdf_path = row_data["sdf_path"]

        # Load ligand data from SDF file
        ligand_data = load_ligand(sdf_path, add_batch_dim=False)

        # Add SMILES information to the data
        ligand_data["smiles"] = smiles
        ligand_data["sdf_filepath"] = sdf_path

        # Calculate RG and SASA properties
        rg, sasa = calc_rg_sasa(ligand_data)
        ligand_data["radius_of_gyration"] = float(rg)
        ligand_data["solvent_accessible_surface_area"] = float(sasa)

        return True, ligand_data, None

    except Exception as e:
        error_msg = f"Error processing ligand {smiles[:50]}...: {str(e)}"
        return False, None, error_msg


def save_conformers_group(smiles, conformers_data, save_path):
    """
    Save all conformers for a given SMILES string to a single PT file.

    Args:
        smiles: SMILES string
        conformers_data: List of ligand data dictionaries for all conformers
        save_path: Directory to save the processed PT file

    Returns:
        tuple: (success: bool, filename: str, error: str or None)
    """
    try:
        # Create a unique filename based on SMILES hash
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:8]
        filename = f"ligand_{smiles_hash}.pt"
        output_path = os.path.join(save_path, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"File {filename} already exists, skipping")
            return True, filename, None

        # Combine all conformers into a single data structure
        combined_data = {"smiles": smiles, "num_conformers": len(conformers_data), "conformers": conformers_data}
        # Save to PT file
        torch.save(combined_data, output_path)

        return True, filename, None

    except Exception as e:
        error_msg = f"Error saving conformers for {smiles[:50]}...: {str(e)}"
        return False, None, error_msg


def process_smiles_group(smiles, conformer_rows, fs, save_path):
    """
    Process all conformers for a single SMILES group and save to PT file.

    Args:
        smiles: SMILES string
        conformer_rows: List of row data for all conformers of this SMILES
        fs: S3 filesystem object
        save_path: Directory to save the processed PT file

    Returns:
        tuple: (success: bool, num_conformers: int, error_count: int, error_msg: str or None)
    """
    try:
        # Process all conformers for this SMILES sequentially
        conformers_data = []
        errors = 0

        for row in conformer_rows:
            success, ligand_data, error = process_single_ligand(row, fs)

            if success:
                conformers_data.append(ligand_data)
            else:
                errors += 1
                logger.error(error)

        # Save all conformers for this SMILES to a single PT file
        if conformers_data:
            success, filename, error = save_conformers_group(smiles, conformers_data, save_path)

            if success:
                return True, len(conformers_data), errors, None
            else:
                return False, 0, errors + 1, error
        else:
            error_msg = f"No conformers successfully processed for SMILES {smiles[:50]}..."
            return False, 0, errors + 1, error_msg

    except Exception as e:
        error_msg = f"Error processing SMILES group {smiles[:50]}...: {str(e)}"
        return False, 0, 1, error_msg


def process_geom_ligands(
    s3_root: str = "s3://prescient-lobster/ume/datasets/geom/processed/test",
    save_path: str = "/data/bucket/lisanza/structures/GEOM/processed/",
    max_workers: int = 8,
    testing: bool = False,
    shuffle_files: bool = True,
):
    """
    Process all ligand data from GEOM dataset parquet files and save as PT files.
    Groups conformers by SMILES string and saves all conformers in a single PT file.
    Skips parquet files where all corresponding PT files already exist.

    Args:
        s3_root: S3 path to the directory containing parquet files
        save_path: Local directory to save processed PT files
        max_workers: Maximum number of parallel workers
        testing: If True, only process a small subset for testing
        shuffle_files: If True, shuffle the order of parquet files for better load distribution
    """

    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Output directory: {save_path}")

    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem()

    # Find all parquet files in S3
    try:
        all_files = fs.find(s3_root)
        parquet_files = [f"s3://{file_path}" for file_path in all_files if file_path.endswith(".parquet")]
        logger.info(f"Found {len(parquet_files)} parquet files in S3")

        if testing:
            parquet_files = parquet_files[:2]  # Only process first 2 files for testing
            logger.info(f"Testing mode: processing only {len(parquet_files)} files")

        # Shuffle the parquet files for better load distribution
        if shuffle_files:
            random.shuffle(parquet_files)
            logger.info("Shuffled parquet files order for better load distribution")

    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        raise

    # Process each parquet file
    total_processed = 0
    total_errors = 0
    total_skipped = 0

    for parquet_file in tqdm.tqdm(parquet_files, desc="Processing parquet files"):
        try:
            # Check if this parquet file has already been fully processed
            if check_parquet_already_processed(parquet_file, save_path):
                total_skipped += 1
                continue

            # Load parquet file
            df = load_geom_parquet_from_s3(parquet_file)

            # Convert to list of dictionaries for processing
            rows = df.to_dict(orient="records")

            if testing:
                rows = rows[:50]  # Only process first 50 rows for testing

            logger.info(f"Processing {len(rows)} ligands from {parquet_file}")

            # Group rows by SMILES string
            smiles_groups = defaultdict(list)
            for row in rows:
                smiles_groups[row["smiles"]].append(row)

            logger.info(f"Found {len(smiles_groups)} unique SMILES strings")

            # Process all SMILES groups in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all SMILES group processing tasks
                future_to_smiles = {
                    executor.submit(process_smiles_group, smiles, conformer_rows, fs, save_path): smiles
                    for smiles, conformer_rows in smiles_groups.items()
                }

                # Process completed tasks with progress bar
                for future in tqdm.tqdm(
                    as_completed(future_to_smiles),
                    total=len(smiles_groups),
                    desc=f"Processing SMILES groups from {os.path.basename(parquet_file)}",
                ):
                    _ = future_to_smiles[future]  # Keep mapping for debugging if needed
                    success, num_conformers, error_count, error_msg = future.result()

                    if success:
                        total_processed += 1
                        # logger.info(f"Saved {num_conformers} conformers for SMILES {smiles[:50]}...")
                    else:
                        total_errors += 1
                        logger.error(error_msg)

                    # Add any errors from conformer processing to total
                    total_errors += error_count

            logger.info(f"Completed {parquet_file}")

        except Exception as e:
            logger.error(f"Error processing parquet file {parquet_file}: {e}")
            total_errors += 1

    logger.info("Processing complete!")
    logger.info(f"Total SMILES processed: {total_processed}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total parquet files skipped: {total_skipped}")
    logger.info(f"Output directory: {save_path}")


def main():
    """Main function to run the ligand processing."""

    # Configuration
    s3_root = "s3://prescient-lobster/ume/datasets/geom/processed/test"
    # save_path = "/data/bucket/lisanza/structures/GEOM/processed/train/"
    save_path = "/data/bucket/shmilovk/structures/GEOM/processed/test/"
    max_workers = max(1, multiprocessing.cpu_count())
    testing = False  # Set to False for full processing
    shuffle_files = True  # Shuffle parquet files for better load distribution

    logger.info("Starting GEOM ligand processing...")
    logger.info(f"S3 root: {s3_root}")
    logger.info(f"Output directory: {save_path}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Testing mode: {testing}")
    logger.info(f"Shuffle files: {shuffle_files}")

    # Process the ligands
    process_geom_ligands(
        s3_root=s3_root, save_path=save_path, max_workers=max_workers, testing=testing, shuffle_files=shuffle_files
    )


if __name__ == "__main__":
    main()
