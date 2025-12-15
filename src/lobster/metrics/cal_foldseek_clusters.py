import subprocess
import os
import logging
from pathlib import Path
from biotite.sequence.io import fasta
import shutil
import pandas as pd

logger = logging.getLogger(__name__)


def setup_foldseek_path(foldseek_bin_path):
    """
    Add Foldseek binary directory to PATH environment variable.

    Args:
        foldseek_bin_path: Path to directory containing Foldseek binary

    Returns:
        Original PATH value for restoration if needed
    """
    if foldseek_bin_path is None:
        return os.environ.get("PATH", "")

    original_path = os.environ.get("PATH", "")
    foldseek_bin_path = str(foldseek_bin_path)

    # Check if already in PATH
    if foldseek_bin_path not in original_path:
        os.environ["PATH"] = f"{foldseek_bin_path}:{original_path}"
        logger.debug(f"Added Foldseek binary path to PATH: {foldseek_bin_path}")

    return original_path


def run_easy_cluster(designable_dir, output_dir, tmscore_threshold=0.5, foldseek_bin_path=None):
    """
    Run Foldseek easy-cluster on a directory of PDB files.

    Args:
        designable_dir: Directory containing PDB files to cluster
        output_dir: Directory where clustering results will be saved
        tmscore_threshold: TM-score threshold for clustering (default: 0.5)
        foldseek_bin_path: Path to Foldseek binary directory (optional)

    Returns:
        Tuple of (num_clusters, total_proteins) or (None, None) on error
    """
    designable_dir = Path(designable_dir)
    output_dir = Path(output_dir)

    # Setup Foldseek PATH if provided
    if foldseek_bin_path:
        setup_foldseek_path(foldseek_bin_path)

    # Check if input directory exists and has PDB files
    if not designable_dir.exists():
        logger.error(f"Input directory does not exist: {designable_dir}")
        return None, None

    pdb_files = list(designable_dir.glob("*.pdb"))
    if not pdb_files:
        logger.warning(f"No PDB files found in {designable_dir}")
        return 0, 0

    total_proteins = len(pdb_files)
    logger.info(f"Found {total_proteins} PDB files in {designable_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build Foldseek command
    easy_cluster_args = [
        "foldseek",
        "easy-cluster",
        str(designable_dir),
        str(output_dir / "res"),
        str(output_dir),
        "--alignment-type",
        "1",
        "--cov-mode",
        "0",
        "--min-seq-id",
        "0",
        "--tmscore-threshold",
        str(tmscore_threshold),
    ]

    try:
        logger.info(f"Running Foldseek clustering with TM-score threshold {tmscore_threshold}")
        process = subprocess.Popen(easy_cluster_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Foldseek failed with return code {process.returncode}")
            logger.error(f"stderr: {stderr}")
            return None, None

        # Read clustering results
        rep_seq_fasta_path = output_dir / "res_rep_seq.fasta"
        if not rep_seq_fasta_path.exists():
            logger.error(f"Foldseek output file not found: {rep_seq_fasta_path}")
            return None, None

        rep_seq_fasta = fasta.FastaFile.read(str(rep_seq_fasta_path))
        num_clusters = len(rep_seq_fasta)

        logger.info(f"Foldseek clustering complete: {num_clusters} clusters from {total_proteins} structures")
        return num_clusters, total_proteins

    except FileNotFoundError:
        logger.error("Foldseek binary not found. Please check foldseek_bin_path or PATH.")
        return None, None
    except Exception as e:
        logger.error(f"Error running Foldseek: {e}")
        return None, None


def copy_structures_by_rmsd(output_dir, length, rmsd_threshold=2.0):
    """
    Copy ESMFold-validated structures that pass RMSD threshold to temporary directory.

    Args:
        output_dir: Generation output directory containing PDB files and CSV
        length: Sequence length to filter
        rmsd_threshold: RMSD threshold for filtering (default: 2.0)

    Returns:
        Tuple of (temp_dir_path, num_copied_files) or (None, 0) on error
    """
    output_dir = Path(output_dir)

    # Find the CSV metrics file
    csv_files = list(output_dir.glob("*_metrics_*.csv"))
    if not csv_files:
        logger.error(f"No metrics CSV file found in {output_dir}")
        return None, 0

    # Use most recent CSV
    csv_path = max(csv_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Reading metrics from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)

        # Filter by sequence length first
        if "sequence_length" not in df.columns:
            logger.error("sequence_length column not found in CSV")
            return None, 0

        df_length = df[df["sequence_length"] == length].copy()
        logger.info(f"Found {len(df_length)} total structures for length {length}")

        if len(df_length) == 0:
            logger.warning(f"No structures found for length {length}")
            return None, 0

        # Filter by RMSD threshold
        if "rmsd" not in df_length.columns:
            logger.error("RMSD column not found in CSV")
            return None, 0

        passing_df = df_length[df_length["rmsd"] < rmsd_threshold].copy()
        logger.info(f"Found {len(passing_df)} structures with RMSD < {rmsd_threshold} for length {length}")

        if len(passing_df) == 0:
            logger.warning("No structures pass RMSD threshold")
            return None, 0

        # Create temporary directory with hierarchical structure
        temp_base_dir = output_dir / "foldseek_temp_dir"
        temp_dir = temp_base_dir / f"length_{length}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Copy PDB files
        num_copied = 0
        for idx, row in passing_df.iterrows():
            # Find the ESMFold PDB file
            # Pattern: generated_structure_length_{length}_{###}_esmfold_000.pdb
            run_id = row.get("run_id", "")

            # Try to construct filename from run_id
            if run_id:
                # Extract iteration number from run_id
                # Example: "unconditional_length_500_iter_001" or "self_reflection_length_500_iter_001"
                parts = run_id.split("_")
                if len(parts) >= 5:
                    iter_num = parts[-1]  # Get last part (e.g., "001")

                    # Construct ESMFold filename
                    esmfold_filename = f"generated_structure_length_{length}_{iter_num}_esmfold_000.pdb"
                    source_path = output_dir / esmfold_filename

                    if source_path.exists():
                        dest_path = temp_dir / esmfold_filename
                        shutil.copy(source_path, dest_path)
                        num_copied += 1
                    else:
                        logger.debug(f"PDB file not found: {source_path}")

        logger.info(f"Copied {num_copied} PDB files to {temp_dir}")
        return temp_dir, num_copied

    except Exception as e:
        logger.error(f"Error copying structures: {e}")
        return None, 0


def calculate_diversity_for_generation(
    output_dir, length, rmsd_threshold=2.0, foldseek_bin_path=None, tmscore_threshold=0.5
):
    """
    Complete workflow for diversity analysis on generated structures.

    This function:
    1. Identifies ESMFold structures with RMSD < threshold from CSV
    2. Creates temp directory: output_dir/foldseek_temp_dir/length_{L}/
    3. Copies filtered PDB files to temp directory
    4. Creates results directory: output_dir/foldseek_results/length_{L}/
    5. Runs Foldseek clustering with specified TM-score threshold
    6. Parses clustering results and calculates diversity metrics
    7. Returns diversity metrics dictionary

    Args:
        output_dir: Path to generation output directory
        length: Sequence length to analyze
        rmsd_threshold: RMSD threshold for filtering (default: 2.0)
        foldseek_bin_path: Path to Foldseek binary directory (optional)
        tmscore_threshold: TM-score threshold for clustering (default: 0.5)

    Returns:
        Dictionary with diversity metrics:
        - total_structures: Number of structures passing RMSD threshold
        - num_clusters: Number of Foldseek clusters found
        - diversity_percentage: (num_clusters / total_structures) * 100
        - tmscore_threshold: TM-score threshold used
        - rmsd_threshold: RMSD threshold used

        Returns None if analysis fails.
    """
    output_dir = Path(output_dir)

    logger.info(f"Starting diversity analysis for length {length}")
    logger.info(f"RMSD threshold: {rmsd_threshold}, TM-score threshold: {tmscore_threshold}")

    # Step 1: Copy structures by RMSD threshold
    temp_dir, num_copied = copy_structures_by_rmsd(output_dir, length, rmsd_threshold)

    if temp_dir is None or num_copied == 0:
        logger.warning("No structures available for diversity analysis")
        return None

    # Step 2: Create results directory
    results_base_dir = output_dir / "foldseek_results"
    results_dir = results_base_dir / f"length_{length}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Run Foldseek clustering
    num_clusters, total_proteins = run_easy_cluster(
        temp_dir, results_dir, tmscore_threshold=tmscore_threshold, foldseek_bin_path=foldseek_bin_path
    )

    if num_clusters is None:
        logger.error("Foldseek clustering failed")
        return None

    # Step 4: Calculate diversity metrics
    diversity_percentage = (num_clusters / total_proteins * 100) if total_proteins > 0 else 0.0

    metrics = {
        "total_structures": total_proteins,
        "num_clusters": num_clusters,
        "diversity_percentage": diversity_percentage,
        "tmscore_threshold": tmscore_threshold,
        "rmsd_threshold": rmsd_threshold,
    }

    logger.info(f"Diversity analysis complete for length {length}:")
    logger.info(f"  Total structures: {total_proteins}")
    logger.info(f"  Number of clusters: {num_clusters}")
    logger.info(f"  Diversity percentage: {diversity_percentage:.2f}%")

    return metrics


if __name__ == "__main__":
    import os
    from pathlib import Path

    lengths_to_process = [500]

    # Define base paths to avoid repetition
    # base_design_dir = Path("/homefs/home/lisanzas/scratch/Develop/lobster/examples/generated_unconditional_self_reflection_all_lengths_plots/esmfolded_pdb_files")
    # base_output_dir = Path("/homefs/home/lisanzas/scratch/Develop/lobster/examples/generated_unconditional_self_reflection_all_lengths_plots/diversity_foldseek")
    base_design_dir = Path(
        "/homefs/home/lisanzas/scratch/Develop/lobster/examples/generated_unconditional_self_reflection_500_2_plots/esmfolded_pdb_files"
    )
    base_output_dir = base_design_dir / "diversity_foldseek"
    # Dictionary to store results for the final summary
    all_results = {}

    # --- Main Loop ---

    for length in lengths_to_process:
        print(f"--- Processing Length: {length} ---")

        # 1. Define paths dynamically for the current length
        designable_dir = base_design_dir / f"length_{length}"
        output_dir = base_output_dir / f"length_{length}"

        # 2. Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # 3. Check if the source directory exists before processing
        if not designable_dir.is_dir():
            print(f"Directory not found, skipping: {designable_dir}\n")
            all_results[length] = {"total_proteins": 0, "num_clusters": 0, "percentage": 0}
            continue

        # 4. Run clustering
        num_clusters, total_proteins = run_easy_cluster(designable_dir, output_dir)

        # Handle case where clustering failed
        if num_clusters is None:
            print(f"Clustering failed for: {designable_dir}")
            all_results[length] = {"total_proteins": 0, "num_clusters": 0, "percentage": 0}
            continue

        # 6. Print stats and handle division by zero if no proteins are found
        if total_proteins > 0:
            percentage = (num_clusters / total_proteins) * 100
            print(f"Total proteins: {total_proteins}")
            print(f"Number of clusters: {num_clusters}")
            print(f"Percentage of clusters: {percentage:.2f}%")

            # Store results for summary
            all_results[length] = {
                "total_proteins": total_proteins,
                "num_clusters": num_clusters,
                "percentage": percentage,
            }
        else:
            print(f"No PDB files found in: {designable_dir}")
            all_results[length] = {"total_proteins": 0, "num_clusters": 0, "percentage": 0}

        print("-" * (25 + len(str(length))) + "\n")  # Print separator

    # --- Final Summary Table ---
    print("\n====================")
    print("  Final Summary")
    print("====================")
    print(f"{'Length':<10} | {'Total Proteins':<15} | {'Num Clusters':<15} | {'Percentage':<10}")
    print("-" * 59)

    for length, data in all_results.items():
        print(f"{length:<10} | {data['total_proteins']:<15} | {data['num_clusters']:<15} | {data['percentage']:.2f}%")
