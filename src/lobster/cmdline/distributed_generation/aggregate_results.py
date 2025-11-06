#!/usr/bin/env python3
"""
Aggregate results from distributed generation jobs.
Includes Foldseek clustering on complete dataset per length.
"""

import pandas as pd
from pathlib import Path
from loguru import logger
import shutil

from lobster.metrics.cal_foldseek_clusters import run_easy_cluster


def aggregate_distributed_results(
    base_output_dir: str,
    num_jobs: int,
    run_foldseek: bool = True,
    foldseek_bin_path: str = None,
    foldseek_tmscore_threshold: float = 0.5,
    rmsd_threshold: float = 2.0,
):
    """
    Aggregate results from multiple distributed generation jobs.
    Includes Foldseek clustering on aggregated structures per length.

    Args:
        base_output_dir: Base output directory containing job_* subdirectories
        num_jobs: Number of jobs to aggregate
        run_foldseek: Whether to run Foldseek clustering (default: True)
        foldseek_bin_path: Path to Foldseek binary directory
        foldseek_tmscore_threshold: TM-score threshold for clustering
        rmsd_threshold: RMSD threshold for filtering structures
    """
    base_path = Path(base_output_dir)

    # Create aggregated output directory
    agg_dir = base_path / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    all_metrics = []
    all_sequences = []
    structures_by_length = {}  # Track structures by length for Foldseek

    logger.info(f"Aggregating {num_jobs} jobs from {base_output_dir}")

    # Collect from each job
    for job_id in range(num_jobs):
        job_dir = base_path / f"job_{job_id}"

        if not job_dir.exists():
            logger.warning(f"Job {job_id} directory not found: {job_dir}")
            continue

        logger.info(f"Processing job {job_id}")

        # Find metrics CSV
        metrics_files = list(job_dir.glob("*_metrics_*.csv"))
        if metrics_files:
            df = pd.read_csv(metrics_files[0])
            df["job_id"] = job_id
            all_metrics.append(df)

        # Find sequences CSV
        seq_files = list(job_dir.glob("sequences_*.csv"))
        if seq_files:
            df_seq = pd.read_csv(seq_files[0])
            df_seq["job_id"] = job_id
            all_sequences.append(df_seq)

        # Collect PDB files organized by length
        # ONLY collect ESMFold structures for Foldseek analysis
        pdb_files = list(job_dir.glob("*.pdb"))
        esmfold_count = 0
        for pdb in pdb_files:
            # Filter: Only include ESMFold structures (contain "_esmfold_" in filename)
            if "_esmfold_" not in pdb.name:
                continue
            esmfold_count += 1

            # Extract length from filename if possible
            # Assuming format like: unconditional_length_500_sample_0_esmfold_000.pdb
            try:
                parts = pdb.stem.split("_")
                if "length" in parts:
                    length_idx = parts.index("length") + 1
                    length = int(parts[length_idx])
                else:
                    # Fallback: read PDB to get length
                    length = get_pdb_length(pdb)

                if length not in structures_by_length:
                    structures_by_length[length] = []

                # Copy with unique name
                new_name = f"job_{job_id}_{pdb.name}"
                dest = agg_dir / new_name
                shutil.copy2(pdb, dest)
                structures_by_length[length].append(dest)

            except Exception as e:
                logger.warning(f"Could not determine length for {pdb.name}: {e}")
                # Copy anyway
                new_name = f"job_{job_id}_{pdb.name}"
                shutil.copy2(pdb, agg_dir / new_name)

        if esmfold_count > 0:
            logger.info(f"  Found {esmfold_count} ESMFold structures (out of {len(pdb_files)} total)")

    # Combine metrics
    diversity_results = {}
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        output_metrics = agg_dir / "combined_metrics.csv"
        combined_metrics.to_csv(output_metrics, index=False)
        logger.info(f"Saved combined metrics: {output_metrics}")
        logger.info(f"Total samples: {len(combined_metrics)}")

        # Print summary statistics
        logger.info("\n=== Summary Statistics ===")
        for col in ["plddt", "tm_score", "rmsd", "predicted_aligned_error"]:
            if col in combined_metrics.columns:
                logger.info(f"{col}:")
                logger.info(f"  Mean: {combined_metrics[col].mean():.3f}")
                logger.info(f"  Std:  {combined_metrics[col].std():.3f}")
                logger.info(f"  Min:  {combined_metrics[col].min():.3f}")
                logger.info(f"  Max:  {combined_metrics[col].max():.3f}")

        # Run Foldseek clustering per length
        if run_foldseek and structures_by_length:
            total_esmfold = sum(len(pdbs) for pdbs in structures_by_length.values())
            logger.info("\n=== Running Foldseek Clustering ===")
            logger.info(f"Found {total_esmfold} ESMFold structures at {len(structures_by_length)} different lengths")
            logger.info("Note: Only ESMFold-validated structures are used for diversity analysis")

            diversity_results = run_foldseek_clustering(
                structures_by_length=structures_by_length,
                output_dir=agg_dir,
                combined_metrics=combined_metrics,
                foldseek_bin_path=foldseek_bin_path,
                tmscore_threshold=foldseek_tmscore_threshold,
                rmsd_threshold=rmsd_threshold,
            )

            # Log diversity results
            logger.info("\n=== Diversity Results ===")
            for length, results in diversity_results.items():
                logger.info(f"Length {length}:")
                logger.info(f"  Total structures: {results['total_structures']}")
                logger.info(f"  Structures passing RMSD < {rmsd_threshold}: {results['structures_passing_rmsd']}")
                logger.info(f"  Number of clusters: {results['num_clusters']}")
                logger.info(f"  Diversity: {results['diversity_percentage']:.1f}%")

        # Create comprehensive summary table per length
        logger.info("\n=== Creating Summary Table ===")
        summary_table = create_summary_table(
            combined_metrics=combined_metrics,
            diversity_results=diversity_results,
            structures_by_length=structures_by_length,
            rmsd_threshold=rmsd_threshold,
        )

        if summary_table is not None:
            # Save to CSV
            summary_csv = agg_dir / "summary_per_length.csv"
            summary_table.to_csv(summary_csv, index=False)
            logger.info(f"Saved summary table: {summary_csv}")

            # Print table
            logger.info("\n=== Summary Per Length ===")
            logger.info(f"\n{summary_table.to_string(index=False)}")
        else:
            logger.warning("Could not create summary table")

    # Combine sequences
    if all_sequences:
        combined_sequences = pd.concat(all_sequences, ignore_index=True)
        output_sequences = agg_dir / "combined_sequences.csv"
        combined_sequences.to_csv(output_sequences, index=False)
        logger.info(f"Saved combined sequences: {output_sequences}")

    logger.info(f"\nAggregation complete! Results in: {agg_dir}")

    return {
        "aggregated_dir": str(agg_dir),
        "total_samples": len(all_metrics[0]) if all_metrics else 0,
        "diversity_results": diversity_results,
    }


def create_summary_table(
    combined_metrics: pd.DataFrame, diversity_results: dict, structures_by_length: dict, rmsd_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Create a comprehensive summary table with metrics per length.

    Args:
        combined_metrics: DataFrame with all metrics
        diversity_results: Dictionary with diversity results per length
        structures_by_length: Dict mapping length -> list of PDB files
        rmsd_threshold: RMSD threshold used for filtering

    Returns:
        DataFrame with summary statistics per length
    """
    if combined_metrics is None or len(combined_metrics) == 0:
        logger.warning("No metrics available for summary table")
        return None

    summary_rows = []

    # Get unique lengths from metrics
    if "sequence_length" in combined_metrics.columns:
        lengths = sorted(combined_metrics["sequence_length"].unique())
    else:
        lengths = sorted(structures_by_length.keys())

    for length in lengths:
        # Filter metrics for this length
        length_metrics = combined_metrics[combined_metrics["sequence_length"] == length]
        # Filter rows where rmsd is not found
        length_metrics = length_metrics[length_metrics["rmsd"].notna()]

        if len(length_metrics) == 0:
            continue

        # Calculate basic metrics
        total_structures = len(length_metrics)
        structures_passing_rmsd = len(length_metrics[length_metrics["rmsd"] < rmsd_threshold])
        pct_passing_rmsd = (structures_passing_rmsd / total_structures * 100) if total_structures > 0 else 0

        # Get diversity metrics if available
        num_clusters = 0
        diversity_pct = 0.0
        if length in diversity_results:
            num_clusters = diversity_results[length]["num_clusters"]
            diversity_pct = diversity_results[length]["diversity_percentage"]

        # Calculate average metrics
        avg_tm = length_metrics["tm_score"].mean() if "tm_score" in length_metrics.columns else 0
        avg_rmsd = length_metrics["rmsd"].mean() if "rmsd" in length_metrics.columns else 0
        avg_plddt = length_metrics["plddt"].mean() if "plddt" in length_metrics.columns else 0

        summary_rows.append(
            {
                "Length": int(length),
                "Total_Structures": total_structures,
                f"Structures_RMSD<{rmsd_threshold}": structures_passing_rmsd,
                f"Pct_RMSD<{rmsd_threshold}": round(pct_passing_rmsd, 2),
                "Num_Clusters": num_clusters,
                "Diversity_Pct": round(diversity_pct, 2),
                "Avg_TM_Score": round(avg_tm, 4),
                "Avg_RMSD": round(avg_rmsd, 4),
                "Avg_pLDDT": round(avg_plddt, 4),
            }
        )

    if not summary_rows:
        logger.warning("No data to create summary table")
        return None

    return pd.DataFrame(summary_rows)


def get_pdb_length(pdb_path: Path) -> int:
    """
    Get sequence length from PDB file by counting CA atoms.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Number of residues
    """
    try:
        import biotite.structure.io.pdb as pdb

        structure = pdb.PDBFile.read(str(pdb_path))
        atom_array = structure.get_structure()[0]

        # Count CA atoms
        ca_mask = atom_array.atom_name == "CA"
        return ca_mask.sum()
    except Exception as e:
        logger.warning(f"Failed to get length from {pdb_path}: {e}")
        return 0


def run_foldseek_clustering(
    structures_by_length: dict,
    output_dir: Path,
    combined_metrics: pd.DataFrame,
    foldseek_bin_path: str = None,
    tmscore_threshold: float = 0.5,
    rmsd_threshold: float = 2.0,
) -> dict:
    """
    Run Foldseek clustering on aggregated structures, organized by length.

    Args:
        structures_by_length: Dict mapping length -> list of PDB file paths
        output_dir: Output directory for Foldseek results
        combined_metrics: DataFrame with all metrics (used for RMSD filtering)
        foldseek_bin_path: Path to Foldseek binary directory
        tmscore_threshold: TM-score threshold for clustering
        rmsd_threshold: RMSD threshold for filtering

    Returns:
        Dictionary with diversity metrics per length
    """
    if foldseek_bin_path is None:
        foldseek_bin_path = "/homefs/home/lisanzas/scratch/Develop/lobster/src/lobster/metrics/foldseek/bin"

    diversity_results = {}

    for length, pdb_files in structures_by_length.items():
        logger.info(f"\nProcessing length {length}: {len(pdb_files)} structures")

        # Create length-specific directory
        length_dir = output_dir / "foldseek_results" / f"length_{length}"
        length_dir.mkdir(parents=True, exist_ok=True)

        # Filter structures by RMSD threshold using metrics DataFrame
        # Only include structures that have valid RMSD < threshold in combined_metrics
        filtered_pdbs = []

        # Get metrics for this length with valid RMSD
        length_metrics = combined_metrics[
            (combined_metrics["sequence_length"] == length)
            & (combined_metrics["rmsd"].notna())
            & (combined_metrics["rmsd"] < rmsd_threshold)
        ]

        # Determine which column to use for matching filenames
        id_column = None
        if "structure_file" in combined_metrics.columns:
            id_column = "structure_file"
        elif "run_id" in combined_metrics.columns:
            id_column = "run_id"
        else:
            logger.warning("  No 'structure_file' or 'run_id' column in metrics, cannot filter structures")
            continue

        # Build mapping from (job_id, sample_idx) pairs that passed RMSD threshold
        # We need BOTH job_id and sample number to uniquely identify structures
        # run_id format: unconditional_length_100_iter_000
        # filename format: job_0_generated_structure_length_100_000_esmfold_000.pdb
        passing_job_sample_pairs = set()

        for idx, row in length_metrics.iterrows():
            identifier = row[id_column]
            job_id = row.get("job_id", None)

            if pd.notna(identifier) and pd.notna(job_id):
                # Extract sample number from run_id
                # Pattern: unconditional_length_XXX_iter_YYY -> YYY is the sample number
                try:
                    parts = str(identifier).split("_")
                    if "iter" in parts:
                        iter_idx = parts.index("iter") + 1
                        sample_num = int(parts[iter_idx])
                        passing_job_sample_pairs.add((int(job_id), sample_num))
                except (ValueError, IndexError):
                    pass

        logger.info(f"  Found {len(passing_job_sample_pairs)} (job_id, sample) pairs passing RMSD threshold")

        # Filter PDB files to only include those with passing (job_id, sample_num) pairs
        for pdb_path in pdb_files:
            pdb_name = pdb_path.stem  # Get filename without extension

            # Extract job_id and sample number from filename
            # Pattern: job_0_generated_structure_length_100_000_esmfold_000
            try:
                parts = pdb_name.split("_")

                # Extract job_id (first part after "job")
                job_id = None
                if "job" in parts:
                    job_idx = parts.index("job") + 1
                    if job_idx < len(parts):
                        job_id = int(parts[job_idx])

                # Extract sample number (comes after "length_XXX")
                sample_num = None
                if "length" in parts:
                    length_idx = parts.index("length") + 1
                    # Skip the length value, next number is the sample index
                    if length_idx + 1 < len(parts):
                        sample_num = int(parts[length_idx + 1])

                # Check if this (job_id, sample_num) pair passed RMSD threshold
                if job_id is not None and sample_num is not None:
                    if (job_id, sample_num) in passing_job_sample_pairs:
                        filtered_pdbs.append(pdb_path)

            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse filename {pdb_name}: {e}")

        structures_passing_rmsd = len(filtered_pdbs)
        logger.info(f"  Structures passing RMSD < {rmsd_threshold}: {structures_passing_rmsd}")

        if structures_passing_rmsd == 0:
            logger.warning(f"  No structures passed RMSD filter for length {length}")
            continue

        # Create temp directory with filtered PDBs for Foldseek
        temp_dir = length_dir / "foldseek_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Copy filtered PDB files to temp directory
        for pdb in filtered_pdbs:
            shutil.copy2(pdb, temp_dir / pdb.name)

        logger.info(f"  Copied {structures_passing_rmsd} structures to temp directory")

        # Run Foldseek clustering using the existing function from lobster.metrics
        try:
            logger.info(f"  Running Foldseek with TM-score threshold {tmscore_threshold}")

            num_clusters, total_proteins = run_easy_cluster(
                designable_dir=temp_dir,
                output_dir=length_dir,
                tmscore_threshold=tmscore_threshold,
                foldseek_bin_path=foldseek_bin_path,
            )

            if num_clusters is not None:
                diversity_pct = (num_clusters / total_proteins) * 100 if total_proteins > 0 else 0

                diversity_results[length] = {
                    "total_structures": len(pdb_files),
                    "structures_passing_rmsd": structures_passing_rmsd,
                    "num_clusters": num_clusters,
                    "diversity_percentage": diversity_pct,
                }

                logger.info(f"  ✓ Clustering complete: {num_clusters} clusters ({diversity_pct:.1f}% diversity)")
            else:
                logger.error(f"  Foldseek clustering failed for length {length}")

        except Exception as e:
            logger.error(f"  Foldseek clustering failed: {e}")

    return diversity_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate distributed generation results with Foldseek clustering")
    parser.add_argument("base_output_dir", help="Base output directory with job_* subdirectories")
    parser.add_argument("num_jobs", type=int, help="Number of jobs to aggregate")
    parser.add_argument("--no-foldseek", action="store_true", help="Skip Foldseek clustering")
    parser.add_argument("--foldseek-bin", help="Path to Foldseek binary directory")
    parser.add_argument("--tmscore-threshold", type=float, default=0.5, help="TM-score threshold for clustering")
    parser.add_argument("--rmsd-threshold", type=float, default=2.0, help="RMSD threshold for filtering")

    args = parser.parse_args()

    results = aggregate_distributed_results(
        base_output_dir=args.base_output_dir,
        num_jobs=args.num_jobs,
        run_foldseek=not args.no_foldseek,
        foldseek_bin_path=args.foldseek_bin,
        foldseek_tmscore_threshold=args.tmscore_threshold,
        rmsd_threshold=args.rmsd_threshold,
    )

    print("\n=== Aggregation Complete ===")
    print(f"Results saved to: {results['aggregated_dir']}")
    print(f"Total samples: {results['total_samples']}")
    if results["diversity_results"]:
        print("\nDiversity Summary:")
        for length, metrics in results["diversity_results"].items():
            print(
                f"  Length {length}: {metrics['num_clusters']} clusters ({metrics['diversity_percentage']:.1f}% diversity)"
            )
