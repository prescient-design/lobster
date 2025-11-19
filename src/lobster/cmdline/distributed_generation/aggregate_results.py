#!/usr/bin/env python3
"""
Aggregate results from distributed generation jobs.

Supports three modes:
1. Unconditional: Groups by length, runs Foldseek diversity analysis
2. Inverse Folding: Groups by input structure, calculates AAR and structural metrics
3. Forward Folding: Groups by input structure, calculates structural metrics
"""

import pandas as pd
from pathlib import Path
from loguru import logger
import shutil

from lobster.metrics.cal_foldseek_clusters import run_easy_cluster


def detect_generation_mode(metrics_df: pd.DataFrame) -> str:
    """
    Detect generation mode from metrics DataFrame.

    CSV columns by mode:
    - Inverse folding: has 'percent_identity' and 'input_file' columns, mode='inverse_folding'
    - Forward folding: has 'input_file' but no 'percent_identity', mode='forward_folding'
    - Unconditional: has 'sequence_length' for grouping by length

    Args:
        metrics_df: DataFrame with metrics from a single job

    Returns:
        "unconditional", "inverse_folding", or "forward_folding"
    """
    # Check for mode column (most reliable)
    if "mode" in metrics_df.columns:
        mode_value = metrics_df["mode"].iloc[0]
        if mode_value in ["inverse_folding", "forward_folding", "unconditional"]:
            logger.info(f"Detected mode from 'mode' column: {mode_value}")
            return mode_value

    # Fallback: check for mode-specific columns
    if "percent_identity" in metrics_df.columns and "input_file" in metrics_df.columns:
        logger.info("Detected mode from columns: inverse_folding")
        return "inverse_folding"
    elif "input_file" in metrics_df.columns:
        logger.info("Detected mode from columns: forward_folding")
        return "forward_folding"
    elif "sequence_length" in metrics_df.columns:
        logger.info("Detected mode from columns: unconditional")
        return "unconditional"
    else:
        logger.warning("Could not detect mode from columns, defaulting to unconditional")
        return "unconditional"


def aggregate_distributed_results(
    base_output_dir: str,
    num_jobs: int,
    mode: str = None,
    run_foldseek: bool = None,
    foldseek_bin_path: str = None,
    foldseek_tmscore_threshold: float = 0.5,
    rmsd_threshold: float = 2.0,
):
    """
    Aggregate results from multiple distributed generation jobs.

    Supports three modes:
    - unconditional: Groups by length, runs Foldseek diversity analysis
    - inverse_folding: Groups by input structure, reports AAR and structural metrics
    - forward_folding: Groups by input structure, reports structural metrics

    Args:
        base_output_dir: Base output directory containing job_* subdirectories
        num_jobs: Number of jobs to aggregate
        mode: Generation mode ("unconditional", "inverse_folding", "forward_folding")
              If None, will auto-detect from metrics CSV
        run_foldseek: Whether to run Foldseek clustering
                     If None, auto-set based on mode (True for unconditional only)
        foldseek_bin_path: Path to Foldseek binary directory
        foldseek_tmscore_threshold: TM-score threshold for clustering
        rmsd_threshold: RMSD threshold for filtering structures

    Returns:
        Dictionary with aggregation results
    """
    base_path = Path(base_output_dir)

    # Auto-detect mode if not provided
    if mode is None:
        logger.info("Mode not specified, attempting auto-detection...")
        for job_id in range(num_jobs):
            job_dir = base_path / f"job_{job_id}"
            if not job_dir.exists():
                continue

            metrics_files = list(job_dir.glob("*_metrics_*.csv"))
            if metrics_files:
                df = pd.read_csv(metrics_files[0])
                mode = detect_generation_mode(df)
                break

        if mode is None:
            logger.warning("Could not auto-detect mode, defaulting to unconditional")
            mode = "unconditional"

    logger.info(f"Aggregating results in {mode} mode")

    # Auto-set Foldseek based on mode if not explicitly provided
    if run_foldseek is None:
        run_foldseek = mode == "unconditional"
        if not run_foldseek:
            logger.info(f"Foldseek disabled for {mode} mode (diversity analysis not applicable)")

    # Branch to mode-specific aggregation
    if mode == "unconditional":
        return aggregate_unconditional(
            base_path=base_path,
            num_jobs=num_jobs,
            run_foldseek=run_foldseek,
            foldseek_bin_path=foldseek_bin_path,
            foldseek_tmscore_threshold=foldseek_tmscore_threshold,
            rmsd_threshold=rmsd_threshold,
        )
    elif mode == "inverse_folding":
        return aggregate_inverse_folding(
            base_path=base_path,
            num_jobs=num_jobs,
            rmsd_threshold=rmsd_threshold,
        )
    elif mode == "forward_folding":
        return aggregate_forward_folding(
            base_path=base_path,
            num_jobs=num_jobs,
            rmsd_threshold=rmsd_threshold,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def aggregate_unconditional(
    base_path: Path,
    num_jobs: int,
    run_foldseek: bool,
    foldseek_bin_path: str,
    foldseek_tmscore_threshold: float,
    rmsd_threshold: float,
) -> dict:
    """
    Aggregate unconditional generation results.
    Groups by length, runs Foldseek diversity analysis.

    This is the original aggregation logic for unconditional generation.
    """

    # Create aggregated output directory
    agg_dir = base_path / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    all_metrics = []
    all_sequences = []
    structures_by_length = {}  # Track structures by length for Foldseek

    logger.info(f"Aggregating {num_jobs} jobs from {base_path}")

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
        "mode": "unconditional",
        "aggregated_dir": str(agg_dir),
        "total_samples": len(combined_metrics) if all_metrics else 0,
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


def aggregate_inverse_folding(
    base_path: Path,
    num_jobs: int,
    rmsd_threshold: float = 2.0,
) -> dict:
    """
    Aggregate inverse folding results.
    Groups by input structure, calculates AAR and structural metrics.

    Key metrics (already in CSV):
    - AAR (Amino Acid Recovery): from 'percent_identity' column
    - TM-score: from 'tm_score' column
    - RMSD: from 'rmsd' column
    - pLDDT: from 'plddt' column
    """
    agg_dir = base_path / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    all_metrics = []
    all_sequences = []

    logger.info(f"Aggregating {num_jobs} jobs from {base_path}")

    # Collect from each job
    for job_id in range(num_jobs):
        job_dir = base_path / f"job_{job_id}"
        if not job_dir.exists():
            logger.warning(f"Job {job_id} directory not found")
            continue

        logger.info(f"Processing job {job_id}")

        # Find inverse folding metrics CSV
        metrics_files = list(job_dir.glob("*inverse_folding*metrics*.csv"))
        if not metrics_files:
            metrics_files = list(job_dir.glob("*_metrics_*.csv"))

        if metrics_files:
            df = pd.read_csv(metrics_files[0])
            df["job_id"] = job_id
            all_metrics.append(df)
            logger.info(f"  Found {len(df)} designs")

        # Find sequences CSV
        seq_files = list(job_dir.glob("sequences_inverse_folding*.csv"))
        if not seq_files:
            seq_files = list(job_dir.glob("sequences_*.csv"))

        if seq_files:
            df_seq = pd.read_csv(seq_files[0])
            df_seq["job_id"] = job_id
            all_sequences.append(df_seq)

        # Copy PDB files (both generated and ESMFold)
        pdb_files = list(job_dir.glob("*.pdb"))
        for pdb in pdb_files:
            new_name = f"job_{job_id}_{pdb.name}"
            shutil.copy2(pdb, agg_dir / new_name)

        if pdb_files:
            logger.info(f"  Copied {len(pdb_files)} PDB files")

    # Combine sequences
    if all_sequences:
        combined_sequences = pd.concat(all_sequences, ignore_index=True)
        output_sequences = agg_dir / "combined_inverse_folding_sequences.csv"
        combined_sequences.to_csv(output_sequences, index=False)
        logger.info(f"Saved combined sequences: {output_sequences}")

    # Combine metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # Merge with sequences to get actual structure names
        if all_sequences and "input_structure" in combined_sequences.columns:
            # Merge on BOTH run_id AND job_id to avoid many-to-many joins
            # (each job restarts batch numbering from batch_000)
            structure_map = combined_sequences[["run_id", "job_id", "input_structure"]].drop_duplicates()
            combined_metrics = combined_metrics.merge(structure_map, on=["run_id", "job_id"], how="left")
            # Replace input_file with actual structure names where available
            if "input_structure" in combined_metrics.columns:
                combined_metrics["input_file"] = combined_metrics["input_structure"].fillna(
                    combined_metrics["input_file"]
                )
                combined_metrics = combined_metrics.drop(columns=["input_structure"])
                logger.info("Replaced generic batch identifiers with actual structure names from sequences CSV")

        output_metrics = agg_dir / "combined_inverse_folding_metrics.csv"
        combined_metrics.to_csv(output_metrics, index=False)

        logger.info(f"Saved combined metrics: {output_metrics}")
        logger.info(f"Total designs: {len(combined_metrics)}")

        # Print summary statistics
        logger.info("\n=== Inverse Folding Summary ===")

        # AAR (already calculated as percent_identity in the CSV)
        if "percent_identity" in combined_metrics.columns:
            aar = combined_metrics["percent_identity"].mean()
            logger.info(f"Average AAR (Amino Acid Recovery): {aar:.2f}%")
            logger.info(f"  Min: {combined_metrics['percent_identity'].min():.2f}%")
            logger.info(f"  Max: {combined_metrics['percent_identity'].max():.2f}%")
            logger.info(f"  Median: {combined_metrics['percent_identity'].median():.2f}%")

        # TM-score
        if "tm_score" in combined_metrics.columns:
            tm = combined_metrics["tm_score"].mean()
            logger.info(f"Average TM-score: {tm:.3f}")
            logger.info(f"  Min: {combined_metrics['tm_score'].min():.3f}")
            logger.info(f"  Max: {combined_metrics['tm_score'].max():.3f}")

        # RMSD
        if "rmsd" in combined_metrics.columns:
            rmsd = combined_metrics["rmsd"].mean()
            logger.info(f"Average RMSD: {rmsd:.3f} Å")
            logger.info(f"  Min: {combined_metrics['rmsd'].min():.3f}")
            logger.info(f"  Max: {combined_metrics['rmsd'].max():.3f}")

            # Count structures passing RMSD threshold
            passing = len(combined_metrics[combined_metrics["rmsd"] < rmsd_threshold])
            pct = (passing / len(combined_metrics)) * 100
            logger.info(f"Designs with RMSD < {rmsd_threshold}: {passing}/{len(combined_metrics)} ({pct:.1f}%)")

        # pLDDT
        if "plddt" in combined_metrics.columns:
            plddt = combined_metrics["plddt"].mean()
            logger.info(f"Average pLDDT: {plddt:.3f}")

        # Create per-structure summary table
        logger.info("\n=== Creating Per-Structure Summary ===")
        summary_table = create_inverse_folding_summary(combined_metrics, rmsd_threshold)
        if summary_table is not None:
            summary_csv = agg_dir / "summary_per_structure.csv"
            summary_table.to_csv(summary_csv, index=False)
            logger.info(f"Saved per-structure summary: {summary_csv}")

            # Print table (limit to first 20 rows for readability)
            logger.info("\n=== Summary Per Structure ===")
            if len(summary_table) > 20:
                logger.info(f"\n{summary_table.head(20).to_string(index=False)}")
                logger.info(f"... and {len(summary_table) - 20} more structures")
            else:
                logger.info(f"\n{summary_table.to_string(index=False)}")
        else:
            logger.warning("Could not create per-structure summary table")

        # Create overall summary table (single row with aggregate stats)
        logger.info("\n=== Creating Overall Summary ===")
        overall_summary = create_overall_summary(combined_metrics, rmsd_threshold)
        if overall_summary is not None:
            overall_csv = agg_dir / "overall_summary.csv"
            overall_summary.to_csv(overall_csv, index=False)
            logger.info(f"Saved overall summary: {overall_csv}")
            logger.info(f"\n{overall_summary.to_string(index=False)}")
        else:
            logger.warning("Could not create overall summary table")

    logger.info(f"\nAggregation complete! Results in: {agg_dir}")

    return {
        "mode": "inverse_folding",
        "aggregated_dir": str(agg_dir),
        "total_designs": len(combined_metrics) if all_metrics else 0,
        "average_aar": combined_metrics["percent_identity"].mean()
        if all_metrics and "percent_identity" in combined_metrics.columns
        else None,
        "average_tm_score": combined_metrics["tm_score"].mean()
        if all_metrics and "tm_score" in combined_metrics.columns
        else None,
    }


def create_inverse_folding_summary(combined_metrics: pd.DataFrame, rmsd_threshold: float = 2.0) -> pd.DataFrame:
    """
    Create summary table for inverse folding results, grouped by input structure.

    CSV has 'input_file' column with values like 'batch_000', 'batch_001', etc.

    Returns:
        DataFrame with columns:
        - Input_Structure: from 'input_file' column
        - Num_Designs
        - Avg_AAR: from 'percent_identity' column
        - Avg_TM_Score: from 'tm_score' column
        - Avg_RMSD: from 'rmsd' column
        - Avg_pLDDT: from 'plddt' column
        - Designs_Passing_RMSD (count)
        - Pct_Passing_RMSD
    """
    if "input_file" not in combined_metrics.columns:
        logger.warning("No 'input_file' column found in metrics")
        return None

    summary_rows = []

    # Group by input structure (input_file column)
    for structure_file, group in combined_metrics.groupby("input_file"):
        num_designs = len(group)

        row = {
            "Input_Structure": structure_file,
            "Num_Designs": num_designs,
        }

        # AAR
        if "percent_identity" in group.columns:
            row["Avg_AAR"] = round(group["percent_identity"].mean(), 2)
            row["Min_AAR"] = round(group["percent_identity"].min(), 2)
            row["Max_AAR"] = round(group["percent_identity"].max(), 2)

        # TM-score
        if "tm_score" in group.columns:
            row["Avg_TM_Score"] = round(group["tm_score"].mean(), 4)
            row["Min_TM_Score"] = round(group["tm_score"].min(), 4)
            row["Max_TM_Score"] = round(group["tm_score"].max(), 4)

        # RMSD
        if "rmsd" in group.columns:
            row["Avg_RMSD"] = round(group["rmsd"].mean(), 4)
            row["Min_RMSD"] = round(group["rmsd"].min(), 4)
            row["Max_RMSD"] = round(group["rmsd"].max(), 4)

            passing = len(group[group["rmsd"] < rmsd_threshold])
            row[f"Designs_RMSD<{rmsd_threshold}"] = passing
            row[f"Pct_RMSD<{rmsd_threshold}"] = round((passing / num_designs) * 100, 2)

        # pLDDT
        if "plddt" in group.columns:
            row["Avg_pLDDT"] = round(group["plddt"].mean(), 4)

        summary_rows.append(row)

    if not summary_rows:
        logger.warning("No data to create summary table")
        return None

    return pd.DataFrame(summary_rows)


def create_overall_summary(
    combined_metrics: pd.DataFrame,
    rmsd_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Create a single-row overall summary table with aggregate statistics across all structures.

    Columns:
    - Total_Structures: Number of unique structures
    - Avg_TM_Score, Std_TM_Score, Min_TM_Score, Max_TM_Score
    - Avg_RMSD, Std_RMSD, Min_RMSD, Max_RMSD
    - Structures_RMSD<{threshold}: Count of structures with RMSD below threshold
    - Pct_RMSD<{threshold}: Percentage of structures passing threshold
    """
    if combined_metrics.empty:
        logger.warning("Empty metrics DataFrame, cannot create overall summary")
        return None

    # Count unique structures
    total_structures = (
        combined_metrics["input_file"].nunique() if "input_file" in combined_metrics.columns else len(combined_metrics)
    )

    summary = {
        "Total_Structures": total_structures,
    }

    # TM-score statistics
    if "tm_score" in combined_metrics.columns:
        summary["Avg_TM_Score"] = round(combined_metrics["tm_score"].mean(), 4)
        summary["Std_TM_Score"] = round(combined_metrics["tm_score"].std(), 4)
        summary["Min_TM_Score"] = round(combined_metrics["tm_score"].min(), 4)
        summary["Max_TM_Score"] = round(combined_metrics["tm_score"].max(), 4)

    # RMSD statistics
    if "rmsd" in combined_metrics.columns:
        summary["Avg_RMSD"] = round(combined_metrics["rmsd"].mean(), 4)
        summary["Std_RMSD"] = round(combined_metrics["rmsd"].std(), 4)
        summary["Min_RMSD"] = round(combined_metrics["rmsd"].min(), 4)
        summary["Max_RMSD"] = round(combined_metrics["rmsd"].max(), 4)

        # Count structures passing RMSD threshold
        passing_count = len(combined_metrics[combined_metrics["rmsd"] < rmsd_threshold])
        summary[f"Structures_RMSD<{rmsd_threshold}"] = passing_count
        summary[f"Pct_RMSD<{rmsd_threshold}"] = round((passing_count / len(combined_metrics)) * 100, 2)

    # AAR statistics (for inverse folding)
    if "percent_identity" in combined_metrics.columns:
        summary["Avg_AAR"] = round(combined_metrics["percent_identity"].mean(), 2)
        summary["Std_AAR"] = round(combined_metrics["percent_identity"].std(), 2)
        summary["Min_AAR"] = round(combined_metrics["percent_identity"].min(), 2)
        summary["Max_AAR"] = round(combined_metrics["percent_identity"].max(), 2)

    # pLDDT statistics (if available)
    if "plddt" in combined_metrics.columns:
        summary["Avg_pLDDT"] = round(combined_metrics["plddt"].mean(), 4)
        summary["Std_pLDDT"] = round(combined_metrics["plddt"].std(), 4)

    return pd.DataFrame([summary])


def aggregate_forward_folding(
    base_path: Path,
    num_jobs: int,
    rmsd_threshold: float = 2.0,
) -> dict:
    """
    Aggregate forward folding results.
    Groups by input structure, calculates structural metrics.

    Key metrics (already in CSV):
    - TM-score: from 'tm_score' column
    - RMSD: from 'rmsd' column
    - pLDDT: from 'plddt' column (if ESMFold validation used)
    """
    # Very similar to inverse_folding, but no AAR metric
    agg_dir = base_path / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    all_metrics = []
    all_sequences = []

    logger.info(f"Aggregating {num_jobs} jobs from {base_path}")

    # Collect from each job
    for job_id in range(num_jobs):
        job_dir = base_path / f"job_{job_id}"
        if not job_dir.exists():
            logger.warning(f"Job {job_id} directory not found")
            continue

        logger.info(f"Processing job {job_id}")

        # Find forward folding metrics CSV
        metrics_files = list(job_dir.glob("*forward_folding*metrics*.csv"))
        if not metrics_files:
            metrics_files = list(job_dir.glob("*_metrics_*.csv"))

        if metrics_files:
            df = pd.read_csv(metrics_files[0])
            df["job_id"] = job_id
            all_metrics.append(df)
            logger.info(f"  Found {len(df)} structures")

        # Find sequences CSV (for actual structure names)
        seq_files = list(job_dir.glob("sequences_forward_folding*.csv"))
        if not seq_files:
            seq_files = list(job_dir.glob("sequences_*.csv"))

        if seq_files:
            df_seq = pd.read_csv(seq_files[0])
            df_seq["job_id"] = job_id
            all_sequences.append(df_seq)

        # Copy PDB files
        pdb_files = list(job_dir.glob("*.pdb"))
        for pdb in pdb_files:
            new_name = f"job_{job_id}_{pdb.name}"
            shutil.copy2(pdb, agg_dir / new_name)

        if pdb_files:
            logger.info(f"  Copied {len(pdb_files)} PDB files")

    # Combine sequences
    if all_sequences:
        combined_sequences = pd.concat(all_sequences, ignore_index=True)
        output_sequences = agg_dir / "combined_forward_folding_sequences.csv"
        combined_sequences.to_csv(output_sequences, index=False)
        logger.info(f"Saved combined sequences: {output_sequences}")

    # Combine metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # Merge with sequences to get actual structure names
        if all_sequences and "input_structure" in combined_sequences.columns:
            # Merge on BOTH run_id AND job_id to avoid many-to-many joins
            # (each job restarts batch numbering from batch_000)
            structure_map = combined_sequences[["run_id", "job_id", "input_structure"]].drop_duplicates()
            combined_metrics = combined_metrics.merge(structure_map, on=["run_id", "job_id"], how="left")
            # Replace input_file with actual structure names where available
            if "input_structure" in combined_metrics.columns:
                combined_metrics["input_file"] = combined_metrics["input_structure"].fillna(
                    combined_metrics["input_file"]
                )
                combined_metrics = combined_metrics.drop(columns=["input_structure"])
                logger.info("Replaced generic batch identifiers with actual structure names from sequences CSV")

        output_metrics = agg_dir / "combined_forward_folding_metrics.csv"
        combined_metrics.to_csv(output_metrics, index=False)

        logger.info(f"Saved combined metrics: {output_metrics}")
        logger.info(f"Total structures: {len(combined_metrics)}")

        # Print summary statistics
        logger.info("\n=== Forward Folding Summary ===")

        # TM-score
        if "tm_score" in combined_metrics.columns:
            tm = combined_metrics["tm_score"].mean()
            logger.info(f"Average TM-score: {tm:.3f}")
            logger.info(f"  Min: {combined_metrics['tm_score'].min():.3f}")
            logger.info(f"  Max: {combined_metrics['tm_score'].max():.3f}")

        # RMSD
        if "rmsd" in combined_metrics.columns:
            rmsd = combined_metrics["rmsd"].mean()
            logger.info(f"Average RMSD: {rmsd:.3f} Å")
            logger.info(f"  Min: {combined_metrics['rmsd'].min():.3f}")
            logger.info(f"  Max: {combined_metrics['rmsd'].max():.3f}")

            passing = len(combined_metrics[combined_metrics["rmsd"] < rmsd_threshold])
            pct = (passing / len(combined_metrics)) * 100
            logger.info(f"Structures with RMSD < {rmsd_threshold}: {passing}/{len(combined_metrics)} ({pct:.1f}%)")

        # pLDDT
        if "plddt" in combined_metrics.columns:
            plddt = combined_metrics["plddt"].mean()
            logger.info(f"Average pLDDT: {plddt:.3f}")

        # Create per-structure summary if input_file column exists
        if "input_file" in combined_metrics.columns:
            logger.info("\n=== Creating Per-Structure Summary ===")
            summary_table = create_forward_folding_summary(combined_metrics, rmsd_threshold)
            if summary_table is not None:
                summary_csv = agg_dir / "summary_per_structure.csv"
                summary_table.to_csv(summary_csv, index=False)
                logger.info(f"Saved per-structure summary: {summary_csv}")

                logger.info("\n=== Summary Per Structure ===")
                if len(summary_table) > 20:
                    logger.info(f"\n{summary_table.head(20).to_string(index=False)}")
                    logger.info(f"... and {len(summary_table) - 20} more structures")
                else:
                    logger.info(f"\n{summary_table.to_string(index=False)}")

        # Create overall summary table (single row with aggregate stats)
        logger.info("\n=== Creating Overall Summary ===")
        overall_summary = create_overall_summary(combined_metrics, rmsd_threshold)
        if overall_summary is not None:
            overall_csv = agg_dir / "overall_summary.csv"
            overall_summary.to_csv(overall_csv, index=False)
            logger.info(f"Saved overall summary: {overall_csv}")
            logger.info(f"\n{overall_summary.to_string(index=False)}")
        else:
            logger.warning("Could not create overall summary table")

    logger.info(f"\nAggregation complete! Results in: {agg_dir}")

    return {
        "mode": "forward_folding",
        "aggregated_dir": str(agg_dir),
        "total_structures": len(combined_metrics) if all_metrics else 0,
        "average_tm_score": combined_metrics["tm_score"].mean()
        if all_metrics and "tm_score" in combined_metrics.columns
        else None,
    }


def create_forward_folding_summary(combined_metrics: pd.DataFrame, rmsd_threshold: float = 2.0) -> pd.DataFrame:
    """
    Create summary table for forward folding results, grouped by input structure.

    Returns:
        DataFrame with columns similar to inverse folding but without AAR
    """
    if "input_file" not in combined_metrics.columns:
        logger.warning("No 'input_file' column found in metrics")
        return None

    summary_rows = []

    for structure_file, group in combined_metrics.groupby("input_file"):
        num_structures = len(group)

        row = {
            "Input_Structure": structure_file,
            "Num_Structures": num_structures,
        }

        # TM-score
        if "tm_score" in group.columns:
            row["Avg_TM_Score"] = round(group["tm_score"].mean(), 4)
            row["Min_TM_Score"] = round(group["tm_score"].min(), 4)
            row["Max_TM_Score"] = round(group["tm_score"].max(), 4)

        # RMSD
        if "rmsd" in group.columns:
            row["Avg_RMSD"] = round(group["rmsd"].mean(), 4)
            row["Min_RMSD"] = round(group["rmsd"].min(), 4)
            row["Max_RMSD"] = round(group["rmsd"].max(), 4)

            passing = len(group[group["rmsd"] < rmsd_threshold])
            row[f"Structures_RMSD<{rmsd_threshold}"] = passing
            row[f"Pct_RMSD<{rmsd_threshold}"] = round((passing / num_structures) * 100, 2)

        # pLDDT
        if "plddt" in group.columns:
            row["Avg_pLDDT"] = round(group["plddt"].mean(), 4)

        summary_rows.append(row)

    if not summary_rows:
        return None

    return pd.DataFrame(summary_rows)


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

    parser = argparse.ArgumentParser(
        description="Aggregate distributed generation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect mode
  python aggregate_results.py ./examples/output 90

  # Unconditional generation (with Foldseek)
  python aggregate_results.py ./examples/generated_unconditional 20 --mode unconditional

  # Inverse folding (no Foldseek needed)
  python aggregate_results.py ./examples/generated_inverse_folding_450M 90 --mode inverse_folding

  # Forward folding (no Foldseek needed)
  python aggregate_results.py ./examples/generated_forward_folding 90 --mode forward_folding
        """,
    )

    parser.add_argument("base_output_dir", help="Base output directory with job_* subdirectories")
    parser.add_argument("num_jobs", type=int, help="Number of jobs to aggregate")

    parser.add_argument(
        "--mode",
        choices=["unconditional", "inverse_folding", "forward_folding"],
        help="Generation mode (auto-detected if not provided)",
    )

    parser.add_argument("--no-foldseek", action="store_true", help="Skip Foldseek clustering")
    parser.add_argument("--foldseek-bin", help="Path to Foldseek binary directory")
    parser.add_argument("--tmscore-threshold", type=float, default=0.5, help="TM-score threshold for clustering")
    parser.add_argument("--rmsd-threshold", type=float, default=2.0, help="RMSD threshold for filtering")

    args = parser.parse_args()

    # Handle run_foldseek logic
    run_foldseek = None if not args.no_foldseek else False

    results = aggregate_distributed_results(
        base_output_dir=args.base_output_dir,
        num_jobs=args.num_jobs,
        mode=args.mode,
        run_foldseek=run_foldseek,
        foldseek_bin_path=args.foldseek_bin,
        foldseek_tmscore_threshold=args.tmscore_threshold,
        rmsd_threshold=args.rmsd_threshold,
    )

    print("\n=== Aggregation Complete ===")
    print(f"Mode: {results.get('mode', 'unknown')}")
    print(f"Results saved to: {results['aggregated_dir']}")

    # Mode-specific summary
    if results.get("mode") == "inverse_folding":
        print(f"Total designs: {results.get('total_designs', 0)}")
        if results.get("average_aar") is not None:
            print(f"Average AAR: {results['average_aar']:.2f}%")
        if results.get("average_tm_score") is not None:
            print(f"Average TM-score: {results['average_tm_score']:.3f}")

    elif results.get("mode") == "forward_folding":
        print(f"Total structures: {results.get('total_structures', 0)}")
        if results.get("average_tm_score") is not None:
            print(f"Average TM-score: {results['average_tm_score']:.3f}")

    elif results.get("mode") == "unconditional":
        print(f"Total samples: {results.get('total_samples', 0)}")
        if results.get("diversity_results"):
            print("\nDiversity Summary:")
            for length, metrics in results["diversity_results"].items():
                print(
                    f"  Length {length}: {metrics['num_clusters']} clusters ({metrics['diversity_percentage']:.1f}% diversity)"
                )
