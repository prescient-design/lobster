#!/usr/bin/env python3
"""
Script to monitor the pairs Arrow IPC file while it's being written.
This demonstrates how to read the file in the middle of computation.
"""

import time
import polars as pl
import pyarrow as pa
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def read_arrow_ipc_streaming(pairs_file_path: str):
    """
    Read Arrow IPC streaming file using record batches.
    Returns a list of all batches read so far.
    """
    batches = []
    try:
        with pa.ipc.open_stream(pairs_file_path) as reader:
            for batch in reader:
                batches.append(batch)
    except Exception as e:
        logger.debug(f"Could not read Arrow IPC stream: {e}")
        return []
    
    return batches


def monitor_pairs_file(pairs_file_path: str, check_interval: int = 30):
    """
    Monitor a pairs Arrow IPC file while it's being written.
    
    Args:
        pairs_file_path: Path to the pairs.arrow file
        check_interval: How often to check the file (in seconds)
    """
    pairs_file = Path(pairs_file_path)
    
    logger.info(f"Monitoring pairs file: {pairs_file}")
    logger.info(f"Checking every {check_interval} seconds...")
    
    last_row_count = 0
    last_file_size = 0
    
    while True:
        try:
            # Check if file exists
            if not pairs_file.exists():
                logger.debug("File does not exist yet, waiting...")
                time.sleep(check_interval)
                continue
            
            # Check file size
            current_file_size = pairs_file.stat().st_size
            if current_file_size == 0:
                logger.debug("File exists but is empty, waiting...")
                time.sleep(check_interval)
                continue
            
            # Try to read the Arrow IPC streaming file
            batches = read_arrow_ipc_streaming(pairs_file)
            
            if batches:
                # Combine all batches into a single table
                combined_table = pa.concat_tables(batches)
                pairs_df = pl.from_arrow(combined_table)
                current_row_count = len(pairs_df)
                
                if current_row_count > last_row_count or current_file_size != last_file_size:
                    logger.info(f"📊 Current progress: {current_row_count:,} pairs (file size: {current_file_size:,} bytes, batches: {len(batches)})")
                    
                    # Calculate some basic stats
                    if current_row_count > 0:
                        morgan_distances = pairs_df['morgan_tanimoto_distance'].to_list()
                        shape_distances = [d for d in pairs_df['shape_tanimoto_distance'].to_list() if not pl.is_nan(d)]
                        
                        logger.info(f"  Morgan distance - Mean: {pl.Series(morgan_distances).mean():.4f}, "
                                  f"Std: {pl.Series(morgan_distances).std():.4f}")
                        
                        if shape_distances:
                            logger.info(f"  Shape distance - Mean: {pl.Series(shape_distances).mean():.4f}, "
                                      f"Std: {pl.Series(shape_distances).std():.4f}")
                    
                    last_row_count = current_row_count
                    last_file_size = current_file_size
                else:
                    logger.debug("No new data since last check")
            else:
                logger.debug("No batches available yet")
                
        except Exception as e:
            if pairs_file.exists():
                file_size = pairs_file.stat().st_size
                logger.debug(f"Could not read file (size: {file_size:,} bytes): {e}")
            else:
                logger.debug(f"File not ready yet: {e}")
        
        time.sleep(check_interval)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python monitor_pairs.py <pairs_file_path> [check_interval]")
        print("Example: python monitor_pairs.py /path/to/pairs.arrow 30")
        sys.exit(1)
    
    pairs_file_path = sys.argv[1]
    check_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    try:
        monitor_pairs_file(pairs_file_path, check_interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}") 