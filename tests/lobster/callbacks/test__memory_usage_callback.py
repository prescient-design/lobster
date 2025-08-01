import torch
from typing import Any


class MemoryUsageCallback:
    """
    Callback for measuring memory usage patterns during model embedding operations.

    Tests memory scaling across different batch sizes and provides detailed metrics.

    Parameters:
    -----------
    batch_sizes : List[int], optional
        List of batch sizes to test. Default is [1, 5, 10, 25, 50, 100].
    verbose : bool, optional
        Whether to print detailed results. Default is True.
    """

    def __init__(self, batch_sizes: list[int] = None, verbose: bool = True):
        self.batch_sizes = batch_sizes or [1, 5, 10, 25, 50, 100]
        self.verbose = verbose
        self.memory_data: dict[int, dict[str, Any]] = {}
        self.max_successful_batch = 0

    def run_test(self, model, sequence: str, sequence_type: str = "amino_acid") -> dict[int, dict[str, Any]]:
        """
        Run the complete memory usage scaling test.

        Parameters:
        -----------
        model : Any
            The model instance to test (e.g., UME)
        sequence : str
            The test sequence to use for memory measurements
        sequence_type : str
            The type of sequence ("amino_acid", "SMILES", etc.)

        Returns:
        --------
        Dict[int, Dict[str, Any]]
            Memory usage data for each batch size tested
        """
        self.memory_data = {}
        self.max_successful_batch = 0

        for batch_size in self.batch_sizes:
            sequences = [sequence] * batch_size

            try:
                # Measure memory if on GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()

                embeddings = model.embed_sequences(sequences, sequence_type)

                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_used_mb = (memory_after - memory_before) / (1024**2)
                    memory_per_seq = memory_used_mb / batch_size
                else:
                    memory_used_mb = None
                    memory_per_seq = None

                self.memory_data[batch_size] = {
                    "status": "success",
                    "total_memory_mb": memory_used_mb,
                    "memory_per_seq_mb": memory_per_seq,
                    "output_shape": embeddings.shape,
                }

                self.max_successful_batch = batch_size

                # Clean up
                del embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                self.memory_data[batch_size] = {
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error_msg": str(e)[:100],
                }
                break  # Stop at first failure

        if self.verbose:
            self._print_results(sequence)

        return self.memory_data

    def _print_results(self, sequence: str) -> None:
        """Print the memory usage test results."""
        print(f"Test sequence length: {len(sequence)}")

        for batch_size, data in self.memory_data.items():
            if data["status"] == "success":
                if data["total_memory_mb"] is not None:
                    print(
                        f"Batch {batch_size:4d}:  {data['total_memory_mb']:.1f} MB total ({data['memory_per_seq_mb']:.2f} MB/seq)"
                    )
                else:
                    print(f"Batch {batch_size:4d}: (CPU - memory not measured)")
            else:
                print(f"Batch {batch_size:4d}: {data['error_type']}")

        print(f"\nMax successful batch size: {self.max_successful_batch}")
