import logging
import random
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

logger = logging.getLogger(__name__)


class ContrastiveRetrievalAccuracy(Metric):
    """
    Contrastive Retrieval Accuracy Metric

    This metric measures how well a model can retrieve the correct matching sequence
    from a different modality given a query sequence. It uses a user-provided transform
    function to generate pairs of sequences in different modalities (e.g., protein -> SMILES,
    nucleotide -> amino acid) and then evaluates retrieval accuracy.

    The metric works by:
    1. Taking a set of query sequences and using the transform function to generate their paired sequences
    2. For each query, computing embeddings for both query and all candidate sequences
    3. Ranking candidates by similarity to the query
    4. Computing top-k retrieval accuracy

    Parameters
    ----------
    query_sequences : list[str]
        List of sequences to use as queries
    embedding_function : Callable[[list[str], str], Tensor]
        Function that takes a list of sequences and modality, returns embeddings.
        Should return tensor of shape (n_sequences, embedding_dim)
    transform_function : Callable[[str], tuple[str, str | None, str, str]]
        Function that takes a query sequence and returns a tuple of:
        (query_sequence, target_sequence, query_modality, target_modality)
        If transformation fails, target_sequence should be None.
    k_values : list[int], optional
        List of k values for top-k accuracy computation. Default is [1, 5, 10].
    distance_metric : {'cosine', 'euclidean'}, optional
        Distance metric for similarity computation. Default is 'cosine'.
    batch_size : int, optional
        Batch size for processing sequences through the embedding function. Default is 32.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.

    Examples
    --------
    >>> sequences = ["MKLLVVVGG", "ARNDCQEGH", "FILVYWKST"]
    >>> def embedding_fn(seqs, modality):
    ...     # Your embedding function here
    ...     return torch.randn(len(seqs), 128)
    >>> def transform_fn(seq):
    ...     # Your transform function here - amino acid to SMILES
    ...     return seq, convert_to_smiles(seq), "amino_acid", "smiles"
    >>> metric = ContrastiveRetrievalAccuracy(
    ...     query_sequences=sequences,
    ...     embedding_function=embedding_fn,
    ...     transform_function=transform_fn
    ... )
    >>> metric.update()
    >>> results = metric.compute()
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        query_sequences: list[str],
        embedding_function: Callable[[list[str], str], Tensor],
        transform_function: Callable[[str], tuple[str, str | None, str, str]],
        k_values: list[int] | None = None,
        distance_metric: Literal["cosine", "euclidean"] = "cosine",
        batch_size: int = 32,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not query_sequences:
            raise ValueError("query_sequences cannot be empty")

        self.query_sequences = query_sequences
        self.embedding_function = embedding_function
        self.transform_function = transform_function
        self.k_values = k_values if k_values is not None else [1, 5, 10]
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        self.random_state = random_state

        # Set random seed for reproducibility
        random.seed(random_state)
        torch.manual_seed(random_state)

        # Generate pairs during initialization
        self.query_target_pairs = self._generate_pairs()

        # State for storing retrieval results
        self.add_state(
            "retrieval_ranks",
            default=[],
            dist_reduce_fx="cat",
        )

    def _generate_pairs(self) -> list[tuple[str, str, str, str]]:
        """Generate pairs of sequences using the specified transform function."""
        pairs = []

        for query_seq in self.query_sequences:
            try:
                # Apply transform to get the pair
                query_result, target_result, query_modality, target_modality = self.transform_function(query_seq)

                # Only add pairs where transformation was successful
                if target_result is not None:
                    pairs.append((query_result, target_result, query_modality, target_modality))
                else:
                    logger.warning(f"Transform failed for sequence: {query_seq}")

            except Exception as e:
                logger.warning(f"Error transforming sequence {query_seq}: {e}")
                continue

        logger.info(f"Generated {len(pairs)} valid pairs out of {len(self.query_sequences)} query sequences")
        return pairs

    def _process_sequences_in_batches(self, sequences: list[str], modality: str) -> Tensor:
        """Process a list of sequences through the embedding function in batches."""
        all_embeddings = []

        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i : i + self.batch_size]
            batch_embeddings = self.embedding_function(batch_sequences, modality)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _compute_distances(self, query_embeddings: Tensor, candidate_embeddings: Tensor) -> Tensor:
        """Compute distances between query and candidate embeddings."""
        if self.distance_metric == "cosine":
            query_norm = F.normalize(query_embeddings, p=2, dim=1, eps=1e-8)
            candidate_norm = F.normalize(candidate_embeddings, p=2, dim=1, eps=1e-8)

            similarity = torch.mm(query_norm, candidate_norm.t())
            similarity = torch.clamp(similarity, -1.0, 1.0)
            return 1 - similarity  # Convert similarity to distance
        else:
            return torch.cdist(query_embeddings, candidate_embeddings, p=2)

    def update(self) -> None:
        """
        Update state with contrastive retrieval analysis results.
        This method processes the query-target pairs and computes retrieval ranks.
        """
        if not self.query_target_pairs:
            logger.warning("No valid query-target pairs found. Skipping update.")
            return

        logger.info(f"Running contrastive retrieval analysis on {len(self.query_target_pairs)} pairs")

        # Extract queries, targets, and modalities
        query_sequences = [pair[0] for pair in self.query_target_pairs]
        target_sequences = [pair[1] for pair in self.query_target_pairs]
        query_modalities = [pair[2] for pair in self.query_target_pairs]
        target_modalities = [pair[3] for pair in self.query_target_pairs]

        # Verify that all query modalities are the same and all target modalities are the same
        if len(set(query_modalities)) > 1:
            raise ValueError(f"All query modalities must be the same. Found: {set(query_modalities)}")
        if len(set(target_modalities)) > 1:
            raise ValueError(f"All target modalities must be the same. Found: {set(target_modalities)}")

        query_modality = query_modalities[0]
        target_modality = target_modalities[0]

        logger.info(f"Input modality: {query_modality}, Output modality: {target_modality}")

        # Get embeddings for queries and targets
        logger.info("Computing embeddings for query sequences...")
        query_embeddings = self._process_sequences_in_batches(query_sequences, query_modality)

        logger.info("Computing embeddings for target sequences...")
        target_embeddings = self._process_sequences_in_batches(target_sequences, target_modality)

        # Compute distances between all queries and all targets
        logger.info("Computing distances...")
        distances = self._compute_distances(query_embeddings, target_embeddings)

        # For each query, find the rank of its correct target
        ranks = []
        for i in range(len(query_sequences)):
            query_distances = distances[i]  # Distances from query i to all targets

            # Sort distances and find the rank of the correct target (index i)
            sorted_indices = torch.argsort(query_distances)
            correct_target_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
            ranks.append(correct_target_rank)

        self.retrieval_ranks.extend(ranks)

        logger.info(f"Computed retrieval ranks for {len(ranks)} queries")

    def compute(self) -> dict[str, float]:
        """
        Compute contrastive retrieval accuracy metrics.

        Returns
        -------
        dict[str, float]
            Dictionary containing retrieval accuracy metrics:
            - top_k_accuracy: Top-k accuracy for each k in k_values
            - mean_rank: Mean rank of correct retrievals
            - median_rank: Median rank of correct retrievals
            - mean_reciprocal_rank: Mean reciprocal rank
        """
        if not self.retrieval_ranks:
            raise ValueError("No retrieval results available. Call update() first.")

        ranks = torch.tensor(self.retrieval_ranks, dtype=torch.float32)
        n_queries = len(ranks)

        # Compute top-k accuracies
        results = {}
        for k in self.k_values:
            top_k_accuracy = torch.sum(ranks <= k).item() / n_queries
            results[f"top_{k}_accuracy"] = top_k_accuracy

        # Compute additional metrics
        results["mean_rank"] = torch.mean(ranks).item()
        results["median_rank"] = torch.median(ranks).item()
        results["mean_reciprocal_rank"] = torch.mean(1.0 / ranks).item()

        # Log results
        logger.info("Contrastive Retrieval Accuracy Results:")
        for k in self.k_values:
            logger.info(f"  Top-{k} accuracy: {results[f'top_{k}_accuracy']:.3f}")
        logger.info(f"  Mean rank: {results['mean_rank']:.3f}")
        logger.info(f"  Median rank: {results['median_rank']:.3f}")
        logger.info(f"  Mean reciprocal rank: {results['mean_reciprocal_rank']:.3f}")

        # Reset state
        self.retrieval_ranks.clear()

        return results
