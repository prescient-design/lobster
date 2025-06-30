import logging
import random
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor
from torchmetrics import Metric
from upath import UPath

from lobster.constants import Modality

logger = logging.getLogger(__name__)


class PerturbationScore(Metric):
    """
    Perturbation Score Metric

    This metric quantifies how sensitive a model's embeddings are to different types of perturbations
    by measuring cosine distances between original and perturbed sequence embeddings.

    1. Compute embedding distance of randomly shuffled sequences.
    2. Compute embedding distance of single-point mutations.
    3. Compute perturbation score as the ratio of shuffling to mutation distances: d(shuffled) / d(mutated)

    Higher perturbation scores → greater sensitivity to perturbations; interpretation depends on context.

    Parameters
    ----------
    sequence : str
        Single sequence to analyze
    embedding_function : Callable[[list[str], str], Tensor]
        Function that takes a list of sequences and modality, returns  embeddings.
        Should return tensor of shape (n_sequences, embedding_dim)
    modality : str
        Modality to use for embedding extraction and default tokens.
        Must be one of the modalities defined in lobster.constants.Modality.
    num_shuffles : int, optional
        Number of shuffled versions to generate per sequence. Default is 10.
    mutation_tokens : list[str] | None, optional
        List of tokens to use for mutations. If None, defaults will be used based on modality
        if available.
    batch_size : int, optional
        Batch size for processing sequences through the embedding function. Default is 32.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    save_heatmap : bool, optional
        Whether to save a perturbation heatmap. Default is False.
    output_file : UPath | None, optional
        Path to save the heatmap image. Required if save_heatmap is True.

    Reference
    ---------
    Credits: Josh Southern for the original perturbation analysis methodology.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        sequence: str,
        embedding_function: Callable[[list[str], str], Tensor],
        modality: str,
        num_shuffles: int = 10,
        mutation_tokens: list[str] | None = None,
        batch_size: int = 32,
        random_state: int = 0,
        save_heatmap: bool = False,
        output_file: UPath | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not sequence:
            raise ValueError("sequence cannot be empty")

        self.sequence = sequence
        self.embedding_function = embedding_function
        self.modality = modality
        self.num_shuffles = num_shuffles
        self.batch_size = batch_size
        self.random_state = random_state
        self.save_heatmap = save_heatmap
        self.output_file = output_file

        self.mutation_tokens = mutation_tokens if mutation_tokens is not None else get_default_mutation_tokens(modality)

        random.seed(random_state)
        torch.manual_seed(random_state)

        self.add_state(
            "shuffling_distances",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "mutation_distances",
            default=[],
            dist_reduce_fx="cat",
        )

    def _process_sequences_in_batches(self, sequences: list[str]) -> Tensor:
        """Process a list of sequences through the embedding function in batches."""
        all_embeddings = []

        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i : i + self.batch_size]
            batch_embeddings = self.embedding_function(batch_sequences, self.modality)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _compute_shuffling_distances(self) -> list[float]:
        """Compute distances between original and shuffled sequence embeddings."""
        # Create all shuffled sequences first
        shuffled_sequences = []
        for _ in range(self.num_shuffles):
            shuffled_chars = list(self.sequence)
            random.shuffle(shuffled_chars)
            shuffled_sequences.append("".join(shuffled_chars))

        # Get embeddings for original and all shuffled sequences
        all_sequences = [self.sequence] + shuffled_sequences
        all_embeddings = self._process_sequences_in_batches(all_sequences)

        # original_embedding: (embedding_dim,)
        original_embedding = all_embeddings[0]
        # shuffled_embeddings: (num_shuffles, embedding_dim)
        shuffled_embeddings = all_embeddings[1:]

        # Compute cosine distances
        distances = 1 - torch.nn.functional.cosine_similarity(
            original_embedding.unsqueeze(0).expand(shuffled_embeddings.shape[0], -1), shuffled_embeddings
        )

        return distances.tolist()

    def _compute_mutation_distances(self) -> torch.Tensor:
        """Compute distances between original and single-point mutation embeddings."""
        # Create all mutated sequences first
        mutated_sequences = []
        sequence_positions = []
        mutation_indices = []

        for i in range(len(self.sequence)):
            for j, token in enumerate(self.mutation_tokens):
                mutated_sequence = self.sequence[:i] + token + self.sequence[i + 1 :]
                mutated_sequences.append(mutated_sequence)
                sequence_positions.append(i)
                mutation_indices.append(j)

        # Get embeddings for original and all mutated sequences
        all_sequences = [self.sequence] + mutated_sequences
        all_embeddings = self._process_sequences_in_batches(all_sequences)

        # original_embedding: (embedding_dim,)
        original_embedding = all_embeddings[0]
        # mutated_embeddings: (num_mutations, embedding_dim)
        mutated_embeddings = all_embeddings[1:]

        # Compute cosine distances
        distances = 1 - torch.nn.functional.cosine_similarity(
            original_embedding.unsqueeze(0).expand(mutated_embeddings.shape[0], -1), mutated_embeddings
        )

        # Reshape to (sequence_length, num_mutation_tokens)
        mutation_distances = distances.reshape(len(self.sequence), len(self.mutation_tokens))

        return mutation_distances

    def _create_perturbation_heatmap(self, mutation_distances: torch.Tensor) -> None:
        """Create and save a heatmap of mutation perturbations."""
        if self.output_file is None:
            raise ValueError("output_file must be provided when save_heatmap is True")

        # mutation_distances: (sequence_length, num_mutation_tokens)
        # Convert to numpy for pandas DataFrame
        df_perturbations = pd.DataFrame(
            mutation_distances.cpu().numpy(),
            index=[self.sequence[i] for i in range(mutation_distances.shape[0])],
            columns=self.mutation_tokens,
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            df_perturbations.T,
            cmap="viridis",
            cbar_kws={"label": "Cosine Distance"},
            vmax=df_perturbations.values.max(),
            vmin=df_perturbations.values.min(),
        )

        plt.xlabel("Sequence Position")
        plt.ylabel(f"{self.modality.replace('_', ' ').title()} Token")
        plt.title(f"{self.modality.replace('_', ' ').title()} Mutation Perturbation Score")
        plt.tight_layout()
        plt.savefig(self.output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def update(self) -> None:
        """
        Update state with perturbation analysis results.
        This method processes the single sequence and computes the perturbation scores.
        """
        logger.info(f"Running perturbation score analysis on sequence of length {len(self.sequence)}")
        logger.info(f"Processing sequences in batches of size {self.batch_size}")

        # Compute shuffling embedding distances
        logger.info("Computing shuffling distances...")
        shuffling_distances = self._compute_shuffling_distances()
        self.shuffling_distances.append(torch.tensor(shuffling_distances))

        # Compute mutation distances
        logger.info("Computing mutation distances...")
        mutation_distances = self._compute_mutation_distances()
        self.mutation_distances.append(mutation_distances.flatten())

        # Create and save heatmap
        if self.save_heatmap:
            self._create_perturbation_heatmap(mutation_distances)
            logger.info(f"Perturbation heatmap saved to {self.output_file}")

    def compute(self) -> dict[str, float]:
        """
        Compute perturbation score metrics.

        Returns
        -------
        dict[str, float]
            Dictionary containing perturbation score metrics:
            - avg_shuffling_distance: Average cosine distance for shuffled sequences
            - avg_mutation_distance: Average cosine distance for mutations
            - distance_ratio: Ratio of shuffling to mutation distances
        """
        if not self.shuffling_distances or not self.mutation_distances:
            raise ValueError("No perturbation analysis results available. Call update() first.")

        shuffling_distances = torch.cat(self.shuffling_distances, dim=0)
        mutation_distances = torch.cat(self.mutation_distances, dim=0)

        # Calculate metrics
        avg_shuffling_distance = torch.mean(shuffling_distances).item()
        avg_mutation_distance = torch.mean(mutation_distances).item()

        # Use the stored perturbation score ratio
        distance_ratio = avg_shuffling_distance / avg_mutation_distance

        metrics = {
            "avg_shuffling_embedding_distance": avg_shuffling_distance,
            "avg_mutation_embedding_distance": avg_mutation_distance,
            "shuffling_mutation_ratio": distance_ratio,
        }

        # Log metrics
        logger.info("Perturbation Score Results:")
        logger.info(f"  Average shuffling embedding distance: {avg_shuffling_distance:.3f}")
        logger.info(f"  Average mutation embedding distance: {avg_mutation_distance:.3f}")
        logger.info(f"  Perturbation score (shuffling/mutation): {distance_ratio:.3f}")

        # Reset state
        self.shuffling_distances.clear()
        self.mutation_distances.clear()

        return metrics


def get_default_mutation_tokens(modality: str) -> list[str]:
    """Get default mutation tokens for supported modalities.

    Parameters
    ----------
    modality : str
        The modality type (must be one of the supported modalities)

    Returns
    -------
    list[str]
        List of default tokens for the given modality

    Raises
    ------
    ValueError
        If modality is not supported for default tokens
    """
    modality_defaults = {
        Modality.AMINO_ACID: list("RKHDESTNQAVILMFYWGP"),
        Modality.NUCLEOTIDE: list("ATCG"),
        Modality.SMILES: list("CHNOSPFIBrCl()[]=#@+-.1234567890"),
    }

    if modality not in modality_defaults:
        raise ValueError(
            f"Modality '{modality}' does not have default mutation tokens. "
            f"Supported modalities with defaults: {list(modality_defaults.keys())}. "
            f"For other modalities, please provide explicit mutation_tokens."
        )

    return modality_defaults[modality]
