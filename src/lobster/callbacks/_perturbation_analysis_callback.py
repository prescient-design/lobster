"""
Perturbation Analysis Callback for evaluating model robustness through sequence perturbations.

This callback analyzes how sensitive a model's embeddings are to different types of perturbations
by measuring cosine distances between original and perturbed sequence embeddings.

Credits: Josh Southern for the original perturbation analysis methodology.
"""

import logging
import random

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import Callback
from scipy.spatial.distance import cosine
from tqdm import tqdm
from upath import UPath

from lobster.constants import Modality

logger = logging.getLogger(__name__)


class PerturbationAnalysisCallback(Callback):
    """Callback for analyzing model robustness through sequence perturbations.

    This callback evaluates how sensitive a model's embeddings are to different types
    of perturbations by measuring the cosine distance between original and perturbed
    sequence embeddings. It supports both shuffling and single-point mutations.

    Credits: Josh Southern for the original perturbation analysis methodology.

    Parameters
    ----------
    output_dir : str | UPath
        Directory to save perturbation analysis visualizations
    sequences : list[str]
        List of sequences to analyze
    num_shuffles : int
        Number of shuffled versions to generate per sequence
    mutation_tokens : list[str] | None
        List of tokens to use for mutations. Required for SMILES and 3D_COORDINATES modalities.
        For AMINO_ACID and NUCLEOTIDE, defaults will be used if not provided.
    run_every_n_epochs : int | None
        If set, runs analysis every n epochs. If None, only runs on request
    random_state : int
        Random seed for reproducibility
    modality : str
        Modality to use for embedding extraction and default tokens.
        Must be one of the modalities defined in lobster.constants.Modality.

    Examples
    --------
    >>> # Analyze protein sequences (amino acid modality)
    >>> from lobster.constants import Modality
    >>> protein_sequences = ["QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS"]
    >>> callback = PerturbationAnalysisCallback(
    ...     output_dir="perturbation_analysis",
    ...     sequences=protein_sequences,
    ...     modality=Modality.AMINO_ACID,
    ...     num_shuffles=10
    ... )

    >>> # Analyze nucleotide sequences
    >>> nucleotide_sequences = ["ATCGATCG", "GCTAGCTA"]
    >>> callback = PerturbationAnalysisCallback(
    ...     output_dir="perturbation_analysis",
    ...     sequences=nucleotide_sequences,
    ...     modality=Modality.NUCLEOTIDE,
    ...     num_shuffles=10
    ... )

    >>> # Analyze SMILES sequences (requires explicit mutation_tokens)
    >>> smiles_sequences = ["CCO", "CC(C)O"]
    >>> callback = PerturbationAnalysisCallback(
    ...     output_dir="perturbation_analysis",
    ...     sequences=smiles_sequences,
    ...     modality=Modality.SMILES,
    ...     mutation_tokens=list("CHNOSPFIBrCl()[]=#@+-.1234567890"),
    ...     num_shuffles=10
    ... )
    """

    def get_default_mutation_tokens(self, modality: str) -> list[str]:
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
        }

        if modality not in modality_defaults:
            raise ValueError(
                f"Modality '{modality}' does not have default mutation tokens. "
                f"Supported modalities with defaults: {list(modality_defaults.keys())}. "
                f"For other modalities, please provide explicit mutation_tokens."
            )

        return modality_defaults[modality]

    def __init__(
        self,
        output_dir: str | UPath,
        sequences: list[str],
        num_shuffles: int = 10,
        mutation_tokens: list[str] | None = None,
        run_every_n_epochs: int | None = None,
        random_state: int = 42,
        modality: str = Modality.AMINO_ACID,
    ):
        super().__init__()
        self.output_dir = UPath(output_dir)
        self.sequences = sequences
        self.num_shuffles = num_shuffles
        self.modality = modality
        self.run_every_n_epochs = run_every_n_epochs
        self.random_state = random_state

        # Set mutation tokens based on modality
        if mutation_tokens is not None:
            self.mutation_tokens = mutation_tokens
        else:
            try:
                self.mutation_tokens = self.get_default_mutation_tokens(modality)
            except ValueError as e:
                raise ValueError(
                    f"Modality '{modality}' requires explicit mutation_tokens. "
                    f"Please provide the mutation_tokens parameter."
                ) from e

        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip analysis this epoch."""
        # Don't skip if run_every_n_epochs is not set
        if self.run_every_n_epochs is None:
            return False

        # Skip if not in the main process
        if trainer.global_rank != 0:
            return True

        return trainer.current_epoch % self.run_every_n_epochs != 0

    def _embed_sequences(self, model: L.LightningModule, sequences: list[str]) -> torch.Tensor:
        """Get embeddings for a list of sequences."""
        model.eval()
        with torch.no_grad():
            if hasattr(model, "embed_sequences"):
                # Use the model's embed_sequences method
                embeddings = model.embed_sequences(sequences, modality=self.modality, aggregate=True)
            elif hasattr(model, "model") and hasattr(model.model, "embed_sequences"):
                # Use the underlying model's embed_sequences method
                embeddings = model.model.embed_sequences(sequences, modality=self.modality, aggregate=True)
            else:
                raise NotImplementedError("Model must implement embed_sequences method or have model.embed_sequences")
        return embeddings

    def _compute_shuffling_distances(self, model: L.LightningModule, sequence: str) -> list[float]:
        """Compute distances between original and shuffled sequence embeddings."""
        original_embedding = self._embed_sequences(model, [sequence]).squeeze(0)
        shuffling_distances = []

        for _ in range(self.num_shuffles):
            # Create shuffled sequence
            shuffled_chars = list(sequence)
            random.shuffle(shuffled_chars)
            shuffled_sequence = "".join(shuffled_chars)

            # Get embedding for shuffled sequence
            shuffled_embedding = self._embed_sequences(model, [shuffled_sequence]).squeeze(0)

            # Compute cosine distance
            distance = cosine(original_embedding.cpu().numpy(), shuffled_embedding.cpu().numpy())
            shuffling_distances.append(distance)

        return shuffling_distances

    def _compute_mutation_distances(self, model: L.LightningModule, sequence: str) -> np.ndarray:
        """Compute distances between original and single-point mutation embeddings."""
        original_embedding = self._embed_sequences(model, [sequence]).squeeze(0)
        mutation_distances = []

        for i in range(len(sequence)):
            position_distances = []
            for token in self.mutation_tokens:
                # Create mutated sequence
                mutated_sequence = sequence[:i] + token + sequence[i + 1 :]

                # Get embedding for mutated sequence
                mutated_embedding = self._embed_sequences(model, [mutated_sequence]).squeeze(0)

                # Compute cosine distance
                distance = cosine(original_embedding.cpu().numpy(), mutated_embedding.cpu().numpy())
                position_distances.append(distance)

            mutation_distances.append(position_distances)

        return np.array(mutation_distances)

    def _create_perturbation_heatmap(self, mutation_distances: np.ndarray, sequence: str, output_file: UPath) -> None:
        """Create and save a heatmap of mutation perturbations."""
        df_perturbations = pd.DataFrame(
            mutation_distances,
            index=[sequence[i] for i in range(mutation_distances.shape[0])],
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
        plt.title(f"{self.modality.replace('_', ' ').title()} Mutation Perturbation Analysis")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _run_analysis(
        self,
        model: L.LightningModule,
        sequences: list[str],
        save_heatmap: bool = True,
        output_dir: UPath | None = None,
        epoch: int | None = None,
    ) -> dict[str, float]:
        """Run the perturbation analysis and return metrics."""
        if not sequences:
            logger.warning("No sequences provided for analysis")
            return {}

        all_shuffling_distances = []
        all_mutation_distances = []

        # Use first sequence for detailed analysis (heatmap)
        main_sequence = sequences[0]

        logger.info(f"Running perturbation analysis on {len(sequences)} sequences")

        for i, sequence in enumerate(tqdm(sequences, desc="Analyzing sequences")):
            # Compute shuffling distances
            shuffling_distances = self._compute_shuffling_distances(model, sequence)
            all_shuffling_distances.extend(shuffling_distances)

            # Compute mutation distances (only for first sequence to save time)
            if i == 0:
                mutation_distances = self._compute_mutation_distances(model, sequence)
                all_mutation_distances = mutation_distances

        # Calculate metrics
        avg_shuffling_distance = np.mean(all_shuffling_distances)
        avg_mutation_distance = np.mean(all_mutation_distances)
        distance_ratio = avg_shuffling_distance / avg_mutation_distance if avg_mutation_distance > 0 else 0

        metrics = {
            "avg_shuffling_distance": avg_shuffling_distance,
            "avg_mutation_distance": avg_mutation_distance,
            "distance_ratio": distance_ratio,
        }

        # Log metrics
        logger.info("Perturbation Analysis Results:")
        logger.info(f"  Average shuffling distance: {avg_shuffling_distance:.6f}")
        logger.info(f"  Average mutation distance: {avg_mutation_distance:.6f}")
        logger.info(f"  Distance ratio (shuffle/mutation): {distance_ratio:.6f}")

        # Create and save heatmap
        if save_heatmap:
            heatmap_dir = output_dir if output_dir is not None else self.output_dir
            heatmap_file = (
                heatmap_dir / f"perturbation_heatmap_epoch_{epoch}.png"
                if epoch
                else heatmap_dir / "perturbation_heatmap.png"
            )
            self._create_perturbation_heatmap(all_mutation_distances, main_sequence, heatmap_file)
            logger.info(f"Perturbation heatmap saved to {heatmap_file}")

        return metrics

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run perturbation analysis at the end of validation epochs."""
        if self._skip(trainer):
            return

        # Run analysis
        metrics = self._run_analysis(pl_module, self.sequences, save_heatmap=True, epoch=trainer.current_epoch)

        # Log metrics to trainer
        for key, value in metrics.items():
            trainer.logger.experiment.add_scalar(f"perturbation_analysis/{key}", value, trainer.current_epoch)

    def evaluate(
        self,
        module: L.LightningModule,
        save_heatmap: bool = True,
        output_dir: str | UPath | None = None,
    ) -> dict[str, float]:
        """Evaluate the model using perturbation analysis.

        Parameters
        ----------
        module : L.LightningModule
            The model to evaluate
        save_heatmap : bool
            Whether to save the perturbation heatmap image
        output_dir : str | UPath | None
            Directory to save the heatmap image. If None, uses the default output_dir

        Returns
        -------
        dict[str, float]
            Dictionary containing perturbation analysis metrics
        """
        # Run analysis
        metrics = self._run_analysis(
            module,
            self.sequences,
            save_heatmap=save_heatmap,
            output_dir=UPath(output_dir) if output_dir is not None else None,
        )

        return metrics
