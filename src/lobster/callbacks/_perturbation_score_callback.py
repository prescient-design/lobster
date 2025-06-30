import logging

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from upath import UPath

from lobster.constants import Modality
from lobster.metrics import PerturbationScore

logger = logging.getLogger(__name__)


class PerturbationScoreCallback(Callback):
    """Callback for analyzing model robustness through sequence perturbations.

    This callback evaluates how sensitive a model's embeddings are to different types
    of perturbations by measuring the cosine distance between original and perturbed
    sequence embeddings. It supports both shuffling and single-point mutations.

    Credits: Josh Southern for the original perturbation analysis notebook.

    Parameters
    ----------
    sequence : str
        Single sequence to analyze
    num_shuffles : int
        Number of shuffled versions to generate per sequence
    mutation_tokens : list[str] | None
        List of tokens to use for mutations. Required for SMILES and 3D_COORDINATES modalities.
        For AMINO_ACID, SMILES, and NUCLEOTIDE, defaults will be used if not provided.
    run_every_n_epochs : int | None
        If set, runs analysis every n epochs. If None, only runs on request
    random_state : int
        Random seed for reproducibility
    modality : str
        Modality to use for embedding extraction and default tokens.
        Must be one of the modalities defined in lobster.constants.Modality.
    output_dir : str | None
        Directory to save perturbation analysis visualizations. If None, no heatmaps will be saved.
    save_heatmap : bool
        Whether to save the perturbation heatmap image

    Examples
    --------
    >>> # Analyze protein sequence (amino acid modality)
    >>> from lobster.constants import Modality
    >>> protein_sequence = "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNT"
    >>> callback = PerturbationScoreCallback(
    ...     sequence=protein_sequence,
    ...     modality=Modality.AMINO_ACID,
    ...     num_shuffles=10,
    ...     output_dir="perturbation_analysis",
    ...     save_heatmap=True,
    ... )

    >>> # Analyze SMILES sequence
    >>> smiles_sequence = "CCO"
    >>> callback = PerturbationScoreCallback(
    ...     sequence=smiles_sequence,
    ...     modality=Modality.SMILES,
    ...     mutation_tokens=list("CHNOSPFIBrCl()[]=#@+-.1234567890"),
    ...     num_shuffles=10
    ... )
    """

    def __init__(
        self,
        sequence: str,
        num_shuffles: int = 10,
        mutation_tokens: list[str] | None = None,
        run_every_n_epochs: int | None = None,
        random_state: int = 42,
        modality: str = Modality.AMINO_ACID,
        save_heatmap: bool = False,
        output_dir: str | None = None,
    ):
        super().__init__()

        if save_heatmap and output_dir is None:
            raise ValueError("output_dir must be provided when save_heatmap is True")

        self.output_dir = UPath(output_dir) if output_dir is not None else None
        self.sequence = sequence
        self.num_shuffles = num_shuffles
        self.modality = modality
        self.run_every_n_epochs = run_every_n_epochs
        self.random_state = random_state
        self.mutation_tokens = mutation_tokens
        self.save_heatmap = save_heatmap

        # Create output directory only if save_heatmap is True
        if self.save_heatmap and self.output_dir is not None:
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

    def _create_embedding_function(self, model: L.LightningModule):
        """Create an embedding function from the model."""

        def embedding_function(sequences, modality):
            model.eval()
            with torch.no_grad():
                if hasattr(model, "embed_sequences"):
                    return model.embed_sequences(sequences, modality=modality, aggregate=True)
                else:
                    raise NotImplementedError("Model must implement embed_sequences method")

        return embedding_function

    def _compute_scores(
        self,
        model: L.LightningModule,
        output_dir: UPath | None = None,
        step: int | None = None,
    ) -> dict[str, float]:
        """Run the perturbation analysis and return metrics."""
        # Determine output file for heatmap
        output_file = None
        if self.save_heatmap:
            heatmap_dir = output_dir if output_dir is not None else self.output_dir
            output_file = (
                heatmap_dir / f"perturbation_heatmap_step_{step}.png"
                if step
                else heatmap_dir / "perturbation_heatmap.png"
            )

        # Create embedding function
        embedding_function = self._create_embedding_function(model)

        # Create and run the PerturbationScore metric
        metric = PerturbationScore(
            sequence=self.sequence,
            embedding_function=embedding_function,
            modality=self.modality,
            num_shuffles=self.num_shuffles,
            mutation_tokens=self.mutation_tokens,
            random_state=self.random_state,
            save_heatmap=self.save_heatmap,
            output_file=output_file,
        )

        metric.update()

        return metric.compute()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run perturbation analysis at the end of validation epochs."""
        if self._skip(trainer):
            return

        metrics = self._compute_scores(pl_module, step=trainer.global_step)

        for key, value in metrics.items():
            trainer.logger.experiment.log(
                {f"perturbation/{key}": value},
            )

    def evaluate(
        self,
        module: L.LightningModule,
        output_dir: str | UPath | None = None,
    ) -> dict[str, float]:
        """Evaluate the model using perturbation analysis.

        Parameters
        ----------
        module : L.LightningModule
            The model to evaluate
        output_dir : str | UPath | None
            Directory to save the heatmap image. If None, uses the default output_dir

        Returns
        -------
        dict[str, float]
            Dictionary containing perturbation analysis metrics
        """
        return self._compute_scores(
            module,
            output_dir=UPath(output_dir) if output_dir is not None else None,
        )
