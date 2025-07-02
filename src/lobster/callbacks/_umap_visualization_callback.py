import logging
from collections import defaultdict
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
from upath import UPath

try:
    import umap

    UMAP_INSTALLED = True
except ImportError:
    UMAP_INSTALLED = False

logger = logging.getLogger(__name__)


class UmapVisualizationCallback(Callback):
    """Callback for generating UMAP visualizations of model embeddings.

    This callback creates visualizations of model embeddings using UMAP
    dimensionality reduction, colored by a specified grouping (e.g., dataset origin).
    It extracts embeddings from validation data and generates plots at specified
    intervals during training.

    Parameters
    ----------
    output_dir : str | UPath
        Directory to save the UMAP visualizations
    max_samples : int
        Maximum number of samples to use per group
    run_every_n_epochs : int | None
        If set, runs visualization every n epochs. If None, only runs on request
    n_neighbors : int
        UMAP parameter: number of neighbors to consider
    min_dist : float
        UMAP parameter: minimum distance between points
    random_state : int
        Random seed for reproducibility
    group_by : str | None
        Key in batch to group embeddings by (e.g., 'dataset'). If None, no grouping is done.
    group_colors : dict[str, str] | None
        Custom color map for groups. If None, uses default color palette
    requires_tokenization : bool
        Whether the model requires tokenized input for embedding

    Examples
    --------
    >>> # Group by dataset
    >>> callback = UmapVisualizationCallback(
    ...     output_dir="my_eval",
    ...     group_by="dataset"
    ... )
    >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        output_dir: str | UPath,
        max_samples: int = 1000,
        run_every_n_epochs: int | None = None,
        n_neighbors: int = 300,
        min_dist: float = 1.0,
        random_state: int = 42,
        group_by: str | None = None,
        group_colors: dict[str, str] | None = None,
        requires_tokenization: bool = True,
    ):
        super().__init__()
        self.output_dir = UPath(output_dir)
        self.max_samples = max_samples
        self.run_every_n_epochs = run_every_n_epochs
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.group_by = group_by
        self.group_colors = group_colors or {}
        self.requires_tokenization = requires_tokenization
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dict to store embeddings
        self.all_embeddings = {}

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip visualization this epoch.

        Parameters
        ----------
        trainer : L.Trainer
            The Lightning trainer instance

        Returns
        -------
        bool
            True if visualization should be skipped, False otherwise
        """
        # Don't skip if run_every_n_epochs is not set
        if self.run_every_n_epochs is None:
            return False

        # Skip if not in the main process
        if trainer.global_rank != 0:
            return True

        return trainer.current_epoch % self.run_every_n_epochs != 0

    def _extract_embeddings(
        self,
        model: L.LightningModule,
        dataloader: DataLoader,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Extract embeddings for different groups in the dataloader.

        Parameters
        ----------
        model : L.LightningModule
            The model to use
        dataloader : DataLoader
            Validation dataloader

        Returns
        -------
        dict[str, torch.Tensor] | torch.Tensor
            Embeddings per group (if group_by is set) or all embeddings (if group_by is None)
            Example per group:
                {
                    "amino_acid": torch.Tensor([...]),
                    "SMILES": torch.Tensor([...]),
                }
        """
        model.eval()

        with torch.no_grad():
            if self.group_by:
                return self._extract_grouped_embeddings(model, dataloader)
            else:
                return self._extract_all_embeddings(model, dataloader)

    def _extract_grouped_embeddings(self, model: L.LightningModule, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Extract embeddings grouped by a key in the batch.

        Parameters
        ----------
        model : L.LightningModule
            Model to extract embeddings from
        dataloader : DataLoader
            Dataloader with batches

        Returns
        -------
        dict[str, torch.Tensor]
            Embeddings grouped by the specified key

        Raises
        ------
        ValueError
            If group_by key is not found in batch or if requires_tokenization=False
            and sequence key is missing
        """
        grouped_embeddings = defaultdict(list)
        group_sample_counts = defaultdict(int)

        # Single pass to collect balanced samples per group
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if self.group_by not in batch:
                raise ValueError(f"Group key '{self.group_by}' not found in batch: {batch.keys()}")

            embeddings = self._get_embeddings_from_batch(model, batch)
            groups = batch[self.group_by]

            # Group embeddings by their corresponding group
            unique_groups = set(groups)
            for group in unique_groups:
                # Skip groups that have reached the sample limit
                if group_sample_counts[group] >= self.max_samples:
                    continue

                mask = torch.tensor([g == group for g in groups], device=embeddings.device)
                if mask.any():
                    group_embeddings = embeddings[mask].cpu()

                    # Only take up to max_samples
                    remaining_samples = self.max_samples - group_sample_counts[group]
                    if len(group_embeddings) > remaining_samples:
                        group_embeddings = group_embeddings[:remaining_samples]

                    grouped_embeddings[group].append(group_embeddings)
                    group_sample_counts[group] += len(group_embeddings)

            # Check if all groups have reached the sample limit
            if all(count >= self.max_samples for count in group_sample_counts.values() if count > 0):
                logger.info("Collected maximum samples from all encountered groups. Stopping.")
                break

        # Concatenate embeddings for each group
        result = {
            group: torch.cat(group_embeddings, dim=0)
            for group, group_embeddings in grouped_embeddings.items()
            if group_embeddings
        }

        # Log collection stats
        for group, embedding in result.items():
            logger.info(f"Group '{group}': collected {len(embedding)} samples")

        return result

    def _extract_all_embeddings(self, model: L.LightningModule, dataloader: DataLoader) -> torch.Tensor:
        """Extract all embeddings without grouping.

        Parameters
        ----------
        model : L.LightningModule
            Model to extract embeddings from
        dataloader : DataLoader
            Dataloader with batches

        Returns
        -------
        torch.Tensor
            All embeddings concatenated
        """
        all_embeddings = []
        total_samples = 0

        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch_embeddings = self._get_embeddings_from_batch(model, batch)

            # Only take up to max_samples in total
            remaining_samples = self.max_samples - total_samples
            if len(batch_embeddings) > remaining_samples:
                batch_embeddings = batch_embeddings[:remaining_samples]

            all_embeddings.append(batch_embeddings.cpu())
            total_samples += len(batch_embeddings)

            if total_samples >= self.max_samples:
                break

        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

    def _get_embeddings_from_batch(self, model: L.LightningModule, batch: dict[str, Any]) -> torch.Tensor:
        """Get embeddings from a batch based on tokenization requirements.

        Parameters
        ----------
        model : L.LightningModule
            Model to extract embeddings from
        batch : dict[str, Any]
            Batch of data

        Returns
        -------
        torch.Tensor
            Embeddings from the batch

        Raises
        ------
        ValueError
            If requires_tokenization=False and sequence key is missing
        """
        if self.requires_tokenization:
            return model.embed(batch)
        else:
            if batch.get("sequence") is None:
                raise ValueError(
                    f"The batch does not contain a 'sequence' key, which is required when requires_tokenization is False. Got keys: {batch.keys()}"
                )
            return model.embed(batch["sequence"])

    def _plot_umap(
        self,
        embeddings: dict[str, torch.Tensor] | torch.Tensor,
        output_file: str | UPath,
        model_name: str = "model",
    ) -> None:
        """Create UMAP visualization of embeddings.

        Parameters
        ----------
        embeddings : dict[str, torch.Tensor] | torch.Tensor
            Embeddings to visualize, either grouped or ungrouped
        output_file : str | UPath
            Path to save the visualization
        model_name : str
            Name of the model for the plot title
        """
        # Importing here and setting Agg is necessary to avoid issues with interactive backends
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        logger.info(f"Creating UMAP visualization for {model_name}")

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        # Process embeddings and run UMAP
        if isinstance(embeddings, dict):
            self._plot_grouped_embeddings(embeddings, ax)
        else:
            self._plot_ungrouped_embeddings(embeddings, ax)

        # Add labels and styling
        ax.set_title(f"UMAP of {model_name} Embeddings", fontsize=16, weight="bold")
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Save visualization
        self._save_figure(fig, output_file)

        logger.info(f"Saved UMAP visualization to {output_file}")

    def _plot_grouped_embeddings(self, embeddings: dict[str, torch.Tensor], ax) -> None:
        """Plot grouped embeddings on matplotlib axes.

        Parameters
        ----------
        embeddings : dict[str, torch.Tensor]
            Embeddings grouped by keys
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on
        """
        # Import matplotlib colors here to avoid issues with backends
        from matplotlib import pyplot as plt

        umap_embeddings, group_labels, available_groups = self._process_grouped_embeddings(embeddings)

        if umap_embeddings is None:
            logger.warning("No embeddings available for UMAP visualization")
            return

        # Generate color palette for groups without defined colors
        palette = plt.get_cmap("tab10").colors  # Use tab10 color palette (10 distinct colors)

        # Plot each group
        for i, group in enumerate(available_groups):
            mask = np.array(group_labels) == group
            # Use custom color if defined, otherwise use color from palette
            color = self.group_colors.get(group, palette[i % len(palette)])

            ax.scatter(
                umap_embeddings[mask, 0],
                umap_embeddings[mask, 1],
                c=[color],
                s=50,
                alpha=0.4,
                edgecolors="white",
                linewidth=0.5,
                label=group,
            )

        # Add legend
        ax.legend(
            title="Groups",
            title_fontsize=12,
            fontsize=10,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
        )

    def _plot_ungrouped_embeddings(self, embeddings: torch.Tensor, ax) -> None:
        """Plot ungrouped embeddings on matplotlib axes.

        Parameters
        ----------
        embeddings : torch.Tensor
            Tensor of embeddings
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings available for UMAP visualization")
            return

        embeddings_array = embeddings.cpu().numpy()
        umap_embeddings = self._run_umap(embeddings_array)

        # Single scatter plot without grouping
        ax.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            c="steelblue",
            s=50,
            alpha=0.4,
            edgecolors="white",
            linewidth=0.5,
        )

    def _save_figure(self, fig, output_file: str | UPath) -> None:
        """Save a figure to a file with S3 support.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        output_file : str | UPath
            Path to save the figure to

        Raises
        ------
        NotImplementedError
            If trying to save to S3
        """
        import matplotlib.pyplot as plt

        plt.tight_layout()

        # Convert to string to handle both string and UPath
        output_path = str(output_file)

        if output_path.startswith("s3://"):
            raise NotImplementedError("S3 support is not implemented yet")
        else:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")

        plt.close(fig)

    def _process_grouped_embeddings(
        self, embeddings: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray | None, list[str], list[str]]:
        """
        Process grouped embeddings for UMAP visualization.

        Parameters
        ----------
        embeddings : dict[str, torch.Tensor]
            Dictionary of embeddings grouped by keys

        Returns
        -------
        tuple[np.ndarray | None, list[str], list[str]]
            UMAP embeddings, group labels, and available groups
        """
        all_embeddings = []
        group_labels = []
        available_groups = []

        # Sort groups to ensure consistent colors
        ordered_groups = list(self.group_colors.keys()) if self.group_colors else list(embeddings.keys())

        # Extract embeddings for each group
        for group in ordered_groups:
            if group not in embeddings:
                continue

            group_embeddings = embeddings[group]
            if isinstance(group_embeddings, torch.Tensor):
                group_embeddings = group_embeddings.cpu().numpy()

            all_embeddings.append(group_embeddings)
            group_labels.extend([group] * len(group_embeddings))
            available_groups.append(group)

        if not all_embeddings:
            return None, [], []

        all_embeddings_array = np.vstack(all_embeddings)
        umap_embeddings = self._run_umap(all_embeddings_array)

        return umap_embeddings, group_labels, available_groups

    def _run_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run UMAP dimensionality reduction.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings array

        Returns
        -------
        np.ndarray
            UMAP-transformed embeddings
        """
        if not UMAP_INSTALLED:
            raise ImportError("UMAP is not installed. Please install it to use this callback.")

        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            metric="cosine",
            n_components=2,
        )

        return reducer.fit_transform(embeddings)

    def _generate_visualization(
        self,
        model: L.LightningModule,
        dataloader: DataLoader,
        output_file: str | UPath | None = None,
        model_name: str | None = None,
        epoch: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> UPath:
        """
        Extract embeddings and create UMAP visualization.

        Parameters
        ----------
        model : L.LightningModule
            The model to visualize embeddings for
        dataloader : DataLoader
            Dataloader containing data for embeddings
        output_file : str | UPath | None
            Custom output file path. If None, uses default naming
        model_name : str | None
            Name of the model. If None, inferred from model type
        epoch : int | None
            Current epoch number for output filename and title
        trainer : L.Trainer | None
            Trainer instance for logging metrics

        Returns
        -------
        UPath
            Path to the generated visualization
        """
        # Set model to eval mode
        model.eval()

        # Get model name if not provided
        if model_name is None:
            model_name = type(model).__name__

        # Extract embeddings
        embeddings = self._extract_embeddings(model, dataloader)

        # Store embeddings
        self.all_embeddings[model_name] = embeddings

        # Generate visualization filename
        if output_file is None:
            if epoch is not None:
                output_file = self.output_dir / f"umap_epoch_{epoch:04d}.png"
            else:
                output_file = self.output_dir / f"umap_{model_name}.png"
        else:
            output_file = UPath(output_file)

        # Set plot title based on epoch
        plot_title = f"{model_name} (Epoch {epoch})" if epoch is not None else model_name

        # Generate visualization
        self._plot_umap(embeddings, output_file, model_name=plot_title)

        # Log visualization path if requested
        if trainer is not None and trainer.logger:
            trainer.logger.log_metrics(
                {"umap_visualization_UPath": str(output_file)},
                step=trainer.global_step,
            )

        return output_file

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Extract embeddings and create UMAP visualization at specified epochs.

        Parameters
        ----------
        trainer : L.Trainer
            The Lightning trainer instance
        pl_module : L.LightningModule
            The Lightning module
        """
        if self._skip(trainer):
            return

        # Get validation dataloader and current epoch
        epoch = trainer.current_epoch

        try:
            val_dataloader = trainer.datamodule.val_dataloader()

            # Generate visualization using the shared method
            self._generate_visualization(
                model=pl_module,
                dataloader=val_dataloader,
                epoch=epoch,
                trainer=trainer,
            )
        except Exception as e:
            logger.error(f"Error in UMAP visualization: {str(e)}")

    def evaluate(
        self,
        module: L.LightningModule,
        dataloader: DataLoader,
        output_file: str | UPath | None = None,
    ) -> UPath:
        """
        Manually trigger UMAP visualization for a model.

        Parameters
        ----------
        module : L.LightningModule
            The model to visualize embeddings for
        dataloader : DataLoader
            Dataloader containing validation data
        output_file : str | UPath | None
            Custom output file UPath. If None, uses default naming

        Returns
        -------
        UPath
            Path to the generated visualization
        """
        return self._generate_visualization(model=module, dataloader=dataloader, output_file=output_file)
