import logging
from collections import defaultdict

import lightning as L
import numpy as np
import torch
import umap
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
from upath import UPath

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
        Custom color map for groups. If None, uses default colors
    enable_grouping : bool
        Whether to enable grouping of embeddings by the group_by key

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
        group_by: str | None = "dataset",
        group_colors: dict[str, str] | None = None,
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dict to store embeddings
        self.all_embeddings = {}

    def _skip(self, trainer: L.Trainer) -> bool:
        """Determine if we should skip visualization this epoch."""
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
                # Group embeddings by the specified key
                grouped_embeddings = defaultdict(list)

                for batch in tqdm(dataloader, desc="Extracting embeddings"):
                    if self.group_by not in batch:
                        raise ValueError(f"Group key '{self.group_by}' not found in batch")

                    embeddings = model.embed(batch)
                    groups = batch[self.group_by]

                    # Group embeddings by their corresponding group
                    for group in set(groups):
                        mask = torch.tensor([g == group for g in groups], device=embeddings.device)
                        if mask.any():
                            grouped_embeddings[group].append(embeddings[mask].cpu())

                # Concatenate embeddings for each group
                return {
                    group: torch.cat(group_embeddings, dim=0)
                    for group, group_embeddings in grouped_embeddings.items()
                    if group_embeddings
                }
            else:
                # Extract all embeddings without grouping
                all_embeddings = [model.embed(batch).cpu() for batch in tqdm(dataloader, desc="Extracting embeddings")]

                return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

    def _plot_umap(
        self,
        embeddings: dict[str, torch.Tensor] | torch.Tensor,
        output_file: str | UPath,
        model_name: str = "model",
    ) -> None:
        """Create UMAP visualization of embeddings."""
        # Importing here and setting Agg is necessary to avoid issues with interactive backends
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        logger.info(f"Creating UMAP visualization for {model_name}")

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        # Process embeddings and run UMAP
        if isinstance(embeddings, dict):
            umap_embeddings, group_labels, available_groups = self._process_grouped_embeddings(embeddings)

            # Plot each group
            for group in available_groups:
                mask = np.array(group_labels) == group
                ax.scatter(
                    umap_embeddings[mask, 0],
                    umap_embeddings[mask, 1],
                    c=[self.group_colors.get(group, "gray")],
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
        else:
            # Process ungrouped embeddings
            embeddings_array = self._subsample_embeddings(embeddings.cpu().numpy())
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

    def _save_figure(self, fig, output_file: str | UPath) -> None:
        """Save a figure to a file with S3 support."""
        import matplotlib.pyplot as plt

        plt.tight_layout()

        if str(output_file).startswith("s3://"):
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

            # Subsample if needed
            group_embeddings = self._subsample_embeddings(group_embeddings)

            all_embeddings.append(group_embeddings)
            group_labels.extend([group] * len(group_embeddings))
            available_groups.append(group)

        if not all_embeddings:
            return None, [], []

        all_embeddings_array = np.vstack(all_embeddings)
        umap_embeddings = self._run_umap(all_embeddings_array)

        return umap_embeddings, group_labels, available_groups

    def _subsample_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Subsample embeddings if they exceed the maximum sample count.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings array

        Returns
        -------
        np.ndarray
            Subsampled embeddings
        """
        if len(embeddings) > self.max_samples:
            indices = np.random.choice(len(embeddings), int(self.max_samples), replace=False)
            return embeddings[indices]
        return embeddings

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
        """Extract embeddings and create UMAP visualization at specified epochs."""
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
