import torch
import lightning
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import glob
from lobster.model.latent_generator.datasets import StructureBackboneTransform

logger = logging.getLogger(__name__)

# DSSP secondary structure states
DSSP_STATES = ["a", "b", "c"]  # Helix, Sheet, Coil
DSSP_MAP = {state: i for i, state in enumerate(DSSP_STATES)}


class DSSPLinearProbe(lightning.Callback):
    """Linear probe callback for predicting DSSP secondary structure from learned tokens.

    This callback trains a linear classifier on top of the learned tokens to predict
    secondary structure states (helix, sheet, coil). It computes accuracy metrics
    during training and saves predictions.
    """

    def __init__(
        self,
        token_dim: int,
        train_paths: str = None,
        val_paths: str = None,
        max_num_structures: int = 1000,
        save_every_n: int = 1000,
        learning_rate: float = 1,
        weight_decay: float = 1e-4,
        batch_size: int = 200,
        run_every_n_steps: int = 1000,
        num_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.001,
        max_seq_len: int = 512,
    ):
        """Initialize the DSSP linear probe.

        Args:
            token_dim: Dimension of the learned tokens
            train_paths: Directory containing training structure .pt files with DSSP annotations
            val_paths: Directory containing validation structure .pt files with DSSP annotations
            max_num_structures: Maximum number of structures to use for training and validation
            save_every_n: Save predictions every N batches
            learning_rate: Learning rate for the linear probe
            weight_decay: Weight decay for the linear probe
            batch_size: Number of structures to process in each batch
            run_every_n_steps: Run the callback every N training steps
            num_epochs: Maximum number of epochs to train the probe
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in validation accuracy to be considered as improvement
        """
        super().__init__()
        self.token_dim = token_dim
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.max_num_structures = max_num_structures
        self.save_every_n = save_every_n
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.run_every_n_steps = run_every_n_steps
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = StructureBackboneTransform()
        self.max_seq_len = max_seq_len
        # Load structure paths
        self.train_files = sorted(glob.glob(f"{self.train_paths}/*.pt"))
        self.val_files = sorted(glob.glob(f"{self.val_paths}/*.pt"))

        # limit the number of structures
        self.train_files = self.train_files[: self.max_num_structures]
        self.val_files = self.val_files[: self.max_num_structures]

        if not self.train_files:
            logger.warning(f"No training files found in {self.train_paths}")
        if not self.val_files:
            logger.warning(f"No validation files found in {self.val_paths}")

        # Initialize cached data
        self.train_batches = None
        self.val_batches = None

    def _load_structure(self, file_path: str) -> dict:
        """Load and process a structure file.

        Args:
            file_path: Path to the structure .pt file

        Returns:
            Processed structure dictionary
        """
        # Load the structure
        batch = torch.load(file_path, weights_only=False)

        # Process the structure
        batch = self.transform(batch)

        batch["coords_res"] = batch["coords_res"].to(self.device)[None]
        batch["sequence"] = batch["sequence"].to(self.device)[None]
        batch["dssp"] = batch["dssp"].to(self.device)[None]
        batch["mask"] = batch["mask"].to(self.device)[None]
        batch["indices"] = batch["indices"].to(self.device)[None]

        return batch

    def _prepare_batches(self, pl_module: lightning.LightningModule, file_paths: list[str]) -> list[dict]:
        """Prepare and cache batches of data.

        Args:
            pl_module: PyTorch Lightning module
            file_paths: List of paths to structure files

        Returns:
            List[Dict]: List of batches containing tokens and DSSP labels
        """
        batches = []

        # Process files in batches
        for i in range(0, len(file_paths), self.batch_size):
            batch_paths = file_paths[i : i + self.batch_size]
            batch_structures = []
            batch_dssp = []
            max_len = 0

            # First pass: get max length and process structures
            for file_path in batch_paths:
                try:
                    structure_batch = self._load_structure(file_path)

                    # Get sequence length
                    B, L = structure_batch["sequence"].shape[:2]
                    if L > self.max_seq_len:  # Skip if sequence is too long
                        continue

                    # Get tokens from the tokenizer
                    tokens = pl_module.forward(structure_batch)[:B]
                    tokens = torch.argmax(tokens, dim=-1)
                    tokens = F.one_hot(tokens, num_classes=self.token_dim)
                    tokens = tokens.float().to(self.device)
                    dssp = structure_batch["dssp"].to(self.device)

                    # Update max length
                    max_len = max(max_len, tokens.shape[1])

                    batch_structures.append(tokens)
                    batch_dssp.append(dssp)

                except Exception as e:
                    logger.error(f"Error processing structure {file_path}: {str(e)}")
                    continue

            if not batch_structures:
                continue

            # Second pass: pad all sequences to max_len
            padded_structures = []
            padded_dssp = []
            for tokens, dssp in zip(batch_structures, batch_dssp):
                # Pad tokens
                pad_len = max_len - tokens.shape[1]
                if pad_len > 0:
                    tokens = F.pad(tokens, (0, 0, 0, pad_len))
                padded_structures.append(tokens)

                # Pad DSSP
                if pad_len > 0:
                    dssp = F.pad(dssp, (0, pad_len), value=-100)
                padded_dssp.append(dssp)

            # Stack into batches
            tokens_batch = torch.cat(padded_structures, dim=0)
            dssp_batch = torch.cat(padded_dssp, dim=0)

            batches.append({"tokens": tokens_batch, "dssp": dssp_batch})

        return batches

    def _train_epoch(self, pl_module: lightning.LightningModule) -> float:
        """Train the probe for one epoch using cached batches.

        Args:
            pl_module: PyTorch Lightning module

        Returns:
            float: Average training accuracy for the epoch
        """
        self.probe.train()
        epoch_acc = []

        for batch in self.train_batches:
            tokens_batch = batch["tokens"]
            dssp_batch = batch["dssp"]

            # Forward pass
            logits = self.probe(tokens_batch)
            loss = F.cross_entropy(logits.view(-1, len(DSSP_STATES)), dssp_batch.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy (ignoring padding)
            preds = torch.argmax(logits, dim=-1)
            mask = dssp_batch != -100
            acc = ((preds == dssp_batch) & mask).float().sum() / mask.sum()
            epoch_acc.append(acc.item())

            # Log metrics
            logger.info(f"Batch loss: {loss.item():.4f}, accuracy: {acc.item():.4f}")
            pl_module.log_dict(
                {"dssp_probe_train_loss": loss.item(), "dssp_probe_train_acc": acc.item()},
                batch_size=tokens_batch.shape[0],
            )

        return np.mean(epoch_acc) if epoch_acc else 0.0

    def _validate(self, pl_module: lightning.LightningModule) -> float:
        """Validate the probe using cached batches.

        Args:
            pl_module: PyTorch Lightning module

        Returns:
            float: Validation accuracy
        """
        self.probe.eval()
        val_acc = []

        with torch.no_grad():
            for batch in self.val_batches:
                tokens_batch = batch["tokens"]
                dssp_batch = batch["dssp"]

                # Forward pass
                logits = self.probe(tokens_batch)
                loss = F.cross_entropy(logits.view(-1, len(DSSP_STATES)), dssp_batch.view(-1))

                # Compute accuracy (ignoring padding)
                preds = torch.argmax(logits, dim=-1)
                mask = dssp_batch != -100
                acc = ((preds == dssp_batch) & mask).float().sum() / mask.sum()
                val_acc.append(acc.item())

                # Log metrics
                logger.info(f"Validation batch loss: {loss.item():.4f}, accuracy: {acc.item():.4f}")
                pl_module.log_dict(
                    {"dssp_probe_val_loss": loss.item(), "dssp_probe_val_acc": acc.item()},
                    batch_size=tokens_batch.shape[0],
                )

        return np.mean(val_acc) if val_acc else 0.0

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Train the linear probe on all training structures.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Model outputs
            batch: Input batch
            batch_idx: Batch index
        """
        # Only run every run_every_n_steps
        if trainer.global_step % self.run_every_n_steps != 0:
            return

        logger.info("Starting DSSP probe training...")

        # Initialize linear probe and move to device
        self.probe = nn.Linear(self.token_dim, len(DSSP_STATES)).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Prepare batches
        self.train_batches = self._prepare_batches(pl_module, self.train_files)
        logger.info(f"Prepared {len(self.train_batches)} training batches")

        self.val_batches = self._prepare_batches(pl_module, self.val_files)
        logger.info(f"Prepared {len(self.val_batches)} validation batches")

        # Early stopping variables
        best_val_acc = 0.0
        patience_counter = 0

        # Train for specified number of epochs
        for epoch in range(self.num_epochs):
            # Train one epoch
            train_acc = self._train_epoch(pl_module)

            # Validate
            val_acc = self._validate(pl_module)

            # Log epoch metrics
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping check
            if val_acc > best_val_acc + self.min_delta:
                best_val_acc = val_acc
                best_train_acc = train_acc
                patience_counter = 0
                # best_model_state = self.probe.state_dict().copy()
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs")

            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        pl_module.log_dict(
            {"dssp_probe_train_acc": best_train_acc, "dssp_probe_val_acc": best_val_acc}, batch_size=self.batch_size
        )

        # Restore best model
        # if best_model_state is not None:
        #    self.probe.load_state_dict(best_model_state)
        #    logger.info(f"Restored best model with validation accuracy: {best_val_acc:.4f}")

        logger.info("Completed DSSP probe training")

    def _save_predictions(self, preds: torch.Tensor, targets: torch.Tensor, batch_idx: int) -> None:
        """Save predictions and targets to file.

        Args:
            preds: Predicted DSSP states
            targets: Target DSSP states
            batch_idx: Batch index
        """
        # Convert predictions and targets to DSSP states
        pred_states = [DSSP_STATES[p] for p in preds.cpu().numpy()]
        target_states = [DSSP_STATES[t] for t in targets.cpu().numpy()]

        # Save to file
        with open(f"dssp_predictions_{batch_idx}.txt", "w") as f:
            f.write("Pred\tTarget\n")
            for p, t in zip(pred_states, target_states):
                f.write(f"{p}\t{t}\n")
