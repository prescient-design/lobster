"""
Reward functions for reinforcement learning training.

This module provides reward functions that can be used with RL training frameworks
like TRL, particularly reward functions based on UME models.
"""

import logging
import re

import torch

from lobster.constants import Modality
from lobster.model import UME

logger = logging.getLogger(__name__)


def detect_modality(text: str) -> Modality:
    """
    Detect the modality of a sequence based on its content.

    Parameters:
    -----------
    text : str
        The text sequence to analyze

    Returns:
    --------
    Modality
        The detected modality (SMILES, AMINO_ACID, or NUCLEOTIDE)
    """
    text = text.strip().upper()

    # Skip empty or very short sequences
    if len(text) < 3:
        return Modality.SMILES

    # DNA patterns: contains only A, T, G, C (check this FIRST since it's more specific)
    dna_pattern = re.compile(r"^[ATGC]+$")
    if dna_pattern.match(text):
        return Modality.NUCLEOTIDE

    # Amino acid patterns: contains standard amino acid codes (but not just DNA bases)
    aa_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    if aa_pattern.match(text):
        return Modality.AMINO_ACID

    # SMILES patterns: contains molecular symbols and structures
    smiles_pattern = re.compile(r"[CNOSPFIHBrCl()\[\]=#@+\-\.\\\/]")
    if smiles_pattern.search(text) and any(c in text for c in "()[]=#@"):
        return Modality.SMILES

    # Default to SMILES if uncertain
    return Modality.SMILES


def compute_pseudo_likelihood(ume_model: UME, sequences: list[str], modality: Modality) -> list[float]:
    """
    Compute pseudo-likelihood for a batch of sequences.

    Parameters:
    -----------
    ume_model : UME
        The UME model to use for likelihood computation
    sequences : List[str]
        List of sequences to evaluate
    modality : Modality
        The modality of the sequences

    Returns:
    --------
    List[float]
        List of pseudo-likelihood scores for each sequence
    """
    with torch.no_grad():
        try:
            # Filter out empty sequences
            valid_sequences = [seq for seq in sequences if seq.strip()]
            if not valid_sequences:
                return [0.0] * len(sequences)

            logger.debug(f"Processing {len(valid_sequences)} sequences for modality {modality.value}")

            # Tokenize the sequences using the appropriate tokenizer
            tokenizer_transform = ume_model.tokenizer_transforms[modality]
            logger.debug(f"Using tokenizer transform: {type(tokenizer_transform)}")

            encoded_batch = tokenizer_transform(valid_sequences)
            logger.debug(f"Encoded batch keys: {list(encoded_batch.keys())}")

            # Get input_ids and attention_mask
            input_ids = encoded_batch["input_ids"]  # Shape: (batch_size, 1, seq_len) or (batch_size, seq_len)
            attention_mask = encoded_batch["attention_mask"]  # Shape: (batch_size, 1, seq_len) or (batch_size, seq_len)

            # Move tensors to the same device as the model
            device = next(ume_model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Debug: print shapes to understand the actual dimensions
            logger.debug(f"input_ids shape: {input_ids.shape}")
            logger.debug(f"attention_mask shape: {attention_mask.shape}")
            logger.debug(f"Device: {device}")

            # Handle different possible shapes
            if input_ids.dim() == 3:
                # Shape is (batch_size, 1, seq_len)
                batch_size, _, seq_len = input_ids.shape
                input_ids_3d = input_ids  # Already in correct format
                attention_mask_3d = attention_mask
            elif input_ids.dim() == 2:
                # Shape is (batch_size, seq_len) - need to add middle dimension
                batch_size, seq_len = input_ids.shape
                input_ids_3d = input_ids.unsqueeze(1)  # Add middle dimension: (batch_size, 1, seq_len)
                attention_mask_3d = attention_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

            logger.debug(f"After shape handling - batch_size: {batch_size}, seq_len: {seq_len}")
            logger.debug(f"input_ids_3d shape: {input_ids_3d.shape}")

            # Prepare inputs for the model (similar to _compute_mlm_loss)
            # _prepare_inputs expects (batch_size, 1, seq_len) format
            input_ids_flat, attention_mask_flat, cu_seqlens = ume_model.model._prepare_inputs(
                input_ids_3d, attention_mask_3d
            )

            logger.debug(f"After _prepare_inputs - input_ids_flat shape: {input_ids_flat.shape}")

            # Get model outputs without masking (we want the full sequence)
            hidden_states = ume_model.model.model(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat,
                cu_seqlens=cu_seqlens,
                max_seqlen=ume_model.max_length,
            )

            logger.debug(f"Hidden states shape: {hidden_states.shape}")

            # Get logits from decoder
            logits = ume_model.model.decoder(hidden_states)
            logits = logits.view(-1, ume_model.model.config.vocab_size)  # (batch_size * seq_len, vocab_size)

            logger.debug(f"Logits shape: {logits.shape}")

            # Reshape input_ids for probability computation
            input_ids_reshaped = input_ids_flat.view(-1)  # (batch_size * seq_len)

            # Convert to log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[torch.arange(len(input_ids_reshaped)), input_ids_reshaped]

            # Reshape back to (batch_size, seq_len)
            token_log_probs = token_log_probs.view(batch_size, seq_len)

            # Average over sequence length to get per-sequence pseudo-likelihood
            # Exclude padding tokens (token_id == ume_model.model.pad_token_id)
            mask = input_ids_3d[:, 0, :] != ume_model.model.pad_token_id
            masked_log_probs = token_log_probs * mask.float()

            # Compute average log probability per sequence
            pseudo_likelihoods = masked_log_probs.sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)

            logger.debug(f"Computed pseudo-likelihoods: {pseudo_likelihoods.shape}")

            # Handle case where some sequences were filtered out
            if len(valid_sequences) < len(sequences):
                result = [0.0] * len(sequences)
                valid_idx = 0
                for i, seq in enumerate(sequences):
                    if seq.strip():
                        result[i] = float(pseudo_likelihoods[valid_idx].cpu().numpy())
                        valid_idx += 1
                return result

            return pseudo_likelihoods.cpu().numpy()

        except Exception as e:
            logger.error(f"Error computing pseudo-likelihood: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return zero rewards for failed computations
            return [0.0] * len(sequences)


class UMERewardFunction:
    """
    Reward function that uses UME to reward completions that are high likelihood.

    This class provides a reward function that can be used with RL training frameworks
    like TRL. It computes rewards based on the pseudo-likelihood of sequences according
    to a UME model.
    """

    def __init__(self, ume_model: UME, temperature: float = 0.1, batch_size: int = 8):
        """
        Initialize the UME reward function.

        Parameters:
        -----------
        ume_model : UME
            The UME model to use for reward computation
        temperature : float, optional
            Temperature scaling for rewards (lower = more extreme rewards), default 0.1
        batch_size : int, optional
            Batch size for processing sequences, default 8
        """
        self.ume_model = ume_model
        self.temperature = temperature
        self.batch_size = batch_size

        # Ensure model is in eval mode and frozen
        self.ume_model.eval()
        self.ume_model.freeze()

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """
        Compute rewards for a list of completions.

        Parameters:
        -----------
        completions : List[str]
            List of completion strings to evaluate
        **kwargs : dict
            Additional keyword arguments (unused)

        Returns:
        --------
        List[float]
            List of reward scores for each completion
        """
        if not completions:
            return []

        logger.info(f"Computing rewards for {len(completions)} completions")

        rewards = []

        # Process completions in batches for efficiency
        for i in range(0, len(completions), self.batch_size):
            batch_completions = completions[i : i + self.batch_size]

            # Detect modality for each completion in the batch
            modalities = [detect_modality(comp) for comp in batch_completions]

            # Group by modality for efficient processing
            modality_groups = {}
            for j, (comp, modality) in enumerate(zip(batch_completions, modalities)):
                if modality not in modality_groups:
                    modality_groups[modality] = []
                modality_groups[modality].append((j, comp))

            # Initialize batch rewards
            batch_rewards = [0.0] * len(batch_completions)

            # Process each modality group
            for modality, items in modality_groups.items():
                indices, sequences = zip(*items)

                logger.debug(f"Processing {len(sequences)} {modality.value} sequences")

                # Compute pseudo-likelihoods for this modality group
                likelihoods = compute_pseudo_likelihood(self.ume_model, sequences, modality)

                # Apply temperature scaling to make rewards more suitable for RL
                scaled_likelihoods = [likelihood / self.temperature for likelihood in likelihoods]

                # Assign rewards back to their positions
                for idx, likelihood in zip(indices, scaled_likelihoods):
                    batch_rewards[idx] = float(likelihood)

            rewards.extend(batch_rewards)

        logger.info(
            f"Computed rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards) / len(rewards):.3f}"
        )
        return rewards


def create_ume_reward_wrapper(ume_model: UME, temperature: float = 0.1, batch_size: int = 8):
    """
    Create a reward function wrapper that captures the ume_model.

    This function creates a closure that captures the UME model and returns
    a function with the correct signature for TRL frameworks.

    Parameters:
    -----------
    ume_model : UME
        The UME model to use for reward computation
    temperature : float, optional
        Temperature scaling for rewards, default 0.1
    batch_size : int, optional
        Batch size for processing sequences, default 8

    Returns:
    --------
    callable
        A reward function with signature (completions, **kwargs) -> List[float]
    """
    reward_function = UMERewardFunction(ume_model, temperature, batch_size)

    def reward_wrapper(completions, **kwargs):
        return reward_function(completions, **kwargs)

    return reward_wrapper
