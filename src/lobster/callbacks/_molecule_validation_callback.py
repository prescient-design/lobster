import random
import time
from collections.abc import Callable

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.nn import CrossEntropyLoss
import torch
from rdkit import RDLogger
# disable logs
RDLogger.DisableLog('rdApp.*')

# Import RDKit and atomic_datasets for validity checking

from rdkit import Chem
from atomic_datasets.utils.rdkit import is_molecule_sane
RDKIT_AVAILABLE = True

# Import distance functions from qm9_pair_gen
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import qm9_pair_gen
from qm9_pair_gen.utils_mol import get_tanimoto_distance, get_shape_tanimoto_distance

def shorten_str(possible_smiles: str, n: int = 300) -> str:
    """
    Truncate a string to a specified length for better printing.
    
    This function takes a (e.g. SMILES) string and returns a truncated version that is more 
    suitable for printing. If the string is longer than n characters, it truncates 
    to the first n characters and adds an ellipsis. It also removes spaces from the 
    input string before processing.
    
    Parameters
    ----------
    possible_smiles : str
        The string to be truncated. Can be any string, but typically a 
        molecular SMILES representation.
    n : int, default=300
        The maximum number of characters to include in the output string.
        
    Returns
    -------
    str
        The truncated SMILES string. If the input is longer than n characters,
        returns the first n characters followed by '...'. If the input is not
        a string, returns the input unchanged.
        
    Examples
    --------
    >>> shorten_str("CCO", 5)
    'CCO'
    >>> shorten_str("CC(C)(C)C1=CC=C(C=C1)C(C)C(=O)O", 10)
    'CC(C)(C)C...'
    """
    if type(possible_smiles) != str:
        return possible_smiles
    return ''.join(possible_smiles.split(' '))[:n] + '...' if len(possible_smiles) > n else possible_smiles

class MoleculeValidationCallback(Callback):
    """
    PyTorch Lightning callback for molecule validation during training.
    
    This callback performs molecule generation and validity checking during validation
    epochs. It supports both unconditional generation (starting from CLS token) and
    conditional generation (using input sequences as conditioning). The callback
    generates molecules, validates them using RDKit, and computes various molecular
    similarity metrics.
    
    The callback operates in two modes:
    1. **Conditional generation**: Extracts conditioning sequences from input batches
       and generates new molecules based on those sequences. --> called during validation batch (since needs access to data)
        Currently, operates on whole batches at once, but should yield approximately num_validation_generations molecules
    2. **Unconditional generation**: Generates molecules from scratch starting with
       a CLS token. --> called at end of validation epoch
        Currently, generates num_validation_generations molecules precisely
    
    Both modes include comprehensive timing measurements and validity checking.

    This class also contains many helper functions as static class functions.
    
    Attributes
    ----------
    num_validation_generations : int
        The number of unconditional molecules to generate during validation.
        Default is 100.
    """
    
    def __init__(self, num_validation_generations: int = 100):
        """
        Initialize the MoleculeValidationCallback.
        
        Parameters
        ----------
        num_validation_generations : int, default=100
            The number of unconditional molecules to generate during each validation epoch.
            This controls how many molecules are generated from scratch starting with
            the CLS token.
        """
        super().__init__()
        self.num_validation_generations = num_validation_generations
    
    def _should_run_validation_batch(self, trainer: pl.Trainer, batch_idx: int, batch_size: int = None) -> bool:
        """
        Determine if a validation batch should be processed, based on desired number of validation samples. It calculates how many batches are needed
        to achieve the desired number of validation samples and selects batches with
        appropriate stride to distribute them evenly across the validation set.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance containing validation information.
        batch_idx : int
            The index of the current validation batch being considered.
        batch_size : int, optional
            The size of the current batch. If provided, used to calculate exact
            number of batches needed. If None, uses a default estimate of 16.
            
        Returns
        -------
        bool
            True if the batch should be processed, False otherwise.
            
        Notes
        -----
        The method ensures at least one batch is always processed and distributes
        the selected batches evenly across the validation set using stride calculation.
        """
        num_val_batches = trainer.num_val_batches[0]
        
        # If we have the batch size, use it to compute exact number of batches needed
        if batch_size is not None:
            batches_needed = max(1, (self.num_validation_generations + batch_size - 1) // batch_size)  # Ceiling division
            batches_to_use = min(batches_needed, num_val_batches)
            batch_stride = max(1, num_val_batches // batches_to_use)
            print(f"Using {batches_to_use} batches of size {batch_size} for {self.num_validation_generations} samples, stride is {batch_stride}, num_val_batches is {num_val_batches}")
        else:
            # Fallback: use a reasonable default batch size estimate
            # This will be refined when we actually process the batch
            estimated_batch_size = 16  # Common default batch size
            batches_needed = max(1, (self.num_validation_generations + estimated_batch_size - 1) // estimated_batch_size)
            batches_to_use = min(batches_needed, num_val_batches)
            batch_stride = max(1, num_val_batches // batches_to_use)
        
        return batch_idx % batch_stride == 0 and batch_idx < batches_to_use * batch_stride
    
    @staticmethod
    def _extract_conditioning_from_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> tuple:
        """
        Extract conditioning sequence from input batch for conditional generation.
        
        This method extracts the conditioning sequence from input tokens by finding
        the sequence up to the separator token. It uses the attention mask to
        identify valid tokens and decodes them to a string representation.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs, shape [seq_len] or [batch_size, seq_len].
            Contains the tokenized input sequence.
        attention_mask : torch.Tensor
            Attention mask indicating valid tokens, shape [seq_len] or [batch_size, seq_len].
            Values are 1 for valid tokens, 0 for padding tokens.
        tokenizer
            The tokenizer object with methods:
            - sep_token_id: int, separator token ID
            - eos_token_id: int, end-of-sequence token ID
            - decode(): method to convert token IDs to string
            
        Returns
        -------
        tuple
            A tuple containing:
            - unmasked_ids: torch.Tensor, shape [cond_seq_len]
                The conditioning sequence token IDs (valid tokens only)
            - cond_str: str
                The decoded conditioning sequence string
            If no valid conditioning sequence is found, returns (None, None).
        """
        cond_ids, cond_mask = MoleculeValidationCallback._extract_conditioning_sequence(
            input_ids, attention_mask, tokenizer.sep_token_id, tokenizer.eos_token_id
        )
        if cond_ids is None:
            return None, None
        
        unmasked_ids = cond_ids[cond_mask.bool()]
        cond_str = tokenizer.decode(unmasked_ids.tolist())
        return unmasked_ids, cond_str
    
    @staticmethod
    def _generate_molecule_conditionally(pl_module, unmasked_ids: torch.Tensor) -> tuple:
        """
        Generate a molecule using conditional generation from a conditioning sequence.
        
        This method performs conditional molecule generation by using the provided
        conditioning sequence as input to the model's generate method. It measures
        generation time and returns both the raw model output and processed results.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - model: huggingface-compatible model with .generate() method
            - tokenizer: tokenizer with pad_token_id, eos_token_id, and decode() method
            - _max_length: int, maximum number of new tokens to generate
        unmasked_ids : torch.Tensor, shape [seq_len]
            The conditioning sequence token IDs to use as input for generation.
            
        Returns
        -------
        tuple
            A tuple containing:
            - generated: torch.LongTensor, shape [1, full_seq_len]
                Raw model output tensor
            - gen_ids: torch.LongTensor, shape [full_seq_len]
                Flattened token IDs of the complete generation (conditioning + new)
            - gen_full_str: str
                Decoded string representation of the complete generation
            - gen_new_str: str
                Decoded string representation of only the newly generated tokens
                
        Notes
        -----
        The method includes timing measurement and prints generation time with
        the [ValidationTime] prefix.
        """
        start_time = time.time()
        generated = pl_module.model.generate(
            input_ids=unmasked_ids.unsqueeze(0),
            max_new_tokens=pl_module._max_length, 
            pad_token_id=pl_module.tokenizer.pad_token_id,
            eos_token_id=pl_module.tokenizer.eos_token_id
        )
        generation_time = time.time() - start_time
        
        gen_ids = generated[0].detach().cpu()
        gen_full_str = pl_module.tokenizer.decode(gen_ids.tolist()) # space separated string
        gen_new_ids = gen_ids[len(unmasked_ids):]
        gen_new_str = pl_module.tokenizer.decode(gen_new_ids.tolist()) if gen_new_ids.numel() > 0 else ""
        
        print(f"[ValidationTime] Conditional generation took {generation_time:.3f}s")
        
        return generated, gen_ids, gen_full_str, gen_new_str
    
    @staticmethod
    def _compute_input_distance_metrics(pl_module, input_ids: torch.Tensor, full_input_smiles1: str, full_input_smiles2: str) -> None:
        """
        Compute distance metrics between original input SMILES strings.
        
        This method calculates molecular similarity metrics (Tanimoto and Shape Tanimoto)
        between two input SMILES strings. It only computes metrics if both molecules
        are valid according to RDKit validation. The results are logged to the
        PyTorch Lightning module for tracking during training.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module with logging capabilities.
        input_ids : torch.Tensor
            Input token IDs (unused in this method, kept for interface consistency).
        full_input_smiles1 : str
            First SMILES string to compare.
        full_input_smiles2 : str
            Second SMILES string to compare.
            
        Returns
        -------
        None
            Results are logged to the pl_module instead of being returned.
            
        Notes
        -----
        The method logs two metrics:
        - val_tanimoto_distance_input: Tanimoto similarity between the molecules
        - val_shape_tanimoto_distance_input: Shape-based Tanimoto similarity
        """
        if (full_input_smiles1 is not None and full_input_smiles2 is not None and 
            MoleculeValidationCallback._is_molecule_valid(full_input_smiles1) and 
            MoleculeValidationCallback._is_molecule_valid(full_input_smiles2)):
            
            input_tanimoto_dist = get_tanimoto_distance(full_input_smiles1, full_input_smiles2)
            pl_module.log("val_tanimoto_distance_input", input_tanimoto_dist, sync_dist=True, on_step=False, on_epoch=True)
            
            input_shape_tanimoto_dist = get_shape_tanimoto_distance(full_input_smiles1, full_input_smiles2, verbose=False)
            pl_module.log("val_shape_tanimoto_distance_input", input_shape_tanimoto_dist, sync_dist=True, on_step=False, on_epoch=True)
            
            print(f"Input data distance metrics - Tanimoto: {input_tanimoto_dist:.3f}, Shape Tanimoto: {input_shape_tanimoto_dist:.3f}")
    
    @staticmethod
    def _compute_generated_distance_metrics(pl_module, conditioned_smiles: str, new_smiles: str) -> None:
        """
        Compute distance metrics between conditioning and generated molecules.
        
        This method calculates molecular similarity metrics (Tanimoto and Shape Tanimoto)
        between a conditioning molecule and a newly generated molecule. It only computes
        metrics if both molecules are valid according to RDKit validation. The results
        are logged to the PyTorch Lightning module for tracking during training.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module with logging capabilities.
        conditioned_smiles : str
            SMILES string of the conditioning molecule (input molecule).
        new_smiles : str
            SMILES string of the newly generated molecule.
            
        Returns
        -------
        None
            Results are logged to the pl_module instead of being returned.
            
        Notes
        -----
        The method logs two metrics:
        - val_tanimoto_distance_conditional: Tanimoto similarity between conditioning and generated
        - val_shape_tanimoto_distance_conditional: Shape-based Tanimoto similarity
        """
        if (MoleculeValidationCallback._is_molecule_valid(conditioned_smiles) and 
            MoleculeValidationCallback._is_molecule_valid(new_smiles)):
            
            tanimoto_dist = get_tanimoto_distance(conditioned_smiles, new_smiles)
            pl_module.log("val_tanimoto_distance_conditional", tanimoto_dist, sync_dist=True, on_step=False, on_epoch=True)
            
            shape_tanimoto_dist = get_shape_tanimoto_distance(conditioned_smiles, new_smiles, verbose=False)
            pl_module.log("val_shape_tanimoto_distance_conditional", shape_tanimoto_dist, sync_dist=True, on_step=False, on_epoch=True)
            
            print(f"Generated distance metrics - Tanimoto: {tanimoto_dist:.3f}, Shape Tanimoto: {shape_tanimoto_dist:.3f}")
    
    @staticmethod
    def _process_conditional_generation_sample_post_generation(pl_module, input_ids: torch.Tensor, attention_mask: torch.Tensor, generated_sequence: torch.Tensor) -> bool:
        """
        Process a single generated molecule for conditional generation validation.
        
        This method processes a generated molecule sequence for conditional generation
        validation. It extracts the conditioning sequence, validates the generated
        molecule, computes distance metrics, and determines if the generation was
        successful. This method is called after batched generation to process
        individual samples.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer with decode() method and special token attributes
            - model: model with generation capabilities
        input_ids : torch.Tensor, shape [seq_len]
            Original input token IDs for the sample.
        attention_mask : torch.Tensor, shape [seq_len]
            Attention mask indicating valid tokens in the input.
        generated_sequence : torch.Tensor, shape [generated_seq_len]
            The generated token sequence from the model.
            
        Returns
        -------
        bool
            True if the generated molecule is valid, False otherwise.
            
        Notes
        -----
        This method performs several operations:
        1. Extracts conditioning sequence from input
        2. Converts generated tokens to SMILES
        3. Validates the generated molecule using RDKit
        4. Computes distance metrics between conditioning and generated molecules
        5. Logs results and prints generation details
        """
        if generated_sequence.numel() == 0:
            return False
        
        # Extract conditioning sequence
        unmasked_ids, cond_str = MoleculeValidationCallback._extract_conditioning_from_batch(input_ids, attention_mask, pl_module.tokenizer)
        if unmasked_ids is None:
            return False
        
        # Convert generated sequence to the expected format
        gen_ids = generated_sequence.detach().cpu()
        gen_full_str = pl_module.tokenizer.decode(gen_ids.tolist())
        gen_new_ids = gen_ids[len(unmasked_ids):]
        gen_new_str = pl_module.tokenizer.decode(gen_new_ids.tolist()) if gen_new_ids.numel() > 0 else ""
        
        # Print generation details
        print(f"\n[MoleculeValidationCallback] Conditional generate | cond='{shorten_str(cond_str)}'\n\n generated_full='{shorten_str(gen_full_str)}'\n\n generated_new='{shorten_str(gen_new_str)}'\n")
        
        # Convert to SMILES
        generated_as_list = [int(g) for g in generated_sequence.reshape(-1)]
        conditioned_smiles, new_smiles = MoleculeValidationCallback._tokens_to_smiles(pl_module.tokenizer, generated_as_list)
        print(f"conditioned_smiles: {shorten_str(conditioned_smiles)}\n\n new_smiles: {shorten_str(new_smiles)}")
        
        # Validate conditioning SMILES matches input
        MoleculeValidationCallback._validate_conditioning_smiles(pl_module.tokenizer, unmasked_ids, conditioned_smiles)
        
        # Compute input distance metrics
        full_input_smiles1, full_input_smiles2 = MoleculeValidationCallback._tokens_to_smiles(pl_module.tokenizer, input_ids.reshape(-1).tolist())
        MoleculeValidationCallback._compute_input_distance_metrics(pl_module, input_ids, full_input_smiles1, full_input_smiles2)
        
        # Check validity and compute generated distance metrics
        is_valid = False
        if new_smiles is not None:
            if MoleculeValidationCallback._is_molecule_valid(new_smiles):
                is_valid = True
            MoleculeValidationCallback._compute_generated_distance_metrics(pl_module, conditioned_smiles, new_smiles)
        
        return is_valid

    @staticmethod
    def _validate_conditioning_smiles(tokenizer, unmasked_ids: torch.Tensor, conditioned_smiles: str) -> None:
        """
        Validate that conditioning SMILES matches input SMILES.
        
        This method performs a validation check to ensure that the extracted conditioning
        SMILES string matches the original input SMILES. It decodes the unmasked token
        IDs, extracts the SMILES portion between CLS and SEP tokens, and compares it
        with the provided conditioned SMILES. If they don't match, it raises an assertion
        error.
        
        Parameters
        ----------
        tokenizer
            The tokenizer object with methods:
            - decode(): method to convert token IDs to string
            - cls_token: str, CLS token string
            - sep_token: str, SEP token string
        unmasked_ids : torch.Tensor, shape [cond_seq_len]
            The conditioning sequence token IDs (valid tokens only).
        conditioned_smiles : str
            The SMILES string extracted from the conditioning sequence.
            
        Returns
        -------
        None
        
        Raises
        ------
        AssertionError
            If the conditioning SMILES does not match the input SMILES.
            
        Notes
        -----
        This validation ensures that the tokenization and SMILES extraction process
        is working correctly by comparing the decoded tokens with the expected SMILES.
        """
        decoded_unmasked = tokenizer.decode(unmasked_ids.tolist())
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        if cls_token in decoded_unmasked and sep_token in decoded_unmasked:
            between_tokens = decoded_unmasked.split(cls_token)[1].split(sep_token)[0].strip().split(' ')
            between_tokens = ''.join(between_tokens)
            assert between_tokens == conditioned_smiles, f"Conditioned SMILES does not match input SMILES: between_tokens='{shorten_str(between_tokens)}' != conditioned_smiles='{shorten_str(conditioned_smiles)}'   "
    
    @staticmethod
    def _get_cls_token_id(tokenizer) -> int:
        """
        Get the CLS token ID from the tokenizer.
        
        This method retrieves the CLS token ID from the tokenizer. It first checks
        if the tokenizer has a cls_token_id attribute, and if not, it converts
        the '<cls>' token string to its corresponding ID.
        
        Parameters
        ----------
        tokenizer
            The tokenizer object with methods:
            - cls_token_id: int, CLS token ID (if available)
            - convert_tokens_to_ids(): method to convert token strings to IDs
            
        Returns
        -------
        int
            The CLS token ID.
            
        Notes
        -----
        The method handles different tokenizer implementations that may or may not
        have a direct cls_token_id attribute.
        """
        cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None
        if cls_token_id is None:
            cls_token_id = tokenizer.convert_tokens_to_ids('<cls>')
        return cls_token_id
    
    @staticmethod
    def _log_unconditional_validation_results(pl_module, first_valid_molecules: int, second_valid_molecules: int, 
                                            total_first_checked: int, total_second_checked: int, trainer: pl.Trainer) -> None:
        """
        Log the final results of unconditional validation.
        
        This method calculates and logs the validity accuracies for unconditional
        molecule generation. It computes overall validity accuracy as well as
        separate accuracies for first and second molecules in each generation.
        Results are logged to the PyTorch Lightning module and printed to console.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module with logging capabilities.
        first_valid_molecules : int
            Number of valid first molecules generated.
        second_valid_molecules : int
            Number of valid second molecules generated.
        total_first_checked : int
            Total number of first molecules checked for validity.
        total_second_checked : int
            Total number of second molecules checked for validity.
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
            
        Returns
        -------
        None
            Results are logged to the pl_module and printed to console.
            
        Notes
        -----
        The method logs three metrics:
        - val_molecule_validity: Overall validity accuracy (all molecules)
        - val_molecule_validity_first: Validity accuracy for first molecules only
        - val_molecule_validity_second: Validity accuracy for second molecules only
        """
        validity_accuracy = float(first_valid_molecules + second_valid_molecules) / (total_first_checked + total_second_checked) if (total_first_checked + total_second_checked) > 0 else 0.0
        first_validity_accuracy = float(first_valid_molecules) / total_first_checked if total_first_checked > 0 else 0.0
        second_validity_accuracy = float(second_valid_molecules) / total_second_checked if total_second_checked > 0 else 0.0

        pl_module.log("val_molecule_validity", validity_accuracy, sync_dist=True)
        pl_module.log("val_molecule_validity_first", first_validity_accuracy, sync_dist=True)
        pl_module.log("val_molecule_validity_second", second_validity_accuracy, sync_dist=True)
        
        print(f"Validation epoch {trainer.current_epoch}: {first_valid_molecules + second_valid_molecules}/{total_first_checked + total_second_checked} molecules valid ({validity_accuracy:.3f}), {first_valid_molecules}/{total_first_checked} first molecules valid ({first_validity_accuracy:.3f}), {second_valid_molecules}/{total_second_checked} second molecules valid ({second_validity_accuracy:.3f})")

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch: dict, batch_idx: int) -> None:
        """
        Called after each validation batch to perform conditional molecule generation validation.
        
        This method is triggered by PyTorch Lightning after each validation batch.
        It performs batched conditional molecule generation, processes the results,
        and logs validity metrics. The method includes comprehensive timing
        measurements for performance analysis.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer object with sep_token_id, eos_token_id, pad_token_id, cls_token and decode()
            - model: huggingface-compatible model exposing .generate()
            - _max_length: int, maximum tokens to generate
        outputs
            Outputs from the validation step (unused in this callback).
        batch : dict
            Dictionary containing the batch data with keys:
            - "input_ids": torch.LongTensor, shape [batch_size, seq_len]
            - "attention_mask": torch.LongTensor or BoolTensor, shape [batch_size, seq_len]
        batch_idx : int
            Index of the current validation batch.
            
        Returns
        -------
        None
            Results are logged to the pl_module.
            
        Notes
        -----
        This method performs conditional generation by:
        1. Extracting conditioning sequences from input batches
        2. Generating new molecules using batched model forward pass
        3. Processing each generated molecule individually for validity checking
        4. Computing distance metrics between conditioning and generated molecules
        5. Logging results with timing information
        """
        print(f"[MoleculeValidationCallback] on_validation_batch_end called: batch_idx={batch_idx}")
            
        # Only run this occasionally to avoid slowing down training too much
        # Only run on a subset of batches to get desired number of total generations
        if not self._should_run_validation_batch(trainer, batch_idx, batch["input_ids"].shape[0]):
            return
            
        batch_start_time = time.time()
        
        # Use the whole batch for conditional generation
        batch_size = batch["input_ids"].shape[0]
        print(f"[ValidationTime] Processing conditional validation batch {batch_idx} with {batch_size} samples")
        
        # Batch generate all samples in this batch
        generation_start_time = time.time()
        all_generated_sequences = MoleculeValidationCallback._batch_generate_molecules_conditionally(
            pl_module, batch["input_ids"], batch["attention_mask"]
        )
        generation_time = time.time() - generation_start_time
        print(f"[ValidationTime] Batched conditional generation took {generation_time:.3f}s for {batch_size} samples")
        
        # Process each generated sequence individually for validity checks
        processing_start_time = time.time()
        valid_generations = 0
        for i, (input_ids, attention_mask, generated_sequence) in enumerate(zip(
            batch["input_ids"], batch["attention_mask"], all_generated_sequences
        )):
            is_valid = MoleculeValidationCallback._process_conditional_generation_sample_post_generation(
                pl_module, input_ids, attention_mask, generated_sequence
            )
            if is_valid:
                valid_generations += 1
        
        processing_time = time.time() - processing_start_time
        print(f"[ValidationTime] Conditional post-generation processing took {processing_time:.3f}s for {batch_size} samples")

        batch_total_time = time.time() - batch_start_time
        print(f"[ValidationTime] Conditional validation batch {batch_idx} took {batch_total_time:.3f}s for {batch_size} samples")

        frac_valid = float(valid_generations) / batch_size
        pl_module.log("val_molecule_validity_conditional", frac_valid, sync_dist=True, on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Compute validity accuracy over num_validation_generations unconditional samples.
        
        This method is triggered by PyTorch Lightning at the end of each validation epoch.
        It performs batched unconditional molecule generation starting from CLS tokens,
        processes the results, and logs comprehensive validity metrics. The method
        includes detailed timing measurements for performance analysis.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer with cls_token_id, pad/eos tokens and decode()
            - model: huggingface-compatible model exposing .generate()
            - _max_length: int, maximum tokens to generate
            
        Returns
        -------
        None
            Results are logged to the pl_module and printed to console.
            
        Notes
        -----
        This method performs unconditional generation by:
        1. Generating molecules from scratch starting with CLS tokens
        2. Converting generated tokens to SMILES strings
        3. Validating molecules using RDKit
        4. Computing distance metrics between generated molecules
        5. Logging comprehensive validity statistics with timing information
        
        The method logs three main metrics:
        - val_molecule_validity: Overall validity accuracy
        - val_molecule_validity_first: Validity accuracy for first molecules
        - val_molecule_validity_second: Validity accuracy for second molecules
        """
        print(f"[MoleculeValidationCallback] on_validation_epoch_end called: epoch={trainer.current_epoch}")
        
        if not hasattr(self, 'num_validation_generations') or self.num_validation_generations <= 0:
            print("[MoleculeValidationCallback] Skipping: num_validation_generations not set or <= 0")
            return
        
        epoch_start_time = time.time()
        
        # Set to evaluation mode
        pl_module.eval()
        
        first_valid_molecules = 0
        second_valid_molecules = 0
        total_checked = 0
        
        # Get CLS token ID
        cls_token_id = MoleculeValidationCallback._get_cls_token_id(pl_module.tokenizer)
        if cls_token_id is None:
            print("Warning: Could not find CLS token, skipping validity check")
            return
        
        print(f"Starting validity check with {self.num_validation_generations} unconditional generations...")
        
        # Batch generate all molecules at once
        generation_start_time = time.time()
        all_sampled_tokens = MoleculeValidationCallback._batch_sample_molecules_from_cls(pl_module, cls_token_id, self.num_validation_generations)
        generation_time = time.time() - generation_start_time
        print(f"[ValidationTime] Batched unconditional generation took {generation_time:.3f}s for {self.num_validation_generations} samples")
        
        # Process each generated sequence individually for validity checks
        processing_start_time = time.time()
        for i, sampled_tokens in enumerate(all_sampled_tokens):
            first_valid, second_valid = MoleculeValidationCallback._process_unconditional_generation_sample_post_generation(
                pl_module, sampled_tokens, cls_token_id
            )
            total_checked += 1

            # Update counters
            if first_valid:
                first_valid_molecules += 1
            
            if second_valid:
                second_valid_molecules += 1

            # Progress update every 10 generations
            if (i + 1) % 10 == 0:
                current_first_accuracy = first_valid_molecules / total_checked if total_checked > 0 else 0.0
                current_second_accuracy = second_valid_molecules / total_checked if total_checked > 0 else 0.0
                print(f"Progress on unconditional generations: {i+1}/{self.num_validation_generations}, Current validity [sanity check] (1): {current_first_accuracy:.3f} (2) {current_second_accuracy:.3f}, as fractions (1): {first_valid_molecules}/{total_checked} (2) {second_valid_molecules}/{total_checked}")

        processing_time = time.time() - processing_start_time
        print(f"[ValidationTime] Unconditional post-generation processing took {processing_time:.3f}s for {self.num_validation_generations} samples")

        epoch_total_time = time.time() - epoch_start_time
        print(f"[ValidationTime] Unconditional validation epoch took {epoch_total_time:.3f}s for {self.num_validation_generations} samples")
        
        # Calculate final validity accuracies
        MoleculeValidationCallback._log_unconditional_validation_results(pl_module, first_valid_molecules, second_valid_molecules, 
                                            total_checked, total_checked, trainer)
        
        # Set back to training mode
        pl_module.train()

    @staticmethod
    def _batch_sample_molecules_from_cls(pl_module, cls_token_id: int, num_samples: int) -> list:
        """
        Batch generate a list of molecules starting from CLS token.
        
        This method performs batched unconditional molecule generation by creating
        a tensor of CLS tokens and generating all molecules in a single model
        forward pass. This is much more efficient than generating molecules
        one at a time.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - model: huggingface-compatible model with .generate() method
            - tokenizer: tokenizer with pad_token_id, eos_token_id
            - _max_length: int, maximum number of new tokens to generate
        cls_token_id : int
            Token ID for the CLS token used as the starting point for generation.
        num_samples : int
            Number of molecules to generate in the batch.
            
        Returns
        -------
        list
            A list of generated token sequences, where each element is a list of integers
            representing the token IDs for one generated molecule.
            Shape: List[List[int]], where each inner list has length [generated_seq_len]
            
        Notes
        -----
        The method creates a tensor of shape [num_samples, 1] filled with cls_token_id
        and passes it to the model's generate method for efficient batched generation.
        """
        # Create a tensor of cls_token_id repeated num_samples times
        # Shape: [num_samples, 1] - each row is a single cls token
        cls_token_tensor = torch.full((num_samples, 1), cls_token_id, dtype=torch.long, device=pl_module.model.device)

        # Generate all molecules in a single batch
        generated_sequences = pl_module.model.generate(
            input_ids=cls_token_tensor,
            max_new_tokens=pl_module._max_length, 
            pad_token_id=pl_module.tokenizer.pad_token_id,
            eos_token_id=pl_module.tokenizer.eos_token_id
        )

        # Convert to list of lists - each inner list is one generated sequence
        if generated_sequences is None:
            return [[] for _ in range(num_samples)]
        
        # Shape: [num_samples, seq_len] -> List[List[int]]
        return [seq.tolist() for seq in generated_sequences]

    @staticmethod
    def _process_unconditional_generation_sample_post_generation(pl_module, sampled_tokens: list, cls_token_id: int) -> tuple:
        """
        Process a single generated molecule for unconditional generation validation.
        
        This method processes a generated molecule sequence for unconditional generation
        validation. It converts the token sequence to SMILES, validates the molecules,
        computes distance metrics, and determines if the generation was successful.
        This method is called after batched generation to process individual samples.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer with cls_token_id, pad/eos tokens and decode()
            - model: model with generation capabilities
        sampled_tokens : list
            The generated token IDs as a list of integers, shape [generated_seq_len].
        cls_token_id : int
            Token ID for the CLS token used as the starting point.
            
        Returns
        -------
        tuple
            A tuple containing:
            - first_valid: bool, True if the first molecule is valid
            - second_valid: bool, True if the second molecule is valid (if present)
            
        Notes
        -----
        This method performs several operations:
        1. Converts generated tokens to SMILES strings
        2. Validates molecules using RDKit
        3. Computes distance metrics between generated molecules
        4. Logs results and prints generation details
        """
        # Get conditioning string
        cond_str = getattr(pl_module.tokenizer, "cls_token", None)
        if cond_str is None:
            cond_str = pl_module.tokenizer.decode([cls_token_id])
        
        # Convert tokens to SMILES
        smiles_string, second_smiles_string = MoleculeValidationCallback._tokens_to_smiles(pl_module.tokenizer, sampled_tokens)
        
        # Print generation details
        if sampled_tokens is not None and len(sampled_tokens) > 0:
            gen_ids = sampled_tokens
            gen_full_str = pl_module.tokenizer.decode(gen_ids)
            gen_new_ids = gen_ids[1:]
            gen_new_str = pl_module.tokenizer.decode(gen_new_ids)
            print(f"\n[MoleculeValidationCallback] Unconditional generate: cond='{shorten_str(cond_str)}'\n\ngenerated_full='{shorten_str(gen_full_str)}'\n\ngenerated_new='{shorten_str(gen_new_str)}'\n")
            print(f"first smiles: {shorten_str(smiles_string)}\n\nsecond smiles: {shorten_str(second_smiles_string)}")
        
        # Check validity
        first_valid = MoleculeValidationCallback._is_molecule_valid(smiles_string)
        second_valid = False
        if second_smiles_string is not None:
            second_valid = MoleculeValidationCallback._is_molecule_valid(second_smiles_string)
            
            # Compute distance metrics
            if (first_valid and second_valid):
                tanimoto_dist = get_tanimoto_distance(smiles_string, second_smiles_string)
                pl_module.log("val_tanimoto_distance_unconditional", tanimoto_dist, sync_dist=True, on_step=False, on_epoch=True)
                
                shape_tanimoto_dist = get_shape_tanimoto_distance(smiles_string, second_smiles_string, verbose=False)
                pl_module.log("val_shape_tanimoto_distance_unconditional", shape_tanimoto_dist, sync_dist=True, on_step=False, on_epoch=True)
                
                print(f"Unconditional distance metrics - Tanimoto: {tanimoto_dist:.3f}, Shape Tanimoto: {shape_tanimoto_dist:.3f}")
        return first_valid, second_valid

    @staticmethod
    def _tokens_to_smiles(tokenizer, tokens) -> tuple:
        """
        Convert token IDs to SMILES string(s).
        
        This method converts a sequence of token IDs to one or two SMILES strings.
        It handles special tokens (CLS, EOS, PAD, SEP) by removing them and
        splitting on the SEP token to extract multiple SMILES if present.
        
        Parameters
        ----------
        tokenizer
            The tokenizer object with methods:
            - decode(): method to convert token IDs to string
            - cls_token: str, CLS token string
            - eos_token: str, EOS token string
            - pad_token: str, PAD token string
            - sep_token: str, SEP token string
        tokens : Union[list, torch.Tensor]
            Token IDs to convert. Can be:
            - List[int]: list of token IDs
            - torch.Tensor: tensor of token IDs, shape [seq_len] or [batch_size, seq_len]
            
        Returns
        -------
        tuple
            A tuple containing:
            - smiles1: str, first SMILES string (cleaned, spaces removed)
            - smiles2: str or None, second SMILES string if SEP token was present, None otherwise
            
        Notes
        -----
        The method performs the following operations:
        1. Decodes token IDs to string
        2. Removes special tokens (CLS, EOS, PAD)
        3. Splits on SEP token if present to extract multiple SMILES
        4. Removes spaces from the resulting strings
        5. Returns cleaned SMILES strings
        """
        # Decode tokens to string
  
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.reshape(-1).tolist()

        decoded = tokenizer.decode(tokens)
        
        # Remove special tokens and clean up
        # Remove CLS token (usually at the beginning)
        if tokenizer.cls_token in decoded:
            decoded = decoded.replace(tokenizer.cls_token, "")
        
        # Remove EOS token (usually at the end)
        if tokenizer.eos_token in decoded:
            decoded = decoded.replace(tokenizer.eos_token, "")
        
        # Remove padding tokens
        if tokenizer.pad_token in decoded:
            decoded = decoded.replace(tokenizer.pad_token, "")
            
        # Split on SEP token if present
        if tokenizer.sep_token in decoded:
            various_possible_smiles = decoded.split(tokenizer.sep_token)
            smiles1 = ''.join(various_possible_smiles[0].split(' '))
            smiles2 = ''.join(various_possible_smiles[1].split(' '))
            return (smiles1.strip(), smiles2.strip())
        
        # If no SEP token, just return the single cleaned string
        return (''.join(decoded.split(' ')).strip(), None)
    
    @staticmethod
    def _is_molecule_valid(smiles_string: str, verbose: bool = False) -> bool:
        """
        Check if a SMILES string represents a valid molecule.
        
        This method validates a SMILES string using RDKit and atomic_datasets.
        It parses the SMILES string into an RDKit molecule and then performs
        a sanity check to ensure the molecule is chemically reasonable.
        
        Parameters
        ----------
        smiles_string : str
            The SMILES string to validate.
        verbose : bool, default=False
            Whether to print detailed error messages if validation fails.
            
        Returns
        -------
        bool
            True if the SMILES parses to a valid RDKit molecule and passes sanity check,
            False otherwise.
            
        Notes
        -----
        The method performs two validation steps:
        1. RDKit parsing: Attempts to parse the SMILES string into an RDKit molecule
        2. Sanity check: Uses atomic_datasets.utils.rdkit.is_molecule_sane() to verify
           the molecule is chemically reasonable
        
        If RDKit is not available, the method returns False.
        The method suppresses RDKit output during validation to avoid cluttering logs.
        """
        if not RDKIT_AVAILABLE:
            return False
            
        # Suppress RDKit output by temporarily redirecting stderr
        import os, sys
        stderr = sys.stderr 
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull

        try:
            # Parse SMILES with RDKit
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                if verbose:
                    print(f"Failed to parse SMILES: {shorten_str(smiles_string)}")
                return False
            
            # Check if molecule is sane using atomic_datasets
            is_sane = is_molecule_sane(mol)
            if not is_sane and verbose:
                print(f"Molecule failed sanity check: {shorten_str(smiles_string)}")
            return is_sane
            
        except Exception as e:
            if verbose:
                print(f"Exception processing SMILES: {shorten_str(smiles_string, n=500)}")
                print(f"Error: {str(e)}")
            return False
        
        finally:
            # Restore stderr
            sys.stderr = stderr
            devnull.close()

    @staticmethod
    def _extract_conditioning_sequence(input_ids: torch.Tensor, attention_mask: torch.Tensor, sep_token_id: int, eos_token_id: int) -> tuple:
        """
        Extract conditioning sequence up to separator token or end token.
        
        This method extracts a conditioning sequence from input tokens by finding
        the sequence up to the first separator token or end token. It handles
        various edge cases and ensures the returned tensors are properly shaped.
        
        Parameters
        ----------
        input_ids : torch.Tensor, shape [seq_len] or [batch_size, seq_len]
            Input token IDs.
        attention_mask : torch.Tensor, shape [seq_len] or [batch_size, seq_len]
            Attention mask indicating valid tokens.
        sep_token_id : int
            Token ID for the separator token.
        eos_token_id : int
            Token ID for the end-of-sequence token.
            
        Returns
        -------
        tuple
            A tuple containing:
            - cond_ids: torch.LongTensor, shape [cond_seq_len]
                The conditioning sequence token IDs
            - cond_mask: torch.Tensor, shape [cond_seq_len]
                The attention mask for the conditioning sequence
            If no valid conditioning segment can be formed, returns (None, None).
            
        Notes
        -----
        Behavior:
        - If a separator token is present, returns tokens up to and including the first SEP
        - If no SEP token is found, returns tokens up to but excluding the first EOS
        - Handles edge cases like empty sequences, tokens at position 0, etc.
        - Ensures returned tensors are 1D and properly typed
        """
        # Handle edge cases
        if input_ids is None or input_ids.numel() == 0:
            return None, None
        # Ensure 1D tensors
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.squeeze(0)
            
        # Find separator position
        sep_nonzero = (input_ids == sep_token_id).nonzero(as_tuple=True)
        sep_pos = sep_nonzero[0] if len(sep_nonzero) > 0 else torch.tensor([], device=input_ids.device, dtype=torch.long)
        
        if len(sep_pos) == 0:
            # No separator found - use full sequence but exclude end token
            # Find the first end token position (assumed to exist)
            eos_nonzero = (input_ids == eos_token_id).nonzero(as_tuple=True)
            eos_pos = eos_nonzero[0] if len(eos_nonzero) > 0 else torch.tensor([], device=input_ids.device, dtype=torch.long)
            if len(eos_pos) == 0:
                print(f"[WARNING] No end token found in sequence, this shouldn't happen")
                return input_ids, attention_mask
                
            # Exclude end token and everything after it
            end_pos = eos_pos[0]
            if end_pos == 0:
                # End token is at the beginning - return empty sequence
                print(f"[DEBUG] End token found at position 0, returning None")
                return None, None
            print(f"[DEBUG] Excluding end token at position {end_pos}, returning sequence of length {end_pos}")
            return input_ids[:end_pos].long(), attention_mask[:end_pos]
        
        # Separator found - extract up to and including separator
        sep_pos = sep_pos[0]
        if sep_pos == 0:
            # Separator is at the beginning - return just the separator
            print(f"[DEBUG] Separator found at position 0, returning just separator")
            return input_ids[:1].long(), attention_mask[:1]
        print(f"[DEBUG] Separator found at position {sep_pos}, returning sequence of length {sep_pos+1}")
        return input_ids[:sep_pos+1].long(), attention_mask[:sep_pos+1]

    @staticmethod
    def _batch_generate_molecules_conditionally(pl_module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list:
        """
        Batch generate molecules using conditional generation.
        
        This method performs batched conditional molecule generation by extracting
        conditioning sequences from all samples in a batch, padding them to the
        same length, and generating all molecules in a single model forward pass.
        This is much more efficient than generating molecules one at a time.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - model: huggingface-compatible model with .generate() method
            - tokenizer: tokenizer with pad_token_id, eos_token_id, sep_token_id
            - _max_length: int, maximum number of new tokens to generate
        input_ids : torch.Tensor, shape [batch_size, seq_len]
            Input token IDs for the entire batch.
        attention_mask : torch.Tensor, shape [batch_size, seq_len]
            Attention mask indicating valid tokens for the entire batch.
            
        Returns
        -------
        list
            A list of generated sequences, where each element is a torch.Tensor
            representing the generated token sequence for one sample.
            Shape: List[torch.Tensor], where each tensor has shape [generated_seq_len]
            Invalid samples (those without valid conditioning sequences) return
            empty tensors.
            
        Notes
        -----
        The method performs the following steps:
        1. Extracts conditioning sequences from each sample in the batch
        2. Filters out samples without valid conditioning sequences
        3. Pads conditioning sequences to the same length for batched generation
        4. Generates all molecules in a single model forward pass
        5. Returns results with empty tensors for invalid samples
        
        This approach significantly improves performance by batching the model
        forward pass instead of processing samples individually.
        """
        batch_size = input_ids.shape[0]
        
        # Extract conditioning sequences for all samples in the batch
        conditioning_sequences = []
        valid_indices = []
        
        for i in range(batch_size):
            cond_ids, cond_mask = MoleculeValidationCallback._extract_conditioning_sequence(
                input_ids[i], attention_mask[i], pl_module.tokenizer.sep_token_id, pl_module.tokenizer.eos_token_id
            )
            if cond_ids is not None:
                unmasked_ids = cond_ids[cond_mask.bool()]
                conditioning_sequences.append(unmasked_ids)
                valid_indices.append(i)
        
        if not conditioning_sequences:
            return [torch.tensor([]) for _ in range(batch_size)]
        
        # Pad conditioning sequences to the same length
        max_cond_len = max(len(seq) for seq in conditioning_sequences)
        padded_conditioning = torch.zeros(len(conditioning_sequences), max_cond_len, dtype=torch.long, device=input_ids.device)
        
        for i, seq in enumerate(conditioning_sequences):
            padded_conditioning[i, :len(seq)] = seq
        
        # Generate all molecules in a single batch
        generated_sequences = pl_module.model.generate(
            input_ids=padded_conditioning,
            max_new_tokens=pl_module._max_length, 
            pad_token_id=pl_module.tokenizer.pad_token_id,
            eos_token_id=pl_module.tokenizer.eos_token_id
        )
        
        # Create result list with empty tensors for invalid samples
        result = [torch.tensor([]) for _ in range(batch_size)]
        
        # Fill in the valid generated sequences
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = generated_sequences[i]
        
        return result