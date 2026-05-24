# Plan for Autoregressive Training with Gen-UME

## Overview

This plan describes implementing autoregressive training for gen_ume as a **modular interpolant** within the `bionemo.moco.interpolants` package. The key insight is that both flow matching and autoregressive generation can be abstracted as "interpolants" that define how to:
1. Sample training conditions (timesteps for flow matching, prefix lengths for autoregressive)
2. Create masked inputs based on those conditions
3. Compute losses for training
4. Generate samples during inference

By creating `AutoregressiveInterpolant` as a drop-in replacement for `DiscreteFlowMatcher`, we achieve:
- **Minimal code changes** to existing gen_ume infrastructure
- **Interface compatibility** enabling easy switching between approaches
- **Clean separation** of interpolant logic from model architecture
- **Easy comparison** of flow matching vs autoregressive via configuration

## 1. High-Level Strategy

**Goal**: Create a new autoregressive interpolant within `bionemo.moco.interpolants` that enables left-to-right generation where the model predicts position `i` given positions `1...i-1`. This will be a parallel option to `DiscreteFlowMatcher`, following the same interface pattern.

**Architecture**: 
- New module: `bionemo.moco.interpolants.AutoregressiveInterpolant`
- Follows same interface as `DiscreteFlowMatcher` for seamless integration
- Pluggable into existing gen_ume infrastructure via configuration

**Key Design Decision**: Whether to use:
- **Joint autoregressive**: Generate sequence and structure tokens simultaneously in lockstep (position by position)
- **Sequential autoregressive**: Generate sequence first, then structure conditioned on sequence (or vice versa)
- **Hybrid approach**: Configurable per batch/sample

## 2. Core Components to Create/Modify

### 2.1 New Module: `bionemo.moco.interpolants.AutoregressiveInterpolant`

**Location**: Create new file in `bionemo.moco.interpolants` package
- `bionemo/moco/interpolants/autoregressive.py`

**Interface**: Must match `DiscreteFlowMatcher` API for drop-in compatibility:
- `__init__(time_distribution, prior_distribution, device)`
- `sample_time(batch_size)` → sample prefix lengths instead of time
- `sample_prior(shape)` → sample mask tokens for positions beyond prefix
- `interpolate(x_1, t, x_0)` → create masked input based on prefix length
- `loss(logits, targets, t)` → compute autoregressive loss
- `step(logits, t, x_t, dt, stochasticity, temperature)` → generate next token

### 2.2 Attention Masking System
- **Location**: Within `AutoregressiveInterpolant` class
- **Purpose**: Causal attention masks that prevent positions from attending to future positions
  
Components:
- Causal mask generator: Create lower-triangular mask for positions 
- Per-sample prefix lengths: Allow variable-length prefixes per batch item
- Modality-specific masks: Independent causal masks for sequence vs structure tokens
- Padding-aware masks: Properly handle variable-length sequences in batch

### 2.3 Training Objective
- **Location**: `AutoregressiveInterpolant.loss()` method
- **Type**: Next-token prediction loss (vs flow matching loss in `DiscreteFlowMatcher`)
  
Components:
- Position-wise cross-entropy: Sum loss over all positions i predicting token at i+1
- Masking strategy per batch:
  - Random start position per sample (via `sample_time()`)
  - Predict all subsequent positions given prefix
- Loss weighting options:
  - Uniform weighting across all positions
  - Position-dependent weighting (e.g., emphasize later positions)
  - Length-normalized loss

## 3. Batch-Level Masking Strategy

### 3.1 Training Time Masking
For each batch, need to determine:

**Option A: Random Prefix Lengths (Recommended)**
- For each sample in batch, randomly sample prefix length `k ~ Uniform(0, L-1)`
- Visible context: positions `[0, k]`
- Prediction targets: positions `[k+1, L]`
- Loss computed on all positions after prefix

**Option B: Curriculum Learning**
- Start with short prefixes (0-10% of sequence)
- Gradually increase prefix length during training
- Final stage: train on all prefix lengths

**Option C: Position Dropout**
- Similar to prefix but with non-contiguous masking
- Keep first position, then probabilistically keep/mask subsequent positions
- Maintains autoregressive property but more flexible

### 3.2 Batch Construction Details

```python
# Per sample in batch:
for i in range(batch_size):
    # 1. Sample prefix_length[i] for sample i
    prefix_length[i] = random.randint(0, L-1)
    
    # 2. Create causal_mask[i] = lower_triangular up to position L
    causal_mask[i] = torch.tril(torch.ones(L, L))
    
    # 3. Create prefix_mask[i] = unmask positions [0:prefix_length[i]]
    prefix_mask[i] = torch.zeros(L)
    prefix_mask[i, :prefix_length[i]] = 1
    
    # 4. Create target_mask[i] = positions [prefix_length[i]+1:L]
    target_mask[i] = torch.zeros(L)
    target_mask[i, prefix_length[i]+1:] = 1
    
    # 5. Combined attention_mask = causal_mask & prefix_mask & padding_mask
    attention_mask[i] = causal_mask[i] & prefix_mask[i] & padding_mask[i]
```

## 4. Model Forward Pass Modifications

### 4.1 Input Preparation
- **Sequence tokens**: Ground truth tokens for positions 0 to k, mask tokens for k+1 to L
- **Structure tokens**: Same strategy (or different if using sequential approach)
- **Attention mask**: Causal + prefix mask as computed above
- **Position IDs**: Standard sequential position encoding (0, 1, 2, ..., L-1)

### 4.2 Embedding Layer
- **No change needed**: Current embedding approach should work
- Ensure mask tokens have learnable embeddings

### 4.3 Transformer Encoder (NeoBERT)
- **Attention modification required**: 
  - NeoBERT needs to respect causal attention mask
  - Check if current implementation supports custom attention masks
  - May need to modify attention mechanism to enforce causality

## 5. Loss Computation

```python
# For each sample i in batch:
for i in range(batch_size):
    prefix_len = prefix_length[i]
    
    # Sequence loss
    seq_logits = model_output["sequence_logits"][i]
    seq_targets = batch["sequence"][i]
    seq_loss = CrossEntropy(seq_logits[prefix_len:], seq_targets[prefix_len:])
    
    # Structure loss  
    struct_logits = model_output["structure_logits"][i]
    struct_targets = batch["structure_tokens"][i]
    struct_loss = CrossEntropy(struct_logits[prefix_len:], struct_targets[prefix_len:])
    
    # Combined
    total_loss += seq_loss + struct_loss
```

## 6. Generation/Inference Modifications

### 6.1 Autoregressive Sampling
```python
# Start with empty sequence (all mask tokens)
tokens = torch.full((batch_size, L), mask_token_id)

# For position i = 0 to L-1:
for i in range(L):
    # Forward pass with tokens [0:i] unmasked, [i+1:L] masked
    # (causal attention naturally handles this)
    output = model.forward(tokens, attention_mask=causal_mask)
    
    # Sample token at position i from predicted distribution
    logits_i = output["logits"][:, i, :]
    tokens[:, i] = sample_from_logits(logits_i, temperature=temp)
    
    # Repeat for next position
```

### 6.2 Generation Modes Support
- **Unconditional**: Start with all masks, generate left-to-right
- **Prefix conditioning**: Provide fixed prefix, generate continuation
- **Inverse folding**: Structure tokens as conditioning, generate sequence autoregressively
- **Forward folding**: Sequence tokens as conditioning, generate structure autoregressively

## 7. Implementation Roadmap

### Phase 1: Create `AutoregressiveInterpolant` Module
**Location**: `bionemo/moco/interpolants/autoregressive.py`

1. Create base `AutoregressiveInterpolant` class matching `DiscreteFlowMatcher` interface
2. Implement `sample_time()` → samples prefix lengths per batch
3. Implement `sample_prior()` → returns mask tokens for initialization
4. Implement `interpolate()` → creates autoregressive masked inputs
5. Implement `loss()` → computes next-token prediction loss
6. Implement `step()` → generates next token during sampling
7. Add causal attention mask utilities within the class
8. Add unit tests for all methods

### Phase 2: Prefix Length Sampling Strategies
**Location**: `bionemo/moco/interpolants/autoregressive.py`

1. Implement `UniformPrefixDistribution` class (samples uniform random prefix lengths)
2. Implement `CurriculumPrefixDistribution` class (gradually increases prefix difficulty)
3. Implement `FixedPrefixDistribution` class (fixed prefix length for testing)
4. Add configuration options for prefix sampling strategies

### Phase 3: Gen-UME Integration  
**Location**: `src/lobster/model/gen_ume/_gen_ume_sequence_structure_encoder_lightning_module.py`

1. Add `interpolant` parameter to accept either `DiscreteFlowMatcher` or `AutoregressiveInterpolant`
2. Modify `__init__()` to conditionally instantiate interpolant based on config
3. Ensure `step()` method works with both interpolants (should be interface-compatible)
4. Add causal attention mask support to `UMESequenceStructureEncoderModule.forward()`
5. Verify NeoBERT respects causal masks (may need attention mechanism updates)

### Phase 4: Inference/Generation
**Location**: `AutoregressiveInterpolant` class + gen_ume module

1. Implement autoregressive `generate()` method in interpolant
2. Add support for partial prefix conditioning
3. Implement KV caching for efficiency (optional, advanced)
4. Add temperature/top-k/top-p sampling controls
5. Ensure generation works for all three modes (unconditional, inverse folding, forward folding)

### Phase 5: Training & Evaluation
1. Train autoregressive model on same data as flow matching model
2. Compare generation quality (TM-score, AAR, etc.)
3. Evaluate sampling speed (autoregressive is typically slower)
4. Ablation studies on prefix length strategies
5. Compare joint vs sequential autoregressive approaches

## 8. Key Design Considerations

### 8.1 Sequence vs Structure Ordering
**Question**: Which should be generated first, or should they be joint?

**Options**:
- **Joint** (parallel): Generate seq[i] and struct[i] together
  - Pros: Maintains coupling, simpler implementation
  - Cons: May not capture sequential dependencies well
  
- **Sequence-first**: Generate full sequence, then structure conditioned on it
  - Pros: Mirrors natural folding process, could use frozen sequence model
  - Cons: Two-stage process, no sequence feedback from structure
  
- **Structure-first**: Generate structure, then sequence (inverse folding style)
  - Pros: Could leverage strong structure priors
  - Cons: Less biologically motivated
  
- **Flexible/Learned**: Model learns to alternate or prioritize based on context
  - Pros: Maximum flexibility
  - Cons: Complex implementation, harder to interpret

**Recommendation**: Start with **joint autoregressive** (simplest), then experiment with sequential if needed.

### 8.2 Prefix Length Distribution
**Question**: How to sample prefix lengths during training?

**Options**:
- Uniform(0, L): All prefix lengths equally likely
- Beta distribution: Bias toward certain prefix lengths
- Curriculum: Start with longer prefixes (easier), move to shorter
- Task-specific: Different distributions for different generation modes

**Recommendation**: Start with **Uniform(0, L-1)**, add curriculum if training unstable.

### 8.3 Compatibility with Existing Checkpoints
- Autoregressive training requires different attention patterns
- Likely cannot directly fine-tune from flow matching checkpoints
- May want to keep flow matching as separate mode
- Consider adding `training_mode` parameter: "flow_matching" vs "autoregressive"

## 9. Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| NeoBERT may not support causal attention | Modify attention mechanism or add attention_mask parameter |
| Slower generation (L forward passes) | Implement KV caching for efficiency |
| Training instability with short prefixes | Use curriculum learning, start with longer prefixes |
| Joint vs sequential generation unclear | Make configurable, experiment with both |
| Integration with existing codebase | Add mode parameter, keep flow matching functional |

## 10. Success Metrics

Compare autoregressive vs flow matching on:
- Training convergence speed
- Final loss values
- Generation quality (TM-score, AAR, RMSD)
- Sampling diversity
- Computational efficiency (training & inference)
- Sample quality vs number of steps

## 10A. Advantages of the Modular `bionemo.moco.interpolants` Approach

### 10A.1 Clean Separation of Concerns
- **Interpolant logic**: All autoregressive-specific logic lives in `AutoregressiveInterpolant` class
- **Model architecture**: Gen-UME architecture remains unchanged
- **Training loop**: Minimal modifications to existing training code
- **Maintainability**: Easy to debug, test, and extend independently

### 10A.2 Interface Compatibility
- **Drop-in replacement**: Swap `DiscreteFlowMatcher` with `AutoregressiveInterpolant` via config
- **Consistent API**: Same method signatures enable code reuse
- **Backward compatibility**: Flow matching remains fully functional
- **Easy experimentation**: Compare approaches by changing one parameter

### 10A.3 Extensibility
- **Future interpolants**: Easy to add new interpolant types (e.g., diffusion-based, hierarchical)
- **Hybrid approaches**: Could combine multiple interpolants in sequence
- **Research-friendly**: Clean abstraction for experimenting with different generation strategies
- **Modularity**: Changes to interpolant don't affect model architecture

### 10A.4 Code Reuse
- **Shared infrastructure**: Both interpolants use same:
  - Prior distributions (`DiscreteMaskedPrior`, `DiscreteUniformPrior`)
  - Model architecture (`UMESequenceStructureEncoderModule`)
  - Training pipeline (Lightning training loop)
  - Evaluation metrics
- **Reduced duplication**: No need to fork entire codebase for autoregressive version

### 10A.5 Production Benefits
- **Single codebase**: One repository supports multiple training paradigms
- **Easy A/B testing**: Compare methods with identical infrastructure
- **Checkpoint compatibility**: Model checkpoints only differ in training method
- **Deployment flexibility**: Choose interpolant based on use case (speed vs quality tradeoffs)

## 11. `AutoregressiveInterpolant` Class Structure

### 11.1 Class Definition

**File**: `bionemo/moco/interpolants/autoregressive.py`

```python
from typing import Callable
import torch
from torch import Tensor

class AutoregressiveInterpolant:
    """
    Autoregressive interpolant for discrete token generation.
    
    Compatible interface with DiscreteFlowMatcher for drop-in replacement in gen_ume.
    Instead of flow matching with time t, uses prefix lengths for autoregressive generation.
    """
    
    def __init__(
        self,
        time_distribution: Callable | None = None,  # Not used, kept for interface compatibility
        prior_distribution: Callable | None = None,
        device: torch.device = "cpu",
        prefix_length_strategy: str = "uniform",
        min_prefix_length: int = 0,
        curriculum_schedule: dict | None = None,
    ):
        """
        Initialize autoregressive interpolant.
        
        Args:
            time_distribution: Ignored (kept for interface compatibility)
            prior_distribution: Distribution for sampling prior tokens (mask tokens)
            device: Device to run on
            prefix_length_strategy: "uniform", "curriculum", or "fixed"
            min_prefix_length: Minimum prefix length (for curriculum learning)
            curriculum_schedule: Schedule for curriculum learning
        """
        self.prior_distribution = prior_distribution
        self.device = device
        self.prefix_length_strategy = prefix_length_strategy
        self.min_prefix_length = min_prefix_length
        self.curriculum_schedule = curriculum_schedule or {}
        self.training_step = 0
    
    def sample_time(self, batch_size: int) -> Tensor:
        """
        Sample prefix lengths for batch (replaces time sampling).
        
        Returns tensor of shape (batch_size,) with prefix lengths.
        These act as 't' in the flow matching interface.
        """
        # Implementation will sample prefix lengths based on strategy
        pass
    
    def sample_prior(self, shape: tuple) -> Tensor:
        """
        Sample prior tokens (mask tokens) for initialization.
        
        Args:
            shape: (batch_size, seq_len)
            
        Returns:
            Tensor of mask token IDs
        """
        return self.prior_distribution.sample(shape)
    
    def interpolate(self, x_1: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
        """
        Create autoregressive masked input.
        
        Args:
            x_1: Ground truth tokens (batch_size, seq_len)
            t: Prefix lengths (batch_size,) - acts as 'time' parameter
            x_0: Prior tokens (mask tokens)
            
        Returns:
            Masked input where positions < prefix_length[i] are x_1, rest are x_0
        """
        batch_size, seq_len = x_1.shape
        x_t = x_0.clone()
        
        for i in range(batch_size):
            prefix_len = int(t[i].item())
            x_t[i, :prefix_len] = x_1[i, :prefix_len]
        
        return x_t
    
    def loss(self, logits: Tensor, targets: Tensor, t: Tensor) -> Tensor:
        """
        Compute autoregressive loss (next-token prediction).
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            targets: Ground truth tokens (batch_size, seq_len)
            t: Prefix lengths (batch_size,)
            
        Returns:
            Loss per sample (batch_size,)
        """
        import torch.nn.functional as F
        
        batch_size, seq_len, vocab_size = logits.shape
        loss = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            prefix_len = int(t[i].item())
            if prefix_len < seq_len:
                # Compute loss on positions after prefix
                loss[i] = F.cross_entropy(
                    logits[i, prefix_len:],
                    targets[i, prefix_len:],
                    reduction='mean'
                )
        
        return loss
    
    def step(
        self, 
        logits: Tensor, 
        t: Tensor, 
        x_t: Tensor, 
        dt: Tensor,
        stochasticity: float = 1.0,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate next token (for inference).
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            t: Current position to generate (batch_size,)
            x_t: Current token sequence (batch_size, seq_len)
            dt: Step size (typically 1 for autoregressive)
            stochasticity: Sampling randomness (not used, kept for interface)
            temperature: Temperature for sampling
            
        Returns:
            Updated token sequence
        """
        import torch.nn.functional as F
        
        batch_size, seq_len, vocab_size = logits.shape
        x_next = x_t.clone()
        
        for i in range(batch_size):
            pos = int(t[i].item())
            if pos < seq_len:
                # Sample from temperature-scaled distribution
                probs = F.softmax(logits[i, pos] / temperature, dim=-1)
                x_next[i, pos] = torch.multinomial(probs, 1).item()
        
        return x_next
    
    def create_causal_attention_mask(
        self, 
        batch_size: int, 
        seq_len: int, 
        prefix_lengths: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Create causal attention masks for autoregressive generation.
        
        Returns:
            attention_mask: (batch_size, seq_len, seq_len) for transformer
            target_mask: (batch_size, seq_len) indicating positions to predict
        """
        # Implementation in detailed section below
        pass
```

### 11.2 Integration with Gen-UME

**Modified**: `src/lobster/model/gen_ume/_gen_ume_sequence_structure_encoder_lightning_module.py`

```python
class UMESequenceStructureEncoderLightningModule(LightningModule):
    def __init__(
        self,
        ...,
        # Modified: Allow AutoregressiveInterpolant as option
        interpolant: Callable[..., DiscreteFlowMatcher | AutoregressiveInterpolant] = DiscreteFlowMatcher,
        training_mode: Literal["flow_matching", "autoregressive"] = "flow_matching",
        **kwargs
    ):
        ...
        
        # Instantiate interpolant (works for both DiscreteFlowMatcher and AutoregressiveInterpolant)
        if training_mode == "autoregressive":
            from bionemo.moco.interpolants import AutoregressiveInterpolant
            interpolant_seq = AutoregressiveInterpolant(
                prior_distribution=prior_seq,
                device=device,
                prefix_length_strategy="uniform",
            )
            interpolant_struc = AutoregressiveInterpolant(
                prior_distribution=prior_struc,
                device=device,
                prefix_length_strategy="uniform",
            )
        else:
            # Existing flow matching setup
            interpolant_seq = self.interpolant(
                time_distribution=time_distribution_seq, 
                prior_distribution=prior_seq, 
                device=device
            )
            ...
```

### 11.3 Usage Example

```python
# Training with AutoregressiveInterpolant
from bionemo.moco.interpolants import AutoregressiveInterpolant
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

model = UMESequenceStructureEncoderLightningModule(
    ...,
    interpolant=AutoregressiveInterpolant,  # Use autoregressive instead of DiscreteFlowMatcher
    training_mode="autoregressive",
    ...
)

# Training step automatically uses AutoregressiveInterpolant.loss()
# which computes next-token prediction instead of flow matching loss
```

## 12. Detailed Implementation Notes

### 12.1 Causal Attention Mask Format

The attention mask needs to be in the format expected by the transformer:
- Shape: `(batch_size, 1, seq_len, seq_len)` or `(batch_size, seq_len, seq_len)`
- Values: `0` for masked positions (cannot attend), `1` for unmasked positions (can attend)
- Causal mask: Lower triangular matrix where position i can only attend to positions ≤ i

```python
def create_causal_mask(seq_len, device):
    """Create causal attention mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask  # (seq_len, seq_len)
```

### 12.2 Combined Mask Construction

Need to combine multiple mask types:
1. **Causal mask**: Prevents future token attention
2. **Padding mask**: Masks out padding tokens
3. **Prefix mask**: Masks tokens beyond prefix length

```python
def create_autoregressive_batch_masks(
    batch_size, 
    seq_len, 
    prefix_lengths, 
    padding_mask,
    device
):
    """
    Create combined masks for autoregressive training.
    
    Args:
        batch_size: Number of samples in batch
        seq_len: Sequence length
        prefix_lengths: Tensor of shape (batch_size,) with prefix length per sample
        padding_mask: Tensor of shape (batch_size, seq_len) indicating valid positions
        device: torch device
        
    Returns:
        attention_mask: (batch_size, seq_len, seq_len) combined attention mask
        target_mask: (batch_size, seq_len) indicating which positions to compute loss on
    """
    # Create base causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Create prefix mask (which positions are visible as input)
    prefix_mask = torch.zeros(batch_size, seq_len, device=device)
    for i in range(batch_size):
        prefix_mask[i, :prefix_lengths[i]] = 1
    
    # Expand prefix mask for attention: (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
    # A position can only attend to prefix positions
    prefix_attention_mask = prefix_mask.unsqueeze(1).expand(-1, seq_len, -1)
    
    # Combine: causal AND prefix AND padding
    attention_mask = causal_mask * prefix_attention_mask
    if padding_mask is not None:
        # padding_mask: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        padding_attention_mask = padding_mask.unsqueeze(1)
        attention_mask = attention_mask * padding_attention_mask
    
    # Create target mask (which positions to predict)
    target_mask = torch.zeros(batch_size, seq_len, device=device)
    for i in range(batch_size):
        # Predict positions after prefix up to sequence end
        if prefix_lengths[i] < seq_len:
            target_mask[i, prefix_lengths[i]:] = 1
    
    # Mask out padding positions from targets
    if padding_mask is not None:
        target_mask = target_mask * padding_mask
    
    return attention_mask, target_mask
```

### 12.3 Modified Training Step

```python
def autoregressive_step(self, batch, batch_idx):
    """Training step for autoregressive mode."""
    device = batch["sequence"].device
    batch_size, seq_len = batch["sequence"].shape
    
    # Sample random prefix lengths for this batch
    prefix_lengths = torch.randint(0, seq_len, (batch_size,), device=device)
    
    # Get ground truth tokens
    seq_gt = batch["sequence"]
    struct_tokens_gt = self.encode_structure(*batch["input"])[0].argmax(dim=-1)
    
    # Create masks
    padding_mask = batch["mask"]
    attention_mask, target_mask = create_autoregressive_batch_masks(
        batch_size, seq_len, prefix_lengths, padding_mask, device
    )
    
    # Prepare input tokens (use ground truth up to prefix, masks after)
    seq_input = seq_gt.clone()
    struct_input = struct_tokens_gt.clone()
    for i in range(batch_size):
        seq_input[i, prefix_lengths[i]:] = self.mask_token_id
        struct_input[i, prefix_lengths[i]:] = self.mask_index_struc_tokens
    
    # Forward pass with causal attention
    output = self.encoder(
        sequence_input_ids=seq_input,
        structure_input_ids=struct_input,
        position_ids=batch["indices"],
        attention_mask=attention_mask,
        conditioning_tensor=None,
        timesteps=None,
        return_auxiliary_tasks=False,
    )
    
    # Compute loss only on target positions
    seq_logits = output["sequence_logits"]
    struct_logits = output["structure_logits"]
    
    # Flatten and apply target mask
    seq_logits_flat = seq_logits.view(-1, self.vocab_size)
    seq_targets_flat = seq_gt.view(-1)
    struct_logits_flat = struct_logits.view(-1, self.num_struc_classes)
    struct_targets_flat = struct_tokens_gt.view(-1)
    target_mask_flat = target_mask.view(-1)
    
    # Compute losses only on target positions
    seq_loss = F.cross_entropy(
        seq_logits_flat[target_mask_flat.bool()],
        seq_targets_flat[target_mask_flat.bool()],
        reduction='mean'
    )
    struct_loss = F.cross_entropy(
        struct_logits_flat[target_mask_flat.bool()],
        struct_targets_flat[target_mask_flat.bool()],
        reduction='mean'
    )
    
    total_loss = seq_loss + struct_loss
    
    return {
        "loss": total_loss,
        "seq_loss": seq_loss,
        "struct_loss": struct_loss,
    }
```

### 12.4 Autoregressive Generation

```python
def generate_autoregressive(
    self,
    length,
    num_samples,
    temperature_seq=1.0,
    temperature_struct=1.0,
    prefix_seq=None,
    prefix_struct=None,
):
    """
    Generate sequences autoregressively.
    
    Args:
        length: Target sequence length
        num_samples: Number of sequences to generate
        temperature_seq: Sampling temperature for sequence tokens
        temperature_struct: Sampling temperature for structure tokens
        prefix_seq: Optional sequence prefix (num_samples, prefix_len)
        prefix_struct: Optional structure prefix (num_samples, prefix_len)
    """
    device = next(self.parameters()).device
    
    # Initialize with mask tokens
    seq_tokens = torch.full(
        (num_samples, length), 
        self.mask_token_id, 
        device=device
    )
    struct_tokens = torch.full(
        (num_samples, length), 
        self.mask_index_struc_tokens, 
        device=device
    )
    
    # Set prefix if provided
    start_pos = 0
    if prefix_seq is not None:
        prefix_len = prefix_seq.shape[1]
        seq_tokens[:, :prefix_len] = prefix_seq
        struct_tokens[:, :prefix_len] = prefix_struct
        start_pos = prefix_len
    
    # Create causal attention mask
    causal_mask = torch.tril(torch.ones(length, length, device=device))
    causal_mask = causal_mask.unsqueeze(0).expand(num_samples, -1, -1)
    
    # Generate position by position
    for pos in tqdm(range(start_pos, length), desc="Generating"):
        # Forward pass
        output = self.encoder(
            sequence_input_ids=seq_tokens,
            structure_input_ids=struct_tokens,
            position_ids=torch.arange(length, device=device).unsqueeze(0).expand(num_samples, -1),
            attention_mask=causal_mask,
            conditioning_tensor=torch.zeros(num_samples, length, 1, device=device),
            timesteps=None,
            return_auxiliary_tasks=False,
        )
        
        # Sample next tokens
        seq_logits = output["sequence_logits"][:, pos, :] / temperature_seq
        struct_logits = output["structure_logits"][:, pos, :] / temperature_struct
        
        seq_probs = F.softmax(seq_logits, dim=-1)
        struct_probs = F.softmax(struct_logits, dim=-1)
        
        seq_tokens[:, pos] = torch.multinomial(seq_probs, 1).squeeze(-1)
        struct_tokens[:, pos] = torch.multinomial(struct_probs, 1).squeeze(-1)
    
    return {
        "sequence_tokens": seq_tokens,
        "structure_tokens": struct_tokens,
    }
```

## 13. Configuration Changes

### 13.1 Lightning Module Parameters

Add new parameters to support `AutoregressiveInterpolant`:

```python
class UMESequenceStructureEncoderLightningModule(LightningModule):
    def __init__(
        self,
        ...,
        # Modified: interpolant can be DiscreteFlowMatcher or AutoregressiveInterpolant
        interpolant: Callable[..., DiscreteFlowMatcher | AutoregressiveInterpolant] = DiscreteFlowMatcher,
        
        # New autoregressive parameters
        training_mode: Literal["flow_matching", "autoregressive"] = "flow_matching",
        prefix_length_strategy: Literal["uniform", "curriculum", "fixed"] = "uniform",
        min_prefix_length: int = 0,
        curriculum_schedule: dict | None = None,
        **kwargs
    ):
        ...
        
        # Conditional instantiation based on training_mode
        if training_mode == "autoregressive":
            from bionemo.moco.interpolants import AutoregressiveInterpolant
            
            interpolant_seq = AutoregressiveInterpolant(
                prior_distribution=prior_seq,
                device=device,
                prefix_length_strategy=prefix_length_strategy,
                min_prefix_length=min_prefix_length,
                curriculum_schedule=curriculum_schedule,
            )
            interpolant_struc = AutoregressiveInterpolant(
                prior_distribution=prior_struc,
                device=device,
                prefix_length_strategy=prefix_length_strategy,
                min_prefix_length=min_prefix_length,
                curriculum_schedule=curriculum_schedule,
            )
        else:
            # Existing flow matching setup
            interpolant_seq = self.interpolant(
                time_distribution=time_distribution_seq,
                prior_distribution=prior_seq,
                device=device
            )
            interpolant_struc = self.interpolant(
                time_distribution=time_distribution_struc,
                prior_distribution=prior_struc,
                device=device
            )
        
        self.interpolant_seq = interpolant_seq
        self.interpolant_struc = interpolant_struc
```

### 13.2 Example Configurations

**Flow Matching (Current Default)**:
```python
model = UMESequenceStructureEncoderLightningModule(
    mask_token_id=32,
    pad_token_id=1,
    vocab_size=33,
    training_mode="flow_matching",  # Use DiscreteFlowMatcher
    interpolant=DiscreteFlowMatcher,
    ...
)
```

**Autoregressive Training**:
```python
from bionemo.moco.interpolants import AutoregressiveInterpolant

model = UMESequenceStructureEncoderLightningModule(
    mask_token_id=32,
    pad_token_id=1,
    vocab_size=33,
    training_mode="autoregressive",  # Use AutoregressiveInterpolant
    interpolant=AutoregressiveInterpolant,
    prefix_length_strategy="uniform",  # Random prefix lengths
    ...
)
```

**Autoregressive with Curriculum Learning**:
```python
model = UMESequenceStructureEncoderLightningModule(
    mask_token_id=32,
    pad_token_id=1,
    vocab_size=33,
    training_mode="autoregressive",
    interpolant=AutoregressiveInterpolant,
    prefix_length_strategy="curriculum",
    min_prefix_length=0,
    curriculum_schedule={
        "start_step": 0,
        "end_step": 50000,
        "start_prefix_ratio": 0.8,  # Start with 80% prefix visible
        "end_prefix_ratio": 0.0,    # End with 0% prefix (full autoregressive)
    },
    ...
)
```

## 14. Module Structure

### 14.1 File Organization

**New files to create**:
```
bionemo/moco/interpolants/
├── __init__.py                    # Export AutoregressiveInterpolant
├── autoregressive.py              # Main AutoregressiveInterpolant class
└── prefix_distributions.py        # Prefix length sampling strategies (optional)
```

**Updated file**:
```
bionemo/moco/interpolants/__init__.py

# Add to exports:
from .autoregressive import AutoregressiveInterpolant

__all__ = [
    "DiscreteFlowMatcher",
    "AutoregressiveInterpolant",  # New export
    ...
]
```

### 14.2 Key Interface Compatibility Points

The `AutoregressiveInterpolant` must match the following `DiscreteFlowMatcher` interface:

| Method | DiscreteFlowMatcher | AutoregressiveInterpolant |
|--------|---------------------|---------------------------|
| `__init__()` | Takes time_distribution, prior_distribution, device | Takes prior_distribution, device, prefix strategy params |
| `sample_time(batch_size)` | Samples time t ∈ [0,1] | Samples prefix lengths ∈ [0, seq_len] |
| `sample_prior(shape)` | Samples from prior (uniform/masked) | Samples mask tokens |
| `interpolate(x_1, t, x_0)` | Masks tokens based on time t | Masks tokens after prefix length |
| `loss(logits, targets, t)` | Flow matching loss | Next-token prediction loss |
| `step(...)` | Discrete flow step | Autoregressive token generation |

This interface compatibility ensures that gen_ume training code requires **minimal changes** - primarily just swapping the interpolant class.

## 15. Summary

### 15.1 What We're Building
A new **`AutoregressiveInterpolant`** class in `bionemo.moco.interpolants` that:
- Implements left-to-right autoregressive generation for protein sequences and structures
- Provides the same interface as `DiscreteFlowMatcher` for seamless integration
- Enables next-token prediction training instead of flow matching
- Supports variable prefix lengths for flexible training strategies

### 15.2 Key Benefits
1. **Modular design**: Interpolant logic separate from model architecture
2. **Drop-in replacement**: Change one parameter to switch training paradigms
3. **Minimal modifications**: Existing gen_ume code mostly unchanged
4. **Backward compatible**: Flow matching remains fully functional
5. **Research-friendly**: Easy to experiment and compare approaches

### 15.3 Integration Points
- **New module**: `bionemo/moco/interpolants/autoregressive.py`
- **Modified module**: `src/lobster/model/gen_ume/_gen_ume_sequence_structure_encoder_lightning_module.py`
- **Configuration**: Add `training_mode` and `prefix_length_strategy` parameters
- **No changes needed**: Model architecture, data loading, evaluation code

### 15.4 Estimated Effort
- **Phase 1** (Core interpolant): 2-3 days
- **Phase 2** (Prefix strategies): 1-2 days  
- **Phase 3** (Integration): 2-3 days
- **Phase 4** (Inference): 2-3 days
- **Phase 5** (Training & eval): Ongoing
- **Total**: ~1-2 weeks for initial implementation

## 16. Next Steps

1. **Review and approve this plan**
2. **Prioritize implementation phases** - which to tackle first?
3. **Decide on design choices**:
   - Joint vs sequential autoregressive? (Recommendation: **joint** for simplicity)
   - Prefix length sampling strategy? (Recommendation: **uniform** to start)
   - Keep flow matching alongside or replace? (Recommendation: **keep both**, make switchable)
4. **Begin Phase 1 implementation** once approved:
   - Create `bionemo/moco/interpolants/autoregressive.py`
   - Implement core `AutoregressiveInterpolant` class with interface compatibility
   - Add unit tests for all public methods
   - Verify interface matches `DiscreteFlowMatcher`

