import math
from dataclasses import dataclass

from lobster.constants import GPUType, ModelType


@dataclass
class Architecture:
    """Architecture configuration for a transformer model."""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int | None = None  # If None, assumes 4*hidden_size
    sequence_length: int = 2048
    batch_size: int = 32
    model_type: ModelType = ModelType.DECODER_ONLY
    tensor_parallel_size: int = 1
    name: str | None = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


class ArchitectureAnalyzer:
    """
    Analyzes transformer model architectures for GPU hardware efficiency.

    This class implements the analysis algorithms from "The Case for Co-Designing Model
    Architectures with Hardware" (Anthony et al., 2024) to identify inefficient
    model dimensions and provide optimization suggestions.

    Parameters
    ----------
    config : Architecture
        The transformer model configuration to analyze.
    gpu_type : GPUType, default=GPUType.A100
        The GPU hardware to target for optimization.

    Attributes
    ----------
    target_alignment : int
        The alignment requirement for optimal Tensor Core usage (64 for FP16).
    issues : List[str]
        List of identified efficiency issues.
    suggestions : List[str]
        List of optimization suggestions.

    Examples
    --------
    Analyzing GPT-3 2.7B:

    >>> from lobster.model.utils import Architecture, ArchitectureAnalyzer, ModelType
    >>> config = Architecture(
    ...     hidden_size=2560,
    ...     num_attention_heads=32,
    ...     num_hidden_layers=32,
    ...     vocab_size=50257,
    ...     model_type=ModelType.ENCODER_ONLY,
    ...     name="My Model"
    ... )
    >>> analyzer = ArchitectureAnalyzer(config)
    >>> result = analyzer.analyze()
    >>> ================================================================================
        GPU EFFICIENCY ANALYSIS: My Model
        ================================================================================
        Model Configuration:
        Model Type:          encoder_only
        Hidden Size:         2560
        Attention Heads:     32
        Head Dimension:      80
        Hidden Layers:       32
        Vocabulary Size:     50257
        Intermediate Size:   10240
        Tensor Parallel:     1

        Efficiency Score: 78.8/100 (Good)

        Identified Issues:
        1. Head dimension (80) is not divisible by 64
        2. Vocabulary size (50257) is not divisible by 64

        Optimization Suggestions:
        1. Change attention heads from 32 to 1 to get head dimension of 2560
        2. Change attention heads from 32 to 2 to get head dimension of 1280
        3. Change attention heads from 32 to 4 to get head dimension of 640
        4. Change attention heads from 32 to 5 to get head dimension of 512
        5. Change attention heads from 32 to 8 to get head dimension of 320
        6. Change attention heads from 32 to 10 to get head dimension of 256
        7. Change attention heads from 32 to 20 to get head dimension of 128
        8. Pad vocabulary size from 50257 to 50304 (+47 tokens)

        Note: These recommendations aim to optimize GPU compute efficiency
            without consideration for model quality or convergence.
        ================================================================================
    """

    def __init__(self, architecture: Architecture, gpu_type: GPUType = GPUType.A100):
        self.config = architecture
        self.gpu_type = gpu_type

        # Set target alignment based on GPU type
        if gpu_type == GPUType.V100:
            self.target_alignment = 8  # For V100 FP16 elements (16 bytes)

        elif gpu_type in [GPUType.A100, GPUType.H100]:
            self.target_alignment = 64  # For A100/H100 FP16 elements (128 bytes)

        else:
            # Default to 64 for newer/unknown GPU types
            self.target_alignment = 64

        self.issues = []
        self.suggestions = []

    def analyze(self) -> dict:
        """Analyze the model configuration and return optimization suggestions."""
        self.issues = []
        self.suggestions = []

        self._check_head_dimension()

        self._check_hidden_dimension()

        self._check_vocab_size()

        self._check_intermediate_size()

        self._check_tensor_parallelism()

        efficiency_score = self._calculate_efficiency_score()

        results = {
            "config": {
                "hidden_size": self.config.hidden_size,
                "num_attention_heads": self.config.num_attention_heads,
                "head_dimension": self.config.hidden_size // self.config.num_attention_heads,
                "num_hidden_layers": self.config.num_hidden_layers,
                "vocab_size": self.config.vocab_size,
                "intermediate_size": self.config.intermediate_size,
                "model_type": self.config.model_type.value,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "name": self.config.name,
            },
            "analysis": {
                "issues": self.issues,
                "suggestions": self.suggestions,
                "efficiency_score": efficiency_score,
            },
        }
        print_analysis_result(results)

        return results

    def _check_head_dimension(self):
        """Check if head dimension is optimal for Tensor Cores."""
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        if head_dim % self.target_alignment != 0:
            self.issues.append(f"Head dimension ({head_dim}) is not divisible by {self.target_alignment}")

            # Find closest valid head dimension
            potential_head_dims = []

            # Try fewer heads (larger head dim)
            for num_heads in range(1, self.config.num_attention_heads):
                potential_head_dim = self.config.hidden_size // num_heads
                if (
                    potential_head_dim % self.target_alignment == 0 and potential_head_dim >= 32
                ):  # Reasonable lower bound
                    potential_head_dims.append((num_heads, potential_head_dim))

            # Try increasing hidden size to get aligned head dim
            for hidden_size in range(self.config.hidden_size, self.config.hidden_size + self.target_alignment * 2, 64):
                if (
                    hidden_size % self.target_alignment == 0
                    and (hidden_size // self.config.num_attention_heads) % self.target_alignment == 0
                ):
                    potential_head_dims.append(
                        (self.config.num_attention_heads, hidden_size // self.config.num_attention_heads)
                    )
                    break

            if potential_head_dims:
                for num_heads, head_dim in potential_head_dims:
                    self.suggestions.append(
                        f"Change attention heads from {self.config.num_attention_heads} to {num_heads} "
                        f"to get head dimension of {head_dim}"
                    )

    def _check_hidden_dimension(self):
        """Check if hidden dimension is optimal for Tensor Cores."""
        if self.config.hidden_size % self.target_alignment != 0:
            self.issues.append(f"Hidden size ({self.config.hidden_size}) is not divisible by {self.target_alignment}")

            # Find closest aligned hidden size
            aligned_hidden_size = math.ceil(self.config.hidden_size / self.target_alignment) * self.target_alignment
            self.suggestions.append(f"Increase hidden size from {self.config.hidden_size} to {aligned_hidden_size}")

    def _check_vocab_size(self):
        """Check if vocabulary size is optimal."""
        if self.config.vocab_size % self.target_alignment != 0:
            self.issues.append(
                f"Vocabulary size ({self.config.vocab_size}) is not divisible by {self.target_alignment}"
            )

            # Find closest aligned vocab size
            aligned_vocab_size = math.ceil(self.config.vocab_size / self.target_alignment) * self.target_alignment
            self.suggestions.append(
                f"Pad vocabulary size from {self.config.vocab_size} to {aligned_vocab_size} "
                f"(+{aligned_vocab_size - self.config.vocab_size} tokens)"
            )

    def _check_intermediate_size(self):
        """Check if intermediate size is optimal."""
        # Check if there's a SwiGLU situation (8/3 * h)
        if (
            abs(self.config.intermediate_size / self.config.hidden_size - 8 / 3) < 0.1
            and self.config.intermediate_size % self.target_alignment != 0
        ):
            self.issues.append(
                f"Intermediate size ({self.config.intermediate_size}) appears to use SwiGLU coefficient (~8/3) "
                f"but is not divisible by {self.target_alignment}"
            )

            # Suggest better coefficient
            better_coef = round(
                math.ceil((8 / 3 * self.config.hidden_size) / self.target_alignment)
                * self.target_alignment
                / self.config.hidden_size,
                4,
            )
            better_intermediate = int(better_coef * self.config.hidden_size)

            self.suggestions.append(
                f"For SwiGLU, use coefficient {better_coef} instead of 8/3 "
                f"to get aligned intermediate size of {better_intermediate}"
            )
        elif self.config.intermediate_size % self.target_alignment != 0:
            self.issues.append(
                f"Intermediate size ({self.config.intermediate_size}) is not divisible by {self.target_alignment}"
            )

            # Find closest aligned intermediate size
            aligned_intermediate = (
                math.ceil(self.config.intermediate_size / self.target_alignment) * self.target_alignment
            )
            self.suggestions.append(
                f"Adjust intermediate size from {self.config.intermediate_size} to {aligned_intermediate}"
            )

    def _check_tensor_parallelism(self):
        """Check tensor parallelism settings."""
        if self.config.tensor_parallel_size > 1:
            # Check hidden size divisibility
            if self.config.hidden_size % self.config.tensor_parallel_size != 0:
                self.issues.append(
                    f"Hidden size ({self.config.hidden_size}) is not divisible by tensor parallel size "
                    f"({self.config.tensor_parallel_size})"
                )

                # Find better tensor parallel size
                for tp_size in range(self.config.tensor_parallel_size - 1, 0, -1):
                    if self.config.hidden_size % tp_size == 0:
                        self.suggestions.append(
                            f"Change tensor parallel size from {self.config.tensor_parallel_size} to {tp_size}"
                        )
                        break

            # Check attention heads divisibility
            if (self.config.num_attention_heads * self.config.batch_size) % self.config.tensor_parallel_size != 0:
                self.issues.append(
                    f"(batch_size * num_attention_heads) = ({self.config.batch_size} * {self.config.num_attention_heads}) "
                    f"is not divisible by tensor parallel size ({self.config.tensor_parallel_size})"
                )

                # Suggest adjusted attention head count
                for head_count in range(self.config.num_attention_heads - 5, self.config.num_attention_heads + 6):
                    if (head_count * self.config.batch_size) % self.config.tensor_parallel_size == 0:
                        self.suggestions.append(
                            f"Adjust attention heads from {self.config.num_attention_heads} to {head_count} "
                            f"for better tensor parallelism efficiency"
                        )
                        break

    def _calculate_efficiency_score(self) -> float:
        """Calculate an efficiency score (0-100) based on how well the model follows optimization rules."""
        score = 100.0

        # Major penalties
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if head_dim % self.target_alignment != 0:
            # Penalty based on how far from alignment
            alignment_ratio = head_dim / self.target_alignment
            if alignment_ratio < 1:
                score -= 25
            else:
                mod_val = head_dim % self.target_alignment
                penalty = 25 * (mod_val / self.target_alignment)
                score -= penalty

        if self.config.hidden_size % self.target_alignment != 0:
            score -= 20

        if self.config.vocab_size % self.target_alignment != 0:
            score -= 15

        # Parallel penalties
        if self.config.tensor_parallel_size > 1:
            if self.config.hidden_size % self.config.tensor_parallel_size != 0:
                score -= 15

            if (self.config.num_attention_heads * self.config.batch_size) % self.config.tensor_parallel_size != 0:
                score -= 10

        # Return bounded score
        return max(0, min(100, score))


def print_analysis_result(result):
    """Print analysis result in a readable format."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    config = result["config"]
    model_name = config.get("name", "Transformer")

    print("=" * 80)
    print("GPU EFFICIENCY ANALYSIS", end="")

    if model_name:
        print(f": {model_name}")
    else:
        print()

    print("=" * 80)

    print("Model Configuration:")
    print(f"  Model Type:          {config['model_type']}")
    print(f"  Hidden Size:         {config['hidden_size']}")
    print(f"  Attention Heads:     {config['num_attention_heads']}")
    print(f"  Head Dimension:      {config['head_dimension']}")
    print(f"  Hidden Layers:       {config['num_hidden_layers']}")
    print(f"  Vocabulary Size:     {config['vocab_size']}")
    print(f"  Intermediate Size:   {config['intermediate_size']}")
    print(f"  Tensor Parallel:     {config['tensor_parallel_size']}")

    analysis = result["analysis"]
    print("\nEfficiency Score: ", end="")
    score = analysis["efficiency_score"]
    if score >= 90:
        print(f"\033[92m{score:.1f}/100\033[0m (Excellent)")
    elif score >= 75:
        print(f"\033[93m{score:.1f}/100\033[0m (Good)")
    elif score >= 50:
        print(f"\033[93m{score:.1f}/100\033[0m (Fair)")
    else:
        print(f"\033[91m{score:.1f}/100\033[0m (Poor)")

    if analysis["issues"]:
        print("\nIdentified Issues:")
        for i, issue in enumerate(analysis["issues"], 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n\033[92mNo issues found! Your model {model_name} is well-optimized.\033[0m")

    if analysis["suggestions"]:
        print("\nOptimization Suggestions:")
        for i, suggestion in enumerate(analysis["suggestions"], 1):
            print(f"  {i}. {suggestion}")

    print("\nNote: These recommendations aim to optimize GPU compute efficiency")
    print("      without consideration for model quality or convergence.")
    print("=" * 80)
