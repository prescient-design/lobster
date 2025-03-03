from lobster.constants import GPUType
from lobster.model.utils import Architecture, ArchitectureAnalyzer


class TestArchitectureAnalyzer:
    """Test suite for ArchitectureAnalyzer class."""

    def test_initialization(self):
        """Test that the analyzer initializes with correct parameters."""
        # Test A100 initialization (default)
        config = Architecture(hidden_size=2560, num_attention_heads=32, num_hidden_layers=32, vocab_size=50257)
        analyzer = ArchitectureAnalyzer(config)

        assert analyzer.config == config
        assert analyzer.gpu_type == GPUType.A100
        assert analyzer.target_alignment == 64
        assert analyzer.issues == []
        assert analyzer.suggestions == []

        # Test V100 initialization
        v100_analyzer = ArchitectureAnalyzer(config, gpu_type=GPUType.V100)
        assert v100_analyzer.target_alignment == 8

        # Test H100 initialization
        h100_analyzer = ArchitectureAnalyzer(config, gpu_type=GPUType.H100)
        assert h100_analyzer.target_alignment == 64

    def test_head_dimension_check_a100(self):
        """Test that head dimension checks work correctly for A100."""
        # Create a configuration with non-aligned head dimension (80)
        config = Architecture(hidden_size=2560, num_attention_heads=32, num_hidden_layers=32, vocab_size=50304)

        analyzer = ArchitectureAnalyzer(config)
        analyzer._check_head_dimension()

        assert len(analyzer.issues) == 1
        assert "Head dimension (80) is not divisible by 64" in analyzer.issues[0]
        assert len(analyzer.suggestions) > 0

        # At least one suggestion should be to change to 20 heads (for 128 dimension)
        assert any("Change attention heads from 32 to 20" in s for s in analyzer.suggestions)

    def test_head_dimension_check_v100(self):
        """Test that head dimension checks work correctly for V100."""
        # Create a configuration with head dimension = 80, which is divisible by 8 (V100)
        config = Architecture(hidden_size=2560, num_attention_heads=32, num_hidden_layers=32, vocab_size=50304)

        analyzer = ArchitectureAnalyzer(config, gpu_type=GPUType.V100)
        analyzer._check_head_dimension()

        # For V100, head dimension of 80 is divisible by 8, so no issues should be found
        assert len(analyzer.issues) == 0
        assert len(analyzer.suggestions) == 0

    def test_vocab_size_check(self):
        """Test that vocabulary size checks work correctly."""
        # Create a configuration with non-aligned vocab size
        config = Architecture(
            hidden_size=2560,
            num_attention_heads=32,
            num_hidden_layers=32,
            vocab_size=50257,  # Not divisible by 64
        )

        analyzer = ArchitectureAnalyzer(config)
        analyzer._check_vocab_size()

        assert len(analyzer.issues) == 1
        assert "Vocabulary size (50257) is not divisible by 64" in analyzer.issues[0]
        assert len(analyzer.suggestions) == 1
        assert "Pad vocabulary size from 50257 to 50304" in analyzer.suggestions[0]
        assert "(+47 tokens)" in analyzer.suggestions[0]

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # Perfect configuration (should score 100)
        perfect_config = Architecture(
            hidden_size=2560,
            num_attention_heads=20,  # Gets head dimension of 128 (divisible by 64)
            num_hidden_layers=32,
            vocab_size=50304,  # Divisible by 64
        )

        analyzer = ArchitectureAnalyzer(perfect_config)
        score = analyzer._calculate_efficiency_score()
        assert score == 100.0

        # GPT-3 2.7B configuration (should score lower)
        gpt3_config = Architecture(
            hidden_size=2560,
            num_attention_heads=32,  # Gets head dimension of 80 (not divisible by 64)
            num_hidden_layers=32,
            vocab_size=50257,  # Not divisible by 64
        )

        analyzer = ArchitectureAnalyzer(gpt3_config)
        score = analyzer._calculate_efficiency_score()
        assert score < 100.0
        assert score > 50.0  # From the example in the docstring, we know it's around 60

        # V100 with the same config should score higher since its requirements are less strict
        v100_analyzer = ArchitectureAnalyzer(gpt3_config, gpu_type=GPUType.V100)
        v100_score = v100_analyzer._calculate_efficiency_score()
        assert v100_score > score

    def test_analyze_method(self):
        """Test the full analyze method."""
        # Configuration with various issues
        config = Architecture(
            hidden_size=2570,  # Not divisible by 64
            num_attention_heads=32,
            num_hidden_layers=32,
            vocab_size=50257,  # Not divisible by 64
            tensor_parallel_size=3,  # Not a divisor of 2570
        )

        analyzer = ArchitectureAnalyzer(config)
        result = analyzer.analyze()

        # Check result structure
        assert "config" in result
        assert "analysis" in result
        assert "issues" in result["analysis"]
        assert "suggestions" in result["analysis"]
        assert "efficiency_score" in result["analysis"]

        # Should have multiple issues
        assert len(result["analysis"]["issues"]) >= 3

        # Should have multiple suggestions
        assert len(result["analysis"]["suggestions"]) >= 3

        # Score should be lower than 75 (multiple issues)
        assert result["analysis"]["efficiency_score"] < 75.0
