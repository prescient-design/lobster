import pytest

from lobster.constants import Modality
from lobster.transforms._modality_aware_transform import ComposedModalityAwareTransform, ModalityAwareTransform


class TestModalityAwareTransform:
    def test_init_default(self):
        """Test default initialization."""
        transform_fn = lambda x: x.upper()  # noqa: E731
        transform = ModalityAwareTransform(transform_fn=transform_fn, input_modality=Modality.AMINO_ACID)
        assert transform.transform_fn == transform_fn
        assert transform.input_modality == Modality.AMINO_ACID
        assert transform.output_modalities == (Modality.AMINO_ACID,)

    def test_init_custom_output(self):
        """Test initialization with custom output modality."""
        transform_fn = lambda x: x.upper()  # noqa: E731
        transform = ModalityAwareTransform(
            transform_fn=transform_fn, input_modality=Modality.AMINO_ACID, output_modalities=[Modality.NUCLEOTIDE]
        )
        assert transform.transform_fn == transform_fn
        assert transform.input_modality == Modality.AMINO_ACID
        assert transform.output_modalities == (Modality.NUCLEOTIDE,)

    @pytest.mark.parametrize(
        "transform_fn, input_modality, output_modalities, input_value, expected_output",
        [
            (lambda x: x.upper(), Modality.AMINO_ACID, None, "hello", "HELLO"),
            (lambda x: x.lower(), Modality.NUCLEOTIDE, None, "HELLO", "hello"),
            (lambda x: x.strip(), Modality.NUCLEOTIDE, None, " hello ", "hello"),
            (lambda x: x.replace("|", "."), Modality.AMINO_ACID, None, "hello|world", "hello.world"),
            (lambda x: f"processed_{x}", Modality.AMINO_ACID, [Modality.NUCLEOTIDE], "test", "processed_test"),
        ],
    )
    def test_transform_valid_inputs(
        self,
        transform_fn: callable,
        input_modality: Modality,
        output_modalities: list[Modality] | None,
        input_value: str,
        expected_output: str,
    ):
        """Test transform with various valid inputs and functions."""
        transform = ModalityAwareTransform(
            transform_fn=transform_fn, input_modality=input_modality, output_modalities=output_modalities
        )
        result = transform(input_value)
        assert result == expected_output

    def test_transform_with_complex_function(self):
        """Test transform with a more complex function."""

        def complex_transform(x: str) -> str:
            return f"processed_{x}_end"

        transform = ModalityAwareTransform(transform_fn=complex_transform, input_modality=Modality.AMINO_ACID)
        result = transform("test")
        assert result == "processed_test_end"


class TestComposedModalityAwareTransform:
    def test_init_default(self):
        """Test default initialization with single transform."""
        transform = ModalityAwareTransform(lambda x: x.upper(), input_modality=Modality.AMINO_ACID)
        composed = ComposedModalityAwareTransform(transform)
        assert composed.input_modality == Modality.AMINO_ACID
        assert composed.output_modalities == (Modality.AMINO_ACID,)
        assert len(composed.transforms) == 1

    def test_init_empty(self):
        """Test initialization with no transforms."""
        with pytest.raises(ValueError, match="At least one transform must be provided"):
            ComposedModalityAwareTransform()

    @pytest.mark.parametrize(
        "transforms, input_value, expected_output",
        [
            (
                [
                    ModalityAwareTransform(lambda x: x.upper(), Modality.AMINO_ACID),
                    ModalityAwareTransform(lambda x: x.strip(), Modality.AMINO_ACID),
                ],
                " hello ",
                "HELLO",
            ),
            (
                [
                    ModalityAwareTransform(lambda x: x.replace("|", "."), Modality.AMINO_ACID),
                    ModalityAwareTransform(lambda x: x.strip(), Modality.AMINO_ACID),
                ],
                " hello|world ",
                "hello.world",
            ),
            (
                [
                    ModalityAwareTransform(lambda x: x.upper(), Modality.AMINO_ACID),
                    ModalityAwareTransform(lambda x: x.lower(), Modality.AMINO_ACID),
                ],
                "HeLLo",
                "hello",
            ),
        ],
    )
    def test_transform_valid_inputs(
        self,
        transforms: list[ModalityAwareTransform],
        input_value: str,
        expected_output: str,
    ):
        """Test composed transform with various valid inputs and transform chains."""
        composed = ComposedModalityAwareTransform(*transforms)
        result = composed(input_value)
        assert result == expected_output

    def test_modality_chaining(self):
        """Test that modalities are properly chained through transforms."""
        transform1 = ModalityAwareTransform(
            lambda x: x.upper(), input_modality=Modality.AMINO_ACID, output_modalities=[Modality.NUCLEOTIDE]
        )
        transform2 = ModalityAwareTransform(
            lambda x: x.lower(), input_modality=Modality.NUCLEOTIDE, output_modalities=[Modality.NUCLEOTIDE]
        )

        composed = ComposedModalityAwareTransform(transform1, transform2)

        assert composed.input_modality == Modality.AMINO_ACID
        assert composed.output_modalities == (Modality.NUCLEOTIDE,)
        assert composed("HeLLo") == "hello"

    def test_transform_with_complex_chain(self):
        """Test transform with a more complex chain of transformations."""

        def add_prefix(x: str) -> str:
            return f"prefix_{x}"

        def add_suffix(x: str) -> str:
            return f"{x}_suffix"

        def clean(x: str) -> str:
            return x.strip()

        transforms = [
            ModalityAwareTransform(clean, Modality.AMINO_ACID),
            ModalityAwareTransform(add_prefix, Modality.AMINO_ACID),
            ModalityAwareTransform(add_suffix, Modality.AMINO_ACID),
        ]

        composed = ComposedModalityAwareTransform(*transforms)
        result = composed(" test ")
        assert result == "prefix_test_suffix"
