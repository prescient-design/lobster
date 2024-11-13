from lobster.tokenization import HyenaTokenizerTransform
from torch import Size


class TestHyenaTokenizerTransform:
    def test_hyena_tokenizer_transform(self):
        inputs = 4 * ["ATCGTACGATCGTACGATCGUN"]

        transform_fn = HyenaTokenizerTransform(
            tokenizer_dir="hyena_tokenizer", truncation=True, padding="max_length", max_length=32
        )

        tokenized = transform_fn.transform(inputs, {})

        assert tokenized["input_ids"].shape == Size([4, 32])
        assert tokenized["labels"].shape == Size([4, 32])

    def test_aa_to_dna(self):
        inputs = 4 * ["GYDPETGTWG"]

        transform_fn = HyenaTokenizerTransform(
            tokenizer_dir="hyena_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=32,
            aa_to_dna=True,
        )

        dna_inputs = [transform_fn.translate_aa_to_dna(seq) for seq in inputs]
        assert all(len(s) == 30 for s in dna_inputs)

        tokenized = transform_fn.transform(inputs, {})
        assert tokenized["input_ids"].shape == Size([4, 32])
        assert tokenized["labels"].shape == Size([4, 32])
