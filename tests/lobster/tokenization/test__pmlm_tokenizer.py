import importlib.resources

from lobster.tokenization import PmlmTokenizer, PmlmTokenizerTransform
from torch import Size


class TestPmlmTokenizer:
    def test_3Di_tokenizer(self):
        inputs = ["GdPfQaPfIlSvRvLvEcQvClGpId"]

        path = importlib.resources.files("lobster") / "assets" / "3di_tokenizer"

        tokenizer = PmlmTokenizer.from_pretrained(path)
        tokenized_inputs = tokenizer(inputs)
        decoded_outputs = tokenizer.decode(
            token_ids=tokenized_inputs["input_ids"][0], skip_special_tokens=True
        ).replace(" ", "")

        assert inputs[0] == decoded_outputs

    def test_cdna_tokenizer(self):
        inputs = 4 * ["ATCGTACGATCGTACGATCGUN"]

        path = importlib.resources.files("lobster") / "assets" / "cdna_tokenizer"

        tokenizer = PmlmTokenizer.from_pretrained(path)
        tokenized_inputs = tokenizer(inputs)
        decoded_outputs = tokenizer.decode(
            token_ids=tokenized_inputs["input_ids"][0], skip_special_tokens=True
        ).replace(" ", "")

        assert inputs[0] == decoded_outputs

        transform_fn = PmlmTokenizerTransform(
            tokenizer_dir="cdna_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=32,
        )

        tokenized = transform_fn.transform(inputs, {})

        assert tokenized["input_ids"].shape == Size([4, 32])
        assert tokenized["labels"].shape == Size([4, 32])
