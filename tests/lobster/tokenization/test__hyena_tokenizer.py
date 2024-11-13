import importlib.resources

from lobster.tokenization import HyenaTokenizer


class TestHyenaTokenizer:
    def test_hyena_tokenizer(self):
        inputs = ["ATCGTACGATCGTACGATCGUN"]

        path = importlib.resources.files("lobster") / "assets" / "hyena_tokenizer"

        tokenizer = HyenaTokenizer.from_pretrained(path)
        tokenized_inputs = tokenizer(
            inputs, add_special_tokens=False, padding="max_length", max_length=512, truncation=True
        )
        # print(tokenized_inputs)

        decoded_outputs = tokenizer.decode(
            token_ids=tokenized_inputs["input_ids"][0], skip_special_tokens=True
        ).replace(" ", "")
        # print(decoded_outputs)

        assert inputs[0] == decoded_outputs
