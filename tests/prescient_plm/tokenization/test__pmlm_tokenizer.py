import importlib.resources

from prescient_plm.tokenization import PmlmTokenizer


class TestPmlmTokenizer:
    def test_3Di_tokenizer(self):
        inputs = ["GdPfQaPfIlSvRvLvEcQvClGpId"]

        path = importlib.resources.files("prescient_plm") / "assets" / "3di_tokenizer"

        tokenizer = PmlmTokenizer.from_pretrained(path)
        tokenized_inputs = tokenizer(inputs)
        decoded_outputs = tokenizer.decode(
            token_ids=tokenized_inputs["input_ids"][0], skip_special_tokens=True
        ).replace(" ", "")

        assert inputs[0] == decoded_outputs
