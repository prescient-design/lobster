from lobster.datasets import AMPLIFYIterableDataset


class TestAMPLIFYIterableDataset:
    def test__iter__(self):
        dataset = AMPLIFYIterableDataset(shuffle=False, download=False)
        example = next(iter(dataset))

        assert isinstance(example, str)
        assert "|" not in example
        assert "." in example
