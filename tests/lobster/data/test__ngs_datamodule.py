from lobster.data import GREDBulkNGSLightningDataModule


class TestFastaLightningDatamodule:
    def test_init(self):
        dm = GREDBulkNGSLightningDataModule(
            root=".",
            batch_size=4,
            max_length=512,
            lengths=(0.8, 0.1, 0.1),
        )

        assert dm._batch_size == 4
