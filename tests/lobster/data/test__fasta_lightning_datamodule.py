import os

from lobster.data import FastaLightningDataModule

CUR_DIR = os.path.dirname(__file__)


class TestFastaLightningDatamodule:
    def test_setup(self):
        path_to_test_data = os.path.join(os.path.dirname(__file__), "../../../test_data/query.fasta")

        dm = FastaLightningDataModule(
            path_to_fasta=[path_to_test_data],
            batch_size=4,
            max_length=512,
            lengths=(0.8, 0.1, 0.1),
        )

        dm.setup(stage="fit")

        assert len(dm._train_dataset) == 15
        assert len(dm._val_dataset) == 2
        assert len(dm._test_dataset) == 1

        batch = next(iter(dm.train_dataloader()))

        assert batch is not None

        # assert batch["input_ids"].shape == Size([4, 1, 512])
        # assert batch["attention_mask"].shape == Size([4, 1, 512])
        # assert batch["labels"].shape == Size([4, 1, 512])
