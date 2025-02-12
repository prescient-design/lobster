from torch.utils.data import Dataset, IterableDataset


class DatasetToIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]
