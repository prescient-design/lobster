import random

from torch.utils.data import IterableDataset, get_worker_info


class ShuffledIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: IterableDataset,
        buffer_size: int = 10000,
        seed: int | None = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

        self._worker_info = None
        self._shared_seed = None

    def _get_shared_seed(self) -> int:
        if self._shared_seed is None:
            base_seed = self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
            self._shared_seed = base_seed

        return self._shared_seed

    def _set_seed(self):
        shared_seed = self._get_shared_seed()

        if self._worker_info is None:
            random.seed(shared_seed)
            return

        # Multiple workers
        worker_id = self._worker_info.id
        worker_seed = shared_seed + worker_id
        random.seed(worker_seed)

    def __iter__(self):
        self._worker_info = get_worker_info()
        self._set_seed()

        # Fill the shuffle buffer
        shuffle_buffer = []

        try:
            dataset_iter = iter(self.dataset)

            for _ in range(self.buffer_size):
                shuffle_buffer.append(next(dataset_iter))

        except StopIteration:
            self.buffer_size = len(shuffle_buffer)

        # Shuffle the buffer
        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)

                    yield shuffle_buffer[evict_idx]

                    shuffle_buffer[evict_idx] = item
                except StopIteration:
                    break

            while len(shuffle_buffer) > 0:
                yield shuffle_buffer.pop()

        except GeneratorExit:
            pass
