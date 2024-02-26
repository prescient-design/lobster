import torch


class FarthestFirstTraversal:
    def __init__(
        self, num_samples: int, k: int = 10, random_seed: int = 0xDEADBEEF, p_norm: int = 2
    ):
        """
        Parameters
        ----------
        num_samples: number of samples to return in the FFT
        k: number of centroids to initialize the FFT
        p_norm: 1 (Manhattan) or 2 (Euclidean)

        """
        torch.manual_seed(random_seed)
        self._num_samples = num_samples
        self._k = k
        self._p_norm = p_norm
        self.distance_matrix = None

    def tensor_fft(self, inputs: torch.Tensor):
        """
        inputs: N x D where N is number of samples and D is embedding dimension
        """
        N = inputs.shape[0]
        perm = torch.randperm(N)
        inputs = inputs[perm]
        centroids = inputs[: self._k]
        while len(centroids) < self._num_samples:
            diff = inputs.unsqueeze(1) - centroids.unsqueeze(
                0
            )  # create a third dimension for both tensors to perform broadcasting
            distances = torch.norm(diff, p=self._p_norm, dim=2)
            max_distances, _max_indices = torch.max(distances, dim=1)
            furthest_vector_index = torch.argmax(max_distances)
            furthest_vector = inputs[furthest_vector_index]
            if furthest_vector in centroids:
                break
            centroids = torch.cat([centroids, furthest_vector.unsqueeze(0)], 0)
        return centroids

    def str_fft(self, inputs: list[str]):
        perm = torch.randperm(len(inputs))
        inputs = [inputs[i] for i in perm]
        centroids = [inputs[i] for i in range(self._k)]
        while len(centroids) < self._num_samples:
            dist = [min(self._levenshtein(str1, str2) for str2 in centroids) for str1 in inputs]
            farthest = dist.index(max(dist))
            if inputs[farthest] in centroids:
                break
            centroids.append(inputs[farthest])
        return centroids

    @staticmethod
    def _levenshtein(s1, s2):
        if len(s1) < len(s2):
            return FarthestFirstTraversal._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
