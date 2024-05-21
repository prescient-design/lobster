import torch
from lobster.data import FarthestFirstTraversal


class TestFarthestFirstTraversal:
    def test_fft(self):
        input_vectors = torch.rand(100, 2)
        k = 1
        num_samples = 3
        random_seed = 42
        p_norm = 2  # set 1 for Manhattan, 2 for Euclidean, etc.

        fft = FarthestFirstTraversal(num_samples, k, random_seed, p_norm)
        centroids = fft.tensor_fft(input_vectors)

        assert centroids.shape[0] == 2

        sequences = ["ALA", "GLYCU", "ARGPHE", "HIPRO", "LEUTRP"]

        centroids = fft.str_fft(sequences)

        assert len(centroids) == 3
