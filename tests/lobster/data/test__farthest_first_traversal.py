import numpy as np
import torch
from lobster.data import FarthestFirstTraversal, ranked_fft


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestFarthestFirstTraversal:
    def test_fft(self):
        set_random_seeds()
        input_vectors = torch.rand(100, 2)
        k = 1
        num_samples = 3
        random_seed = 42
        p_norm = 2  # set 1 for Manhattan, 2 for Euclidean, etc.

        fft = FarthestFirstTraversal(num_samples, k, random_seed, p_norm)
        centroids = fft.tensor_fft(input_vectors)

        assert centroids.shape[0] == 3

        sequences = ["ALA", "GLYCU", "ARGPHE", "HIPRO", "LEUTRP"]

        centroids = fft.str_fft(sequences)

        assert len(centroids) == 3


def test_ranked_fft():
    library = np.array(
        [
            "AAAA",
            "GGGG",
            "CCCC",
            "TTTT",
        ]
    )
    ranking_scores = np.array([3, 2, 1, 4])
    n = 2
    selected = ranked_fft(library, ranking_scores, n, descending=True)

    assert np.all(selected == np.array([3, 0]))

    selected = ranked_fft(library, ranking_scores, n, descending=False)
    assert np.all(selected == np.array([2, 1]))

    ranking_scores = None
    selected = ranked_fft(library, ranking_scores, n)
    assert np.all(selected == np.array([0, 1]))

    # harder example with wider spread of distances
    library = np.array(
        [
            "AAAA",
            "GGGG",
            "CCCC",
            "TTTT",
            "ACGT",
            "TGCA",
            "ACGT",
            "TGCA",
        ]
    )
    ranking_scores = np.array(list(range(8)))
    n = 3
    selected = ranked_fft(library, ranking_scores, n, descending=True)
    assert np.all(selected == np.array([7, 6, 3]))

    selected = ranked_fft(library, ranking_scores, n, descending=False)
    assert np.all(selected == np.array([0, 1, 2]))
