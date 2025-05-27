import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from lobster.metrics import RandomNeighborScore


class TestRandomNeighborScore:
    def test__init__(self):
        bio_embeddings = torch.randn(100, 10)
        random_embeddings = torch.randn(100, 10)

        metric = RandomNeighborScore(bio_embeddings, random_embeddings, k=10)

        assert metric.k == 10
        assert metric.distance_metric == "cosine"
        assert metric.biological_embeddings.shape == bio_embeddings.shape
        assert metric.random_embeddings.shape == random_embeddings.shape

    @pytest.mark.parametrize(
        "distance_metric, bio_embeddings, random_embeddings, query_embeddings, expected_result",
        [
            pytest.param(
                "cosine",
                torch.ones(500, 128),
                torch.ones(500, 128) * -1,  # Opposite direction of bio
                torch.ones(100, 128),  # Same as bio
                0.0,
                id="cosine_close_to_bio",
            ),
            pytest.param(
                "cosine",
                torch.ones(500, 128),
                torch.ones(500, 128) * -1,  # Opposite direction of bio
                torch.ones(100, 128) * -1,  # Same as random
                1.0,
                id="cosine_close_to_random",
            ),
            pytest.param(
                "euclidean",
                torch.ones(500, 128),
                torch.zeros(500, 128),
                torch.ones(100, 128),  # Same as bio
                0.0,
                id="euclidean_close_to_bio",
            ),
            pytest.param(
                "euclidean",
                torch.ones(500, 128),
                torch.zeros(500, 128),
                torch.zeros(100, 128),  # Same as random
                1.0,
                id="euclidean_close_to_random",
            ),
        ],
    )
    def test_compute(self, distance_metric, bio_embeddings, random_embeddings, query_embeddings, expected_result):
        torch.manual_seed(42)

        metric = RandomNeighborScore(bio_embeddings, random_embeddings, k=10, distance_metric=distance_metric)

        metric.update(query_embeddings)

        rns = metric.compute()

        assert isinstance(rns, Tensor)
        assert_close(rns, torch.tensor(expected_result), rtol=1e-4, atol=1e-4)
