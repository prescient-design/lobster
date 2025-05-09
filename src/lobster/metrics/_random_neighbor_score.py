from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class RandomNeighborScore(Metric):
    """
    Random Neighbor Score (RNS) Metric

    RNS quantifies the fraction of non-biological (randomly shuffled) sequence
    embeddings among the k-nearest neighbours of a protein embedding.
    Higher RNS → greater representational uncertainty; lower is better.

    Parameters
    ----------
    biological_embeddings : Tensor
        Embeddings of the protein sequences being evaluated.
        Shape: (n_bio, embedding_dim)
    random_embeddings : Tensor
        Embeddings of residue-shuffled, non-biological sequences
        Shape: (n_rand, embedding_dim)
        If n_rand > n_bio, the random embeddings are randomly sampled
        to match the size of the biological embeddings or vice versa.
    k : int, optional
        Number of nearest neighbours to inspect. The study tested k=1-2000,
        observed the strongest, most stable correlations for 200≤k≤1000,
        and used k=500 in most analyses (for sets of 2k-15k sequences).
        Default is 500.
    distance_metric : {'cosine', 'euclidean'}, optional
        Distance metric for neighbour search. The paper reported similar trends
        for both cosine and Euclidean distances. Default is 'cosine'.

    Reference
    ---------
    Prabakaran R, Bromberg Y. “Quantifying uncertainty in Protein Representations
    Across Models and Task.” bioRxiv (2025) doi:10.1101/2025.04.30.651545
    """

    is_differentiable: bool = False
    higher_is_better: bool = False  # Lower RNS values are better
    full_state_update: bool = False

    # TODO @zadorozk: Add support for pre-built sets of sequences if embedding function
    # provided
    def __init__(
        self,
        biological_embeddings: Tensor,
        random_embeddings: Tensor,
        k: int = 500,
        distance_metric: Literal["cosine", "euclidean"] = "cosine",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.distance_metric = distance_metric

        biological_embeddings, random_embeddings = self._balance_reference_sets(
            biological_embeddings, random_embeddings
        )

        self.register_buffer("biological_embeddings", biological_embeddings)
        self.register_buffer("random_embeddings", random_embeddings)

        self.add_state(
            "query_embeddings",
            default=[],
            dist_reduce_fx="cat",
        )

        if k >= self.biological_embeddings.shape[0] + self.random_embeddings.shape[0]:
            raise ValueError("k must be smaller than the total number of reference embeddings.")

    def _balance_reference_sets(self, embeddings1: Tensor, embeddings2: Tensor) -> tuple[Tensor, Tensor]:
        """
        Balance reference sets to have equal sizes.
        """
        n1 = embeddings1.shape[0]
        n2 = embeddings2.shape[0]

        # Sample from the larger set to match the smaller one
        if n1 > n2:
            indices = torch.randperm(n1, device=embeddings1.device)[:n2]
            embeddings1 = embeddings1[indices]
        elif n2 > n1:
            indices = torch.randperm(n2, device=embeddings2.device)[:n1]
            embeddings2 = embeddings2[indices]

        return embeddings1, embeddings2

    def update(self, query_embeddings: Tensor) -> None:
        """
        Update state with new query embeddings.
        """
        self.query_embeddings.append(query_embeddings)

    def _compute_distances(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        """
        Compute distances between two sets of embeddings.
        """
        if self.distance_metric == "cosine":
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=1, eps=1e-8)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=1, eps=1e-8)

            similarity = torch.mm(embeddings1_norm, embeddings2_norm.t())
            similarity = torch.clamp(similarity, -1.0, 1.0)
            return 1 - similarity
        else:
            return torch.cdist(embeddings1, embeddings2, p=2)

    def compute(self) -> Tensor:
        """
        Compute RNS for all query embeddings.
        """
        if not self.query_embeddings:
            raise ValueError("No query embeddings have been added to the metric state")

        query_embeddings = torch.cat(self.query_embeddings, dim=0)

        # Compute distances to biological and random embeddings separately
        bio_distances = self._compute_distances(query_embeddings, self.biological_embeddings)
        random_distances = self._compute_distances(query_embeddings, self.random_embeddings)

        # Combine all distances
        all_distances = torch.cat([bio_distances, random_distances], dim=1)
        n_bio = self.biological_embeddings.shape[0]

        # Find k nearest neighbors
        effective_k = min(self.k, n_bio + self.random_embeddings.shape[0])
        _, indices = torch.topk(all_distances, k=effective_k, dim=1, largest=False)

        # Count random neighbors (indices >= n_bio)
        random_mask = indices >= n_bio
        random_counts = torch.sum(random_mask, dim=1).float()

        # Compute RNS
        rns = random_counts / effective_k

        # Reset state
        self.query_embeddings.clear()

        return torch.mean(rns)
