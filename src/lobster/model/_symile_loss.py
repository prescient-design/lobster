import torch
import torch.nn.functional as F


class SymileLoss:
    def __init__(self, negative_sampling: str = "n"):
        """
        Initialize the Symile loss function.

        Args:
            negative_sampling (str, optional): Specifies the negative sampling strategy.
                                               Must be either 'n' (for O(n)) or 'n_squared' (for O(n^2)).
                                               Defaults to 'n'.
        """
        self.negative_sampling = negative_sampling

    def compute_logits_n(self, anchor_rep, non_anchor_reps):
        """
        Computes the logits for anchor modality anchor_rep with bsz - 1 negatives for
        each positive - or (bsz^2 - bsz) total negatives. Returned logits have size
        (bsz, bsz) with bsz positive multilinear inner products (MIPs) and (bsz^2 - bsz)
        negative MIPs. Positive MIPs are along the diagonal of the square logits matrix.

        For example, given anchor_rep x and non_anchor_reps y and z, the second row of
        `logits` might be:

        [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[0], z[1]) MIP(x[1], y[2], z[3]) ].

        Notice that only the second element is the positive MIP; all others are negative.
        There is a small chance of a false negative MIP.

        Args:
            anchor_rep (torch.Tensor): Representation vector for anchor modality (bsz, d).
            non_anchor_reps (list[torch.Tensor]): List of representation tensors for non-anchor
                                                  modalities, each of size (bsz, d). This list
                                                  can contain any number of tensors.
        Returns:
            logits (torch.Tensor): Logits for anchor_rep of size (bsz, bsz).
        """
        # shuffle rows of each tensor in non_anchor_reps and element-wise multiply
        non_anchor_shuff = torch.ones_like(anchor_rep)
        for r in non_anchor_reps:
            # cannot use inplace operations like *= because of autograd
            non_anchor_shuff = non_anchor_shuff * r[torch.randperm(r.shape[0])]

        logits = anchor_rep @ torch.t(non_anchor_shuff)  # (bsz, bsz)

        MIP_of_positive_samples = anchor_rep.clone()
        for r in non_anchor_reps:
            # cannot use inplace operations like *= because of autograd
            MIP_of_positive_samples = MIP_of_positive_samples * r
        MIP_of_positive_samples = MIP_of_positive_samples.sum(axis=1)  # (bsz)

        # insert positive samples along diagonal of shuffled logits
        return torch.where(
            torch.eye(n=anchor_rep.shape[0]).to(anchor_rep.device) > 0.5, MIP_of_positive_samples, logits
        )

    def compute_non_anchor_products(self, tensors):
        """
        Recursively generates all possible element-wise products by shifting the rows
        of each tensor in the `tensors` list and computing products across tensors.

        Args:
            tensors (list[torch.Tensor]): List of tensors with size (bsz, d).

        Returns:
            all_products (list[torch.Tensor]): List of all possible element-wise products of the input
                                               tensors with shifted rows. The length of the list is
                                               bsz^len(tensors).
        """
        # base case
        if len(tensors) == 2:
            y, z = tensors
            y_z = []
            for i in range(y.shape[0]):
                y_z.append(y * z)
                z = torch.roll(z, shifts=1, dims=0)
            return y_z

        x = tensors[0]

        partial_products = self.compute_non_anchor_products(tensors[1:])

        all_products = []
        for i in range(x.shape[0]):
            for partial_product in partial_products:
                all_products.append(partial_product * x)
            x = torch.roll(x, shifts=1, dims=0)

        return all_products

    def compute_logits_n_squared(self, anchor_rep, non_anchor_reps):
        """
        Computes the logits for anchor modality anchor_rep with (bsz^len(non_anchor_reps)) - 1
        negatives for each positive. Returned logits have size (bsz, bsz^len(non_anchor_reps))
        with bsz positive multilinear inner products (MIPs) and (bsz^(len(non_anchor_reps)+1) - bsz)
        negative MIPs. Positive MIPs are along the main diagonal of the (non-square) logits matrix.

        For example, given anchor_rep x and non_anchor_reps y and z, and bsz = 4,
        then the second row of `logits` is:

        [ MIP(x[1], y[0], z[0]) MIP(x[1], y[1], z[1]) MIP(x[1], y[2], z[2]) MIP(x[1], y[3], z[3])
          MIP(x[1], y[0], z[3]) MIP(x[1], y[1], z[0]) MIP(x[1], y[2], z[1]) MIP(x[1], y[3], z[2])
          MIP(x[1], y[0], z[2]) MIP(x[1], y[1], z[3]) MIP(x[1], y[2], z[0]) MIP(x[1], y[3], z[1])
          MIP(x[1], y[0], z[1]) MIP(x[1], y[1], z[2]) MIP(x[1], y[2], z[3]) MIP(x[1], y[3], z[0])  ]

        Notice that only the second element is the positive MIP; all others are negative.

        Args:
            anchor_rep (torch.Tensor): Representation vector for anchor modality (bsz, d).
            non_anchor_reps (list[torch.Tensor]): List of representation tensors for non-anchor
                                                  modalities, each of size (bsz, d). This list
                                                  can contain any number of tensors.
        Returns:
            logits (torch.Tensor): Logits for anchor_rep of size (bsz, bsz^len(non_anchor_reps)).
        """
        non_anchor_products = self.compute_non_anchor_products(non_anchor_reps)

        non_anchor_product = torch.cat(non_anchor_products, 0)

        logits = anchor_rep @ non_anchor_product.T

        return logits

    def forward(self, representations, logit_scale):
        """
        Computes the Symile loss for a batch of representation vectors.

        Args:
            representations (list[torch.Tensor]): List of representation vectors, each of size (bsz, d).
        Returns:
            (torch.Tensor): Symile loss, which is an average over the losses where each modality is
                            treated as the anchor in turn.
        """
        labels = torch.arange(representations[0].shape[0]).to(representations[0].device)
        losses = []

        for i, r in enumerate(representations):
            if self.negative_sampling == "n":
                logits = logit_scale * self.compute_logits_n(
                    r, [rep for j, rep in enumerate(representations) if i != j]
                )
            elif self.negative_sampling == "n_squared":
                logits = logit_scale * self.compute_logits_n_squared(
                    r, [rep for j, rep in enumerate(representations) if i != j]
                )
            else:
                raise ValueError("Invalid value for negative_sampling. Expected 'n' or 'n_squared'.")

            loss = F.cross_entropy(logits, labels)
            losses.append(loss)

        return sum(losses) / len(losses)

    def __call__(self, representations, logit_scale):
        return self.forward(representations, logit_scale)
