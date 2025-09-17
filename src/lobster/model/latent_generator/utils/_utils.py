import torch


def batch_align_on_calpha(x, y):
    aligned_x = []
    for i, xi in enumerate(x):
        xi_calpha = xi[:, 1, :]
        _, (R, t) = kabsch_align(xi_calpha, y[i, :, 1, :])
        xi_ctr = xi - xi_calpha.mean(0, keepdim=True)
        xi_aligned = xi_ctr @ R.t() + t
        aligned_x.append(xi_aligned)
    return torch.stack(aligned_x)


def kabsch_align(p, q):
    if len(p.shape) > 2:
        p = p.reshape(-1, 3)
    if len(q.shape) > 2:
        q = q.reshape(-1, 3)
    p_ctr = p - p.mean(0, keepdim=True)
    t = q.mean(0, keepdim=True)
    q_ctr = q - t
    H = p_ctr.t() @ q_ctr
    U, S, V = torch.svd(H)
    R = V @ U.t()
    I_ = torch.eye(3).to(p)
    I_[-1, -1] = R.det().sign()
    R = V @ I_ @ U.t()
    p_aligned = p_ctr @ R.t() + t
    return p_aligned, (R, t)


def kabsch_torch_batched_old(P, Q, mask):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  #

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # RMSD
    # rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / P.shape[1])

    # apply R and t to P
    P_aligned = torch.matmul(p, R.transpose(1, 2)) + centroid_Q
    return P_aligned


def kabsch_torch_batched(P, Q, mask, return_transform=False):
    """Compute the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner, considering only the masked positions.

    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :param mask: A BxN matrix indicating which positions to consider (1 for include, 0 for exclude)
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the aligned points.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    assert mask.shape == P.shape[:2], "Mask dimensions must match the first two dimensions of P and Q"
    # turn to full precision
    P = P.to(torch.float32)
    Q = Q.to(torch.float32)

    # Expand mask to match the dimensions of P and Q
    mask_expanded = mask.unsqueeze(-1)  # BxNx1

    # Compute weighted centroids
    centroid_P = torch.sum(P * mask_expanded, dim=1, keepdims=True) / torch.sum(
        mask_expanded, dim=1, keepdims=True
    )  # Bx1x3
    centroid_Q = torch.sum(Q * mask_expanded, dim=1, keepdims=True) / torch.sum(
        mask_expanded, dim=1, keepdims=True
    )  # Bx1x3

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Apply mask to centered points
    p = p * mask_expanded
    q = q * mask_expanded

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # Apply R and t to P
    P_aligned = torch.matmul(p, R.transpose(1, 2)) + centroid_Q

    if return_transform:
        return P_aligned, (R, t)
    else:
        return P_aligned


def random_continuous_crops_with_mask(tokens, crop_length_range, mask):
    """Extract random continuous crops of length sampled from `crop_length_range` along the L dimension and update the mask.

    :param tokens: torch tensor of shape (B, L, n_tokens) representing the input tokens.
    :param crop_length_range: Tuple (min_crop_length, max_crop_length) representing the range of crop lengths.
    :param mask: torch tensor of shape (B, L) representing the existing mask with padding information.
    :return: tuple of (cropped_tokens, indices of crops, updated mask)
             cropped tokens: torch tensor of shape (B, crop_length, n_tokens) representing the cropped tokens.
             indices of crops: torch tensor of shape (B, 2) where each row is (start_idx, end_idx)
             updated mask: torch tensor of shape (B, L) where masked out indices are set to 0 and others to 1
    """
    if mask is None:
        return tokens, None, None

    B, L = mask.shape
    min_crop_length, max_crop_length = crop_length_range
    assert min_crop_length <= max_crop_length <= L, "Crop length range must be within the length of the L dimension"

    device = mask.device
    crop_indices = torch.zeros((B, 2), dtype=torch.int, device=device)
    cropped_tokens_list = []
    updated_mask = torch.zeros_like(mask)

    for b in range(B):
        crop_length = torch.randint(min_crop_length, max_crop_length + 1, (1,)).item()
        valid_indices = torch.where(mask[b] == 1)[0]
        if len(valid_indices) < crop_length:
            # Use as many valid indices as possible and pad the rest
            start_idx = valid_indices[0].item() if len(valid_indices) > 0 else 0
            crop_length = len(
                valid_indices
            )  # note if you have the case where non padded regions has zeros, then we will count them as valid indices
        else:
            start_idx = valid_indices[: -(crop_length - 1)].tolist()
            start_idx = start_idx[torch.randint(len(start_idx), (1,)).item()]
        end_idx = start_idx + crop_length
        cropped_tokens = torch.zeros((crop_length, tokens.shape[-1]), device=device)
        cropped_tokens = tokens[b, start_idx:end_idx]
        cropped_tokens_list.append(cropped_tokens)
        crop_indices[b] = torch.tensor([start_idx, end_idx], device=device)
        updated_mask[b, start_idx:end_idx] = 1

    max_crop_length = max(cropped_tokens.shape[0] for cropped_tokens in cropped_tokens_list)
    final_cropped_tokens = torch.zeros((B, max_crop_length, tokens.shape[-1]), device=device)
    for b in range(B):
        crop_length = cropped_tokens_list[b].shape[0]
        final_cropped_tokens[b, :crop_length] = cropped_tokens_list[b]

    updated_mask = updated_mask * mask

    return final_cropped_tokens, crop_indices, updated_mask


def extract_cropped_coordinates(atom_coords, updated_mask):
    """Extract cropped coordinates from atom_coords based on the updated_mask.

    :param atom_coords: torch tensor of shape (B, L, n_atoms, 3) representing atom coordinates.
    :param updated_mask: torch tensor of shape (B, L) representing the updated mask with cropping information.
    :return: tuple of (cropped coordinates, new mask)
             cropped coordinates: torch tensor of shape (B, crop_length, n_atoms, 3)
             new mask: torch tensor of shape (B, crop_length) representing the new mask for the cropped coordinates.
    """
    B, L, n_atoms, _ = atom_coords.shape
    cropped_coords_list = []
    new_mask_list = []

    for b in range(B):
        valid_indices = torch.where(updated_mask[b] == 1)[0]
        crop_length = valid_indices[-1] - valid_indices[0] + 1
        cropped_coords = atom_coords[b, valid_indices[0] : valid_indices[-1] + 1]
        cropped_coords_list.append(cropped_coords)
        c_mask = (
            torch.ones(crop_length, dtype=torch.int, device=atom_coords.device)
            * updated_mask[b, valid_indices[0] : valid_indices[-1] + 1]
        )
        new_mask_list.append(c_mask)

    # Determine the maximum crop length to pad the cropped coordinates
    max_crop_length = max(cropped_coords.shape[0] for cropped_coords in cropped_coords_list)

    # Initialize the final cropped coordinates array with padding
    final_cropped_coords = torch.zeros((B, max_crop_length, n_atoms, 3), device=atom_coords.device)
    final_new_mask = torch.zeros((B, max_crop_length), dtype=torch.int, device=atom_coords.device)

    for b in range(B):
        crop_length = cropped_coords_list[b].shape[0]
        final_cropped_coords[b, :crop_length] = cropped_coords_list[b]
        final_new_mask[b, :crop_length] = new_mask_list[b]

    return final_cropped_coords, final_new_mask
