import torch
import time
from icecream import ic

def get_pairwise_distances(points):
    """
    Calculate pairwise distances between all points.
    Args:
        points: tensor of shape (B, L, 3)
    Returns:
        distances: tensor of shape (B, L, L)
    """
    diff = points.unsqueeze(2) - points.unsqueeze(1)  # (B, L, L, 3)
    distances = torch.norm(diff, dim=-1)  # (B, L, L)
    return distances

def get_neighbor_mask(distances, radius=30.0):
    """
    Create mask for points within radius.
    Args:
        distances: tensor of shape (B, L, L)
        radius: float, cutoff distance
    Returns:
        mask: boolean tensor of shape (B, L, L)
    """
    return distances < radius

def compute_covariance_matrices(points, neighbor_mask):
    """
    Compute local covariance matrices for each point.
    Args:
        points: tensor of shape (B, L, 3)
        neighbor_mask: boolean tensor of shape (B, L, L)
    Returns:
        covariance: tensor of shape (B, L, 3, 3)
        means: tensor of shape (B, L, 3)
    """
    B, L, _ = points.shape

    # Get neighborhood points
    neighbor_points = points.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, 3)

    # Apply mask and handle empty neighborhoods
    valid_neighbors = neighbor_mask.unsqueeze(-1)  # (B, L, L, 1)
    neighbor_count = valid_neighbors.sum(dim=2)  # (B, L, 1)
    valid_count = torch.clamp(neighbor_count, min=1.0)

    # Calculate mean of valid neighbors
    means = (neighbor_points * valid_neighbors).sum(dim=2) / valid_count  # (B, L, 3)

    # Center the points and compute covariance
    centered_points = neighbor_points - means.unsqueeze(2)  # (B, L, L, 3)
    masked_centered = centered_points * valid_neighbors

    # Compute covariance matrices using outer products
    covariance = torch.einsum('bijk,bijl->bikl', masked_centered, masked_centered)

    # Divide by (n-1) for unbiased estimate
    divisor = torch.clamp(neighbor_count - 1, min=1.0)
    covariance = covariance / divisor.unsqueeze(-1)

    return covariance, means, masked_centered

def enforce_consistent_orientation(lrf, points, means, neighbor_mask, mask=None):
    """
    Orient the local reference frame consistently based on point distribution.
    Args:
        lrf: tensor of shape (B, L, 3, 3) containing eigenvectors for lrfs of each point
        points: tensor of shape (B, L, 3) points
        means: tensor of shape (B, L, 3) local means
        neighbor_mask: tensor of shape (B, L, L) for neighbors
        mask: tensor of shape (B, L) for ignoring padding
    Returns:
        oriented_lrf: tensor of shape (B, 3, 3) with consistent orientation
    """
    if mask is None:
        mask = torch.ones(points.shape[0], points.shape[1], device=points.device)
    oriented_lrf = lrf.clone()
    #get local environment per point as indicated by the neighbor mask
    local_env = points.unsqueeze(1).expand(-1, points.shape[1], -1, -1)  # (B, L, L, 3)
    local_env = local_env * neighbor_mask.unsqueeze(-1)  # Apply mask to local environment

    #center locol environmen with respective position means
    local_env = local_env - means.unsqueeze(2)  # Center the points
    local_env = local_env * neighbor_mask.unsqueeze(-1)  # Apply mask to local environment
    local_env = local_env * mask.unsqueeze(-1).unsqueeze(-1)

    # For each batch
    for b in range(oriented_lrf.shape[0]):
        for l in range(oriented_lrf.shape[1]):
            # Get the signs of projections of points onto each principal axis
            frame = oriented_lrf[b,l]  # (3, 3)
            centered_points = local_env[b, l]  # (L, 3)

            # For each axis, compute the skewness of point distributions
            # If skewness is negative, flip the axis
            for i in range(3):
                # Project points onto current axis
                projections = torch.matmul(centered_points, frame[:, i])

                # Compute third moment (skewness)
                valid_points = neighbor_mask[b, l].nonzero(as_tuple=True)[0]

                skewness = torch.mean(projections[valid_points] ** 3)

                # If skewness is negative, flip this axis
                if skewness < 0:
                    frame[:, i] = -frame[:, i]

            # Ensure right-handedness
            cross_prod = torch.cross(frame[:, 0], frame[:, 1], dim = -1)
            if torch.dot(cross_prod, frame[:, 2]) < 0:
                frame[:, 2] = -frame[:, 2]

            oriented_lrf[b,l] = frame

    return oriented_lrf

def generate_rotations_by_sign_flips(rot_matrices):
    """
    Generate all possible proper rotation matrices by applying sign flips to eigenvectors,
    handling an additional batch dimension.

    Parameters:
        rot_matrices (torch.Tensor): Tensor of shape (B, L, 3, 3), batch of rotation matrices.

    Returns:
        torch.Tensor: Tensor of shape (B, L, 4, 3, 3), batch of all possible proper rotation matrices.
    """
    # rot_matrices: tensor of shape (B, L, 3, 3)
    device = rot_matrices.device
    B, L, _, _ = rot_matrices.shape

    # Define sign flip combinations that result in proper rotations (even number of sign flips)
    sign_flips = torch.tensor([
        [1.,  1.,  1.],    # No flips
        [-1., -1.,  1.],   # Flips in axes 0 and 1
        [-1.,  1., -1.],   # Flips in axes 0 and 2
        [1., -1., -1.],    # Flips in axes 1 and 2
    ], device=device)

    num_flips = sign_flips.shape[0]  # 4

    # Create diagonal sign flip matrices
    D = torch.stack([torch.diag(flip) for flip in sign_flips])  # (4, 3, 3)

    # Expand dimensions for batch matrix multiplication
    rot_matrices = rot_matrices.unsqueeze(2)  # (B, L, 1, 3, 3)
    D = D.unsqueeze(0).unsqueeze(0)           # (1, 1, 4, 3, 3)

    # Apply sign flips to the rotation matrices
    # The result has shape (B, L, 4, 3, 3)
    rotations = torch.matmul(rot_matrices, D)

    return rotations  # Tensor of shape (B, L, 4, 3, 3)

def apply_stochastic_fa(lrf, give_all=False):
    """
    Apply stochastic frame averaging.
    Args:
        lrf: tensor of shape (B, L, 3, 3) containing eigenvectors for lrfs of each point
    Returns:
        lrf: tensor of shape (B, L, 3, 3) with stochastic frame selected from all possible SE(3) transformations

    """
    lrf_expanded = generate_rotations_by_sign_flips(lrf)  # (B, L, 4, 3, 3)
    if give_all:
        return lrf_expanded
    #compute random number between lrf_expanded.shape[2] for each point
    random_indices = torch.randint(0, lrf_expanded.shape[2], (lrf_expanded.shape[0], lrf_expanded.shape[1]), device=lrf.device)  # (B, L)
    # Select the random frames
    selected_frames = lrf_expanded[torch.arange(lrf_expanded.shape[0]).unsqueeze(1), torch.arange(lrf_expanded.shape[1]).unsqueeze(0), random_indices]  # (B, L, 3, 3)
    return selected_frames

def get_non_zero_neighbors_fast(lrf_points_nb, neighbor_mask, max_neighbors):
    """
    Optimized version to get all non-zero neighbors using vectorized operations.

    Args:
        lrf_points_nb (torch.Tensor): Tensor of shape (B, L, L, 3)
        neighbor_mask (torch.Tensor): mask of shape (B, L, L)
        max_neighbors (int): Maximum number of neighbors to consider

    Returns:
        torch.Tensor: A tensor of shape (B, L, max_neighbors, 3)
    """
    device = lrf_points_nb.device
    B, L, _, _ = lrf_points_nb.shape

    # Convert mask to boolean and get number of neighbors
    neighbor_mask = neighbor_mask.bool()
    num_neighbors = neighbor_mask.sum(dim=-1)

    # Create indices for sorting
    mask_indices = torch.arange(L, device=device).expand(B, L, L)

    # Set indices where mask is False to a large number (for sorting)
    mask_indices = torch.where(neighbor_mask, mask_indices,
                             torch.tensor(L, device=device))

    # Sort indices based on mask
    _, sorted_indices = mask_indices.sort(dim=-1)

    # Create batch and point indices for gathering
    batch_idx = torch.arange(B, device=device).view(-1, 1, 1).expand(-1, L, L)
    point_idx = torch.arange(L, device=device).view(1, -1, 1).expand(B, -1, L)

    # Gather valid neighbors in sorted order
    gathered_points = lrf_points_nb[batch_idx, point_idx, sorted_indices]

    # Take only up to max_neighbors
    output = gathered_points[:, :, :max_neighbors]

    # Create final mask for zeroing out invalid neighbors
    final_mask = torch.arange(max_neighbors, device=device).expand(B, L, -1)
    final_mask = final_mask < num_neighbors.unsqueeze(-1)

    # Zero out invalid neighbors
    output = output * final_mask.unsqueeze(-1)

    return output


def compute_geometric_features(points, radius=30.0, max_neighbors=30, stochastic_fa= True, mask = None, mask_probability = 0.8, give_all_frames=False):
    """
    Compute geometric features and consistently oriented LRF.
    Args:
        points: tensor of shape (B, L, 3) or (B, L, x, 3)
        radius: float, cutoff distance for neighbors
    Returns:
        dict containing geometric features and oriented LRF
    """
    B, L = points.shape[:2]
    transform_full = False
    if points.dim() == 4:
        points_full = points.clone()
        points = points[:,:, 1, :]  # (B, L, 3)#get CA coordinates
        transform_full = True
    if mask is None:
        mask = torch.ones(points.shape[0], points.shape[1], device=points.device)

    # Get pairwise distances and neighbor mask
    distances = get_pairwise_distances(points)
    neighbor_mask = get_neighbor_mask(distances, radius)
    neighbor_mask = neighbor_mask * mask.unsqueeze(-1) * mask.unsqueeze(-2)

    # Compute covariance matrices
    #start_time = time.time()
    covariance, local_means, masked_centered = compute_covariance_matrices(points, neighbor_mask)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)


    # Sort in descending order
    eigenvalues = eigenvalues.flip(-1)  # Now λ1 ≥ λ2 ≥ λ3
    eigenvectors = eigenvectors.flip(-1)  # Corresponding eigenvectors

    # Compute geometric features
    lambda1 = torch.clamp(eigenvalues[..., 0], min=1e-10)
    lambda2 = eigenvalues[..., 1]
    lambda3 = eigenvalues[..., 2]

    anisotropy = (lambda1 - lambda3) / lambda1
    planarity = (lambda2 - lambda3) / lambda1
    omnivariance = torch.pow(lambda1 * lambda2 * lambda3, 1/3)

    # Build consistently oriented Local Reference Frame
    lrf = eigenvectors # B, L, 3, 3
    #ensure right-handedness note shape is (B, L, 3, 3)
    det = torch.linalg.det(lrf)
    #correct for negative determinants
    neg_det_count = torch.sum(det < 0)
    if neg_det_count > 0:
        #print(f"Warning: {neg_det_count} frames have negative determinant")
        lrf = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, -lrf, lrf)
    # Transform points to local coordinate system
    centered_points = points - local_means
    # Orient the basis vectors consistently
    #start_time = time.time()
    if stochastic_fa:
        lrf = apply_stochastic_fa(lrf, give_all=give_all_frames)
    else:
        lrf = enforce_consistent_orientation(lrf, points, local_means, neighbor_mask, mask=mask)
    #end_time = time.time()
    #elapsed_time_or = end_time - start_time
    #ic(elapsed_time_or)
    if not transform_full:
        if give_all_frames:
            lrf_points = torch.einsum('blfdc,blc->blfd', lrf.transpose(-1, -2), centered_points)
            lrf_points = lrf_points * mask.unsqueeze(-1).unsqueeze(-1)  # Apply mask to lrf_points
        else:
            lrf_points = torch.einsum('bldc,blc->bld', lrf.transpose(-1, -2), centered_points)
            lrf_points = lrf_points * mask.unsqueeze(-1)  # Apply mask to lrf_points
    else:
        # Transform points to local coordinate system
        #centered_points = points_full - local_means.unsqueeze(2)
        lrf_points = []
        if give_all_frames:
            for ed in range(points_full.shape[2]):
                centered_points = points_full[:,:,ed,:] - local_means
                lrf_points_ = torch.einsum('blfdc,blc->blfd', lrf.transpose(-1, -2), centered_points)
                #print(lrf_points_.shape)
                lrf_points.append(lrf_points_)
            lrf_points = torch.stack(lrf_points, dim=2)
            lrf_points = lrf_points * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Apply mask to lrf_points
        else:
            for ed in range(points_full.shape[2]):
                centered_points = points_full[:,:,ed,:] - local_means
                lrf_points_ = torch.einsum('bldc,blc->bld', lrf.transpose(-1, -2), centered_points)
                #print(lrf_points_.shape)
                lrf_points.append(lrf_points_)
            lrf_points = torch.stack(lrf_points, dim=2)
            #print(lrf_points.shape)
            lrf_points = lrf_points * mask.unsqueeze(-1).unsqueeze(-1)  # Apply mask to lrf_points

    if give_all_frames:
        lrf_points_nb = torch.einsum('blfdc,blkc->blkfd', lrf.transpose(-1, -2), masked_centered)
        lrf_points_nb = lrf_points_nb * neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        lrf_points_nb_list = []
        for frame in range(lrf_points_nb.shape[3]):
            lrf_points_nb_list.append(get_non_zero_neighbors_fast(lrf_points_nb[:,:,:,frame,:], neighbor_mask, max_neighbors))
        lrf_points_nb_ = torch.stack(lrf_points_nb_list, dim=3)
    else:
        lrf_points_nb = torch.einsum('bldc,blkc->blkd', lrf.transpose(-1, -2), masked_centered)
        lrf_points_nb = lrf_points_nb * neighbor_mask.unsqueeze(-1)  # Apply mask to lrf_points
        lrf_points_nb_ = get_non_zero_neighbors_fast(lrf_points_nb, neighbor_mask, max_neighbors)

    #get grf from lrf for first point
    if give_all_frames:
        lrf_point_0 = lrf[:, 0, :, :]
        centered_points_full = points_full - local_means[:, 0, :].unsqueeze(1).unsqueeze(1)
        centered_points_full = centered_points_full.view(B, -1, 3)
        centered_points_full = centered_points_full.unsqueeze(2)
        centered_points_full = centered_points_full.expand(-1, -1, lrf_points_nb.shape[3], -1)
        grf_points = torch.einsum('bfij,bkfj->bkfi', lrf_point_0.transpose(-1, -2), centered_points_full)
        grf_points = grf_points.reshape(B, L, -1, lrf_points_nb.shape[3], 3)
        grf_points = grf_points * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Apply mask to grf_points

    else:
        lrf_point_0 = lrf[:, 0, :, :]
        centered_points_full = points_full - local_means[:, 0, :].unsqueeze(1).unsqueeze(1)
        centered_points_full = centered_points_full.view(B, -1, 3)
        grf_points = torch.einsum('bij,bkj->bki', lrf_point_0.transpose(-1, -2), centered_points_full)
        grf_points = grf_points.view(B, L, -1, 3)
        grf_points = grf_points * mask.unsqueeze(-1).unsqueeze(-1) # Apply mask to grf_points


    #randomly mask w/ probability 0.8 for grf points to be masked, where 1 is no maske and o is masked
    mask_grf = torch.rand(grf_points.shape[0], grf_points.shape[1], device=grf_points.device) > mask_probability

    if give_all_frames:
        #ic(grf_points.shape)
        #ic(mask_grf.shape)
        grf_points_mask = grf_points * mask_grf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        grf_points_mask = grf_points * mask_grf.unsqueeze(-1).unsqueeze(-1)

    # Verify right-handedness (for debugging)
    #det = torch.linalg.det(lrf.to(torch.float32))
    #neg_det_count = torch.sum(det < 0)
    #if neg_det_count > 0:
    #    print(f"Warning: {neg_det_count} frames have negative determinant")

    return {
        'eigenvalues': eigenvalues,
        'anisotropy': anisotropy,
        'planarity': planarity,
        'omnivariance': omnivariance,
        'lrf': lrf,
        'lrf_points': lrf_points,
        'lrf_points_nb': lrf_points_nb_,
        'grf_points': grf_points,
        'grf_points_mask': grf_points_mask,
        'neighbor_mask': neighbor_mask
    }
