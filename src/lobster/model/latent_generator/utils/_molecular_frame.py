import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def calculate_centroid(positions):
    """Calculate the centroid from a list of positions."""
    return torch.mean(positions, dim=0)


def nan_to_num(vec, num=0.0):
    return torch.nan_to_num(vec, num)


def _normalize(vec, dim=-1):
    return nan_to_num(vec / (torch.norm(vec, dim=dim, keepdim=True) + 1e-8))


def create_local_coordinate_system(point1, point2):
    # Calculate the local Z-axis vector and normalize it
    z_axis = point2 - point1
    z_axis = z_axis / (torch.norm(z_axis) + 1e-8)

    # Find a vector orthogonal to Z-axis for X-axis
    # Check if Z-axis is parallel to global Z-axis
    global_z = torch.tensor([0.0, 0.0, 1.0], device=point1.device)
    global_y = torch.tensor([0.0, 1.0, 0.0], device=point1.device)

    if torch.allclose(z_axis, global_z) or torch.allclose(z_axis, -global_z):
        # Use global Y-axis if Z-axis is parallel/near to global Z-axis
        x_axis = torch.linalg.cross(z_axis, global_y)
    else:
        # Typically, use global Z-axis
        x_axis = torch.linalg.cross(z_axis, global_z)

    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)

    # Calculate Y-axis as a cross product of Z-axis and X-axis
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / (torch.norm(y_axis) + 1e-8)

    return x_axis, y_axis, z_axis


def construct_local_frame(coords, non_col_points=None, use_pca=False):
    """Construct local coordinate frame using PyTorch.

    Args:
        coords (torch.Tensor): Input coordinates of shape (N, 3)
        non_col_points (torch.Tensor, optional): Non-collinear points
        use_pca (bool, optional): Use PCA for frame construction

    Returns:
        tuple: (local_coords, invariant_spherical_coords, rotation_matrix, translation)

    """
    if len(coords.shape) == 3:
        L, n_atoms, _ = coords.shape
        # ic(L,n_atoms)
    if use_pca:
        raise NotImplementedError("PCA is not implemented")

    # Ensure tensor is on GPU if available
    device = coords.device

    # Center coordinates
    if len(coords.shape) == 3:
        coords_flat = coords.view(-1, 3)
        mean = torch.mean(coords_flat, dim=0)
        centered_coords = coords_flat - mean
        # centered_coords = coords_flat - coords[0,1] #intial CA is the origin
        if non_col_points is not None:
            centered_coords = centered_coords.view(L, n_atoms, 3)
    else:
        mean = torch.mean(coords, dim=0)
        # centered_coords = coords - coords[0]
        centered_coords = coords - mean

    # Adjust non-collinear points if provided
    if non_col_points is not None:
        non_col_points = non_col_points - coords[0, 1]  # intial CA is the origin

    num_nodes = len(centered_coords)
    if num_nodes < 3 and non_col_points is None:
        raise ValueError("At least three points are required to construct a local frame")

    # Initialize output tensors
    x, y, z = None, None, None
    local_coords = centered_coords

    # More than two points
    flag = False

    if non_col_points is not None:
        v1 = non_col_points[1] - non_col_points[0]
        v2 = non_col_points[2] - non_col_points[0]
        flag = torch.norm(torch.linalg.cross(v1, v2)) != 0
    else:
        v1 = centered_coords[1] - centered_coords[0]
        for i in range(2, num_nodes):
            v2 = centered_coords[i] - centered_coords[0]
            if torch.norm(torch.linalg.cross(v1, v2)) != 0:
                flag = True
                break

    if not flag and i == num_nodes - 1:
        # Could not find a third non-collinear point
        # local_coords = centered_coords
        raise ValueError("Could not find a third non-collinear point")
    else:
        # Build global frame
        x = _normalize(v1)
        y = _normalize(torch.linalg.cross(v1, v2))
        z = torch.linalg.cross(x, y)

        # Transform coordinates
        if len(coords.shape) == 3:
            # flatten coords
            centered_coords = centered_coords.view(-1, 3)
            local_coords = torch.matmul(centered_coords, torch.stack((x, y, z)).T)
            local_coords = local_coords.view(L, n_atoms, 3)
        else:
            local_coords = torch.matmul(centered_coords, torch.stack((x, y, z)).T)

    # Construct rotation matrix and translation
    rotation_matrix = torch.column_stack((x, y, z)) if x is not None else torch.eye(3, device=device)
    if len(coords.shape) == 3:
        # translation = coords[0,1]
        translation = mean
    else:
        # translation = coords[0]
        translation = mean

    return local_coords, rotation_matrix, translation


def get_lrf_features_old(local_frame):
    """Compute features from local reference frame.

    Args:
        local_frame (torch.Tensor): Local reference frame of shape (N, 3)
    Returns:
        torch.Tensor: Features of [eigenvalues, eigenvectors, principal_components]
    """
    # calulate covariance matrix
    cov_matrix = torch.cov(local_frame.T, correction=0)  # Use correction=0 for sample covariance

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    # Sort in descending order
    eigenvalues = eigenvalues.flip(-1)  # Now λ1 ≥ λ2 ≥ λ3
    eigenvectors = eigenvectors.flip(-1)  # Flip to get descending order

    # Concatenate features
    features = torch.cat([eigenvalues.real, eigenvectors.real.reshape(-1)])

    return features


def get_lrf_features(local_frame):
    """Compute features from local reference frame.

    Args:
        local_frame (np.ndarray): Local reference frame of shape (N, 3)
    Returns:
        np.ndarray: Features of [eigenvalues, eigenvectors, principal_components]
    """
    # Calculate covariance matrix
    cov_matrix = np.cov(local_frame.T, bias=True)  # Use bias=True for sample covariance

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    # Sort in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for sorting in descending order
    eigenvalues = eigenvalues[sorted_indices]  # Now λ1 ≥ λ2 ≥ λ3
    eigenvectors = eigenvectors[:, sorted_indices]  # Sort eigenvectors accordingly

    # Concatenate features
    features = np.concatenate([eigenvalues.real, eigenvectors.real.reshape(-1)])

    return features


def frags_to_seq(lrfs, lrf_means, grf_coords, non_col_points=None, use_pca=False, just_coord=True):
    """Convert fragments to sequence in molecular frame.

    Args:
        lrfs (torch.Tensor): Local reference frames
        lrf_means (torch.Tensor): Mean coordinates of local reference frames
        grf_coords (torch.Tensor): Global reference frame coordinates
        non_col_points (torch.Tensor, optional): Non-collinear points for each fragment
        use_pca (bool, optional): Use PCA for frame construction

    Returns:
        tuple: (coordinates in molecular frame, rotation matrix to world, translation to world)

    """
    # Construct local frame for global reference frame
    frag_centroid_local_coords, R_m2w, t_m2w = construct_local_frame(
        lrf_means, non_col_points=grf_coords, use_pca=use_pca
    )

    # Prepare output list
    coord_out_molframe_list = []
    coord_out_locframe_list = []
    lrf_features_list = []

    # Process each local reference frame
    for i, lrf in enumerate(lrfs):
        # Select non-collinear points if provided
        selected_points = non_col_points[i] if non_col_points is not None else None

        # Construct local frame for fragment
        frag_atom_local_coords, R_f2w, t_f2w = construct_local_frame(
            lrf, non_col_points=selected_points, use_pca=use_pca
        )

        # Compute rotation and translation from fragment to molecule frame
        R_f2m = torch.matmul(R_m2w.T, R_f2w)
        t_f2m = torch.matmul(R_m2w.T, t_f2w - t_m2w)

        # Transform fragment coordinates to molecule frame
        L, n_atoms, _ = frag_atom_local_coords.shape
        frag_atom_local_coords = frag_atom_local_coords.view(-1, 3)
        molecule_frame = torch.matmul(frag_atom_local_coords, R_f2m.T) + t_f2m
        molecule_frame = molecule_frame.view(L, n_atoms, 3)
        local_frame = frag_atom_local_coords
        # compute eigenvalues and eigenvectors of covariance matrix for local frame
        if not just_coord:
            lrf_features = get_lrf_features(local_frame)
            lrf_features_list.append(lrf_features)

        # Add first point of transformed coordinates
        coord_out_molframe_list.append(molecule_frame[0].unsqueeze(0))
        coord_out_locframe_list.append(local_frame)  # [1].unsqueeze(0))

    return coord_out_molframe_list, coord_out_locframe_list, lrf_features_list, R_m2w, t_m2w


def get_local_neighborhood_gpu(coords, angstrong_cutoff=10):
    """GPU-accelerated local neighborhood search using pairwise distance computation.

    Args:
        coords (torch.Tensor): Input coordinates tensor of shape (N, 3)
        angstrong_cutoff (float): Cutoff distance for neighborhood search

    Returns:
        tuple:
            - indices (list): List of neighbor indices for each point
            - coordinate_neighbors (list): List of neighbor coordinates including query point

    """
    coords_n = coords[:, 0, :]
    coords_ca = coords[:, 1, :]
    coords_c = coords[:, 2, :]
    # Compute pairwise distances
    # Shape: (N, N)
    distances = torch.cdist(coords_ca, coords_ca)

    # Create boolean mask for neighborhoods
    # Shape: (N, N)
    neighborhood_mask = distances <= angstrong_cutoff

    # Remove self-connections (diagonal)
    torch.diagonal(neighborhood_mask).fill_(False)

    # Find indices of neighbors for each point
    indices = [torch.where(neighborhood_mask[i])[0].cpu().numpy() for i in range(coords_ca.shape[0])]

    # Prepare coordinate neighbors
    coordinate_neighbors_backbone = []
    for i in range(len(indices)):
        # Get neighbors for current point
        neighbor_coords_n = coords_n[indices[i]]
        neighbor_coords_ca = coords_ca[indices[i]]
        neighbor_coords_c = coords_c[indices[i]]

        # Prepend query point to neighbors
        query_point_n = coords_n[i].unsqueeze(0)
        query_point_ca = coords_ca[i].unsqueeze(0)
        query_point_c = coords_c[i].unsqueeze(0)
        combined_coords_n = torch.cat([query_point_n, neighbor_coords_n])
        combined_coords_ca = torch.cat([query_point_ca, neighbor_coords_ca])
        combined_coords_c = torch.cat([query_point_c, neighbor_coords_c])
        combined_coords_backbone = torch.stack([combined_coords_n, combined_coords_ca, combined_coords_c], dim=1)
        indices[i] = np.append(i, indices[i])  # Include the index of the query point

        coordinate_neighbors_backbone.append(combined_coords_backbone)

    return indices, coordinate_neighbors_backbone


def get_grf_from_lrf(coords, sequence=None):
    indices_lrf, coordinate_neighbors_backbone = get_local_neighborhood_gpu(coords, angstrong_cutoff=10)

    lrf_n_ca_c_means = []
    sequence_lrf = []
    for i in range(len(coordinate_neighbors_backbone)):
        lrf_n_ca_c_means_i = torch.stack(
            [
                coordinate_neighbors_backbone[i][:, 0].mean(axis=0),
                coordinate_neighbors_backbone[i][:, 1].mean(axis=0),
                coordinate_neighbors_backbone[i][:, 2].mean(axis=0),
            ]
        )
        lrf_n_ca_c_means.append(lrf_n_ca_c_means_i)
        # print(indices[i])
        # print(sequence)
        if sequence is not None:
            seq_lrf = [sequence[j] for j in indices_lrf[i]]
            sequence_lrf.append(seq_lrf)
        # print(seq_lrf)
        # lol

    lrf_means = []
    for i in range(len(coordinate_neighbors_backbone)):
        lrf_means_i = coordinate_neighbors_backbone[i].view(-1, 3).mean(axis=0)
        lrf_means.append(lrf_means_i)

    coords_n_mean = coords[:, 0, :].mean(axis=0)
    coords_ca_mean = coords[:, 1, :].mean(axis=0)
    coords_c_mean = coords[:, 2, :].mean(axis=0)

    grf_coords = torch.stack([coords_n_mean, coords_ca_mean, coords_c_mean])

    lrf_means = torch.stack(lrf_means)
    coord_out_molframe_list, coord_out_locframe_list, features_lrf, R_m2w, t_m2w = frags_to_seq(
        coordinate_neighbors_backbone, lrf_means, grf_coords, non_col_points=lrf_n_ca_c_means
    )

    coords_grf = torch.cat(coord_out_molframe_list)
    coords_lrf = coord_out_locframe_list
    return coords_grf, coords_lrf, features_lrf, sequence_lrf, indices_lrf


def get_grf_from_lrf_from_backbone_coords(backbone_coords):
    coords_grf, coords_lrf, features_lrf, seq_lrf, indices_lrf = get_grf_from_lrf(backbone_coords.clone())
    return coords_grf


def get_grf_from_lrf_from_pdb(pdb_path, save_path, device):
    save_path = save_path + pdb_path.split("/")[-1].split(".")[0] + ".pt"
    # if file exists, return
    if os.path.exists(save_path):
        return None, None
    # check size of file if greater than 1GB, skip
    # if os.path.getsize(pdb_path) > 1e9:
    #    return None, None

    pt_file = torch.load(pdb_path)
    backbone_coords = pt_file["backbone_coords"].type(torch.float64).to(device)

    # rondomly rotate backbone_coords
    rot = Rotation.random().as_matrix()
    rot = torch.tensor(rot, dtype=torch.float64, device=device)

    # make sure rotation is valid
    det = torch.linalg.det(rot)
    neg_det_count = torch.sum(det < 0)
    if neg_det_count > 0:
        rot = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, -rot, rot)

    backbone_coords = torch.matmul(backbone_coords, rot)
    coords_grf, coords_lrf, features_lrf, seq_lrf, indices_lrf = get_grf_from_lrf(
        backbone_coords.clone(), pt_file["sequence"]
    )

    # save as pytorch pt file
    save_file = {
        "name": pt_file["name"],
        "sequence": pt_file["sequence"],
        "chains": pt_file["chains"],
        "residue_numbers": pt_file["residue_numbers"],
        "backbone_coords": pt_file["backbone_coords"],
        "coords_grf": coords_grf,
        "coords_lrf": coords_lrf,
        "features_lrf": features_lrf,
        "seq_lrf": seq_lrf,
        "indices_lrf": indices_lrf,
    }
    # ic(save_file['name'])
    # ic(save_file['seq_lrf'][0])
    # ic(save_file['indices_lrf'][0])
    # ic(save_file['features_lrf'][0])

    # save bacbone coordintates to pdb
    # save_backbone_as_pdb(backbone_coords, output_path="protein_backbone.pdb")
    # save_backbone_as_pdb(coords_grf, output_path="protein_backbone_grf.pdb")

    torch.save(save_file, save_path)
    return None, None  # coords_grf, coords_lrf


if __name__ == "__main__":
    import concurrent.futures
    import glob

    import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "/data/lisanzas/latent_generator/studies/data/pinder_raw_pdbs_molecular_frame_2/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pdb_dir = "/data/lisanzas/latent_generator/studies/data/pinder_raw_pdbs_bb_coords/"
    pdb_paths = glob.glob(pdb_dir + "*.pt")
    np.random.shuffle(pdb_paths)
    # for pdb_path in tqdm.tqdm(pdb_paths):
    #    coords_grf, coords_lrf = get_grf_from_lrf_from_pdb(pdb_path, save_path, device)
    #    lol

    def process_pdb(pdb_path):
        try:
            coords_grf, coords_lrf = get_grf_from_lrf_from_pdb(pdb_path, save_path, device)
        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")
            return None, None
        return coords_grf, coords_lrf

    with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
        results = list(tqdm.tqdm(executor.map(process_pdb, pdb_paths), total=len(pdb_paths)))
