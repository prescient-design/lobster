import numpy as np
import torch
from loguru import logger
from ._lrf import apply_stochastic_fa as apply_stochastic_fa_func
from ._molecular_frame import get_grf_from_lrf_from_backbone_coords

PARAMS = {
    "DMIN": 2.0,
    "DMAX": 20.0,
    "DBINS": 36,
    "ABINS": 36,
}


def get_Cb(xyz):
    N = xyz[:, :, 0]
    Ca = xyz[:, :, 1]
    C = xyz[:, :, 2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    return Cb


# ============================================================
def normQ(Q):
    """normalize a quaternions"""
    return Q / torch.linalg.norm(Q, keepdim=True, dim=-1)


# ============================================================
def avgQ(Qs):
    """average a set of quaternions
    input dims:
    Qs - (B,N,R,4)
    averages across 'N' dimension
    """

    def areClose(q1, q2):
        return (q1 * q2).sum(dim=-1) >= 0.0

    N = Qs.shape[1]
    Qsum = Qs[:, 0] / N

    for i in range(1, N):
        mask = areClose(Qs[:, 0], Qs[:, i])
        Qsum[mask] += Qs[:, i][mask] / N
        Qsum[~mask] -= Qs[:, i][~mask] / N

    return normQ(Qsum)


def Rs2Qs(Rs):
    Qs = torch.zeros((*Rs.shape[:-2], 4), device=Rs.device)

    Qs[..., 0] = 1.0 + Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[..., 1] = 1.0 + Rs[..., 0, 0] - Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 2] = 1.0 - Rs[..., 0, 0] + Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 3] = 1.0 - Rs[..., 0, 0] - Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[Qs < 0.0] = 0.0
    Qs = torch.sqrt(Qs) / 2.0
    Qs[..., 1] *= torch.sign(Rs[..., 2, 1] - Rs[..., 1, 2])
    Qs[..., 2] *= torch.sign(Rs[..., 0, 2] - Rs[..., 2, 0])
    Qs[..., 3] *= torch.sign(Rs[..., 1, 0] - Rs[..., 0, 1])

    return Qs


def Qs2Rs(Qs):
    Rs = torch.zeros((*Qs.shape[:-1], 3, 3), device=Qs.device)

    Rs[..., 0, 0] = (
        Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
    )
    Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 1, 1] = (
        Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
    )
    Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 2] = (
        Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] + Qs[..., 3] * Qs[..., 3]
    )

    return Rs


# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points

    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist


# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v * w, dim=-1)

    return torch.acos(vw)


# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)

    return torch.atan2(y, x)


# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance
    and orientation maps

    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps
    """

    batch = xyz.shape[0]
    nres = xyz.shape[1]

    N = xyz[:, :, 0]
    Ca = xyz[:, :, 1]
    Cb = get_Cb(xyz)

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch, nres, nres, 4], dtype=xyz.dtype, device=xyz.device)

    dist = get_pair_dist(Cb, Cb)
    c6d[..., 0] = dist + 999.9 * torch.eye(nres, device=xyz.device).unsqueeze(0)  # [None,...]
    b, i, j = torch.where(c6d[..., 0] < params["DMAX"])

    c6d[b, i, j, torch.full_like(b, 1)] = get_dih(Ca[b, i], Cb[b, i], Cb[b, j], Ca[b, j])
    c6d[b, i, j, torch.full_like(b, 2)] = get_dih(N[b, i], Ca[b, i], Cb[b, i], Cb[b, j])
    c6d[b, i, j, torch.full_like(b, 3)] = get_ang(Ca[b, i], Cb[b, i], Cb[b, j])

    # fix long-range distances
    c6d[..., 0][c6d[..., 0] >= params["DMAX"]] = 999.9
    c6d = torch.nan_to_num(c6d)

    return c6d


def xyz_to_t2d(xyz_t, mask, params=PARAMS):
    """convert template cartesian coordinates into 2d distance
    and orientation maps

    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,natm,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms
    mask: pytorch tensor of shape [batch,templ,nrres,nres]
          indicates whether valid residue pairs or not
    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,37+6+1]
          stores stacked dist,omega,theta,phi 2D maps
    """
    B, T, L = xyz_t.shape[:3]
    c6d = xyz_to_c6d(xyz_t[:, :, :, :3].view(B * T, L, 3, 3), params=params)
    c6d = c6d.view(B, T, L, L, 4)

    # dist to one-hot encoded
    mask = mask[..., None]
    dist = dist_to_onehot(c6d[..., 0], params) * mask
    orien = torch.cat((torch.sin(c6d[..., 1:]), torch.cos(c6d[..., 1:])), dim=-1) * mask  # (B, T, L, L, 6)
    #
    t2d = torch.cat((dist, orien, mask), dim=-1)
    return t2d


def xyz_to_chi1(xyz_t):
    """convert template cartesian coordinates into chi1 angles

    Parameters
    ----------
    xyz_t: pytorch tensor of shape [batch, templ, nres, 14, 3]
           stores Cartesian coordinates of template atoms. For missing atoms, it should be NaN

    Returns
    -------
    chi1 : pytorch tensor of shape [batch, templ, nres, 2]
           stores cos and sin chi1 angle
    """
    B, T, L = xyz_t.shape[:3]
    xyz_t = xyz_t.reshape(B * T, L, 14, 3)

    # chi1 angle: N, CA, CB, CG
    chi1 = get_dih(xyz_t[:, :, 0], xyz_t[:, :, 1], xyz_t[:, :, 4], xyz_t[:, :, 5])  # (B*T, L)
    cos_chi1 = torch.cos(chi1)
    sin_chi1 = torch.sin(chi1)
    mask_chi1 = ~torch.isnan(chi1)
    chi1 = torch.stack((cos_chi1, sin_chi1, mask_chi1), dim=-1)  # (B*T, L, 3)
    chi1[torch.isnan(chi1)] = 0.0
    chi1 = chi1.reshape(B, T, L, 3)
    return chi1


def xyz_to_bbtor(xyz, params=PARAMS):
    # three anchor atoms
    N = xyz[:, :, 0]
    Ca = xyz[:, :, 1]
    C = xyz[:, :, 2]

    # recreate Cb given N,Ca,C
    next_N = torch.roll(N, -1, dims=1)
    prev_C = torch.roll(C, 1, dims=1)
    phi = get_dih(prev_C, N, Ca, C)
    psi = get_dih(N, Ca, C, next_N)
    #
    phi[:, 0] = 0.0
    psi[:, -1] = 0.0
    #
    astep = 2.0 * np.pi / params["ABINS"]
    phi_bin = torch.round((phi + np.pi - astep / 2) / astep)
    psi_bin = torch.round((psi + np.pi - astep / 2) / astep)
    return torch.stack([phi_bin, psi_bin], axis=-1).long()


# ============================================================
def dist_to_onehot(dist, params=PARAMS):
    dstep = (params["DMAX"] - params["DMIN"]) / params["DBINS"]
    dbins = torch.linspace(
        params["DMIN"] + dstep, params["DMAX"], params["DBINS"], dtype=dist.dtype, device=dist.device
    )
    db = torch.bucketize(dist.contiguous(), dbins).long()
    dist = torch.nn.functional.one_hot(db, num_classes=params["DBINS"] + 1).float()
    return dist


def c6d_to_bins(c6d, params=PARAMS):
    """bin 2d distance and orientation maps"""

    dstep = (params["DMAX"] - params["DMIN"]) / params["DBINS"]
    astep = 2.0 * np.pi / params["ABINS"]

    dbins = torch.linspace(params["DMIN"] + dstep, params["DMAX"], params["DBINS"], dtype=c6d.dtype, device=c6d.device)
    ab360 = torch.linspace(-np.pi + astep, np.pi, params["ABINS"], dtype=c6d.dtype, device=c6d.device)
    ab180 = torch.linspace(astep, np.pi, params["ABINS"] // 2, dtype=c6d.dtype, device=c6d.device)

    db = torch.bucketize(c6d[..., 0].contiguous(), dbins)
    ob = torch.bucketize(c6d[..., 1].contiguous(), ab360)
    tb = torch.bucketize(c6d[..., 2].contiguous(), ab360)
    pb = torch.bucketize(c6d[..., 3].contiguous(), ab180)

    ob[db == params["DBINS"]] = params["ABINS"]
    tb[db == params["DBINS"]] = params["ABINS"]
    pb[db == params["DBINS"]] = params["ABINS"] // 2

    return torch.stack([db, ob, tb, pb], axis=-1).long()


# ============================================================
def dist_to_bins(dist, params=PARAMS):
    """bin 2d distance maps"""

    dstep = (params["DMAX"] - params["DMIN"]) / params["DBINS"]
    db = torch.round((dist - params["DMIN"] - dstep / 2) / dstep)

    db[db < 0] = 0
    db[db > params["DBINS"]] = params["DBINS"]

    return db.long()


# ============================================================
def c6d_to_bins2(c6d, same_chain, negative=False, params=PARAMS):
    """bin 2d distance and orientation maps"""

    dstep = (params["DMAX"] - params["DMIN"]) / params["DBINS"]
    astep = 2.0 * np.pi / params["ABINS"]

    db = torch.round((c6d[..., 0] - params["DMIN"] - dstep / 2) / dstep)
    ob = torch.round((c6d[..., 1] + np.pi - astep / 2) / astep)
    tb = torch.round((c6d[..., 2] + np.pi - astep / 2) / astep)
    pb = torch.round((c6d[..., 3] - astep / 2) / astep)

    # put all d<dmin into one bin
    db[db < 0] = 0

    # synchronize no-contact bins
    db[db > params["DBINS"]] = params["DBINS"]
    ob[db == params["DBINS"]] = params["ABINS"]
    tb[db == params["DBINS"]] = params["ABINS"]
    pb[db == params["DBINS"]] = params["ABINS"] // 2

    if negative:
        db = torch.where(same_chain.bool(), db.long(), params["DBINS"])
        ob = torch.where(same_chain.bool(), ob.long(), params["ABINS"])
        tb = torch.where(same_chain.bool(), tb.long(), params["ABINS"])
        pb = torch.where(same_chain.bool(), pb.long(), params["ABINS"] // 2)

    return torch.stack([db, ob, tb, pb], axis=-1).long()


def expand(x, tgt=None, dim=1):
    if tgt is None:
        for _ in range(dim):
            x = x[..., None]
    else:
        while len(x.shape) < len(tgt.shape):
            x = x[..., None]
    return x


def apply_random_se3(coords_in, atom_mask=None, translation_scale=1.0, rotation_mode="svd"):
    """
    Apply random SE(3) transformation to coordinates.

    Args:
        coords_in: Input coordinates tensor of shape [num_res, num_atom_type, 3] or [num_res, 3]
        atom_mask: Optional mask tensor
        translation_scale: Scale factor for random translation
        rotation_mode: Method to generate random rotation ("svd", "quaternion", or "none")
    """
    is_flat = len(coords_in.shape) == 2
    if is_flat:
        # Handle flat input [num_res, 3]
        coords_mean = coords_in.mean(-2, keepdim=True)
    else:
        # Handle structured input [num_res, num_atom_type, 3]
        coords_mean = coords_in[:, 1:2].mean(-3, keepdim=True)

    coords_in -= coords_mean

    if rotation_mode == "svd":
        random_rot, _ = torch.linalg.qr(torch.randn(3, 3))
    elif rotation_mode == "quaternion":
        random_rot = uniform_rand_rotation(1).squeeze(0)
    elif rotation_mode == "none":
        logger.info("no rotation applied in function apply_random_se3")
        random_rot = torch.eye(3)

    coords_in = coords_in @ random_rot.to(coords_in)
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in += random_trans.to(coords_in)

    if atom_mask is not None:
        if is_flat:
            coords_in = coords_in * atom_mask[..., None]
        else:
            coords_in = coords_in * atom_mask[..., None, None]

    return coords_in


def apply_random_se3_2(coords_in, atom_mask=None, translation_scale=1.0, rotation_mode="quaternion"):
    # unbatched. center on the mean of CA coords
    # coords_in: [num_res, 3]
    coords_mean = coords_in.mean(-2, keepdim=True)
    coords_in -= coords_mean
    if rotation_mode == "svd":
        random_rot, _ = torch.linalg.qr(torch.randn(3, 3))
    elif rotation_mode == "quaternion":
        random_rot = uniform_rand_rotation(1).squeeze(0)
    random_rot, _ = torch.linalg.qr(torch.randn(3, 3))
    coords_in = coords_in @ random_rot.to(coords_in)
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in += random_trans.to(coords_in)
    if atom_mask is not None:
        coords_in = coords_in * atom_mask[..., None]
    return coords_in


def apply_random_se3_batched(coords_in, atom_mask=None, translation_scale=1.0, rotation_mode="svd"):
    B = coords_in.shape[0]
    for b in range(B):
        if atom_mask is not None:
            atom_mask_b = atom_mask[b]
        else:
            atom_mask_b = None
        coords_in[b] = apply_random_se3(
            coords_in[b], atom_mask=atom_mask_b, translation_scale=translation_scale, rotation_mode=rotation_mode
        )
    return coords_in


def _graham_schmidt(x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12):
    e1 = xy_plane
    denom = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
    x_axis = x_axis / denom
    dot = (x_axis * e1).sum(dim=-1, keepdim=True)
    e1 = e1 - x_axis * dot
    denom = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
    e1 = e1 / denom
    e2 = torch.cross(x_axis, e1, dim=-1)
    rots = torch.stack([x_axis, e1, e2], dim=-1)
    return rots


def from_graham_schmidt(neg_x_axis: torch.Tensor, origin: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12):
    x_axis = origin - neg_x_axis
    xy_plane = xy_plane - origin
    rots = _graham_schmidt(x_axis, xy_plane, eps=eps)
    return (origin, rots)


def atom3_to_backbone_frames(bb_positions: torch.Tensor):
    N, CA, C = bb_positions.unbind(dim=-2)
    return from_graham_schmidt(C, CA, N)


def enforce_consistent_orientation(lrf, centered_points, mask):
    """
    Orient the local reference frame consistently based on point distribution.
    Args:
        lrf: tensor of shape (B, 3, 3) containing eigenvectors as columns
        centered_points: tensor of shape (B, L, 3) centered points
        mask: tensor of shape (B, L) for ignoring padding
    Returns:
        oriented_lrf: tensor of shape (B, 3, 3) with consistent orientation
    """
    oriented_lrf = lrf.clone()
    centered_points = centered_points * mask.unsqueeze(-1)  # Apply mask to centered points

    # For each batch
    for b in range(oriented_lrf.shape[0]):
        # Get the signs of projections of points onto each principal axis
        frame = oriented_lrf[b]

        # For each axis, compute the skewness of point distributions
        # If skewness is negative, flip the axis
        for i in range(3):
            # Project points onto current axis
            projections = torch.matmul(centered_points[b], frame[:, i])

            # Compute third moment (skewness)
            valid_points = mask[b] > 0
            skewness = torch.mean(projections[valid_points] ** 3)

            # If skewness is negative, flip this axis
            if skewness < 0:
                frame[:, i] = -frame[:, i]

        # Ensure right-handedness
        cross_prod = torch.cross(frame[:, 0], frame[:, 1], dim=-1)
        if torch.dot(cross_prod, frame[:, 2]) < 0:
            frame[:, 2] = -frame[:, 2]

        oriented_lrf[b] = frame

    return oriented_lrf


def compute_global_reference_frame(points, mask, apply_stochastic_fa=False, get_all_frames=False):
    """
    Compute global reference frame from total point cloud covariance.
    Args:
        points: tensor of shape (B, L, 3)
        mask: tensor of shape (B, L) for ignoring padding
    Returns:
        global_frame: tensor of shape (B, 3, 3)
        global_mean: tensor of shape (B, 1, 3)
    """
    # Expand mask for broadcasting
    mask = mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)

    # Calculate mean only over valid positions
    valid_points = points * mask
    sum_mask = mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1, 1)
    mean = valid_points.sum(dim=1, keepdim=True) / sum_mask

    # Center the points
    centered_points = (points - mean) * mask

    # Compute global covariance for each batch
    covariance = torch.einsum("bij,bik->bjk", centered_points, centered_points)

    # Normalize by (N-1) where N is the number of valid points per batch
    n_valid = mask.squeeze(-1).sum(dim=1)  # Shape: (batch_size,)
    covariance = covariance / (n_valid - 1).unsqueeze(-1).unsqueeze(-1)

    # Add small diagonal perturbation to avoid degenerate cases
    eps = 1e-12
    covariance = covariance + eps * torch.eye(3, device=covariance.device)

    # Compute eigenvalues and eigenvectors of global covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    # Sort in descending order
    eigenvalues = eigenvalues.flip(-1)  # Now λ1 ≥ λ2 ≥ λ3
    global_frame = eigenvectors.flip(-1)  # Flip to get descending order

    # ensure right-handedness note shape is (B, L, 3, 3)
    det = torch.linalg.det(global_frame)
    # ic(det.shape)
    # ic(det)
    # correct for negative determinants
    neg_det_count = torch.sum(det < 0)
    if neg_det_count > 0:
        # print(f"Warning: {neg_det_count} frames have negative determinant")
        global_frame = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, -global_frame, global_frame)

    if apply_stochastic_fa:
        global_frame = global_frame.unsqueeze(1)
        global_frame = apply_stochastic_fa_func(global_frame).squeeze(1)
    elif get_all_frames:
        global_frame = global_frame.unsqueeze(1)
        global_frame = apply_stochastic_fa_func(global_frame, give_all=True).squeeze(1)  # (B, 4, 3, 3)

    else:
        global_frame = enforce_consistent_orientation(global_frame, centered_points, mask.squeeze(-1))

    return global_frame, mean


def transform_to_global_frame(points, transform_only=False, mask=None, apply_stochastic_fa=False, get_all_frames=False):
    """
    Transform points to global reference frame.

    Args:
        points: tensor of shape (B, L,x, 3)
        transform_only: bool, if True only returns transformed points
        mask: optional tensor of shape (B, L) for ignoring padding
    Returns:
        If transform_only:
            transformed_points: tensor of shape (B, L, 3)
        else:
            dict containing transformed points and transformation info
    """
    if mask is None:
        mask = torch.ones_like(points[..., 0, 0])
    # Remove the incorrect indexing
    if len(points.shape) == 4:
        points_flat = points[:, :, 1, :]  # get CA
    else:
        # ic(points.shape)
        points_flat = points

    global_frame, global_mean = compute_global_reference_frame(
        points_flat, mask, apply_stochastic_fa=apply_stochastic_fa, get_all_frames=get_all_frames
    )
    # ic(global_mean.shape)
    if len(points.shape) == 4:
        centered_points = points - global_mean.unsqueeze(1)  # Center points around the mean
        centered_points = centered_points * mask.unsqueeze(-1).unsqueeze(-1)
    else:
        centered_points = points - global_mean  # Center points around the mean
        centered_points = centered_points * mask.unsqueeze(-1)
    # set padding position to zero
    # ic(centered_points.shape)

    # Transform to global frame
    centered_points = centered_points.view(points.shape[0], -1, 3)
    # ic(centered_points.shape)
    if get_all_frames:
        transformed_points = []
        num_frames = global_frame.shape[1]
        for g_frame in range(num_frames):
            global_frame_ = global_frame[:, g_frame]
            transformed_points_ = torch.einsum("bij,bkj->bki", global_frame_.transpose(-1, -2), centered_points)
            if len(points.shape) == 4:
                transformed_points_ = transformed_points_.view(points.shape[0], points.shape[1], -1, 3)
                transformed_points_ = transformed_points_ * mask.unsqueeze(-1).unsqueeze(-1)
            else:
                # ic(transformed_points_.shape)
                transformed_points_ = transformed_points_ * mask.unsqueeze(-1)
            # ic(transformed_points_.shape)
            transformed_points.append(transformed_points_)
        # stack along batch dimension
        transformed_points = torch.stack(transformed_points, dim=1).squeeze()  # Shape: (B, num_frames, L, 3)
        ####CHECK THS VISUALLY########
    else:
        transformed_points = torch.einsum("bij,bkj->bki", global_frame.transpose(-1, -2), centered_points)
        if len(points.shape) == 4:
            transformed_points = transformed_points.view(points.shape[0], points.shape[1], -1, 3)
            transformed_points = transformed_points * mask.unsqueeze(-1).unsqueeze(
                -1
            )  # Apply mask to transformed points
        else:
            transformed_points = transformed_points * mask.unsqueeze(-1)  # Apply mask to transformed points

    if transform_only:
        return transformed_points

    return {"transformed_points": transformed_points, "global_frame": global_frame, "global_mean": global_mean}


def apply_global_frame_to_coords(
    coords, frame_type="pca_frame", mask=None, apply_stochastic_fa=False, get_all_frames=False
):
    """
    Apply global frame transformation to coordinates.

    Args:
        coords: Input coordinates tensor
        frame_type: Type of frame to use. Must be one of ["norm_frame", "pca_frame", "mol_frame"]
        mask: Optional mask tensor
        apply_stochastic_fa: Whether to apply stochastic frame alignment (only for pca_frame)
        get_all_frames: Whether to get all frames (only for pca_frame)

    Returns:
        Transformed coordinates
    """
    if apply_stochastic_fa and get_all_frames:
        raise ValueError("apply_stochastic_fa and get_all_frames cannot be both True at the same time.")

    if frame_type not in ["norm_frame", "pca_frame", "mol_frame"]:
        raise ValueError(f"frame_type must be one of ['norm_frame', 'pca_frame', 'mol_frame'], got {frame_type}")

    B = coords.shape[0]
    coords_ = coords.clone()

    if frame_type == "mol_frame":
        for b in range(B):
            coords_[b] = get_grf_from_lrf_from_backbone_coords(coords_[b])
    elif frame_type == "norm_frame":
        for b in range(B):
            if mask is not None:
                mask_b = mask[b]
            else:
                mask_b = torch.ones(coords.shape[1], device=coords.device)
            average_position_per_n_ca_c = coords_[b, :, :3, :].sum(0) / mask_b.sum(0)
            trans, rot = atom3_to_backbone_frames(average_position_per_n_ca_c)
            inv_rot = rot.transpose(-1, -2)
            trans_ = torch.einsum("...ij,...j", inv_rot, trans)
            coords_[b, ...] = torch.einsum("...ij,...j", inv_rot, coords[b]) + trans_
            coords_[b] = coords_[b] * mask_b[..., None, None]
    elif frame_type == "pca_frame":
        if mask is None:
            mask = torch.ones(coords_.shape[0], coords_.shape[1], device=coords_.device)
        coords_ = transform_to_global_frame(
            coords_,
            transform_only=True,
            mask=mask,
            apply_stochastic_fa=apply_stochastic_fa,
            get_all_frames=get_all_frames,
        )

    return coords_


def uniform_rand_rotation(batch_size):  #! 24/1/10 ZH added
    # Creates a shape (batch_size, 3, 3) rotation matrix uniformly at random in SO(3)
    # Uses quaternionic multiplication to generate independent rotation matrices for each batch
    q = torch.randn(batch_size, 4)
    q /= torch.norm(q, dim=1, keepdim=True)
    rotation = torch.zeros(batch_size, 3, 3).to(q)
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation[:, 0, :] = torch.stack([2 * a**2 - 1 + 2 * b**2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c]).T
    rotation[:, 1, :] = torch.stack([2 * b * c + 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d - 2 * a * b]).T
    rotation[:, 2, :] = torch.stack([2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, 2 * a**2 - 1 + 2 * d**2]).T
    return rotation
