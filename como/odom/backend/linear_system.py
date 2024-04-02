import torch

import como.geometry.lie_algebra as lie


def row_col_to_lin_index(row, col, width):
    return row * width + col


# All should be 1D
# NOTE: scatter_add_ is nondeterministic
def accumulate_gradient_scatter(g_batch, grad, grad_inds):
    grad.scatter_add_(0, grad_inds, g_batch)


# All should be 1D except Hessian 2D
# NOTE: scatter_add_ is nondeterministic
def accumulate_hessian_scatter(H_batch, H, H_inds):
    H_flat_view = H.view(-1)
    H_flat_view.scatter_add_(0, H_inds, H_batch)


# J (b,n,r,d), r (b,n,r) -> grad (b,d)
def get_gradient(J, r):
    grad = -torch.sum(J * r[..., None], dim=(1, 2))
    return grad


# J (b,n,r,d) -> H (b,d,d)
def get_hessian_diag_block(J):
    H_block = torch.einsum("bnck,bncl->bkl", J, J)
    return H_block


# J1 (b,n,r,d1), J1 (b,n,r,d2) -> H12 (b,d1,d2)
def get_hessian_off_diag_block(J1, J2):
    H12_blocks = torch.einsum("bnck,bncl->bkl", J1, J2)
    return H12_blocks


def accumulate_gradient_batched(g_batch, grad, grad_inds):
    for b in range(g_batch.shape[0]):
        inds_b = grad_inds[b, :]
        grad[inds_b] += g_batch[b, :]


def accumulate_hessian_diag_batched(H_batch, H, H_inds):
    for b in range(H_batch.shape[0]):
        inds_b = H_inds[b, :]
        H[inds_b[:, None], inds_b[None, :]] += H_batch[b, :, :]


def accumulate_hessian_off_diag_batched(H_batch, H, H_inds1, H_inds2):
    for b in range(H_batch.shape[0]):
        inds1_b = H_inds1[b, :]
        inds2_b = H_inds2[b, :]
        H[inds1_b[:, None], inds2_b[None, :]] += H_batch[b, :, :]
        H[inds2_b[:, None], inds1_b[None, :]] += torch.transpose(
            H_batch[b, :, :], dim0=0, dim1=1
        )


# H_blocks (b,m,d,d)
# NOTE: Creates wasteful block diagonal matrix
def accumulate_hessian_block_diag(H_blocks, H, H_inds):
    for b in range(H_blocks.shape[0]):
        H_block = torch.block_diag(*H_blocks[b, ...])
        inds_b = H_inds[b, :]
        H[inds_b[:, None], inds_b[None, :]] += H_block


def get_batched_pose_inds(num_kf, device):
    pose_dim = 6 * num_kf
    inds_flat = torch.arange(pose_dim, device=device)
    pose_inds = torch.reshape(inds_flat, (num_kf, -1))
    return pose_inds


def landmark_to_batched_3d_point_inds(landmark_inds, num_kf):
    device = landmark_inds.device
    landmark_inds_batched = landmark_inds[:, 1].view(num_kf, -1)
    # Go from landmark inds to Hessian indices (3D points)
    p_inds_batched = 3 * (landmark_inds_batched.repeat_interleave(3, dim=1))
    p_inds_batched += (
        torch.arange(3, device=device)
        .unsqueeze(0)
        .repeat(num_kf, landmark_inds_batched.shape[1])
    )
    return p_inds_batched


def allocate_system(num_poses, num_landmarks, device):
    pose_dim = 6 * num_poses
    geo_dim = 3 * num_landmarks
    dim = pose_dim + geo_dim
    H = torch.zeros((dim, dim), device=device)
    g = torch.zeros((dim,), device=device)
    return H, g


def solve_system(H, g):
    # # Check SVD of Hessian for debugging
    # U, S, Vh = torch.linalg.svd(H)
    # print(S.flatten())
    # print(Vh[-1,:].flatten())
    # print(Vh[-2,:].flatten())

    # L = torch.linalg.cholesky(H, upper=False)
    L, _ = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
    delta = torch.cholesky_solve(g[:, None], L, upper=False)

    return delta


def update_vars(
    delta,
    kf_poses,
    kf_aff_params,
    kf_inds,
    recent_poses,
    recent_aff_params,
    recent_inds,
    P,
    landmark_ind_start,
):
    device = delta.device
    dtype = delta.dtype

    delta = delta.squeeze(-1)

    kf_delta = delta[kf_inds]
    kf_poses_new = lie.batch_se3(kf_poses, kf_delta[:, :6])
    kf_aff_params_new = kf_aff_params + kf_delta[:, 6:, None]

    if recent_inds.shape[0] > 0:
        recent_delta = delta[recent_inds]
        recent_poses_new = lie.batch_se3(recent_poses, recent_delta[:, :6])
        recent_aff_params_new = recent_aff_params + recent_delta[:, 6:, None]
    else:
        recent_poses_new = torch.empty((0), device=device, dtype=dtype)
        recent_aff_params_new = torch.empty((0), device=device, dtype=dtype)

    P_delta = delta[landmark_ind_start:]
    P_new = P + P_delta.view(-1, 3)

    return (
        kf_poses_new,
        kf_aff_params_new,
        recent_poses_new,
        recent_aff_params_new,
        P_new,
    )
