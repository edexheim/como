import torch

from como.geometry.lie_algebra import invertSE3, skew_symmetric
from como.utils.lin_alg import transpose_last
import como.odom.backend.linear_system as lin_sys


def pose_range_factor(range_meas, pose1, pose2, pose_inds1, pose_inds2, H, g, sigma):
    device = pose1.device
    dtype = pose1.dtype
    B = pose1.shape[0]

    info_sqrt = 1.0 / sigma

    # Error calculation
    t2 = pose2[:, :3, 3:4]
    range_pred, dr_dT1, dr_dt2 = pose_point_range(pose1, t2)

    dr_dT2 = torch.empty((B, 1, 6), device=device, dtype=dtype)
    dr_dT2[:, :, :3] = 0.0
    dr_dT2[:, :, 3:] = dr_dt2 @ pose2[:, :3, :3]

    r = -(range_meas - range_pred)

    # Weight system by sigma
    r = info_sqrt * r[:, :, None]
    dr_dT1 = info_sqrt * dr_dT1[:, :, None, :]
    dr_dT2 = info_sqrt * dr_dT2[:, :, None, :]

    # Gradients
    grad_batch1 = lin_sys.get_gradient(dr_dT1, r)
    grad_batch2 = lin_sys.get_gradient(dr_dT2, r)
    lin_sys.accumulate_gradient_batched(grad_batch1, g, pose_inds1)
    lin_sys.accumulate_gradient_batched(grad_batch2, g, pose_inds2)

    # Hessian diag
    H_batch1 = lin_sys.get_hessian_diag_block(dr_dT1)
    H_batch2 = lin_sys.get_hessian_diag_block(dr_dT2)
    lin_sys.accumulate_hessian_diag_batched(H_batch1, H, pose_inds1)
    lin_sys.accumulate_hessian_diag_batched(H_batch2, H, pose_inds2)

    # Hessian off-diag
    H_off_diag_batch = lin_sys.get_hessian_off_diag_block(dr_dT1, dr_dT2)
    lin_sys.accumulate_hessian_off_diag_batched(
        H_off_diag_batch, H, pose_inds1, pose_inds2
    )

    total_err = torch.sum(torch.square(r))

    return total_err


def pose_point_range(T1, tw2):
    device = T1.device
    dtype = T1.dtype
    B = T1.shape[0]

    # Errors
    T1_inv = invertSE3(T1)
    R1_inv = T1_inv[:, :3, :3]
    t1_inv = T1_inv[:, :3, 3:4]
    t12 = R1_inv @ tw2 + t1_inv
    r = torch.linalg.norm(t12, dim=1)

    # Jacobians
    dr_dt12 = (1.0 / r) * transpose_last(t12)

    dt12_dTw1 = torch.empty((B, 3, 6), device=device, dtype=dtype)
    dt12_dTw1[:, :, :3] = skew_symmetric(t12.squeeze(-1))
    dt12_dTw1[:, :, 3:] = -torch.eye(3)

    dt12_dtw2 = R1_inv

    # Chain rule
    dr_dTw1 = dr_dt12 @ dt12_dTw1
    dr_dtw2 = dr_dt12 @ dt12_dtw2

    return r, dr_dTw1, dr_dtw2
