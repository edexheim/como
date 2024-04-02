import torch

import como.odom.backend.linear_system as lin_sys


# Based on marginal likelihood with fixed covariance and unknown training values
def gp_ml_cost(
    logzm,
    log_median_depths,
    L_mm,
    dlogzm_dPw,
    dlogzm_dTwc,
    p_inds_batched,
    pose_inds_batched,
    H,
    g,
    sigma,
):
    b, m, _ = L_mm.shape
    device = L_mm.device

    da_dlogzm = torch.eye(m, device=device).reshape((1, m, m)).repeat(b, 1, 1)
    dr_dlogzm = torch.linalg.solve_triangular(L_mm, da_dlogzm, upper=False)  # L_inv

    # h(x) - z with z = 0
    r = dr_dlogzm @ (logzm - log_median_depths)  # L_inv @ logzm

    # Point Jacobians
    dr_dPw = (
        dr_dlogzm[:, :, :, None] * dlogzm_dPw[:, None, :, 0, :]
    )  # (B.M,M,3) <- (B,M,M) x (B,M,1,3)
    dr_dPw = torch.reshape(dr_dPw, (b, m, 1, 3 * m))  # (B.M,1,3M)
    # Pose Jacobians
    dr_dTwc = (dr_dlogzm @ dlogzm_dTwc[:, :, 0, :]).unsqueeze(
        2
    )  # (B,M,1,6) <- (B,M,M) x (B,M,1,6)

    info_scale = 1.0 / (sigma**2)

    # Gradient g = -Jt @ r
    grad_Pw = info_scale * lin_sys.get_gradient(dr_dPw, r)
    grad_Twc = info_scale * lin_sys.get_gradient(dr_dTwc, r)
    H_Pw = info_scale * lin_sys.get_hessian_diag_block(dr_dPw)
    H_Twc = info_scale * lin_sys.get_hessian_diag_block(dr_dTwc)
    H_Twc_Pw = info_scale * lin_sys.get_hessian_off_diag_block(dr_dTwc, dr_dPw)

    grads_flat = torch.cat((grad_Pw.flatten(), grad_Twc.flatten()))
    grad_inds_flat = torch.cat((p_inds_batched.flatten(), pose_inds_batched.flatten()))
    lin_sys.accumulate_gradient_scatter(grads_flat, g, grad_inds_flat)

    W = H.shape[1]
    H_inds_Pw = lin_sys.row_col_to_lin_index(
        p_inds_batched[:, :, None], p_inds_batched[:, None, :], W
    )
    H_inds_Twc = lin_sys.row_col_to_lin_index(
        pose_inds_batched[:, :, None], pose_inds_batched[:, None, :], W
    )
    H_inds_Twc_Pw = lin_sys.row_col_to_lin_index(
        pose_inds_batched[:, :, None], p_inds_batched[:, None, :], W
    )
    H_inds_Twc_Pw2 = lin_sys.row_col_to_lin_index(
        p_inds_batched[:, :, None], pose_inds_batched[:, None, :], W
    )

    H_flat = torch.cat(
        (H_Pw.flatten(), H_Twc.flatten(), H_Twc_Pw.flatten(), H_Twc_Pw.mT.flatten())
    )
    H_inds_flat = torch.cat(
        (
            H_inds_Pw.flatten(),
            H_inds_Twc.flatten(),
            H_inds_Twc_Pw.flatten(),
            H_inds_Twc_Pw2.flatten(),
        )
    )
    lin_sys.accumulate_hessian_scatter(H_flat, H, H_inds_flat)

    total_err = torch.sum(info_scale * torch.square(r))

    return total_err


def mean_log_depth_cost(
    logzm,
    Knm_Kmminv,
    mean_log_depth_prior,
    dlogzm_dPw,
    dlogzm_dTwc,
    p_inds_batched,
    pose_inds_batched,
    H,
    g,
    sigma,
):
    b, n, m = Knm_Kmminv.shape

    info_scale = 1.0 / (sigma**2)

    # h(x) - z
    mean_pred = torch.mean(Knm_Kmminv @ logzm, dim=(1, 2), keepdim=True)
    r = mean_pred - mean_log_depth_prior

    # Chain rule
    dr_dlogzm = torch.sum(Knm_Kmminv, dim=(1), keepdim=False) / n  # (B,M)
    # Points
    dr_dPw = (dr_dlogzm[:, :, None] * dlogzm_dPw[:, :, 0, :]).view(b, 1, 1, 3 * m)
    # Poses
    dr_dTwc = (dr_dlogzm[:, None, :] @ dlogzm_dTwc[:, :, 0, :]).unsqueeze(2)

    # Gradient g = -Jt @ r
    grad_Pw = info_scale * lin_sys.get_gradient(dr_dPw, r)
    grad_Twc = info_scale * lin_sys.get_gradient(dr_dTwc, r)
    H_Pw = info_scale * lin_sys.get_hessian_diag_block(dr_dPw)
    H_Twc = info_scale * lin_sys.get_hessian_diag_block(dr_dTwc)
    H_Twc_Pw = info_scale * lin_sys.get_hessian_off_diag_block(dr_dTwc, dr_dPw)

    grads_flat = torch.cat((grad_Pw.flatten(), grad_Twc.flatten()))
    grad_inds_flat = torch.cat((p_inds_batched.flatten(), pose_inds_batched.flatten()))
    lin_sys.accumulate_gradient_scatter(grads_flat, g, grad_inds_flat)

    W = H.shape[1]
    H_inds_Pw = lin_sys.row_col_to_lin_index(
        p_inds_batched[:, :, None], p_inds_batched[:, None, :], W
    )
    H_inds_Twc = lin_sys.row_col_to_lin_index(
        pose_inds_batched[:, :, None], pose_inds_batched[:, None, :], W
    )
    H_inds_Twc_Pw = lin_sys.row_col_to_lin_index(
        pose_inds_batched[:, :, None], p_inds_batched[:, None, :], W
    )
    H_inds_Twc_Pw2 = lin_sys.row_col_to_lin_index(
        p_inds_batched[:, :, None], pose_inds_batched[:, None, :], W
    )

    H_flat = torch.cat(
        (H_Pw.flatten(), H_Twc.flatten(), H_Twc_Pw.flatten(), H_Twc_Pw.mT.flatten())
    )
    H_inds_flat = torch.cat(
        (
            H_inds_Pw.flatten(),
            H_inds_Twc.flatten(),
            H_inds_Twc_Pw.flatten(),
            H_inds_Twc_Pw2.flatten(),
        )
    )
    lin_sys.accumulate_hessian_scatter(H_flat, H, H_inds_flat)

    total_err = torch.sum(info_scale * torch.square(r))

    return total_err
