import torch

import como.odom.backend.linear_system as lin_sys


# Enforce log-depth to be close to
def log_depth_prior(
    logzm_curr,
    logzm_mean,
    dlogzm_dPwm,
    dlogzm_dTwc,
    obs_ref_mask,
    p_inds_batched,
    pose_inds_batched,
    H,
    g,
    mode,
    sigma_first=None,
    sigma_all=None,
):
    b, m, _ = logzm_curr.shape
    device = logzm_curr.device

    if sigma_first is not None:
        info_scale_first = 1.0 / (sigma_first**2)
    if sigma_all is not None:
        info_scale_all = 1.0 / (sigma_all**2)

    # Assuming mean is not optimized
    r = logzm_curr - logzm_mean  # (B,M,1)

    # Unscaled Gradient g = -Jt @ r (accounting for negative just as get_gradient function)
    gu_Pw_all = -(dlogzm_dPwm.mT @ r[:, :, :, None])  # (B,M,3,1)
    gu_Twc_all = -(dlogzm_dTwc.mT @ r[:, :, :, None])  # (B,M,6,1)

    # Unscaled H_Pw diagonal blocks for each point involved
    Hu_Pw_blocks = dlogzm_dPwm.mT @ dlogzm_dPwm  # (B,M,3,3)
    Hu_Twc_all = dlogzm_dTwc.mT @ dlogzm_dTwc  # (B,M,6,6)
    Hu_Twc_Pw_blocks = dlogzm_dTwc.mT @ dlogzm_dPwm  # (B,M,6,3)

    def get_first_obs_mask():
        valid_mask = torch.zeros((b, m), device=device, dtype=torch.bool)
        valid_mask[obs_ref_mask] = True  # inds should be num_landmarks
        invalid_mask = torch.logical_not(valid_mask)
        return valid_mask, invalid_mask

    # Decide which factors to apply and turn into scale vector
    scale_vec = torch.empty((b, m, 1, 1), device=device)
    if mode == "first_mean":
        first_mask, rest_mask = get_first_obs_mask()
        scale_vec[first_mask, :, :] = info_scale_first
        scale_vec[rest_mask, :, :] = 0.0
        r[rest_mask, :] = 0.0
    elif mode == "first_curr":
        first_mask, rest_mask = get_first_obs_mask()
        scale_vec[first_mask, :, :] = info_scale_first
        scale_vec[rest_mask, :, :] = 0.0
        gu_Pw_all[:] = 0.0
        gu_Twc_all[:] = 0.0
        r[:] = 0.0
    elif mode == "all_curr":
        scale_vec[:] = info_scale_all
        gu_Pw_all[:] = 0.0
        gu_Twc_all[:] = 0.0
        r[:] = 0.0
    elif mode == "all_mean":
        scale_vec[:] = info_scale_all
    elif mode == "first_plus_rest_mean":  # Keeps gradient wrt pm_mean
        first_mask, rest_mask = get_first_obs_mask()
        scale_vec[first_mask, :, :] = info_scale_first
        scale_vec[rest_mask, :, :] = info_scale_all
    elif mode == "first_plus_rest_curr":  # No gradient since want around pm_curr
        first_mask, rest_mask = get_first_obs_mask()
        scale_vec[first_mask, :, :] = info_scale_first
        scale_vec[rest_mask, :, :] = info_scale_all
        gu_Pw_all[rest_mask, :, :] = 0.0
        gu_Twc_all[rest_mask, :, :] = 0.0
        r[rest_mask, :] = 0.0
    else:
        raise ValueError("pixel_prior_cost mode: " + mode + " is not implemented.")

    # Scale gradients and Hessians by info scale (and those masked out)
    g_Pw_all = scale_vec * gu_Pw_all
    g_Twc_all = scale_vec * gu_Twc_all
    H_Pw_blocks = scale_vec * Hu_Pw_blocks
    H_Twc_all = scale_vec * Hu_Twc_all
    H_Twc_Pw_blocks = scale_vec * Hu_Twc_Pw_blocks

    # Fix gradient shape
    g_Pw = torch.reshape(g_Pw_all, (b, 3 * m))  # (B, 3*M)

    # For poses, accumulate all info
    g_Twc = torch.sum(g_Twc_all, dim=(1, 3))  # (B,6)
    H_Twc = torch.sum(H_Twc_all, dim=1)  # (B,6,6)

    # Reshape off-diag blocks
    H_Twc_Pw = torch.reshape(
        torch.permute(H_Twc_Pw_blocks, (0, 2, 1, 3)), (b, 6, 3 * m)
    )  # (B,6,3*M)

    grads_flat = torch.cat((g_Pw.flatten(), g_Twc.flatten()))
    grad_inds_flat = torch.cat((p_inds_batched.flatten(), pose_inds_batched.flatten()))
    lin_sys.accumulate_gradient_scatter(grads_flat, g, grad_inds_flat)

    W = H.shape[1]
    # Need special handling for block diag
    p_inds_diag = p_inds_batched.view(b, m, 3)  # (b,m,3)
    H_inds_Pw_diag = lin_sys.row_col_to_lin_index(
        p_inds_diag[:, :, :, None], p_inds_diag[:, :, None, :], W
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
        (
            H_Pw_blocks.flatten(),
            H_Twc.flatten(),
            H_Twc_Pw.flatten(),
            H_Twc_Pw.mT.flatten(),
        )
    )
    H_inds_flat = torch.cat(
        (
            H_inds_Pw_diag.flatten(),
            H_inds_Twc.flatten(),
            H_inds_Twc_Pw.flatten(),
            H_inds_Twc_Pw2.flatten(),
        )
    )
    lin_sys.accumulate_hessian_scatter(H_flat, H, H_inds_flat)

    total_err = torch.sum(scale_vec[:, :, :, 0] * torch.square(r))

    return total_err


# Prior that all log-depths close to mean log-depth
def dense_depth_prior(
    logzn,
    logz_mean,
    dlogzn_dlogzm,
    dlogzm_dPwm,
    dlogzm_dTwc,
    p_inds_batched,
    pose_inds_batched,
    H,
    g,
    sigma,
):
    b, n, _, m = dlogzn_dlogzm.shape
    device = logzn.device

    info_scale = 1.0 / (sigma**2)

    r = logzn - logz_mean

    dr_dPw = (
        dlogzn_dlogzm[:, :, 0, :, None] * dlogzm_dPwm[:, None, :, 0, :]
    )  # (B,N,M,3)
    dr_dPw = torch.reshape(dr_dPw, (b, n, 1, 3 * m))  # (B,N,1,3*M)
    dr_dTwc = dlogzn_dlogzm @ torch.permute(dlogzm_dTwc, (0, 2, 1, 3))  # (B,N,1,6)

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
