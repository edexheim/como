import torch

from como.depth_cov.core.gaussian_kernel import interpolate_kernel_params
from como.geometry.camera import backprojection, projection
from como.geometry.depth import (
    depth_to_log_depth,
    log_depth_to_depth,
    predict_log_depth,
)
from como.geometry.lie_algebra import invertSE3_J
from como.geometry.transforms import transform_points
from como.utils.coords import swap_coords_xy, get_test_coords, normalize_coordinates


# Project all 3D landmarks into each keyframes
# poses (B,4,4), Pw (N,3)
# p (B,N,2), z (B,N,1)
def project_landmarks(Twc, Pw, intrinsics, reinit_P, median_depths):
    # Transform points
    Tcw, dTcw_dTwc = invertSE3_J(Twc)

    # Log-depth projection
    Pc, dPc_dTcw, dPc_dPw = transform_points(Tcw, Pw)
    z = Pc[..., 2:3]

    # Check if log-depth is invalid and then use reinit landmarks
    min_depth = (1e-1) * median_depths
    z_mask = (z < min_depth[:, None, None]).squeeze(2)
    if torch.sum(z_mask) > 0:
        # print("Reinitializing: ", torch.sum(z_mask).item(), " points!")
        # print(median_depths.flatten())
        # print(z[z_mask,:])

        # Transform reinit points
        r_Pc, r_dPc_dTcw, r_dPc_dPw = transform_points(Tcw, reinit_P)
        r_z = r_Pc[..., 2:3]

        # dPc_dPw doesn't depend on point value so ignore
        Pc[z_mask, :] = r_Pc[z_mask, :]
        dPc_dTcw[z_mask, :, :] = r_dPc_dTcw[z_mask, :, :]
        z[z_mask, :] = r_z[z_mask, :]

    logz, dlogz_dz = depth_to_log_depth(z)

    # Pixel projection
    p, dp_dPc = projection(intrinsics, Pc)

    # Depth Jacobians
    dPc_dTwc = dPc_dTcw @ dTcw_dTwc[:, None, :, :]

    # Storing dz_dPw because it's constant for all points in a frame
    # We exploit this later on to speed up Hessian and Jacobian batched computation by keeping dim m instead of 3*m
    dz_dPw = dPc_dPw[:, :, 2:3, :]
    dz_dTwc = dPc_dTwc[:, :, 2:3, :]

    # Pixel Jacobians
    dp_dPw = dp_dPc @ dPc_dPw  # (B,M,2,3)
    dp_dTwc = dp_dPc @ dPc_dTwc  # (B,M,2,6)

    return p, logz, z_mask, dlogz_dz, dz_dPw, dz_dTwc, dp_dPw, dp_dTwc


# Point should project into image and have a positive depth
def get_valid_reproj_mask(p, reproj_log_depth, img_size):
    valid_x = torch.logical_and(p[:, :, 0] >= 1, p[:, :, 0] < img_size[-1] - 1)
    valid_y = torch.logical_and(p[:, :, 1] >= 1, p[:, :, 1] < img_size[-2] - 1)
    valid_mask = torch.logical_and(valid_x, valid_y)
    valid_mask = torch.logical_and(valid_mask, ~reproj_log_depth[..., 0].isnan())
    return valid_mask


# Returns function that maps from all landmarks per keyframe to padded valid ones for each keyframe
def get_batch_remap_function(correspondence_mask):
    # Get mapping from boolean mask (num_kf x num_landmarks) mask to indices in (num_kf x max_depths)
    def landmark_inds_to_batched_inds(mask):
        # Get sequential indices of nonzeros for each keyframe row
        seq_inds = -1 + torch.cumsum(mask, dim=1, dtype=torch.long)
        landmark_inds = torch.nonzero(mask)
        batch_inds = torch.stack(
            (landmark_inds[:, 0], seq_inds[landmark_inds[:, 0], landmark_inds[:, 1]]),
            dim=1,
        )
        # Inds for mapping from landmarks to lower dimension padded max_depths tensors
        paired_landmark_batch_inds = [landmark_inds, batch_inds]
        max_num_depth = torch.max(seq_inds[:, -1]) + 1
        return paired_landmark_batch_inds, max_num_depth

    # Assumes variable has dimensions (num_kf, num_landmarks, ...)
    def to_batched_variable(
        variable, paired_landmark_batch_inds, max_depth, default_val=-1
    ):
        num_kf = variable.shape[0]
        batched_variable = torch.full(
            (num_kf, max_depth) + (variable.shape[2:]),
            default_val,
            device=variable.device,
            dtype=variable.dtype,
        )
        landmark_inds, batch_inds = paired_landmark_batch_inds
        batched_variable[batch_inds[:, 0], batch_inds[:, 1], ...] = variable[
            landmark_inds[:, 0], landmark_inds[:, 1], ...
        ]
        return batched_variable

    paired_landmark_batch_inds, max_num_depth = landmark_inds_to_batched_inds(
        correspondence_mask
    )
    remap_variable_to_batch = lambda variable, default_val: to_batched_variable(
        variable, paired_landmark_batch_inds, max_num_depth, default_val
    )

    return remap_variable_to_batch, paired_landmark_batch_inds


# TODO: Right now deterministic based on gradient magnitude, but could sample weighted or uniformly
def subselect_pixels(kf_img_and_grads, photo_window_size):
    device = kf_img_and_grads.device
    num_kf = kf_img_and_grads.shape[0]

    coords_n_all = get_test_coords(
        kf_img_and_grads.shape[-2:], device=device, batch_size=num_kf
    )

    num_samples = coords_n_all.shape[1] // (photo_window_size**2)

    # Pick pixel with largest gradient magnitude in local window
    c = kf_img_and_grads.shape[1] // 3
    gx = kf_img_and_grads[:, c : 2 * c, :, :]
    gy = kf_img_and_grads[:, 2 * c :, :, :]
    grad_norm = torch.sqrt(
        torch.sum(torch.square(gx) + torch.square(gy), dim=1, keepdim=False)
    )
    max_grad_norm, max_indices = torch.nn.functional.max_pool2d(
        grad_norm, kernel_size=photo_window_size, return_indices=True
    )
    sampled_indices = max_indices.flatten(start_dim=1, end_dim=2)

    num_samples = sampled_indices.shape[1]
    batch_inds = torch.arange(num_kf, device=device).unsqueeze(1).repeat(1, num_samples)
    coords_n_sampled = coords_n_all[batch_inds, sampled_indices, :]

    return coords_n_sampled, batch_inds


def get_gp_training(coords_m, cov_params_imgs, model):
    b, m = coords_m.shape[:2]
    img_size = cov_params_imgs.shape[-2:]
    device = coords_m.device

    # Get training covariance matrix
    coords_m_norm = normalize_coordinates(coords_m, img_size)
    E_m = interpolate_kernel_params(cov_params_imgs, coords_m_norm)
    with torch.no_grad():
        level = -1
        K_mm = model.cov_modules[level](coords_m_norm, E_m)
        K_mm += torch.diag_embed(1e-6 * torch.ones(b, m, device=device))

    # Get training covariance inverse
    L_mm, info = torch.linalg.cholesky_ex(K_mm, upper=False)
    I_mm = torch.eye(m, device=device, dtype=K_mm.dtype).unsqueeze(0).repeat(b, 1, 1)
    K_mm_inv = torch.cholesky_solve(I_mm, L_mm, upper=False)

    return K_mm_inv, L_mm


def get_gp_test(coords_m, K_mm_inv, coords_n, cov_params_imgs, model):
    img_size = cov_params_imgs.shape[-2:]
    coords_m_norm = normalize_coordinates(coords_m, img_size)
    E_m = interpolate_kernel_params(cov_params_imgs, coords_m_norm)
    coords_n_norm = normalize_coordinates(coords_n.to(dtype=K_mm_inv.dtype), img_size)
    E_n = interpolate_kernel_params(cov_params_imgs, coords_n_norm)

    # Calculate training covariance and cross-covariance
    with torch.no_grad():
        level = -1
        K_nm = model.cross_cov_modules[level](coords_n_norm, E_n, coords_m_norm, E_m)

    Knm_Kmminv = K_nm @ K_mm_inv

    return Knm_Kmminv


# Backproject
def backproject_cloud(logz_m, Knm_Kmminv, coords_n, intrinsics):
    logz_n, dlogzn_dlogzm = predict_log_depth(logz_m, Knm_Kmminv)
    z_n, dzn_dlogzn = log_depth_to_depth(logz_n)
    p = swap_coords_xy(coords_n)
    Pc_n, dPcn_dzn = backprojection(intrinsics, p, z_n)

    # Jacobians of camera points
    dPcn_dlogzn = dPcn_dzn @ dzn_dlogzn
    dPcn_dlogzm = dPcn_dlogzn * dlogzn_dlogzm

    return Pc_n, dPcn_dlogzm, dlogzn_dlogzm, logz_n


def setup_point_to_frame(
    Pw_all, Twc, remap_variable_to_batch, K, reinit_P, median_depths
):
    num_kf = Twc.shape[0]

    Pwm = remap_variable_to_batch(Pw_all.unsqueeze(0).repeat(num_kf, 1, 1), -1)
    reinit_Pwm = remap_variable_to_batch(reinit_P.unsqueeze(0).repeat(num_kf, 1, 1), -1)

    pm, logzm, z_mask, dlogzm_dzm, dzm_dPwm, dzm_dTwc, dpm_dPwm, dpm_dTwc = (
        project_landmarks(Twc, Pwm, K[0, ...], reinit_Pwm, median_depths)
    )

    return pm, logzm, z_mask, dlogzm_dzm, dzm_dPwm, dzm_dTwc, dpm_dPwm, dpm_dTwc


def setup_test_points(pm, logzm, Twc, Knm_Kmminv, coords_n, K, dlogzm_dTwc, dlogzm_dzm):
    # Predicted point cloud in camera frame
    Pc_n, dPcn_dlogzm, dlogzn_dlogzm, logzn = backproject_cloud(
        logzm, Knm_Kmminv, coords_n, K[0, ...]
    )
    dPcn_dTwc = dPcn_dlogzm @ dlogzm_dTwc[:, None, :, 0, :]
    dPcn_dzm = dPcn_dlogzm[:, :, :, :, None] * dlogzm_dzm[:, None, None, :, 0, :]

    median_depths = torch.median(Pc_n[:, :, 2], dim=1).values

    # Transform points to world frame
    Pw_n, dPwn_dTwc, dPwn_dPcn = transform_points(Twc, Pc_n)
    # NOTE: dPwn_dPwm currently only has rank 1 (z-z components)
    dPwn_dzm = dPwn_dPcn[:, :, None, :, :] @ torch.permute(dPcn_dzm, (0, 3, 4, 2, 1))
    dPwn_dzm = torch.permute(dPwn_dzm, (0, 4, 3, 1, 2))
    # Dense points wrt keyframe poses
    dPwn_dTwc_full = dPwn_dTwc + (dPwn_dPcn @ dPcn_dTwc)

    return Pw_n, dPwn_dTwc_full, dPwn_dzm, median_depths, dlogzn_dlogzm, logzn


def get_full_cross_cov(coords_m, cov_params_img, model):
    b = coords_m.shape[0]
    img_size = cov_params_img.shape[-2:]
    device = coords_m.device

    # Get training vars
    coords_m_norm = normalize_coordinates(coords_m, img_size)
    E_m = interpolate_kernel_params(cov_params_img, coords_m_norm)

    # Get test vars
    coords_n_all = get_test_coords(img_size, device=device, batch_size=b)
    E_n = torch.reshape(torch.permute(cov_params_img, (0, 2, 3, 1)), (b, -1, 2, 2))
    coords_n_norm = normalize_coordinates(coords_n_all, img_size)

    # Calculate kernel matrices
    with torch.no_grad():
        level = -1
        K_nm = model.cross_cov_modules[level](coords_n_norm, E_n, coords_m_norm, E_m)

    return K_nm


def calc_dense_depth(logz_m, K_mm_inv, K_nm, img_size):
    b = logz_m.shape[0]

    Knm_Kmminv = K_nm @ K_mm_inv

    logz_n, _ = predict_log_depth(logz_m, Knm_Kmminv)
    z_n = torch.exp(logz_n)
    z_img = torch.reshape(
        torch.permute(z_n, (0, 2, 1)), (b, 1, img_size[0], img_size[1])
    )

    return z_img
