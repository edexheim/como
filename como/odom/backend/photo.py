import torch

import como.odom.backend.robust_loss as robust
import como.odom.backend.linear_system as lin_sys

from como.odom.backend.graph_pair_construction import setup_photometric_pairs

from como.geometry.camera import projection
from como.geometry.lie_algebra import invertSE3_J
from como.geometry.transforms import transform_points
from como.utils.coords import swap_coords_xy, normalize_coordinates


# Valid mask (in image and z > 0)
def get_valid_mask(p, z, img_size):
    valid_u = torch.logical_and(p[:, :, 0] >= 1, p[:, :, 0] < img_size[-1] - 1)
    valid_v = torch.logical_and(p[:, :, 1] >= 1, p[:, :, 1] < img_size[-2] - 1)
    valid_z = z[..., 0] > 0
    valid_mask = torch.logical_and(valid_u, valid_v)
    valid_mask = torch.logical_and(valid_mask, valid_z)
    return valid_mask


def interp_img(img_and_grads_j, Pcj, K):
    b, c3, h, w = img_and_grads_j.shape
    c = c3 // 3
    img_size = img_and_grads_j.shape[-2:]
    device = img_and_grads_j.device

    # Projection
    pj, dpj_dPcj = projection(K, Pcj)

    # Interpolate
    coords_j = swap_coords_xy(pj)
    coords_norm = normalize_coordinates(coords_j, img_size)
    pj_norm = swap_coords_xy(coords_norm).unsqueeze(2)
    img_and_grads_interp = torch.nn.functional.grid_sample(
        img_and_grads_j,
        pj_norm,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    img_and_grads_interp = img_and_grads_interp.squeeze(3)  # (B,N,C)

    # Separate into values and gradients
    vals_target = torch.permute(img_and_grads_interp[:, 0:c, ...], (0, 2, 1))
    dIt_dw = torch.stack(
        (
            img_and_grads_interp[:, c : 2 * c, ...],
            img_and_grads_interp[:, 2 * c :, ...],
        ),
        dim=3,
    )
    dIt_dw = torch.permute(dIt_dw, (0, 2, 1, 3))

    # Chain rule for image gradients and camera projection
    dIt_dPcj = torch.matmul(dIt_dw, dpj_dPcj)

    # Valid mask (pixel in image and depth in front)
    valid_mask = get_valid_mask(pj, Pcj[:, :, 2:3], img_size)

    return vals_target, dIt_dPcj, valid_mask


def robustify_system_inplace(r, J_list, invalid_mask, sigma):
    # Noise parameters
    info_sqrt = 1.0 / sigma

    whitened_r = r * info_sqrt
    weight = robust.huber(whitened_r)
    weight[invalid_mask, ...] = 0.0
    weight_sqrt = torch.sqrt(weight)

    r *= info_sqrt * weight_sqrt
    for J in J_list:
        J *= info_sqrt[..., None] * weight_sqrt[..., None]

    total_err = torch.sum(torch.square(weight_sqrt * whitened_r))
    return total_err


def batch_photo_cost(
    vals_i,
    aff_params_i,
    Pwn,
    Twcj,
    aff_params_j,
    img_and_grads_j,
    dPwn_dTwci,
    dPwn_dzm,
    dzm_dPwm,
    pose_ref_inds,
    pose_target_inds,
    landmark_inds,
    intrinsics,
    H,
    g,
):
    b, n, _, m, _ = dPwn_dzm.shape
    device = vals_i.device
    dtype = vals_i.dtype

    # Transform world points into target frame
    Tcw_target, dTcwj_dTwcj = invertSE3_J(Twcj)
    Pcjn, dPcjn_dTcjw, dPcjn_dPwn = transform_points(Tcw_target, Pwn)
    dPcjn_dTwcj = torch.matmul(dPcjn_dTcjw, dTcwj_dTwcj[:, None, :, :])

    # Projective data association observations
    vals_target, dIt_dPcjn, valid_mask = interp_img(img_and_grads_j, Pcjn, intrinsics)
    invalid_mask = torch.logical_not(valid_mask)

    # Residuals
    vals_i_scaled = (
        torch.exp(aff_params_j[:, 0:1, :] - aff_params_i[:, 0:1, :]) * vals_i
    )
    photo_bias = aff_params_j[:, 1:2, :] - aff_params_i[:, 1:2, :]
    r_photo = vals_target - vals_i_scaled + photo_bias

    # Affine Jacobians
    dI_daffi = torch.stack((vals_i_scaled, -torch.ones_like(vals_i_scaled)), dim=-1)
    dI_daffj = -dI_daffi

    # Global median (across all frames)
    r_abs = torch.abs(r_photo[valid_mask])
    med_r = torch.median(r_abs)

    sigma_r = 1.4826 * med_r

    # Robustify
    total_err = robustify_system_inplace(
        r_photo, [dIt_dPcjn, dI_daffi, dI_daffj], invalid_mask, sigma=sigma_r
    )

    # Chain rule for geometry parameters
    dIt_dPwn = dIt_dPcjn @ dPcjn_dPwn
    # Points
    dIt_dzm = dIt_dPwn @ dPwn_dzm.view(
        b, n, 3, m
    )  # (B,N,1,M) <- (B,N,1,3) x (B,N,3,M,1)

    # Batched all pose Jacobians together for efficiency
    dr_ij = torch.empty(
        (2 * b, vals_i.shape[1], vals_i.shape[2], 8), device=device, dtype=dtype
    )
    dr_ij[:b, :, :, 6:] = dI_daffi
    dr_ij[b:, :, :, 6:] = dI_daffj
    dr_ij[:b, :, :, :6] = dIt_dPwn @ dPwn_dTwci
    dr_ij[b:, :, :, :6] = dIt_dPcjn @ dPcjn_dTwcj

    r_dup = torch.cat((r_photo, r_photo), dim=0)
    dr_dzm_dup = torch.cat((dIt_dzm, dIt_dzm), dim=0)
    dzm_dPwm_dup = torch.cat((dzm_dPwm, dzm_dPwm), dim=0)
    pose_inds_dup = torch.cat((pose_ref_inds, pose_target_inds), dim=0)
    landmark_inds_dup = torch.cat((landmark_inds, landmark_inds), dim=0)

    # Poses and geometry cross-terms
    grad_ij = lin_sys.get_gradient(dr_ij, r_dup)
    H_ij_diag = lin_sys.get_hessian_diag_block(dr_ij)

    # Off diag terms (pose-pose and pose-geometry cross terms)
    H_ij_off_diag = lin_sys.get_hessian_off_diag_block(dr_ij[:b, ...], dr_ij[b:, ...])
    H_ij_zm = lin_sys.get_hessian_off_diag_block(dr_ij, dr_dzm_dup)

    # Geometry self terms
    grad_zm = lin_sys.get_gradient(dIt_dzm, r_photo)
    H_zm = lin_sys.get_hessian_diag_block(dIt_dzm)

    # Geometry sleight-of-hand trick
    # dzm_dPwm is a constant matrix for all sparse points in a given image
    # Instead of Sum (J_P^T J_P) we split it into  dP_dz Sum (J_z^T J_z) dz_dP
    # The reduced dimensionality in the batching step is important for efficiency
    # dz_dP is shape (b,1,1,3)
    H_ij_Pwm = torch.reshape(H_ij_zm[:, :, :, None] * dzm_dPwm_dup, (2 * b, 8, 3 * m))
    grad_Pwm = torch.reshape(grad_zm[:, :, None] * dzm_dPwm[:, :, 0, :], (b, 3 * m))
    dz_dP_s = dzm_dPwm[:, 0, 0, :]  # (B,3) small version
    H_Pwm = (
        dz_dP_s[:, None, :, None, None]
        * H_zm[:, :, None, :, None]
        * dz_dP_s[:, None, None, None, :]
    )  # (B,M,3,M,3)
    H_Pwm = torch.reshape(H_Pwm, (b, 3 * m, 3 * m))

    grads_flat = torch.cat((grad_ij.flatten(), grad_Pwm.flatten()))
    grad_inds_flat = torch.cat((pose_inds_dup.flatten(), landmark_inds.flatten()))
    lin_sys.accumulate_gradient_scatter(grads_flat, g, grad_inds_flat)

    W = H.shape[1]
    # Diag inds
    H_inds_ij_diag = lin_sys.row_col_to_lin_index(
        pose_inds_dup[:, :, None], pose_inds_dup[:, None, :], W
    )
    H_inds_Pwm_diag = lin_sys.row_col_to_lin_index(
        landmark_inds[:, :, None], landmark_inds[:, None, :], W
    )
    # Off-diag inds
    H_inds_ij_off_diag = lin_sys.row_col_to_lin_index(
        pose_ref_inds[:, :, None], pose_target_inds[:, None, :], W
    )
    H_inds_ij_off_diag2 = lin_sys.row_col_to_lin_index(
        pose_target_inds[:, :, None], pose_ref_inds[:, None, :], W
    )
    H_inds_ij_Pwm = lin_sys.row_col_to_lin_index(
        pose_inds_dup[:, :, None], landmark_inds_dup[:, None, :], W
    )
    H_inds_ij_Pwm2 = lin_sys.row_col_to_lin_index(
        landmark_inds_dup[:, :, None], pose_inds_dup[:, None, :], W
    )

    # NOTE: Off-diags need both sides of matrix!
    H_flat = torch.cat(
        (
            H_ij_diag.flatten(),
            H_Pwm.flatten(),
            H_ij_off_diag.flatten(),
            H_ij_off_diag.mT.flatten(),
            H_ij_Pwm.flatten(),
            H_ij_Pwm.mT.flatten(),
        )
    )
    H_inds_flat = torch.cat(
        (
            H_inds_ij_diag.flatten(),
            H_inds_Pwm_diag.flatten(),
            H_inds_ij_off_diag.flatten(),
            H_inds_ij_off_diag2.flatten(),
            H_inds_ij_Pwm.flatten(),
            H_inds_ij_Pwm2.flatten(),
        )
    )
    lin_sys.accumulate_hessian_scatter(H_flat, H, H_inds_flat)

    return total_err


def create_photo_system(
    kf_poses,
    kf_aff_params,
    recent_poses,
    recent_aff_params,
    Pwn,
    dPwn_dTwc,
    dPwn_dzm,
    dzm_dPwm,
    median_depths,
    vals_n,
    kf_img_and_grads,
    recent_img_and_grads,
    kf_timestamps,
    recent_timestamps,
    intrinsics,
    H,
    g,
    photo_construction_cfg,
    kf_inds,
    recent_inds,
    landmark_inds,
):
    kf_ref_ids, kf_target_ids, one_way_kf_ids, one_way_target_ids = (
        setup_photometric_pairs(
            kf_poses,
            recent_poses,
            kf_timestamps,
            recent_timestamps,
            median_depths,
            photo_construction_cfg,
        )
    )

    photo_error = 0.0

    # Batch constraints
    all_kf_ids = kf_ref_ids + one_way_kf_ids
    num_kf_pairs = len(kf_target_ids)
    num_total_pairs = len(all_kf_ids)
    num_batches = (
        1 + (num_total_pairs - 1) // photo_construction_cfg["pairwise_batch_size"]
    )
    r_start = num_kf_pairs
    r_end = num_total_pairs

    for b in range(num_batches):
        b1 = b * photo_construction_cfg["pairwise_batch_size"]
        b2 = b1 + photo_construction_cfg["pairwise_batch_size"]

        # Reference vars easy to access since all for keyframes
        ref_ids = all_kf_ids[b1:b2]

        # Handle target vars since one-way need different handling (some keyframe targets, some one way)
        if b2 <= num_kf_pairs:
            target_ids = kf_target_ids[b1:b2]
            target_poses = kf_poses[target_ids]
            target_aff_params = kf_aff_params[target_ids]
            target_img_and_grads = kf_img_and_grads[target_ids]
            target_pose_inds = kf_inds[target_ids]
        else:
            kf_target_ids_batch = kf_target_ids[b1:b2]
            r1 = max(b1, r_start) - r_start
            r2 = min(b2, r_end) - r_start
            recent_target_ids = one_way_target_ids[r1:r2]

            target_poses = torch.cat(
                (
                    kf_poses[kf_target_ids_batch, ...],
                    recent_poses[recent_target_ids, ...],
                ),
                dim=0,
            )
            target_aff_params = torch.cat(
                (
                    kf_aff_params[kf_target_ids_batch, ...],
                    recent_aff_params[recent_target_ids, ...],
                ),
                dim=0,
            )
            target_img_and_grads = torch.cat(
                (
                    kf_img_and_grads[kf_target_ids_batch, ...],
                    recent_img_and_grads[recent_target_ids, ...],
                ),
                dim=0,
            )
            target_pose_inds = torch.cat(
                (
                    kf_inds[kf_target_ids_batch, ...],
                    recent_inds[recent_target_ids, ...],
                ),
                dim=0,
            )

        photo_error += batch_photo_cost(
            vals_n[ref_ids],
            kf_aff_params[ref_ids],
            Pwn[ref_ids],
            target_poses,
            target_aff_params,
            target_img_and_grads,
            dPwn_dTwc[ref_ids],
            dPwn_dzm[ref_ids],
            dzm_dPwm[ref_ids],
            kf_inds[ref_ids],
            target_pose_inds,
            landmark_inds[ref_ids],
            intrinsics[0, ...],
            H,
            g,
        )

    return (
        photo_error,
        [kf_ref_ids, kf_target_ids],
        [one_way_kf_ids, one_way_target_ids],
    )
