import torch

from como.depth_cov.core.samplers import sample_sparse_coords
from como.geometry.camera import backprojection, projection
from como.geometry.lie_algebra import invertSE3
from como.geometry.transforms import transform_points
from como.utils.coords import normalize_coordinates, swap_coords_xy, get_test_coords
from como.utils.image_processing import ImageGradientModule

from como.depth_cov.core.distill_depth import (
    distill_depth_from_scratch,
    distill_conditional_depth_from_scratch,
)


# Coords and img_size in row, col form
def filter_reproj_coords(coords, P, img_size, min_depth):
    valid_x = torch.logical_and(
        coords[0, :, 1] >= 1, coords[0, :, 1] < img_size[-1] - 1
    )
    valid_y = torch.logical_and(
        coords[0, :, 0] >= 1, coords[0, :, 0] < img_size[-2] - 1
    )
    valid_mask = torch.logical_and(valid_x, valid_y)
    valid_mask = torch.logical_and(valid_mask, P[0, :, 2] > min_depth)
    coords_filt = coords[:, valid_mask, :]
    P_filt = P[:, valid_mask, :]
    return coords_filt, P_filt, valid_mask


def condition_depth(logz_m, Knm_Kmminv):
    logz_n = Knm_Kmminv @ logz_m
    return logz_n


def reproject_points(coords_i, zi, Tji, K):
    pi = swap_coords_xy(coords_i)
    Pi, _ = backprojection(K[0, ...], pi, zi)
    Pj, _, _ = transform_points(Tji, Pi)
    pj, _ = projection(K[0, ...], Pj)
    coords_j = swap_coords_xy(pj)
    return coords_j, Pj


# Log-depth attenuates errors for far points and penalizes small errors for nearby points
def get_correspondence_errors(P_reproj, P_new, mode):
    if mode == "z":
        err = P_reproj[..., 2:3] - P_new[..., 2:3]
        errors = torch.abs(err)
    elif mode == "logz" or mode == "logr":
        # Same as subtracting log-depths and subtracting log ray distance
        err = torch.log(P_reproj[..., 2:3]) - torch.log(P_new[..., 2:3])
        errors = torch.abs(err)
    elif mode == "3d":
        errors = torch.linalg.norm(P_reproj - P_new, dim=-1, keepdim=True)

    return errors


def track_and_init(
    pose1,
    pose2,
    coords_m1,
    z_m1,
    z_img1,
    cov_params_img2,
    K,
    model,
    corr_params,
    sampling_params,
    rgb_img_size,
    rgb1=None,
    rgb2=None,
):  # TODO: Remove these, just for viz
    device = coords_m1.device
    b, _, h, w = cov_params_img2.shape
    N = rgb_img_size[0] * rgb_img_size[1]
    assert b == 1

    # Project sparse and dense into next frame
    Tji = invertSE3(pose2) @ pose1
    coords_n1 = get_test_coords(z_img1.shape[-2:], device=device, batch_size=b)
    z_n1 = torch.permute(torch.reshape(z_img1, (b, 1, N)), (0, 2, 1))
    coords_j_m1, Pj_m1 = reproject_points(coords_m1, z_m1, Tji, K)
    coords_j_n1, Pj_n1 = reproject_points(coords_n1, z_n1, Tji, K)

    # Remove points outside of boundary
    cov_img_size = cov_params_img2.shape[-2:]
    coords_j_m1_filt, Pj_m1_filt, mask_m1_filt = filter_reproj_coords(
        coords_j_m1, Pj_m1, cov_img_size, corr_params["min_obs_depth"]
    )
    coords_j_n1_filt, Pj_n1_filt, mask_n1_filt = filter_reproj_coords(
        coords_j_n1, Pj_n1, cov_img_size, corr_params["min_obs_depth"]
    )
    zj_n1_filt = Pj_n1_filt[:, :, 2:3]

    # Using next frame's covariance parameters and reprojected sparse coords, solve for latent depths
    logz_m, logz_residuals = distill_depth_from_scratch(
        coords_j_m1_filt,
        coords_j_n1_filt,
        zj_n1_filt,
        cov_params_img2,
        model,
        distill_with_prior=corr_params["distill_with_prior"],
        min_depth=corr_params["min_obs_depth"],
    )
    z_m = torch.exp(logz_m)
    P_m, _ = backprojection(K[0, ...], swap_coords_xy(coords_j_m1_filt), z_m)

    # Check sparse in other direction and interpolate depth
    Tij = invertSE3(Tji)
    coords_i_m1, Pi_m1 = reproject_points(coords_j_m1_filt, z_m, Tij, K)
    coords_i_m1_norm = normalize_coordinates(coords_i_m1, cov_img_size)
    z_proj = torch.nn.functional.grid_sample(
        z_img1,
        swap_coords_xy(coords_i_m1_norm).unsqueeze(1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    z_proj = torch.permute(
        torch.reshape(z_proj, (b, 1, coords_i_m1.shape[1])), (0, 2, 1)
    )
    P_proj, _ = backprojection(K[0, ...], swap_coords_xy(coords_i_m1), z_proj)

    # Check gradient magnitude of reference depth
    image_grad_module = ImageGradientModule(
        channels=1, device=device, dtype=z_img1.dtype
    )
    gx, gy = image_grad_module(torch.log(z_img1))
    grad_mag_logz = torch.sqrt(torch.square(gx) + torch.square(gy))
    coords_m1_filt = coords_m1[:, mask_m1_filt, :]
    coords_m1_filt_norm = normalize_coordinates(coords_m1_filt, cov_img_size)
    grad_mag_ref = torch.nn.functional.grid_sample(
        grad_mag_logz,
        swap_coords_xy(coords_m1_filt_norm).unsqueeze(1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    grad_mag_ref = torch.permute(
        torch.reshape(grad_mag_ref, (b, 1, coords_m1_filt.shape[1])), (0, 2, 1)
    )

    # Old coordinate correspondences denoted by 1, new ones denoted by 2 (coords_m = [coords_1, coords_2])
    # Get correspondences by comparing reprojected and estimated depth
    corr_errors_j = get_correspondence_errors(
        Pj_m1_filt, P_m, mode=corr_params["corr_mode"]
    )
    corr_errors_i = get_correspondence_errors(
        P_proj, Pi_m1, mode=corr_params["corr_mode"]
    )
    corr_errors = torch.maximum(corr_errors_i, corr_errors_j)
    low_grad_mag = grad_mag_ref < corr_params["logz_grad_mag_thresh"]
    corr_mask = corr_errors < corr_params["corr_thresh"]
    corr_mask_filt = torch.logical_and(corr_mask, low_grad_mag)
    corr_mask_filt = corr_mask_filt[0, :, 0]

    coords_1 = coords_j_m1_filt[:, corr_mask_filt, :]
    z1 = Pj_m1_filt[:, corr_mask_filt, 2:3]

    if coords_1.shape[1] > 0:
        with torch.no_grad():
            signal_var = model.get_scale(-1)
            coords_1_keep, domain_inds1_keep = sample_sparse_coords(
                cov_params_img2,
                sampling_params["max_num_coords"],
                "greedy_conditional_entropy",  # sampling_params["mode"],
                sampling_params["max_stdev_thresh"],
                border=sampling_params["border"],
                terminate_early=True,
                dist_thresh=sampling_params["dist_thresh"],
                signal_var=signal_var,
                fixed_var=sampling_params["fixed_var"],
                coords_domain=coords_1,
            )

        # NOTE: Sampling shuffles order of coordinates! So just used sampled mask to index into coords_1, z1
        # Label sampled indices
        sampled_mask = torch.zeros(coords_1.shape[1], device=device, dtype=torch.bool)
        sampled_mask[domain_inds1_keep[0, :]] = True

        # Filter sampled points that we kept
        coords_1 = coords_1[:, sampled_mask, :]
        z1 = z1[:, sampled_mask, :]

        # Label correspondence indices
        corr_mask_filt[corr_mask_filt.clone()] = sampled_mask
        corr_mask = mask_m1_filt.clone()
        corr_mask[mask_m1_filt] = corr_mask_filt
    else:
        coords_1 = coords_j_m1_filt[:, corr_mask_filt, :]
        z1 = Pj_m1_filt[:, corr_mask_filt, 2:3]
        # All tracked are correspondences
        corr_mask = mask_m1_filt.clone()
        corr_mask[mask_m1_filt] = corr_mask_filt

    if coords_1.shape[1] < sampling_params["max_num_coords"]:
        with torch.no_grad():
            # Sample new coords given correspondences
            signal_var = model.get_scale(-1)
            coords_2, _ = sample_sparse_coords(
                cov_params_img2,
                sampling_params["max_num_coords"],
                sampling_params["mode"],
                sampling_params["max_stdev_thresh"],
                border=sampling_params["border"],
                terminate_early=False,
                dist_thresh=sampling_params["dist_thresh"],
                signal_var=signal_var,
                fixed_var=sampling_params["fixed_var"],
                curr_coords=coords_1,
            )

            coords_2 = coords_2.to(dtype=coords_1.dtype)

        sigma_r = torch.std(logz_residuals)

        coords_all = torch.cat((coords_1, coords_2), dim=1)

        # Solved for new depths conditioned on previously observed depths
        logz_2 = distill_conditional_depth_from_scratch(
            coords_all,
            z1,
            coords_j_n1_filt,
            cov_params_img2,
            zj_n1_filt,
            model,
            min_depth=0.0,
            stdev_obs=sigma_r,
        )
        z2 = torch.exp(logz_2)

        z_all = torch.cat((z1, z2), dim=1)

    else:
        coords_all = coords_1.clone()
        z_all = z1.clone()
        coords_2 = torch.empty((1, 0, 2), device=device, dtype=coords_1.dtype)
        z2 = torch.empty((1, 0, 1), device=device, dtype=z1.dtype)

    return coords_2, z2, corr_mask, coords_all, z_all
