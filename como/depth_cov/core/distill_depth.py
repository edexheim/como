import torch

from como.depth_cov.core.gaussian_kernel import interpolate_kernel_params
from como.utils.coords import normalize_coordinates
from como.utils.lin_alg import lstsq_chol


def calc_kernel_matrices(coords_m, coords_n, cov_params_img, model):
    img_size = cov_params_img.shape[-2:]

    # Get training vars
    coords_m_norm = normalize_coordinates(coords_m, img_size)
    E_m = interpolate_kernel_params(cov_params_img, coords_m_norm)

    # Get test vars
    coords_n_norm = normalize_coordinates(coords_n, img_size)
    E_n = interpolate_kernel_params(cov_params_img, coords_n_norm)

    # Calculate training covariance and cross-covariance
    level = -1
    with torch.no_grad():
        K_mm = model.cov_modules[level](coords_m_norm, E_m)
        K_nm = model.cross_cov_modules[level](coords_n_norm, E_n, coords_m_norm, E_m)
        K_nn_diag = model.diagonal_cov_modules[level](coords_n_norm, E_n)

    return K_mm, K_nm, K_nn_diag


def get_predictor(K_mm, K_nm, K_nn_diag):
    # Get training covariance inverse
    L_mm, info = torch.linalg.cholesky_ex(K_mm, upper=False)
    M = L_mm.shape[-1]
    I_mm = (
        torch.eye(M, device=K_mm.device, dtype=K_mm.dtype).unsqueeze(0).repeat(1, 1, 1)
    )
    K_mm_inv = torch.cholesky_solve(I_mm, L_mm, upper=False)

    # Dense image predictor
    Knm_Kmminv = K_nm @ K_mm_inv

    # Dense image variance
    # NOTE: Assuming no point variance, so need to clamp 0
    var_n = K_nn_diag - torch.sum(K_nm * Knm_Kmminv, dim=2)
    var_n += torch.min(var_n) + 1e-8
    var_n = var_n.unsqueeze(-1)
    stdev_inv_n = 1.0 / torch.sqrt(var_n)

    return Knm_Kmminv, L_mm, stdev_inv_n


# argmax p(dn|dm) p(dm)
def distill_depth(Knm_Kmminv, z_obs, with_prior, L_mm=None, stdev_inv_obs=None):
    batch_size, n, m = Knm_Kmminv.shape
    device = Knm_Kmminv.device
    dtype = Knm_Kmminv.dtype

    logz_obs = torch.log(z_obs)

    if not with_prior:
        logz_m = lstsq_chol(Knm_Kmminv, logz_obs)
    else:
        A = torch.empty((batch_size, m + n, m), device=device, dtype=dtype)
        identity_m = (
            torch.eye(m, device=device, dtype=dtype)
            .reshape((1, m, m))
            .repeat(batch_size, 1, 1)
        )
        L_inv = torch.linalg.solve_triangular(L_mm, identity_m, upper=False)
        # Prior
        A[:, :m, :m] = L_inv
        # Conditional
        A[:, m:, :m] = stdev_inv_obs * Knm_Kmminv

        b = torch.empty((batch_size, m + n, 1), device=device, dtype=dtype)
        b[:, :m, :] = 0.0
        b[:, m:, :] = stdev_inv_obs * logz_obs

        logz_m = lstsq_chol(A, b)

    logz_residuals = Knm_Kmminv @ logz_m - logz_obs

    return logz_m, logz_residuals


# Handles kernel matrix calculation and depth filtering
def distill_depth_from_scratch(
    coords_m,
    coords_n,
    z_obs,
    cov_params_img,
    model,
    distill_with_prior,
    min_depth,
    stdev_obs=None,
):
    assert coords_m.shape[0] == 1

    # Calculate kernel matrices for first image
    K_mm, K_nm, K_nn_diag = calc_kernel_matrices(
        coords_m, coords_n, cov_params_img, model
    )
    Knm_Kmminv, L_mm, stdev_inv_n = get_predictor(K_mm, K_nm, K_nn_diag)

    # Overwrite stdev if given
    if stdev_obs is not None:
        stdev_inv_n = (1.0 / stdev_obs) * torch.ones_like(stdev_inv_n)

    # Initiaize inducing depths with ground-truth supervision (just for init)
    valid_depth_mask = z_obs[0, :, 0] > min_depth
    logz_m1, logz_residuals = distill_depth(
        Knm_Kmminv[:, valid_depth_mask, :],
        z_obs[:, valid_depth_mask, :],
        distill_with_prior,
        L_mm=L_mm,
        stdev_inv_obs=stdev_inv_n[:, valid_depth_mask, :],
    )

    return logz_m1, logz_residuals


# Instead of p(d2|d1) prior, just promote p(d2) close to median depth
def distill_conditional_depth_with_scale_prior(Knm_Kmminv, z_obs, z1, stdev_inv_obs):
    batch_size, n, m = Knm_Kmminv.shape
    assert batch_size == 1
    m1 = z1.shape[1]
    m2 = m - m1
    device = Knm_Kmminv.device
    dtype = Knm_Kmminv.dtype

    s = torch.log(torch.median(z_obs))

    sigma_median = 5e-2
    stdev_inv_prior = 1.0 / sigma_median

    logz_obs = torch.log(z_obs)
    logz1 = torch.log(z1)

    A = torch.empty((batch_size, m2 + n, m2), device=device, dtype=dtype)
    A[:, :m2, :] = stdev_inv_prior * torch.eye(m2, device=device, dtype=dtype)
    A[:, m2:, :] = stdev_inv_obs * Knm_Kmminv[:, :, m1:]

    b = torch.empty((batch_size, m2 + n, 1), device=device, dtype=dtype)
    b[:, :m2, :] = stdev_inv_prior * s
    b[:, m2:, :] = stdev_inv_obs * (logz_obs - Knm_Kmminv[:, :, :m1] @ logz1)

    logz2 = lstsq_chol(A, b)

    return logz2


# Handles kernel matrix calculation and depth filtering
def distill_conditional_depth_from_scratch(
    coords_m, z_m1, coords_n, cov_params_img, z_obs, model, min_depth, stdev_obs
):
    assert coords_m.shape[0] == 1

    # Calculate kernel matrices for first image
    K_mm, K_nm, K_nn_diag = calc_kernel_matrices(
        coords_m, coords_n, cov_params_img, model
    )
    Knm_Kmminv, L_mm, stdev_inv_n = get_predictor(K_mm, K_nm, K_nn_diag)

    stdev_inv_n = (1.0 / stdev_obs) * torch.ones_like(stdev_inv_n)

    # Initiaize inducing depths with ground-truth supervision (just for init)
    valid_depth_mask = z_obs[0, :, 0] > min_depth

    logz_m2 = distill_conditional_depth_with_scale_prior(
        Knm_Kmminv[:, valid_depth_mask, :],
        z_obs[:, valid_depth_mask, :],
        z_m1,
        stdev_inv_n[:, valid_depth_mask, :],
    )

    return logz_m2
