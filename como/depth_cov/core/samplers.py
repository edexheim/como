import torch

import como.depth_cov.core.gaussian_kernel as gk
from como.utils.coords import normalize_coordinates

import como_backends


def get_coords_domain(cov_params_img, border=0):
    b, c, h, w = cov_params_img.shape
    device = cov_params_img.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.long, device=device),
        torch.arange(w, dtype=torch.long, device=device),
        indexing="ij",
    )
    coord_img = torch.dstack((y_coords, x_coords))
    coord_img = coord_img[border : h - border, border : w - border, :]
    coord_vec = torch.reshape(coord_img, (-1, 2))
    coord_vec = coord_vec.unsqueeze(0).repeat(b, 1, 1)

    return coord_vec


def get_cov_domain(coord_vec, cov_params_img):
    b, c, h, w = cov_params_img.shape
    gaussian_covs_vec = cov_params_img[:, :, coord_vec[0, :, 0], coord_vec[0, :, 1]]
    gaussian_cov_mats = torch.permute(gaussian_covs_vec, (0, 2, 1))
    gaussian_cov_mats = torch.reshape(gaussian_cov_mats, (b, -1, 2, 2)).contiguous()

    return gaussian_cov_mats


# Returns normalized coordinates
def sample_sparse_coords(
    cov_params_img,
    num_samples,
    mode,
    max_stdev_thresh=-1e8,
    border=0,
    terminate_early=False,
    dist_thresh=0.0,
    signal_var=None,
    fixed_var=None,
    curr_coords=None,
    curr_var=None,
    coords_domain=None,
    dtype=torch.float,
):
    b = cov_params_img.shape[0]
    img_size = cov_params_img.shape[-2:]

    device = cov_params_img.device
    orig_dtype = cov_params_img.dtype

    # TODO: Remove these dtype conversions?
    cov_params_img = cov_params_img.to(device=device, dtype=dtype)

    if curr_coords is None:
        curr_coords = torch.empty((b, 0, 2), device=device, dtype=dtype)

    if curr_var is None:
        curr_var = torch.zeros((b, 0), device=device, dtype=dtype)

    if coords_domain is None:
        coords_domain = get_coords_domain(cov_params_img, border=border)
        coords_domain_norm = normalize_coordinates(coords_domain, img_size).to(dtype)
        E_domain = get_cov_domain(coords_domain, cov_params_img)
    else:
        coords_domain_norm = normalize_coordinates(coords_domain, img_size).to(dtype)
        E_domain = gk.interpolate_kernel_params(cov_params_img, coords_domain_norm)

    if mode == "random_uniform":
        num_curr_coords = curr_coords.shape[-2]
        num_samples_remaining = num_samples - num_curr_coords
        # NOTE: Might still sample redundant coordinates?
        new_coord_vec_inds = random_uniform(num_samples_remaining, coords_domain_norm)
    elif mode == "greedy_conditional_entropy":
        num_samples_clamp = min(num_samples, coords_domain.shape[1])
        curr_coords_norm = normalize_coordinates(curr_coords, img_size).to(dtype)
        new_coord_vec_inds = greedy_conditional_entropy(
            cov_params_img,
            E_domain,
            num_samples_clamp,
            coords_domain_norm,
            curr_coords_norm,
            curr_var,
            fixed_var,
            signal_var,
            max_stdev_thresh,
            terminate_early,
            dist_thresh,
        )
    else:
        raise ValueError("sample_sparse_coords mode: " + mode + " is not implemented.")

    # Put things back in original dtype
    cov_params_img = cov_params_img.to(dtype=orig_dtype)

    domain_inds = new_coord_vec_inds[:, new_coord_vec_inds[0, :] >= 0]
    batch_inds = (
        torch.arange(b, device=device).unsqueeze(1).repeat(1, domain_inds.shape[1])
    )
    new_coords_sampled = coords_domain[batch_inds, domain_inds, :]

    return new_coords_sampled, domain_inds


def random_uniform(n, coords_domain_norm):
    device = coords_domain_norm.device
    weights = torch.ones((coords_domain_norm.shape[:-1]), device=device)
    coord_vec_inds = torch.multinomial(weights, n, replacement=False)
    return coord_vec_inds


def get_obs_info(L, K_mn):
    obs_info = torch.linalg.solve_triangular(L, K_mn, upper=False)
    return obs_info


def calc_var(obs_info, K_diag):
    return K_diag - torch.sum(obs_info * obs_info, dim=1)


def precalc_entropy_vars(
    E_domain,
    gaussian_covs,
    n,
    coords_domain_norm,
    curr_coords_norm,
    curr_var,
    fixed_var,
    scale,
):
    b, m, _ = curr_coords_norm.shape
    device = E_domain.device
    dtype = E_domain.dtype

    domain_size = coords_domain_norm.shape[-2]

    # Incrementally updating variables
    coord_vec_inds = torch.empty((b, n), device=device, dtype=torch.long)
    coords_n_norm = torch.empty((b, n, 2), device=device, dtype=dtype)
    E_n = torch.empty((b, n, 2, 2), device=device, dtype=dtype)
    # Identity allows us to solve triangular system of 0s without worrying about effect on sums (solution is 0s)
    L = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
    obs_info = torch.zeros((b, n, domain_size), device=device, dtype=dtype)

    if m > 0:  # Existing coords
        coord_vec_inds[:, :m] = -1
        coords_n_norm[:, 0:m, :] = curr_coords_norm
        E_n[:, :m, :, :] = gk.interpolate_kernel_params(gaussian_covs, curr_coords_norm)
    else:
        # Heuristic: Pick valid point with largest determinant
        cov_areas = (
            E_domain[..., 0, 0] * E_domain[..., 1, 1]
            - E_domain[..., 0, 1] * E_domain[..., 1, 0]
        )
        best_inds = torch.argmax(cov_areas.view(b, -1), dim=1)
        batch_inds = torch.arange(b, device=device)
        coord_vec_inds[:, 0] = best_inds
        coords_n_norm[:, 0, :] = coords_domain_norm[batch_inds, best_inds, :]
        E_n[:, 0, :, :] = E_domain[batch_inds, best_inds, :, :]
        m = 1

    K_nn = como_backends.cross_covariance(
        coords_n_norm[:, :m, :],
        E_n[:, :m, :, :],
        coords_n_norm[:, :m, :].clone(),
        E_n[:, :m, :, :].clone(),
        scale,
    )
    # Add uncertainty of current known coordinates to diagonal
    if curr_var.shape[1] > 0:
        assert curr_var.shape[1] == curr_coords_norm.shape[1]
        K_nn += torch.diag_embed(curr_var)
    # Add fixed variance if given
    if fixed_var is not None:
        K_nn += torch.diag_embed(fixed_var * torch.ones(b, m, device=device))

    L[:, :m, :m] = torch.linalg.cholesky(K_nn, upper=False)

    K_md_init = como_backends.cross_covariance(
        coords_n_norm[:, :m, :],
        E_n[:, :m, :, :],
        coords_domain_norm.view(b, -1, 2),
        E_domain,
        scale,
    )
    obs_info[:, :m, :] = get_obs_info(L[:, :m, :m], K_md_init)

    return coord_vec_inds, coords_n_norm, E_n, L, obs_info, m


def greedy_loop(
    coord_vec_inds,
    coords_n_norm,
    E_n,
    coords_domain_norm,
    E_domain,
    L,
    obs_info,
    m,
    n,
    signal_var,
    fixed_var,
    max_stdev_thresh,
    terminate_early,
    dist_thresh,
):
    device = coords_n_norm.device
    b = coords_n_norm.shape[0]

    batch_inds = torch.arange(b, device=device)

    dist_thresh_sq = dist_thresh * dist_thresh

    def get_next_inds(gp_var, gp_stdev_thresh, coords_curr_norm):
        # Variances can be slightly negative which creates nans for sqrt
        gp_stdev = torch.sqrt(gp_var)
        nan_mask = gp_stdev.isnan()
        gp_stdev[nan_mask] = 0.0
        gp_stdev += 1e-10

        dists_sq = torch.sum(
            torch.square(
                coords_curr_norm[:, :, None, :] - coords_domain_norm[:, None, :, :]
            ),
            dim=-1,
        )
        valid_dist_mask = (dists_sq > dist_thresh_sq).all(dim=1)

        # Filter out points close to others (nonmax suppression)
        cost = gp_stdev * valid_dist_mask
        best_inds = torch.argmax(cost, dim=1)
        max_gp_stdev = gp_stdev[batch_inds, best_inds]

        return max_gp_stdev, best_inds

    pred_var = calc_var(obs_info[:, :m, :], signal_var)
    i = m - 1
    max_gp_stdev, best_inds = get_next_inds(
        pred_var, max_stdev_thresh, coords_n_norm[:, : (i + 1), :]
    )

    for i in range(m, n):
        # Check for early termination
        # For batching, all batches must reach threshold to terminate early
        if terminate_early:
            all_batches_done = (max_gp_stdev < max_stdev_thresh).all()
            if all_batches_done:
                coord_vec_inds = coord_vec_inds[:, :i]
                coords_n_norm = coords_n_norm[:, :i, :]
                break

        # Add new point/cov
        coord_i_norm = coords_domain_norm[batch_inds, best_inds, :]
        E_i = E_domain[batch_inds, best_inds, :, :]
        coord_vec_inds[:, i] = best_inds
        coords_n_norm[:, i, :] = coord_i_norm
        E_n[:, i, :, :] = E_i

        coord_i_norm = coord_i_norm.unsqueeze(1)
        E_i = E_i.unsqueeze(1)
        k_ni = como_backends.cross_covariance(
            coords_n_norm[:, 0:i, :], E_n[:, 0:i, :, :], coord_i_norm, E_i, signal_var
        )
        k_id = como_backends.cross_covariance(
            coord_i_norm, E_i, coords_domain_norm, E_domain, signal_var
        )
        k_ii = signal_var.clone()  # Need this clone
        if fixed_var is not None:
            k_ii += fixed_var

        como_backends.get_new_chol_obs_info(L, obs_info, pred_var, k_ni, k_id, k_ii, i)

        max_gp_stdev, best_inds = get_next_inds(
            pred_var, max_stdev_thresh, coords_n_norm[:, : (i + 1), :]
        )

    return coord_vec_inds


def greedy_conditional_entropy(
    gaussian_covs,
    E_domain,
    n,
    coords_domain_norm,
    curr_coords_norm,
    curr_var,
    fixed_var,
    signal_var,
    max_stdev_thresh,
    terminate_early,
    dist_thresh,
):
    coord_vec_inds, coords_n_norm, E_n, L, obs_info, m = precalc_entropy_vars(
        E_domain,
        gaussian_covs,
        n,
        coords_domain_norm,
        curr_coords_norm,
        curr_var,
        fixed_var,
        signal_var,
    )

    coord_vec_inds = greedy_loop(
        coord_vec_inds,
        coords_n_norm,
        E_n,
        coords_domain_norm,
        E_domain,
        L,
        obs_info,
        m,
        n,
        signal_var,
        fixed_var,
        max_stdev_thresh,
        terminate_early,
        dist_thresh,
    )

    return coord_vec_inds
