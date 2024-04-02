import torch

from como.geometry.depth import log_depth_to_depth, predict_log_depth
from como.geometry.camera import backprojection, projection
from como.geometry.transforms import transform_points
from como.odom.frontend.photo_utils import img_interp, setup_test_coords
import como.odom.backend.robust_loss as robust
from como.utils.image_processing import IntrinsicsPyramidModule
import como.geometry.lie_algebra as lie
from como.utils.coords import normalize_coordinates, swap_coords_xy
import como.depth_cov.core.gaussian_kernel as gk


# Coarse-to-fine where inputs are lists except Tji_init, sparse_log_depth_init
def two_frame_sfm_pyr(
    Tji_init,
    sparse_log_depth_init,
    aff_init,
    test_coords_i,
    vals_i,
    Knm_Kmminv,
    img_and_grads_j,
    dr_prior_dd,
    H_prior_d_d,
    intrinsics,
    sigmas,
    term_criteria,
    init_cfg,
):
    Tji = Tji_init.clone()
    sparse_log_depth = sparse_log_depth_init.clone()
    aff = aff_init.clone()

    num_levels = len(vals_i)
    for l in range(num_levels):
        Tji, sparse_log_depth, aff, coords_j, depths_j, mean_log_depth = two_frame_sfm(
            Tji,
            sparse_log_depth,
            aff,
            test_coords_i[l],
            vals_i[l],
            Knm_Kmminv[l],
            img_and_grads_j[l],
            dr_prior_dd,
            H_prior_d_d,
            intrinsics[l],
            sigmas,
            term_criteria,
            init_cfg,
        )

    return Tji, sparse_log_depth, aff, coords_j, depths_j, mean_log_depth


def setup_reference(
    img_and_grads, sparse_coords_norm, model, cov_params_img, intrinsics
):
    c3 = img_and_grads[-1].shape[-3]
    c = c3 // 3
    device = img_and_grads[-1].device
    dtype = img_and_grads[-1].dtype

    model_level = -1

    intrinsics_pyr_module = IntrinsicsPyramidModule(0, len(img_and_grads), device)
    intrinsics_pyr = intrinsics_pyr_module(intrinsics, [1.0, 1.0])

    E_m = gk.interpolate_kernel_params(cov_params_img, sparse_coords_norm)
    with torch.no_grad():
        K_mm = model.cov_modules[model_level](sparse_coords_norm, E_m)
        L_mm, info = torch.linalg.cholesky_ex(K_mm, upper=False)

    dr_prior_dd, H_prior_d_d = linearize_sparse_depth_prior(L_mm)

    vals_pyr = []
    test_coords_pyr = []
    Knm_Kmminv_pyr = []
    img_sizes_pyr = []
    for i in range(len(img_and_grads)):
        img = img_and_grads[i][:, :c, :, :]
        img_grads = img_and_grads[i][:, c:, :, :]
        test_coords = setup_test_coords(img_grads, depth=None, grad_mag_thresh=None)

        test_coords_norm = normalize_coordinates(
            test_coords.to(dtype=dtype), img.shape[-2:]
        )
        test_coords_pyr.append(test_coords)

        vals = img[:, :, test_coords[0, :, 0], test_coords[0, :, 1]]
        vals_pyr.append(vals)

        E_n = gk.interpolate_kernel_params(cov_params_img, test_coords_norm)
        with torch.no_grad():
            K_mn = model.cross_cov_modules[model_level](
                sparse_coords_norm, E_m, test_coords_norm, E_n
            )

        Kmminv_Kmn = torch.cholesky_solve(K_mn, L_mm, upper=False)
        Knm_Kmminv = torch.transpose(Kmminv_Kmn, dim0=-2, dim1=-1)
        Knm_Kmminv_pyr.append(Knm_Kmminv)

        img_sizes_pyr.append(img.shape[-2:])

    return (
        vals_pyr,
        test_coords_pyr,
        Knm_Kmminv_pyr,
        img_sizes_pyr,
        intrinsics_pyr,
        dr_prior_dd,
        H_prior_d_d,
    )


def linearize_sparse_depth_prior(L_mm):
    B, N_train, _ = L_mm.shape
    device = L_mm.device

    da_dd = torch.eye(N_train, device=device).unsqueeze(0).repeat(B, 1, 1)
    dr_dd = torch.linalg.solve_triangular(L_mm, da_dd, upper=False)
    # Note: This is K_mm_inv
    H_d_d = torch.einsum("hjk,hjl->hkl", dr_dd, dr_dd)

    return dr_dd, H_d_d


def linearize_mean_log_depth_prior_system(Knm_Kmminv):
    B, N, M = Knm_Kmminv.shape

    dr_dd = torch.sum(Knm_Kmminv, dim=(1), keepdim=True) / N
    H_d_d = torch.einsum("hjk,hjl->hkl", dr_dd, dr_dd)

    return dr_dd, H_d_d


def construct_sparse_depth_prior_system(sparse_log_depth_ref, H, g, dr_dd, H_d_d):
    # h(x) - z with z = 0
    a = sparse_log_depth_ref
    r = torch.matmul(dr_dd, a)
    total_err = torch.sum(torch.square(r), dim=(1, 2))
    # Gradient g = -Jt @ r
    g[6:] -= torch.sum(dr_dd * r, dim=(1)).flatten().squeeze(0)
    # H = Jt @ J
    H[6:, 6:] += H_d_d.squeeze(0)

    return total_err


def construct_mean_log_depth_prior_system(log_depth, H, g, dr_dd, H_d_d, sigma):
    info_sqrt = 1.0 / sigma
    info = info_sqrt * info_sqrt

    # h(x) - z with z = 0
    r = torch.mean(log_depth, dim=(1, 2), keepdim=True)
    # print("TwoFrameSfm pred mean: ", r.item())
    r *= info_sqrt
    total_err = torch.sum(torch.square(r), dim=(1, 2))
    # Gradient g = -Jt @ r
    g[6:] -= torch.sum(info_sqrt * dr_dd * r, dim=(1)).flatten()
    # H = Jt @ J
    H[6:, 6:] += info * torch.block_diag(*H_d_d)

    return total_err


def depth_prior(log_depth, mean, H, g, sigma):
    info = 1.0 / (sigma**2)
    r = log_depth - mean
    I = torch.eye(log_depth.shape[1], device=log_depth.device)
    # Gradient g = -Jt @ r
    g[6:] -= info * r[0, :, 0]
    # H = Jt @ J
    H[6:, 6:] += info * I

    total_err = info * torch.sum(torch.square(r))

    return total_err


def linearize_photo(Tji, vals_i, Pi, aff_params, img_and_grads_j, intrinsics):
    c = vals_i.shape[1]

    Pj, dPj_dTji, dPj_dPi = transform_points(Tji, Pi)
    pj, dpj_dPj = projection(intrinsics, Pj)

    A_norm = 1.0 / torch.as_tensor(
        (img_and_grads_j.shape[-1], img_and_grads_j.shape[-2]),
        device=img_and_grads_j.device,
    )

    img_and_grads_interp, valid_mask = img_interp(img_and_grads_j, pj, A_norm)
    valid_mask = torch.logical_and(valid_mask, Pj[..., 2] > 0)
    vals_j = img_and_grads_interp[:, 0:c, ...]
    dIj_dw = torch.stack(
        (
            img_and_grads_interp[:, c : 2 * c, ...],
            img_and_grads_interp[:, 2 * c : 3 * c, ...],
        ),
        dim=3,
    )
    dIj_dw = torch.permute(dIj_dw, (0, 2, 1, 3))

    # Residuals  h(x) - z
    r = vals_j - vals_i
    r = torch.permute(r, (0, 2, 1))

    # Jacobians for pose and points
    dIj_dPj = torch.matmul(dIj_dw, dpj_dPj)
    dIj_dTji = torch.matmul(dIj_dPj, dPj_dTji)
    dIj_dPi = torch.matmul(dIj_dPj, dPj_dPi)

    test_coords_target = swap_coords_xy(pj)
    coords_j = test_coords_target[0:1, valid_mask[0, :], :]
    depths_j = Pj[0:1, valid_mask[0, :], 2:3]

    return r, valid_mask, dIj_dTji, dIj_dPi, coords_j, depths_j


def fill_photo_system(H, g, r, dIj_dTji, dIj_dd):
    g[:6] += -torch.sum(dIj_dTji * r[..., None], dim=(1, 2)).squeeze(0)
    g[6:] += -torch.sum(dIj_dd * r[..., None], dim=(1, 2)).squeeze(0)

    # Diagonal
    H[:6, :6] += torch.einsum("hijk,hijl->hkl", dIj_dTji, dIj_dTji).squeeze(0)
    H[6:, 6:] += torch.einsum("hijk,hijl->hkl", dIj_dd, dIj_dd).squeeze(0)
    # Off-diagonal
    H_Tji_d = torch.einsum("hijk,hijl->hkl", dIj_dTji, dIj_dd).squeeze(0)
    H[:6, 6:] += H_Tji_d
    H[6:, :6] += torch.transpose(H_Tji_d, dim0=0, dim1=1)


def construct_photo_system(
    Tji,
    sparse_log_depth,
    aff,
    test_coords_i,
    vals_i,
    Knm_Kmminv,
    img_and_grads_j,
    intrinsics,
    photo_sigma,
    H,
    g,
):
    # Reference points
    log_depth_i, dlogz_dd = predict_log_depth(sparse_log_depth, Knm_Kmminv)
    z_i, dz_dlogz = log_depth_to_depth(log_depth_i)
    pi = swap_coords_xy(test_coords_i)
    Pi, dPi_dz = backprojection(intrinsics, pi, z_i)
    dPi_dlogz = torch.matmul(dPi_dz, dz_dlogz)
    dPi_dd = torch.matmul(dPi_dlogz, dlogz_dd)

    r, valid_mask, dIj_dTji, dIj_dPi, coords_j, depths_j = linearize_photo(
        Tji, vals_i, Pi, aff, img_and_grads_j, intrinsics
    )

    dIj_dd = torch.matmul(dIj_dPi, dPi_dd)

    # Get sigma
    r_abs = torch.abs(r[valid_mask])
    med_r = torch.median(r_abs)
    sigma_r = 1.4826 * med_r

    invalid_mask = torch.logical_not(valid_mask)
    total_err = robustify_photo(r, invalid_mask, dIj_dTji, dIj_dd, sigma_r)

    fill_photo_system(H, g, r, dIj_dTji, dIj_dd)

    return total_err, log_depth_i, coords_j, depths_j, valid_mask, Pi


def robustify_photo(r, invalid_mask, dIj_dTji, dIj_dd, sigma):
    info_sqrt = 1.0 / sigma
    whitened_r = r * info_sqrt
    weight = robust.huber(whitened_r)
    weight[invalid_mask[...], :] = 0.0
    weight_sqrt = torch.sqrt(weight)

    total_err = torch.sum(torch.square(weight_sqrt * whitened_r))

    r *= info_sqrt * weight_sqrt
    dIj_dTji *= info_sqrt[..., None] * weight_sqrt[..., None]
    dIj_dd *= info_sqrt[..., None] * weight_sqrt[..., None]

    return total_err


def solve_delta(H, g):
    # U, S, Vh = torch.linalg.svd(H)
    # L = torch.linalg.cholesky(H, upper=False)
    L, _ = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
    delta = torch.cholesky_solve(g[:, None], L, upper=False)
    return delta


def update_vars(T, sparse_log_depth, aff, delta):
    delta_T = delta[:6, 0].unsqueeze(0)
    T_new = lie.batch_se3(T, delta_T)

    delta_d = delta[6:]
    sparse_log_depth_new = sparse_log_depth + delta_d

    return T_new, sparse_log_depth_new, aff


def two_frame_sfm(
    Tji_init,
    sparse_log_depth_init,
    aff_init,
    test_coords_i,
    vals_i,
    Knm_Kmminv,
    img_and_grads_j,
    dr_prior_dd,
    H_prior_d_d,
    intrinsics,
    sigmas,
    term_criteria,
    init_cfg,
):
    device = Tji_init.device
    dtype = Tji_init.dtype

    N_train = sparse_log_depth_init.shape[1]
    D = 6 + N_train  # Pose, aff, N points

    Tji = Tji_init.clone()
    sparse_log_depth = sparse_log_depth_init.clone()
    aff = aff_init.clone()

    # Precomputate linearizations
    dr_mean_dd, H_mean_d_d = linearize_mean_log_depth_prior_system(Knm_Kmminv)

    iter = 0
    done = False
    total_err_prev = float("inf")
    while not done:
        H = torch.zeros((D, D), device=device, dtype=dtype)
        g = torch.zeros((D), device=device, dtype=dtype)

        photo_err, log_depth, coords_j, depths_j, valid_mask, Pi = (
            construct_photo_system(
                Tji,
                sparse_log_depth,
                aff,
                test_coords_i,
                vals_i,
                Knm_Kmminv,
                img_and_grads_j,
                intrinsics,
                sigmas["photo"],
                H,
                g,
            )
        )

        depth_prior_err = construct_sparse_depth_prior_system(
            sparse_log_depth, H, g, dr_prior_dd, H_prior_d_d
        )
        mean_depth_err = construct_mean_log_depth_prior_system(
            log_depth, H, g, dr_mean_dd, H_mean_d_d, sigma=1e0
        )

        total_err = photo_err + depth_prior_err + mean_depth_err
        delta = solve_delta(H, g)
        Tji_new, sparse_log_depth_new, aff_new = update_vars(
            Tji, sparse_log_depth, aff, delta
        )
        Tji = Tji_new
        sparse_log_depth = sparse_log_depth_new
        aff = aff_new

        iter += 1
        delta_norm = torch.norm(delta[:6])
        abs_decrease = total_err_prev - total_err
        rel_decrease = (
            torch.abs(abs_decrease) / total_err_prev
        )  # Check minor changes instead of decrease
        # print(iter, total_err.item(), delta_norm.item(), rel_decrease.item())
        if (
            iter >= init_cfg["max_iter"]
            or delta_norm < init_cfg["delta_norm"]
            or (rel_decrease < init_cfg["rel_tol"] and abs_decrease > 0)
        ):
            # or abs_decrease < term_criteria["abs_tol"] \
            done = True

        total_err_prev = total_err

    mean_log_depth = torch.mean(log_depth, dim=(1, 2), keepdim=True)

    return Tji, sparse_log_depth, aff, coords_j, depths_j, mean_log_depth
