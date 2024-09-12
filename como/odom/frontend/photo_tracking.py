import torch

from como.geometry.camera import projection, transform_project
from como.geometry.lie_algebra import se3_exp, skew_symmetric
from como.odom.frontend.photo_utils import img_interp
import como.odom.backend.robust_loss as robust


# Coarse-to-fine where inputs are lists except Tji_init
def photo_tracking_pyr(
    Tji_init,
    aff_init,
    vals_i,
    Pi,
    dI_dT,
    intrinsics,
    img_j,
    photo_sigma,
    term_criteria,
):
    Tji = Tji_init.clone()
    aff = aff_init.clone()
    num_levels = len(vals_i)
    for l in range(num_levels):
        vals_l = vals_i[l]
        P_l = Pi[l]
        dI_dT_l = dI_dT[l]
        Tji, aff = photo_level_tracking(
            Tji,
            aff,
            vals_l,
            P_l,
            dI_dT_l,
            img_j[l],
            intrinsics[l],
            photo_sigma,
            term_criteria,
        )

    return Tji, aff


# IC precalculate Jacobians at theta=0
def precalc_jacobians(dI_dw, P, vals, intrinsics):
    c = vals.shape[2]
    device = dI_dw.device
    dtype = dI_dw.dtype

    b, n, _ = P.shape
    dPi_dT = torch.empty((b, n, 3, 6), device=device, dtype=dtype)
    dPi_dT[:, :, :, 3:] = (
        torch.eye(3, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(b, n, 1, 1)
    )
    dPi_dT[:, :, :, :3] = -skew_symmetric(P)

    _, dpi_dPi = projection(intrinsics, P)
    dpi_dT = torch.matmul(dpi_dPi, dPi_dT)
    dI_dT = torch.matmul(dI_dw, dpi_dT)

    dI_dp = torch.cat(
        (
            dI_dT,
            vals.unsqueeze(-1),
            torch.ones((dI_dT.shape[0], n, c, 1), device=device, dtype=dtype),
        ),
        dim=-1,
    )

    return dI_dp


def robustify_photo(r, dIt_dT, invalid_mask, photo_sigma):
    info_sqrt = 1.0 / photo_sigma
    whitened_r = r * info_sqrt
    weight = robust.huber(whitened_r)
    weight[invalid_mask[...], :] = 0.0

    total_err = torch.sum(weight * torch.square(whitened_r))
    num_valid = invalid_mask.shape[-1] - torch.count_nonzero(invalid_mask, dim=-1)
    mean_sq_err = total_err / num_valid

    J_W = weight[:, :, :, None] * dIt_dT
    grad = torch.sum(J_W * r[..., None], dim=(1, 2))
    H = torch.einsum("bnck,bncl->bkl", J_W, dIt_dT)

    grad_norm = torch.linalg.norm(grad)

    return H, grad, total_err, mean_sq_err, grad_norm


def solve_delta(H, grad):
    L, _ = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
    delta = torch.cholesky_solve(grad[..., None], L, upper=False)
    return delta


# TODO: Batch
def update_pose_ic(T, aff, delta):
    delta_T = delta[:, :6, 0]
    T_new = torch.matmul(T, se3_exp(-delta_T))

    delta_a = delta[:, 6, 0]
    delta_b = delta[:, 7, 0]
    aff_new = torch.empty_like(aff)

    aff_new[:, 0] = aff[:, 0] - delta_a
    aff_new[:, 1] = aff[:, 1] - delta_b

    return T_new, aff_new


def tracking_iter(Tji, Pi, intrinsics, img_j, aff, vals_i, dI_dT, photo_sigma, A_norm):
    pj, depth_j = transform_project(intrinsics, Tji, Pi)

    vals_target, valid_mask = img_interp(img_j, pj, A_norm)
    valid_mask = torch.logical_and(valid_mask, depth_j[..., 0] > 0)
    invalid_mask = torch.logical_not(valid_mask)

    tmp = torch.exp(-aff[:, None, 0]) * vals_target
    dI_dT[..., 6] = torch.permute(-tmp, (0, 2, 1))
    vals_target = tmp + aff[:, None, 1]

    vals_ref = torch.permute(vals_i, (0, 2, 1))
    r = vals_target - vals_ref
    r = torch.permute(r, (0, 2, 1))

    r_abs = torch.abs(r[valid_mask])
    med_r = torch.median(r_abs)
    sigma_r = 1.4826 * med_r

    H, grad, total_err, mean_sq_err, grad_norm = robustify_photo(
        r, dI_dT, invalid_mask, sigma_r
    )

    delta = solve_delta(H, grad)
    Tji_new, aff_new = update_pose_ic(Tji, aff, delta)

    return Tji_new, aff_new, delta, mean_sq_err, grad_norm, pj, valid_mask, depth_j


# Inverse compositional tracking
def photo_level_tracking(
    Tji_init, aff_init, vals_i, Pi, dI_dT, img_j, intrinsics, photo_sigma, term_criteria
):
    Tji = Tji_init.clone()
    aff = aff_init.clone()

    A_norm = 1.0 / torch.as_tensor(
        (img_j.shape[-1], img_j.shape[-2]), device=img_j.device, dtype=img_j.dtype
    )

    iter = 0
    done = False
    mean_sq_err_prev = float("inf")
    while not done:
        Tji, aff, delta, mean_sq_err, grad_norm, p_j, valid_reproj_mask, depth_j = (
            tracking_iter(
                Tji, Pi, intrinsics, img_j, aff, vals_i, dI_dT, photo_sigma, A_norm
            )
        )

        iter += 1
        delta_norm = torch.norm(delta)
        abs_decrease = mean_sq_err_prev - mean_sq_err

        # NOTE: Checking for convergence, not if error goes up, so want absolute value!
        rel_decrease = torch.abs(abs_decrease / mean_sq_err_prev)

        # print("Tracking: ", iter, mean_sq_err.item(), delta_norm.item(), rel_decrease.item(), grad_norm.item())
        if (
            iter >= term_criteria["max_iter"]
            or delta_norm < term_criteria["delta_norm"]
            or rel_decrease < term_criteria["rel_tol"]
            or grad_norm < term_criteria["grad_norm"]
        ):
            done = True
            # print(iter, abs_decrease, delta_norm, rel_decrease)
        mean_sq_err_prev = mean_sq_err

    return Tji, aff
