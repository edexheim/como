import torch
from como.utils.coords import get_coord_img


# log_depth (B, N, 1)
def log_depth_to_depth(log_depth):
    depth = torch.exp(log_depth)
    dz_dlogz = depth.unsqueeze(-1)
    return depth, dz_dlogz


# depth (B, N, 1)
def depth_to_log_depth(depth):
    log_depth = torch.log(depth)
    dlogz_dz = (1.0 / depth).unsqueeze(-1)
    return log_depth, dlogz_dz


# log_depth_train (B, N_train, 1)
# Knm_Kmminv (B, N_test, N_train)
def predict_log_depth(logz_m, Knm_Kmminv):
    logz_n = Knm_Kmminv @ logz_m
    dlogz_dlogz_m = Knm_Kmminv.unsqueeze(-2)
    return logz_n, dlogz_dlogz_m


def backproject_depth_img(depth_img, K):
    coord_img = get_coord_img(
        depth_img.shape[-2:], device=depth_img.device, batch_size=depth_img.shape[0]
    )
    rx = (coord_img[..., 1] - K[0, 2]) / K[0, 0]
    ry = (coord_img[..., 0] - K[1, 2]) / K[1, 1]
    r = torch.stack((rx, ry, torch.ones_like(rx)), dim=1)
    P = depth_img * r
    return P
