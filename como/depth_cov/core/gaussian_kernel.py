import torch
import numpy as np


# det_eps should be less than min(x*z)!
def normalize_params_cov(kernel_img, det_eps=1e-8, corr_coeff_max=0.99):
    kernel_img_norm = torch.empty(
        (kernel_img.shape[0], 3, kernel_img.shape[2], kernel_img.shape[3]),
        device=kernel_img.device,
    )
    diag_scale = 1.0
    x = torch.clamp(kernel_img[:, 0, :, :], min=np.log(1e-3), max=np.log(1e4))
    z = torch.clamp(kernel_img[:, 1, :, :], min=np.log(1e-3), max=np.log(1e4))
    x = diag_scale * torch.exp(x)
    z = diag_scale * torch.exp(z)
    corr_coeff = corr_coeff_max * torch.tanh(kernel_img[:, 2, :, :])
    kernel_img_norm[:, 0, :, :] = x
    kernel_img_norm[:, 1, :, :] = z
    # Want determinant > thresh
    # |E| = x*z*(1-p^2)
    kernel_img_norm[:, 2, :, :] = torch.sqrt(x * z - det_eps) * corr_coeff
    return kernel_img_norm


def get_kernel_mats_cov(kernel_params):
    device = kernel_params.device
    b, n, _ = kernel_params.shape

    E = torch.empty((b, n, 2, 2), device=device, dtype=kernel_params.dtype)
    E[:, :, 0, 0] = kernel_params[:, :, 0]
    E[:, :, 1, 1] = kernel_params[:, :, 1]
    E[:, :, 0, 1] = kernel_params[:, :, 2].clone()
    E[:, :, 1, 0] = kernel_params[:, :, 2].clone()
    return E


def kernel_params_to_covariance(kernel_img_norm):
    B = kernel_img_norm.shape[0]
    C = kernel_img_norm.shape[1]
    H = kernel_img_norm.shape[2]
    W = kernel_img_norm.shape[3]

    kernel_img_tmp = torch.reshape(kernel_img_norm, (B, 3, -1))
    kernel_img_tmp = torch.permute(kernel_img_tmp, (0, 2, 1))
    E = get_kernel_mats_cov(kernel_img_tmp)
    E = torch.reshape(E, (B, H, W, 4))
    E = torch.permute(E, (0, 3, 1, 2))

    return E


def interpolate_kernel_params(kernel_img, x):
    # x coordinates are normalized [-1,1]
    # NOTE: grid_sample expects (x,y) as in image coordinates (so column then row)
    x_samples = torch.unsqueeze(x, dim=1)
    ind_swap = torch.tensor([1, 0], device=kernel_img.device)
    x_samples = torch.index_select(x_samples, 3, ind_swap)

    assert kernel_img.shape[1] == 4

    # kernel_image shape: B x 3 x H x W
    # x shape: B x N x 2
    # output shape: B x 3 x N
    B = kernel_img.shape[0]
    N = x.shape[1]

    # Get sampled features
    sampled_params = torch.nn.functional.grid_sample(
        kernel_img,
        x_samples,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled_params = torch.permute(torch.squeeze(sampled_params, dim=2), (0, 2, 1))

    kernel_mats = torch.reshape(sampled_params, (B, N, 2, 2))

    return kernel_mats
