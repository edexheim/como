import torch

from como.utils.coords import normalize_coordinates_A


# img is (B, C, H, W)
# sparse coords is (B, N, 2)
# output is (B, C, N)
def img_interp(img, coords, A_norm, mode="bilinear"):
    # Assuming x comes first
    img_size = img.shape
    valid_x = torch.logical_and(
        coords[:, :, 0] >= 1, coords[:, :, 0] < img_size[-1] - 1
    )
    valid_y = torch.logical_and(
        coords[:, :, 1] >= 1, coords[:, :, 1] < img_size[-2] - 1
    )
    valid_mask = torch.logical_and(valid_x, valid_y)

    coords_norm = normalize_coordinates_A(coords, A_norm)

    x_samples = torch.unsqueeze(coords_norm, dim=1)

    # Note: If interpolating gradients, they are zeroed out due to padding mode
    sampled_vals = torch.nn.functional.grid_sample(
        img, x_samples, mode=mode, padding_mode="zeros", align_corners=False
    )
    # Now convert to (B, 3*C, N)
    sampled_vals = torch.squeeze(sampled_vals, dim=2)

    return sampled_vals, valid_mask


def setup_test_coords(img_grads, depth=None, grad_mag_thresh=None):
    b, c2, h, w = img_grads.shape
    device = img_grads.device
    dtype = img_grads.dtype

    if grad_mag_thresh is not None:
        grad_norm = torch.linalg.norm(img_grads, dim=1, keepdim=True)

    if depth is not None and grad_mag_thresh is not None:
        depth_valid = torch.logical_and(depth > 0.0, ~depth.isnan())
        mask = torch.logical_and(depth_valid, grad_norm > grad_mag_thresh)
    elif grad_mag_thresh is not None:
        mask = grad_norm > grad_mag_thresh
    else:
        mask = torch.ones((b, 1, h, w), device=device)

    mask_vec = torch.reshape(mask, (b, -1)).to(dtype)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    coord_img = torch.dstack((y_coords, x_coords)).unsqueeze(0).repeat(b, 1, 1, 1)
    coord_vec = torch.reshape(coord_img, (b, -1, 2))

    num_samples = torch.max(torch.count_nonzero(mask_vec, dim=1))
    inds = torch.multinomial(mask_vec * (1 + 1e-4), num_samples, replacement=False)
    batch_inds = torch.arange(b, device=device).unsqueeze(1).repeat(1, num_samples)
    coords = coord_vec[batch_inds, inds, :]

    return coords
