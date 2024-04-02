import torch


def swap_coords_xy(coords):
    coords_swap = torch.empty_like(coords)
    coords_swap[..., 0] = coords[..., 1]
    coords_swap[..., 1] = coords[..., 0]
    return coords_swap


# Transforms pixel coordiantes to [-1,1] assuming pixels are at fractional coordinates
def normalize_coordinates(x_pixel, dims):
    A = 1.0 / torch.as_tensor(dims, device=x_pixel.device, dtype=x_pixel.dtype)
    x_norm = 2 * A * x_pixel + A - 1
    return x_norm


def normalize_coordinates_A(x_pixel, A):
    x_norm = 2 * A * x_pixel + A - 1
    return x_norm


def unnormalize_coordinates(x_norm, dims):
    A = torch.as_tensor(dims, device=x_norm.device, dtype=x_norm.dtype) / 2.0
    x_pixel = A * x_norm + A - 0.5
    return x_pixel


def get_test_coords(img_size, device, batch_size=1):
    h, w = img_size
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    test_coords = torch.column_stack((torch.flatten(y_coords), torch.flatten(x_coords)))
    test_coords = test_coords.repeat(batch_size, 1, 1)
    return test_coords


def get_coord_img(img_size, device, batch_size=1):
    h, w = img_size
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    coord_img = (
        torch.dstack((y_coords, x_coords)).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )
    return coord_img


def fill_image(coords, vals, img_size, default_val=float("nan")):
    coords_long = coords.long()
    img = default_val * torch.ones(
        (1, img_size[0], img_size[1]), device=coords.device, dtype=vals.dtype
    )
    img[:, coords_long[..., 0], coords_long[..., 1]] = vals[..., 0]
    return img
