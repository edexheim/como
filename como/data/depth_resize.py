import torch
import torch.nn.functional as nnf


# Note: Right now only handling factors of 2
def pyr_depth(depth, mode, kernel_size):
    stride = kernel_size

    if mode == "bilinear":
        new_depth = nnf.avg_pool2d(depth, kernel_size, stride)
    elif mode == "nearest_neighbor":
        new_depth = depth[:, :, 0::stride, 0::stride]
    elif mode == "max":
        new_depth = nnf.max_pool2d(depth, kernel_size)
    elif mode == "min":
        new_depth = -nnf.max_pool2d(-depth, kernel_size)
    elif mode == "masked_bilinear":
        mask = ~depth.isnan()
        depth_masked = torch.zeros_like(depth, device=depth.device)
        depth_masked[mask] = depth[mask]
        depth_sum = nnf.avg_pool2d(
            depth_masked, kernel_size, stride, divisor_override=1
        )
        mask_sum = nnf.avg_pool2d(mask.float(), kernel_size, stride, divisor_override=1)
        new_depth = torch.where(
            mask_sum > 0.0,
            depth_sum / mask_sum,
            torch.tensor(0.0, dtype=depth.dtype, device=depth.device),
        )
    else:
        raise ValueError("pyr_depth mode: " + mode + " is not implemented.")

    return new_depth
