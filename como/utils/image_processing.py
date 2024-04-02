import torch
import torch.nn as nn

from como.data.depth_resize import pyr_depth
from como.geometry.camera import resize_intrinsics


class ImageGradientModule(nn.Module):
    def __init__(self, channels, device, dtype):
        super(ImageGradientModule, self).__init__()

        # Scharr kernel
        kernel_x = (1.0 / 32.0) * torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        kernel_x = kernel_x.view((1, 1, 3, 3))
        self.kernel_x = kernel_x.repeat(channels, 1, 1, 1)

        kernel_y = (1.0 / 32.0) * torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        kernel_y = kernel_y.view((1, 1, 3, 3))
        self.kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    def forward(self, x):
        gx = nn.functional.conv2d(
            nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.kernel_x,
            groups=x.shape[1],
        )

        gy = nn.functional.conv2d(
            nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.kernel_y,
            groups=x.shape[1],
        )

        return gx, gy


class GaussianBlurModule(nn.Module):
    def __init__(self, channels, device, dtype):
        super(GaussianBlurModule, self).__init__()

        # Matches opencv documentation
        gaussian_kernel = (1.0 / 16.0) * torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        self.gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    def forward(self, x):
        x_blur = nn.functional.conv2d(
            nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.gaussian_kernel,
            groups=x.shape[1],
        )
        return x_blur


class ImagePyramidModule(nn.Module):
    def __init__(self, channels, start_level, end_level, device, dtype):
        super(ImagePyramidModule, self).__init__()

        self.blur_module = GaussianBlurModule(
            channels=channels, device=device, dtype=dtype
        )
        self.start_level = start_level
        self.end_level = end_level

    def forward(self, x):
        pyr = []
        x_level = x
        for i in range(self.end_level - 1):
            if i >= self.start_level:
                pyr.insert(0, x_level)
            x_level = self.blur_module(x_level)[:, :, 0::2, 0::2]
        pyr.insert(0, x_level)
        return pyr


class DepthPyramidModule(nn.Module):
    def __init__(self, start_level, end_level, mode, device):
        super(DepthPyramidModule, self).__init__()

        self.start_level = start_level
        self.end_level = end_level
        self.mode = mode

    def forward(self, x):
        pyr = []
        x_level = x
        for i in range(self.end_level - 1):
            if i >= self.start_level:
                pyr.insert(0, x_level)
            x_level = pyr_depth(x_level, self.mode, kernel_size=2)
        pyr.insert(0, x_level)
        return pyr


class IntrinsicsPyramidModule(nn.Module):
    def __init__(self, start_level, end_level, device):
        super(IntrinsicsPyramidModule, self).__init__()

        self.start_level = start_level
        self.end_level = end_level

    def forward(self, K_orig, image_scale_start):
        pyr = []
        for i in range(self.start_level, self.end_level):
            y_scale = image_scale_start[0] * pow(2.0, -i)
            x_scale = image_scale_start[1] * pow(2.0, -i)
            K_level = resize_intrinsics(K_orig, [y_scale, x_scale])
            pyr.insert(0, K_level)
        return pyr
