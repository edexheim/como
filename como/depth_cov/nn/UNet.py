import torch.nn as nn
from torchvision import transforms

from como.depth_cov.nn.layers import DownConv, UpConv, ResidualConv


class UNet(nn.Module):
    def __init__(
        self,
        num_levels,
        in_channels,
        base_feature_channels,
        feature_channels,
        kernel_size,
        padding,
        stride,
        feature_act=None,
    ):
        super(UNet, self).__init__()

        self.num_levels = num_levels

        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.base = ResidualConv(
            in_channels, base_feature_channels, kernel_size, padding, stride
        )

        down_convs = []
        up_convs = []
        feature_convs = []
        c = base_feature_channels
        for i in range(self.num_levels):
            c_next = 2 * c

            down_conv = DownConv(c, c_next, kernel_size, padding, stride)
            down_convs.append(down_conv)

            up_conv = UpConv(c_next, c, c, kernel_size, padding, stride)
            up_convs.append(up_conv)

            if i < self.num_levels - 1:
                feature_conv = nn.Conv2d(c, feature_channels, 1)
                feature_convs.append(feature_conv)

            c = c_next

        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)
        self.feature_convs = nn.ModuleList(feature_convs)

        self.output_act = feature_act

    def forward(self, x):
        # Normalization
        x_norm = self.normalize(x)

        x_enc = []
        # Initial Conv
        x_enc.append(self.base(x_norm))  # 64
        # Encoder
        for i in range(self.num_levels):
            x_enc.append(self.down_convs[i](x_enc[-1]))

        # Decoder
        f_out = []
        x_dec = x_enc[-1]
        for i in range(self.num_levels - 1, -1, -1):
            x_dec = self.up_convs[i](x_dec, x_enc[i])

            if i < self.num_levels - 1:
                f_out_level = self.output_act(self.feature_convs[i](x_dec))
                f_out.append(f_out_level)

        return f_out
