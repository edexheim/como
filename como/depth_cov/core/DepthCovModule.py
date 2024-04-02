import torch
import torch.nn as nn
import pytorch_lightning as pl

from como.depth_cov.core.covariance import (
    CovarianceModule,
    CrossCovarianceModule,
    DiagonalCovarianceModule,
)
import como.depth_cov.core.gaussian_kernel as gk
from como.depth_cov.core.kernels import matern
from como.depth_cov.nn.UNet import UNet


class DepthCovModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        num_levels = 5
        self.depth_var_prior = 1e-2
        kernel_scale_prior = 1e0

        self.gaussian_cov_net = UNet(
            num_levels=num_levels,
            in_channels=3,
            base_feature_channels=16,
            feature_channels=3,
            kernel_size=3,
            padding=1,
            stride=1,
            feature_act=gk.normalize_params_cov,
        )

        # Covariance modules and parameters
        iso_cov_fn = matern

        self.log_depth_var_priors = []
        self.log_depth_var_scales = nn.ParameterList()
        self.cov_modules = nn.ModuleList()
        self.cross_cov_modules = nn.ModuleList()
        self.diagonal_cov_modules = nn.ModuleList()

        for i in range(num_levels - 1):
            self.log_depth_var_priors.append(self.depth_var_prior)
            var_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.log_depth_var_scales.append(var_param)

            kernel_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.cov_modules.append(
                CovarianceModule(
                    iso_cov_fn=iso_cov_fn,
                    scale_param=kernel_scale,
                    scale_prior=kernel_scale_prior,
                )
            )
            self.cross_cov_modules.append(
                CrossCovarianceModule(
                    iso_cov_fn=iso_cov_fn,
                    scale_param=kernel_scale,
                    scale_prior=kernel_scale_prior,
                )
            )
            self.diagonal_cov_modules.append(
                DiagonalCovarianceModule(
                    iso_cov_fn=iso_cov_fn,
                    scale_param=kernel_scale,
                    scale_prior=kernel_scale_prior,
                )
            )

    def get_var(self, level):
        var_level = self.log_depth_var_priors[level] * torch.exp(
            self.log_depth_var_scales[level]
        )
        return var_level

    def get_scale(self, level):
        return self.cov_modules[level].get_scale()

    def forward(self, rgb):
        gaussian_cov_params = self.gaussian_cov_net(rgb)
        num_levels = len(gaussian_cov_params)
        gaussian_covs = []
        for l in range(0, num_levels):
            gaussian_covs.append(gk.kernel_params_to_covariance(gaussian_cov_params[l]))

        return gaussian_covs
