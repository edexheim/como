import torch
import torch.nn as nn
from como.depth_cov.core.kernels import (
    prob_product_quad,
    prob_product_constant,
    diagonal_prob_product,
)


class CovarianceModule(nn.Module):
    def __init__(self, iso_cov_fn, scale_param, scale_prior):
        super().__init__()

        self.iso_cov_fn = iso_cov_fn

        self.scale_param = scale_param
        self.scale_prior = scale_prior

    def get_scale(self):
        return self.scale_prior * torch.exp(self.scale_param)

    def forward(self, coords, E):
        K = self.iso_cov_fn(prob_product_quad(coords, E, coords, E))
        K_scaled = K * prob_product_constant(E, E)
        K_scaled *= self.get_scale()
        return K_scaled


class CrossCovarianceModule(CovarianceModule):
    def __init__(self, iso_cov_fn, scale_param, scale_prior):
        super().__init__(iso_cov_fn, scale_param, scale_prior)

    def forward(self, coords_train, E_train, coords_test, E_test):
        K_train_test = self.iso_cov_fn(
            prob_product_quad(coords_train, E_train, coords_test, E_test)
        )
        K_train_test_scaled = K_train_test * prob_product_constant(E_train, E_test)
        K_train_test_scaled *= self.get_scale()
        return K_train_test_scaled


class DiagonalCovarianceModule(CovarianceModule):
    def __init__(self, iso_cov_fn, scale_param, scale_prior):
        super().__init__(iso_cov_fn, scale_param, scale_prior)

    def forward(self, coords, E):
        Q, C = diagonal_prob_product(coords, E)
        K_diag = C * self.iso_cov_fn(Q)
        K_diag_scaled = K_diag * self.get_scale()
        return K_diag_scaled
