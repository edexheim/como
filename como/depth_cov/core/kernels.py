import torch
import numpy as np
from como.utils.math import safe_sqrt
from como.utils import lin_alg


def quadratic(x, A):
    x_sq = torch.square(x)
    x_corr = x[..., 0] * x[..., 1]
    xtAx = (
        A[..., 0, 0] * x_sq[..., 0]
        + 2 * A[..., 0, 1] * x_corr
        + A[..., 1, 1] * x_sq[..., 1]
    )
    return xtAx


# https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf
# bhattacharyya kernel: p=0.5 gives K(x,x) = 1
# expected likelihood kernel: p=1.0
# Assumess D=2, p=0.5
def prob_product_quad(x1, E1, x2, E2):
    dim1 = len(x1.shape) - 1
    dim2 = len(x2.shape) - 2
    diff = (x1.unsqueeze(dim1) - x2.unsqueeze(dim2)).float()

    Q = (E1[..., 1, 1].unsqueeze(dim1) + E2[..., 1, 1].unsqueeze(dim2)) * torch.square(
        diff[..., 0]
    )
    Q += (
        -2
        * (E1[..., 0, 1].unsqueeze(dim1) + E2[..., 0, 1].unsqueeze(dim2))
        * diff[..., 0]
        * diff[..., 1]
    )
    Q += (E1[..., 0, 0].unsqueeze(dim1) + E2[..., 0, 0].unsqueeze(dim2)) * torch.square(
        diff[..., 1]
    )
    Q /= (
        (E1[..., 0, 0].unsqueeze(dim1) + E2[..., 0, 0].unsqueeze(dim2))
        * (E1[..., 1, 1].unsqueeze(dim1) + E2[..., 1, 1].unsqueeze(dim2))
    ) - torch.square(E1[..., 0, 1].unsqueeze(dim1) + E2[..., 0, 1].unsqueeze(dim2))
    Q *= 0.5

    return Q


# Assumes D=2, p=0.5
def prob_product_constant(E1, E2):
    dim1 = len(E1.shape) - 2
    dim2 = len(E2.shape) - 3

    E1_det_root = lin_alg.det2x2(E1) ** (0.25)
    E2_det_root = lin_alg.det2x2(E2) ** (0.25)
    C = (
        2.0
        * E1_det_root.unsqueeze(dim1)
        * E2_det_root.unsqueeze(dim2)
        / safe_sqrt(
            (E1[..., 0, 0].unsqueeze(dim1) + E2[..., 0, 0].unsqueeze(dim2))
            * (E1[..., 1, 1].unsqueeze(dim1) + E2[..., 1, 1].unsqueeze(dim2))
            - torch.square(
                E1[..., 0, 1].unsqueeze(dim1) + E2[..., 0, 1].unsqueeze(dim2)
            )
        )
    )

    return C


## Diagonal covariance functions


def diagonal_prob_product(coords, E):
    E_det_root = torch.sqrt(lin_alg.det2x2(E))
    E_sum_det = lin_alg.det2x2(2 * E)
    C = 2.0 * E_det_root / safe_sqrt(E_sum_det)
    Q = torch.zeros_like(C)
    return Q, C


def matern(Q):
    Q_sqrt = safe_sqrt(Q)  # Constant term for stability, otherwise nan on backward
    # v=3/2
    tmp = (np.sqrt(3)) * Q_sqrt
    k_v_3_2 = (1 + tmp) * torch.exp(-tmp)

    K = k_v_3_2
    return K
