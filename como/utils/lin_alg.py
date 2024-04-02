import torch


def transpose_last(A):
    return torch.transpose(A, dim0=-2, dim1=-1)


def create_diag_inv_sqrt_mat(d):
    return torch.diag_embed(1.0 / torch.sqrt(d))


# v1 (..., M), v2 (..., N)
def batched_outer(v1, v2):
    return torch.matmul(v1.unsqueeze(-1), v2.unsqueeze(-2))


def det2x2(mats):
    dets = mats[..., 0, 0] * mats[..., 1, 1] - mats[..., 0, 1] * mats[..., 1, 0]
    return dets


def trace2x2(mats):
    return mats[..., 0, 0] + mats[..., 1, 1]


def inv2x2(mats):
    invs = torch.empty_like(mats)
    invs[..., 0, 0] = mats[..., 1, 1]
    invs[..., 1, 1] = mats[..., 0, 0]
    invs[..., 0, 1] = -mats[..., 1, 0]
    invs[..., 1, 0] = -mats[..., 0, 1]

    determinants = det2x2(mats)

    invs *= 1.0 / determinants[..., None, None]

    return invs, determinants


def cholesky2x2(mats, upper=True):
    chol = torch.empty_like(mats)
    if upper:
        chol[..., 1, 0] = 0
        chol[..., 0, 0] = torch.sqrt(mats[..., 0, 0])
        chol[..., 0, 1] = torch.div(mats[..., 1, 0], chol[..., 0, 0])
        chol[..., 1, 1] = torch.sqrt(mats[..., 1, 1] - torch.square(chol[..., 0, 1]))
    else:
        chol[..., 0, 1] = 0
        chol[..., 0, 0] = torch.sqrt(mats[..., 0, 0])
        chol[..., 1, 0] = torch.div(mats[..., 1, 0], chol[..., 0, 0])
        chol[..., 1, 1] = torch.sqrt(mats[..., 1, 1] - torch.square(chol[..., 1, 0]))

    return chol


def chol_log_det(L):
    return 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)


def chol_to_inverse(L):
    b, _, m = L.shape
    I = torch.eye(m, device=L.device, dtype=L.dtype).unsqueeze(0).repeat(b, 1, 1)
    A_inv = torch.cholesky_solve(I, L, upper=False)
    return A_inv


def add_chol_rows(L11, K12, K22):
    b, m1, m2 = K12.shape
    m = m1 + m2

    L12 = torch.triangular_solve(K12, L11, upper=False)
    L22, _ = torch.cholesky_ex(K22 - (L12.mT @ L12))

    L = torch.zeros((b, m, m), device=L11.device, dtype=L11.dtype)
    L[:, :m1, :m1] = L11
    L[:, m1:, :m1] = L12
    L[:, m1:, m1:] = L22

    return L


def lstsq_chol(A, b):
    ATA = A.mT @ A
    ATb = A.mT @ b
    L, info = torch.linalg.cholesky_ex(ATA, upper=False)
    x = torch.cholesky_solve(ATb, L, upper=False)
    return x


def trace(A):
    return torch.sum(torch.diagonal(A, dim1=-2, dim2=-1), dim=1)
