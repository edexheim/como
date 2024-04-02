import torch


def resize_intrinsics(K, image_scale_factors):
    T = torch.tensor(
        [
            [image_scale_factors[1], 0, image_scale_factors[1]],
            [0, image_scale_factors[0], image_scale_factors[0]],
            [0, 0, 1],
        ],
        device=K.device,
        dtype=K.dtype,
    )
    K_new = torch.matmul(T, K)
    return K_new


# K is (3, 3)
# P is (B, N, 3)
def projection(K, P):
    tmp1 = K[0, 0] * P[..., 0] / P[..., 2]
    tmp2 = K[1, 1] * P[..., 1] / P[..., 2]

    p = torch.empty(P.shape[:-1] + (2,), device=P.device, dtype=K.dtype)
    p[..., 0] = tmp1 + K[0, 2]
    p[..., 1] = tmp2 + K[1, 2]

    dp_dP = torch.empty(P.shape[:-1] + (2, 3), device=P.device, dtype=K.dtype)
    dp_dP[..., 0, 0] = K[0, 0]
    dp_dP[..., 0, 1] = 0.0
    dp_dP[..., 0, 2] = -tmp1
    dp_dP[..., 1, 0] = 0.0
    dp_dP[..., 1, 1] = K[1, 1]
    dp_dP[..., 1, 2] = -tmp2
    dp_dP /= P[..., 2, None, None]

    return p, dp_dP


# K is (3, 3)
# p is (B, N, 2)
# z is (B, N, 1)
def backprojection(K, p, z):
    tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
    tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]

    dP_dz = torch.empty(p.shape[:-1] + (3, 1), device=z.device, dtype=K.dtype)
    dP_dz[..., 0, 0] = tmp1
    dP_dz[..., 1, 0] = tmp2
    dP_dz[..., 2, 0] = 1.0

    P = torch.squeeze(z[..., None, :] * dP_dz, dim=-1)

    return P, dP_dz


def transform_project(K, Tji, Pi):
    Pmat = torch.matmul(K[None, :, :], Tji[:, 0:3, :])

    A = Pmat[:, None, :3, :3].contiguous()
    b = Pmat[:, None, :3, 3:4].contiguous()

    p_h = torch.squeeze(torch.matmul(A, Pi[..., None]) + b, dim=-1)

    depth = p_h[:, :, 2:3]
    coords = p_h[:, :, :2] / depth

    return coords, depth
