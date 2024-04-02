import torch
import lietorch

import numpy as np
from scipy.spatial.transform import Rotation


# Numpy
# pose is (B,4,4) or (4,4)
# output is Bx7  (tx, ty, tz, qx, qy, qz, qw)
def pose_to_tq(pose):
    if len(pose.shape) == 2:
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        tq = np.concatenate([pose[:3, 3], q], axis=0)
    elif len(pose.shape) == 3:
        q = Rotation.from_matrix(pose[:, :3, :3]).as_quat()
        tq = np.concatenate([pose[:, :3, 3], q], axis=1)
    return tq


# Numpy
# tq is (B,7) or (7,)
def tq_to_pose(tq):
    if len(tq.shape) == 1:
        t = tq[:3]  # tx, ty, tz
        q = tq[3:]  # qx, qy, qz, qw
        T = np.empty((4, 4))
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3, 3] = t
        T[3, :3] = 0.0
        T[3, 3] = 1.0

    elif len(tq.shape) == 2:
        t = tq[:, :3]  # tx, ty, tz
        q = tq[:, 3:]  # qx, qy, qz, qw
        T = np.empty((tq.shape[0], 4, 4))
        T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
        T[:, :3, 3] = t
        T[:, 3, :3] = 0.0
        T[:, 3, 3] = 1.0

    return T


def se3_exp(delta):
    delta_T_lietorch = torch.cat((delta[:, 3:], delta[:, :3]), dim=1)
    T_lietorch = lietorch.SE3.exp(delta_T_lietorch).matrix()
    # T = SE3_expmap(delta.squeeze(0)).unsqueeze(0)
    return T_lietorch


def batch_se3(poses, delta_T):
    delta_T_lietorch = torch.cat((delta_T[:, 3:], delta_T[:, :3]), dim=1)
    T_lietorch = lietorch.SE3.exp(delta_T_lietorch).matrix()
    poses_new = torch.matmul(poses, T_lietorch)
    return poses_new


# T is (B, 4, 4)
def adjoint_matrix(T):
    B = T.shape[0]
    adj_mat = torch.empty((B, 6, 6), device=T.device, dtype=T.dtype)
    adj_mat[:, :3, :3] = T[:, :3, :3]
    adj_mat[:, :3, 3:] = 0.0
    adj_mat[:, 3:, :3] = torch.matmul(skew_symmetric(T[:, :3, 3]), T[:, :3, :3])
    adj_mat[:, 3:, 3:] = T[:, :3, :3]
    return adj_mat


# T is (..., 4, 4)
def invertSE3(T):
    T_inv = torch.empty_like(T)
    Rt = torch.transpose(
        T[..., :3, :3], dim0=-2, dim1=-1
    )  # Do this explicitly so autograd works
    T_inv[..., :3, :3] = Rt
    T_inv[..., :3, 3:4] = -torch.matmul(Rt, T[..., :3, 3:4])
    T_inv[..., 3, :3] = 0.0
    T_inv[..., 3, 3] = 1.0
    return T_inv


def invertSE3_J(T):
    T_inv = torch.empty_like(T)
    Rt = torch.transpose(
        T[..., :3, :3], dim0=-2, dim1=-1
    )  # Do this explicitly so autograd works
    T_inv[..., :3, :3] = Rt
    T_inv[..., :3, 3:4] = -torch.matmul(Rt, T[..., :3, 3:4])
    T_inv[..., 3, :3] = 0.0
    T_inv[..., 3, 3] = 1.0

    dTinv_dT = -adjoint_matrix(T)

    return T_inv, dTinv_dT


def normalizeSE3_inplace(T):
    R = T[..., :3, :3]
    U, S, Vh = torch.linalg.svd(R)
    T[..., :3, :3] = torch.matmul(U, Vh)


def SO3_expmap(w):
    device = w.device
    dtype = w.dtype

    theta2 = torch.sum(torch.square(w))
    theta = torch.sqrt(theta2)
    sin_theta = torch.sin(theta)
    s2 = torch.sin(0.5 * theta)
    one_minus_cos = 2.0 * s2 * s2

    W = torch.tensor(
        [[0.0, -w[2], w[1], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]],
        device=device,
        dtype=dtype,
    )
    K = W / theta
    KK = torch.matmul(K, K)

    R = torch.eye(3, device=device, dtype=dtype) + sin_theta * K + one_minus_cos * KK

    return R


def SO3_logmap(R, eps=1e-6):
    trace_R = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    tr_3 = trace_R - 3.0
    theta = torch.acos(0.5 * (trace_R - 1))
    # print(trace_R-3.0, theta)
    mag = torch.where(
        tr_3 < -eps,
        theta / (2.0 * torch.sin(theta)),
        0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0,
    )
    tmp_v = torch.stack(
        (R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]),
        dim=1,
    )
    w = mag * tmp_v
    return w


# P is (..., 3)
# Px is (..., 3, 3)
def skew_symmetric(P):
    size = list(P.shape)
    size.append(3)
    Px = torch.zeros(size, device=P.device, dtype=P.dtype)
    Px[..., 0, 1] = -P[..., 2]
    Px[..., 0, 2] = P[..., 1]
    Px[..., 1, 0] = P[..., 2]
    Px[..., 1, 2] = -P[..., 0]
    Px[..., 2, 0] = -P[..., 1]
    Px[..., 2, 1] = P[..., 0]
    return Px


def SE3_logmap(T, eps=1e-6):
    w = SO3_logmap(T[:, :3, :3])
    theta = torch.linalg.norm(w, dim=(1))
    theta = torch.clamp(theta, min=eps)
    w_norm = w / theta
    tan = torch.tan(0.5 * theta)

    t = T[:, :3, 3]
    wnorm_x_t = torch.cross(w_norm, t)
    V_inv_t = (
        t
        - (0.5 * t) * wnorm_x_t
        + (1.0 - theta / (2.0 * tan)) * torch.cross(w_norm, wnorm_x_t)
    )
    xi = torch.cat((w, V_inv_t), dim=-1)

    return xi
