import torch

from como.geometry.lie_algebra import invertSE3, skew_symmetric


def get_T_w_curr(T_w_ref, T_curr_ref):
    T_w_curr = torch.matmul(T_w_ref, invertSE3(T_curr_ref))
    return T_w_curr


def get_rel_pose(pose1, pose2):
    T_12 = torch.matmul(invertSE3(pose1), pose2)
    return T_12


# Pi either same batch as Tji or B=1 and projected into all camera frames
def transform_points(Tji, Pi):
    R = Tji[:, None, :3, :3].contiguous()
    t = Tji[:, None, :3, 3:4].contiguous()

    Pj = torch.matmul(R, Pi[..., None]) + t
    Pj = torch.squeeze(Pj, dim=-1)

    dPj_dT = torch.empty(
        (Tji.shape[0], Pi.shape[1]) + (3, 6), device=Pi.device, dtype=Tji.dtype
    )
    dPj_dT[..., :, :3] = -torch.matmul(R, skew_symmetric(Pi))
    dPj_dT[..., :, 3:] = R

    dPj_dPi = (
        R  # Note: This is the same for all points, so not expanding to Pi.shape[1]
    )

    return Pj, dPj_dT, dPj_dPi
