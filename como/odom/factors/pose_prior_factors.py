import torch
import como.geometry.lie_algebra as lie


def linearize_pose_prior(pose, meas, H, g, Dpose, sigma):
    device = pose.device

    info_sqrt = 1.0 / sigma

    T = torch.matmul(lie.invertSE3(pose), meas)
    xi = -lie.SE3_logmap(T)[0, :]
    J = info_sqrt * torch.eye(6, 6, device=device)
    r = info_sqrt * xi.unsqueeze(-1)

    H[Dpose[0] : Dpose[1], Dpose[0] : Dpose[1]] += J.T @ J
    g[Dpose[0] : Dpose[1]] -= torch.sum(J * r, dim=(1))
    total_err = torch.sum(torch.square(r))

    return total_err
