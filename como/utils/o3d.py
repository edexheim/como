import torch
import open3d as o3d
import numpy as np
import cv2

from como.utils.image_processing import ImageGradientModule
from como.utils.coords import get_coord_img


def enable_widget(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


def torch_to_o3d_rgb(rgb):
    rgb_np = torch.permute(rgb, (1, 2, 0)).numpy()
    rgb_np_uint8 = np.ascontiguousarray((rgb_np * 255).astype(np.uint8))
    rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
    return rgb_img


def torch_to_o3d_rgb_with_points(rgb, coords, radius, colors):
    rgb_np = torch.permute(rgb, (1, 2, 0)).numpy()
    rgb_np_circles = rgb_np.copy()

    colors_np = colors.numpy()

    for i in range(coords.shape[1]):
        pt = (round(coords[0, i, 1].item()), round(coords[0, i, 0].item()))
        cv2.circle(rgb_np_circles, pt, radius, colors_np[i, :].tolist(), thickness=-1)

    rgb_np_uint8 = np.ascontiguousarray((rgb_np_circles * 255).astype(np.uint8))
    rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
    return rgb_img


def torch_to_o3d_spheres(P, radius, resolution, color):
    n = P.shape[0]
    for i in range(n):
        if i == 0:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius, resolution=resolution
            )
            spheres = sphere.translate(P[i, :]).paint_uniform_color(color)
        else:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius, resolution=resolution
            )
            spheres += sphere.translate(P[i, :]).paint_uniform_color(color)
    return spheres


def torch_to_o3d_depth(depth):
    depth_filt = depth.clone()
    depth_filt[depth.isnan()] = 0.0
    depth_np = np.ascontiguousarray(depth_filt[0, ...].numpy())
    depth_img = o3d.t.geometry.Image(depth_np)
    return depth_img


def torch_to_o3d_normal_color(normals):
    normals_color = 0.5 * (1.0 + normals)
    normals_color = torch.permute(normals_color, (1, 2, 0))
    rgb_np_uint8 = np.ascontiguousarray((normals_color.numpy() * 255).astype(np.uint8))
    rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
    return rgb_img


def torch_to_o3d_pcd(points, rgb=None, normals=None):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(points.numpy())
    if rgb is not None:
        pcd.point.colors = o3d.core.Tensor(rgb.numpy())
    if normals is not None:
        pcd.point.normals = o3d.core.Tensor(normals.numpy())

    return pcd


def get_rays(img_size, K, device):
    coords = get_coord_img(img_size[-2:], device, batch_size=1)
    tmp1 = (coords[..., 0] - K[1, 2]) / K[1, 1]
    tmp2 = (coords[..., 1] - K[0, 2]) / K[0, 0]
    rays = torch.empty((3, img_size[-2], img_size[-1]), device=device)
    rays[0, ...] = tmp2
    rays[1, ...] = tmp1
    rays[2, ...] = 1.0
    return rays


def get_points_and_normals(depth, K):
    device = depth.device

    rays = get_rays(depth.shape[-2:], K, device)

    b = depth.shape[0]
    rays = rays.unsqueeze(0).repeat(b, 1, 1, 1)
    P = depth * rays  # (b,3,h,w)

    grad_module = ImageGradientModule(3, device=device, dtype=P.dtype)
    gx, gy = grad_module(P)
    gx = torch.permute(gx, (0, 2, 3, 1))  # (b,h,w,3)
    gy = torch.permute(gy, (0, 2, 3, 1))
    n_dir = torch.linalg.cross(gx, gy, dim=-1)

    normals = torch.nn.functional.normalize(n_dir, p=2.0, dim=-1, eps=1e-12)
    normals = torch.permute(normals, (0, 3, 1, 2))  # (b,3,h,w)

    # Set edges since undefined
    default_normal = torch.tensor([0.0, 0.0, 1.0])[None, :, None]
    normals[:, :, 0, :] = default_normal
    normals[:, :, -1, :] = default_normal
    normals[:, :, :, 0] = default_normal
    normals[:, :, :, -1] = default_normal

    return P, normals, rays


def setup_cloud(rgb, P, normals, rays, pose, cos_thresh):
    b, _, h, w = rgb.shape

    colors_np = rgb.numpy()
    points_np = P.numpy()
    normals_np = normals.numpy()
    rays_np = rays.numpy()

    # Trim edges since gradient undefined
    points_img = points_np[:, :, 1:-1, 1:-1]
    new_size = points_img.shape[2] * points_img.shape[3]
    colors_np = np.resize(colors_np[:, :, 1:-1, 1:-1], (b, 3, new_size)).astype(
        np.float32
    )
    points_np = np.resize(points_img, (b, 3, new_size)).astype(np.float64)
    normals_np = np.resize(normals_np[:, :, 1:-1, 1:-1], (b, 3, new_size)).astype(
        np.float32
    )
    rays_np = np.resize(rays_np[:, :, 1:-1, 1:-1], (b, 3, new_size)).astype(np.float32)

    colors_np = colors_np.transpose(0, 2, 1)
    points_np = points_np.transpose(0, 2, 1)
    normals_np = normals_np.transpose(0, 2, 1)
    rays_np = rays_np.transpose(0, 2, 1)

    rays_norm = rays_np / np.linalg.norm(rays_np, axis=2, keepdims=True)
    # Check dot product between ray and normal
    cos_theta = np.abs(np.sum(normals_np * rays_norm, axis=2))
    invalid_mask = cos_theta < cos_thresh

    # Transform
    pose_np = pose.numpy().astype(np.float64)
    points_np = (
        points_np @ pose_np[:, 0:3, 0:3].transpose(0, 2, 1)
        + pose_np[:, np.newaxis, 0:3, 3]
    )
    points_np = points_np.astype(np.float32)

    return colors_np, points_np, invalid_mask


def rgb_depth_to_pcd(rgb, depth, pose, K, cos_thresh):
    P, normals, rays = get_points_and_normals(depth, K)
    colors_np, points_np, invalid_mask = setup_cloud(
        rgb, P, normals, rays, pose, cos_thresh
    )
    valid_mask = np.logical_not(invalid_mask)

    # Construct t PointCloud
    for b in range(rgb.shape[0]):
        pcd_b = o3d.t.geometry.PointCloud()
        mask = valid_mask[b]
        pcd_b.point.positions = o3d.core.Tensor(points_np[b, mask, :])
        pcd_b.point.colors = o3d.core.Tensor(colors_np[b, mask, :])

        if b == 0:
            pcd = pcd_b
        else:
            pcd = pcd.append(pcd_b)

    return pcd, normals


def frustum_lineset(intrinsics, img_size, pose, scale):
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        img_size[1], img_size[0], intrinsics.numpy(), np.linalg.inv(pose), scale=scale
    )
    return frustum


def poses_to_traj_lineset(poses):
    n = poses.shape[0]
    points = poses[:, :3, 3]
    lines = np.stack(
        (np.arange(0, n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)), axis=1
    )

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


def get_reference_frame_lineset(poses, one_way_poses, ind_pairs):
    n = len(ind_pairs[0])
    points_np = np.concatenate(
        (poses[ind_pairs[0], :3, 3], one_way_poses[ind_pairs[1], :3, 3]), axis=0
    )
    lines_ind1 = np.arange(0, n, dtype=np.int32)
    lines_np = np.stack((lines_ind1, lines_ind1 + n), axis=1)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points_np)
    lineset.lines = o3d.utility.Vector2iVector(lines_np)
    return lineset


def get_one_way_lineset(
    poses, one_way_poses, ref_inds, intrinsics, img_size, color, frustum_scale
):
    lineset = get_reference_frame_lineset(poses, one_way_poses, ref_inds)
    for i in range(one_way_poses.shape[0]):
        lineset += frustum_lineset(
            intrinsics, img_size, one_way_poses[i, ...], frustum_scale
        )
    lineset.paint_uniform_color(color)
    return lineset


def get_traj_lineset(
    poses,
    intrinsics,
    img_size,
    color,
    frustum_scale,
    frustum_mode="last",
    pose_lines=True,
):
    if pose_lines:
        lineset = poses_to_traj_lineset(poses)
    else:
        lineset = o3d.geometry.LineSet()
    if frustum_mode == "all":
        for i in range(poses.shape[0]):
            lineset += frustum_lineset(
                intrinsics, img_size, poses[i, ...], frustum_scale
            )
    elif frustum_mode == "last":
        lineset += frustum_lineset(intrinsics, img_size, poses[-1, ...], frustum_scale)
    elif frustum_mode == "none":
        pass

    lineset.paint_uniform_color(color)
    return lineset


# NOTE: Assumes canonical world frame direction from first pose
def pose_to_camera_setup(pose, pose_init, scale):
    # Assume original negative y axis is up
    up_global = -pose_init[:3, 1]
    # up_global = np.array([0, 0, 1.0])
    # up_global = pose[:3,1]

    # Camera coordinates
    center = scale * np.array([0, 0.0, 0.5])  # Point camera is looking at
    eye = scale * np.array([0, -0.0, -0.5])  # Camera location

    def rot2eul(R):
        beta = -np.arcsin(R[2, 0])
        alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
        gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
        return np.array((alpha, beta, gamma))

    def eul2rot(theta):
        R = np.array(
            [
                [
                    np.cos(theta[1]) * np.cos(theta[2]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2])
                    - np.sin(theta[2]) * np.cos(theta[0]),
                    np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2])
                    + np.sin(theta[0]) * np.sin(theta[2]),
                ],
                [
                    np.sin(theta[2]) * np.cos(theta[1]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2])
                    + np.cos(theta[0]) * np.cos(theta[2]),
                    np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0])
                    - np.sin(theta[0]) * np.cos(theta[2]),
                ],
                [
                    -np.sin(theta[1]),
                    np.sin(theta[0]) * np.cos(theta[1]),
                    np.cos(theta[0]) * np.cos(theta[1]),
                ],
            ]
        )
        return R

    # Transform into world coordinates ()
    R = pose[:3, :3]
    t = pose[:3, 3]

    zyx = rot2eul(R)
    # zyx[0] = 0.0 # Roll
    # zyx[2] = 0.0 # Pitch
    R = eul2rot(zyx)

    center = R @ center + t
    eye = R @ eye + t

    return center, eye, up_global
