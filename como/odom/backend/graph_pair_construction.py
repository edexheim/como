import torch
import numpy as np


def get_forward_edges(B):
    # Get forward consecutive keyframe edges
    ref_ids = [b for b in range(0, B - 1)]
    target_ids = [b for b in range(1, B)]
    return ref_ids, target_ids


def get_backward_edges(B):
    ref_ids = [b for b in range(1, B)]
    target_ids = [b for b in range(0, B - 1)]
    return ref_ids, target_ids


def calc_rotation_cos(poses1, poses2):
    R1 = poses1[:, :3, :3]
    R2 = poses2[:, :3, :3]
    R12 = R1[:, None, :, :].mT @ R2[None, :, :, :]
    trace_R12 = R12[:, :, 0, 0] + R12[:, :, 1, 1] + R12[:, :, 2, 2]
    cos_theta = 0.5 * (trace_R12 - 1)
    return cos_theta


def calc_scaled_dist(poses1, poses2, median_depths1):
    dists = torch.cdist(
        poses1[:, :3, 3],
        poses2[:, :3, 3],
        compute_mode="use_mm_for_euclid_dist_if_necessary",
    )
    scaled_dists = dists / median_depths1[:, None]
    return scaled_dists


def get_pose_pairs(poses1, median_depths1, poses2, cfg, mode):
    scaled_dists = calc_scaled_dist(poses1, poses2, median_depths1)

    if mode == "nearest":
        inds2 = torch.arange(
            scaled_dists.shape[1], device=poses1.device, dtype=torch.long
        )
        min_dists, inds1 = torch.min(scaled_dists, dim=0)
    elif mode == "radius":
        scaled_dists = calc_scaled_dist(poses1, poses2, median_depths1)
        cos_theta = calc_rotation_cos(poses1, poses2)
        cos_thresh = np.cos(cfg["degrees_thresh"] * 3.14159 / 180.0)
        valid_edges = torch.logical_and(
            scaled_dists < cfg["radius_thresh"], cos_theta > cos_thresh
        )
        inds1, inds2 = torch.nonzero(valid_edges, as_tuple=True)
    elif mode == "nearest_and_radius":
        # Ensure at least nearest is included, and then others from radius as well while avoiding duplicates
        inds2_nearest = torch.arange(
            scaled_dists.shape[1], device=poses1.device, dtype=torch.long
        )
        min_dists, inds1_nearest = torch.min(scaled_dists, dim=0)

        scaled_dists = calc_scaled_dist(poses1, poses2, median_depths1)
        cos_theta = calc_rotation_cos(poses1, poses2)
        cos_thresh = np.cos(cfg["degrees_thresh"] * 3.14159 / 180.0)
        valid_edges = torch.logical_and(
            scaled_dists < cfg["radius_thresh"], cos_theta > cos_thresh
        )
        # Mask out nearest for remaining
        valid_edges[inds1_nearest, inds2_nearest] = False

        inds1_radius, inds2_radius = torch.nonzero(valid_edges, as_tuple=True)
        inds1 = torch.cat((inds1_nearest, inds1_radius))
        inds2 = torch.cat((inds2_nearest, inds2_radius))
    else:
        raise ValueError("get_pose_pairs mode: " + mode + " is not implemented.")

    return inds1, inds2


def get_kf_edges(poses, median_depths, cfg):
    inds1, inds2 = get_pose_pairs(poses, median_depths, poses, cfg, mode="radius")
    # Avoid pose with itself and consecutive keyframes
    valid_pairs = torch.abs(inds1 - inds2) > 1
    ref_ids = inds1[valid_pairs].tolist()
    target_ids = inds2[valid_pairs].tolist()
    return ref_ids, target_ids


def get_closest_temporal(timestamps1, timestamps2):
    dists = torch.abs(timestamps1[:, None] - timestamps2[None, :])
    inds2 = torch.arange(dists.shape[1], device=timestamps1.device, dtype=torch.long)
    min_dists, inds1 = torch.min(dists, dim=0)
    return inds1, inds2


# Get edges temporally (each one-way connected to the two keyframes it is between)
# If recent frame is newer than last keyframe, then only one connection
# NOTE: Timestamps ordered sequentially
def get_one_way_temporal_neighbors(kf_timestamps, recent_timestamps):
    num_kf = len(kf_timestamps)
    num_recent = len(recent_timestamps)

    one_way_kf_ids = []
    one_way_ids = []
    kf_ind = -1

    # Find first keyframe where recent frame can attach (kf_ind is kf behind)
    while recent_timestamps[0] > kf_timestamps[kf_ind + 1]:
        kf_ind += 1
        if kf_ind == num_kf - 1:  # Reached last KF
            break

    # Find inds between two keyframes (always checking to kf behind)
    r_ind = 0
    if kf_ind < num_kf - 1:
        while r_ind < num_recent:
            if recent_timestamps[r_ind] > kf_timestamps[kf_ind + 1]:
                kf_ind += 1
            if kf_ind >= num_kf - 1:  # Reached last KF
                break

            one_way_kf_ids.append(kf_ind)  # KF behind
            one_way_ids.append(r_ind)
            one_way_kf_ids.append(kf_ind + 1)  # KF ahead
            one_way_ids.append(r_ind)

            r_ind += 1

    # Rest of recent frames are newer than newest keyframe
    while r_ind < num_recent:
        one_way_kf_ids.append(kf_ind)
        one_way_ids.append(r_ind)
        r_ind += 1

    return one_way_kf_ids, one_way_ids


def get_one_way_edges(
    kf_poses, kf_median_depths, recent_poses, kf_timestamps, recent_timestamps, cfg
):
    device = kf_poses.device

    if cfg["radius_thresh"] > 0.0 and cfg["degrees_thresh"] > 0.0:
        one_way_kf_inds, one_way_inds = get_pose_pairs(
            kf_poses, kf_median_depths, recent_poses, cfg, mode="nearest_and_radius"
        )
        one_way_kf_ids = one_way_kf_inds.tolist()
        one_way_ids = one_way_inds.tolist()
    else:
        one_way_kf_ids, one_way_ids = get_one_way_temporal_neighbors(
            kf_timestamps, recent_timestamps
        )

    return one_way_kf_ids, one_way_ids


def setup_photometric_pairs(
    poses, recent_poses, kf_timestamps, recent_timestamps, median_depths, cfg
):
    num_kf = poses.shape[0]
    num_recent = recent_poses.shape[0]

    ## Graph Construction
    ref_ids_f, target_ids_f = get_forward_edges(num_kf)
    ref_ids_b, target_ids_b = get_backward_edges(num_kf)

    # Get keyframe edges within radius
    if cfg["radius_thresh"] > 0.0 and cfg["degrees_thresh"] > 0.0:
        ref_ids_kf, target_ids_kf = get_kf_edges(poses, median_depths, cfg)
    else:
        ref_ids_kf, target_ids_kf = [], []

    # Get one-way edges
    if num_recent > 0:
        one_way_kf_ids, one_way_target_ids = get_one_way_edges(
            poses, median_depths, recent_poses, kf_timestamps, recent_timestamps, cfg
        )
    else:
        one_way_kf_ids, one_way_target_ids = [], []

    kf_ref_ids = ref_ids_f + ref_ids_b + ref_ids_kf
    kf_target_ids = target_ids_f + target_ids_b + target_ids_kf

    return kf_ref_ids, kf_target_ids, one_way_kf_ids, one_way_target_ids
