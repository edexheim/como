import time

import torch
import torchvision.transforms.functional as TF

from como.depth_cov.core.DepthCovModule import DepthCovModule
from como.depth_cov.core.gaussian_kernel import interpolate_kernel_params

from como.geometry.affine_brightness import get_aff_w_curr
from como.geometry.lie_algebra import normalizeSE3_inplace
from como.geometry.transforms import get_T_w_curr, transform_points
from como.geometry.camera import backprojection

from como.odom.factors.pose_prior_factors import linearize_pose_prior
from como.odom.factors.scalar_prior_factors import (
    linearize_scalar_prior,
    linearize_multi_scalar_prior,
)
from como.odom.factors.pixel_prior import pixel_prior_cost
from como.odom.factors.gp_priors import gp_ml_cost, mean_log_depth_cost
from como.odom.factors.depth_prior import log_depth_prior

from como.odom.frontend.TwoFrameSfm import TwoFrameSfm

from como.utils.config import str_to_dtype
from como.utils.coords import normalize_coordinates, swap_coords_xy, get_test_coords
from como.utils.image_processing import ImageGradientModule
from como.utils.multiprocessing import init_gpu

from como.odom.frontend.corr import track_and_init
from como.odom.backend.sparse_map import (
    setup_point_to_frame,
    get_batch_remap_function,
    setup_test_points,
    subselect_pixels,
)

import como.odom.backend.linear_system as lin_sys
from como.odom.backend.photo import create_photo_system


# Do not declare any CUDA tensors in init function
class Mapping:
    def __init__(self, cfg, intrinsics):
        super().__init__()

        self.cfg = cfg
        self.device = cfg["device"]
        self.dtype = str_to_dtype(cfg["dtype"])

        self.intrinsics = intrinsics.unsqueeze(0)

        self.is_init = False

    def setup(self):
        init_gpu(self.device)
        self.init_basic_vars()
        self.load_model()
        self.init_keyframe_vars()
        self.init_prior_vals()
        self.reset_iteration_vars(new_kf=True, converged=True)
        self.two_frame_sfm = TwoFrameSfm(
            self.cfg,
            self.intrinsics[0, :, :],
            self.model,
            self.cov_level,
            self.network_size,
        )

    def init_basic_vars(self):
        if self.cfg["color"] == "gray":
            c = 1
        elif self.cfg["color"] == "rgb":
            c = 3

        self.intrinsics = self.intrinsics.to(device=self.device, dtype=self.dtype)
        self.gradient_module = ImageGradientModule(
            channels=c, device=self.device, dtype=self.dtype
        )

        self.last_kf_send_time = 0.0

    def init_keyframe_vars(self):
        # Bookkeeping
        self.kf_timestamps = []

        ## Keyframes
        # Inputs
        self.rgb = torch.empty(
            (0), device=self.device, dtype=self.dtype
        )  # For visualization
        self.kf_img_and_grads = torch.empty((0), device=self.device, dtype=self.dtype)
        self.cov_params_img = torch.empty((0), device=self.device, dtype=self.dtype)
        # Pose variables
        self.kf_poses = torch.empty((0), device=self.device, dtype=self.dtype)
        self.kf_aff_params = torch.empty((0), device=self.device, dtype=self.dtype)
        # Sparse pixel variables
        self.depth_dims = []  # Bookkeeping points
        self.pm_first_obs = torch.empty((0), device=self.device, dtype=self.dtype)
        self.pm = torch.empty((0), device=self.device, dtype=self.dtype)
        self.logzm = torch.empty((0), device=self.device, dtype=self.dtype)
        self.L_mm = torch.empty(
            (0), device=self.device, dtype=self.dtype
        )  # Store but updated constantly
        self.Kmm_inv = torch.empty((0), device=self.device, dtype=self.dtype)
        self.Knm_Kmminv = torch.empty((0), device=self.device, dtype=self.dtype)
        # Sparse landmark variables
        self.correspondence_mask = torch.empty(
            (0), device=self.device, dtype=self.dtype
        )
        self.P_m = torch.empty((0), device=self.device, dtype=self.dtype)
        self.obs_ref_mask = torch.empty(
            (0), device=self.device, dtype=torch.bool
        )  # whether point seen first

        ## One-way Frames
        self.recent_timestamps = []
        # Inputs
        self.recent_img_and_grads = torch.empty(
            (0), device=self.device, dtype=self.dtype
        )
        # Pose variables
        self.recent_poses = torch.empty((0), device=self.device, dtype=self.dtype)
        self.recent_aff_params = torch.empty((0), device=self.device, dtype=self.dtype)

        # Storing certain variables for fast queries/visualization
        self.depth_imgs = None

    def init_prior_vals(self):
        self.window_full = False
        self.pose_anchor = torch.empty((0), device=self.device, dtype=self.dtype)
        self.aff_anchor = torch.empty((0), device=self.device, dtype=self.dtype)
        self.sparse_log_depth_anchor = torch.empty(
            (0), device=self.device, dtype=self.dtype
        )

    # Same as add_keyframe but given init depth variables, so don't calculate
    def init_keyframe(
        self, rgb, cov_params_img, coords_m, pose_init, logz_m, aff_init, timestamp
    ):
        img_and_grads = self.get_img_and_grads(rgb)

        cov_params_img = TF.resize(
            cov_params_img,
            rgb.shape[-2:],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

        self.initialize_pose_vars(pose_init, aff_init)
        self.initialize_kf_img_vars_vars(rgb, img_and_grads, cov_params_img)

        # Initialize sparse pixel variables
        Kmm_inv, L_mm, Knm_Kmminv = self.prep_predictor(cov_params_img, coords_m)

        depth_dim = coords_m.shape[1]
        pm = swap_coords_xy(coords_m)
        z_m = torch.exp(logz_m)
        self.initialize_sparse_pixel_vars(pm, z_m, depth_dim, Kmm_inv, L_mm, Knm_Kmminv)

        # Initialize landmark variables
        corr_mask = torch.ones((1, depth_dim), device=self.device, dtype=torch.bool)
        Pc_m, _ = backprojection(self.intrinsics[0, :, :], pm, z_m)
        Pw_m, _, _ = transform_points(pose_init, Pc_m)
        Pw_m = Pw_m.squeeze(0)

        self.initialize_sparse_landmark_vars(corr_mask, Pw_m)

        self.kf_timestamps = [timestamp]

        self.store_vars(pm, logz_m, Knm_Kmminv)

        return

    def add_keyframe(self, rgb, kf_pose_init, kf_aff_init, timestamp):
        img_and_grads = self.get_img_and_grads(rgb)
        cov_params_img = self.run_model(rgb)

        # Get variables from last frame
        kf_pose_last = self.kf_poses[-1, None, :, :]

        pm_last = self.pm[-1, None, :, :]
        coords_m_last = swap_coords_xy(pm_last)
        logzm_last = self.logzm[-1, None, :, :]
        depth_img_last = self.depth_imgs[-1, None, :, :]

        # Track correspondences and initialize new landmarks
        zm_last = torch.exp(logzm_last)
        coords_m_new, z_m_new, corr_mask, coords_m, zm_first_obs = track_and_init(
            kf_pose_last,
            kf_pose_init,
            coords_m_last,
            zm_last,
            depth_img_last,
            cov_params_img,
            self.intrinsics,
            self.model,
            self.cfg["corr"],
            self.cfg["sampling"],
            self.kf_img_and_grads.shape[-2:],
        )  # rgb1=self.rgb[-1,None,:,:], rgb2=rgb)

        p_m_new = swap_coords_xy(coords_m_new).to(dtype=z_m_new.dtype)
        Pc_m_new, _ = backprojection(self.intrinsics[0, :, :], p_m_new, z_m_new)
        Pw_m_new, _, _ = transform_points(kf_pose_init, Pc_m_new)
        Pw_m_new = Pw_m_new.squeeze(0)

        Kmm_inv, L_mm, Knm_Kmminv = self.prep_predictor(cov_params_img, coords_m)
        pm_first_obs = swap_coords_xy(coords_m)
        new_depth_dim = coords_m_new.shape[1]

        self.window_cat_helper_list(
            self.kf_timestamps, timestamp, self.get_kf_start_window_ind()
        )
        self.initialize_pose_vars(kf_pose_init, kf_aff_init)
        self.initialize_kf_img_vars_vars(rgb, img_and_grads, cov_params_img)
        self.initialize_sparse_pixel_vars(
            pm_first_obs, zm_first_obs, new_depth_dim, Kmm_inv, L_mm, Knm_Kmminv
        )
        self.initialize_sparse_landmark_vars(corr_mask, Pw_m_new)

        self.reset_iteration_vars(new_kf=True)

        self.store_vars(self.pm, self.logzm, self.Knm_Kmminv)

        # Prune all one way frames that are older than now oldest keyframe
        self.prune_one_way()

        return

    def prune_one_way(self):
        oldest_kf_ts = self.kf_timestamps[0]

        r_ind = 0
        for i in range(len(self.recent_timestamps)):
            if self.recent_timestamps[i] < oldest_kf_ts:
                r_ind = i + 1

        # Remove oldest one way frames
        self.recent_timestamps = self.recent_timestamps[r_ind:]
        self.recent_img_and_grads = self.recent_img_and_grads[r_ind:]
        self.recent_poses = self.recent_poses[r_ind:]
        self.recent_aff_params = self.recent_aff_params[r_ind:]

        return

    def add_one_way_frame(self, rgb, pose_init, aff_init, timestamp):
        img_and_grads = self.get_img_and_grads(rgb)

        recent_ind = self.get_recent_start_window_ind()
        self.window_cat_helper_list(self.recent_timestamps, timestamp, recent_ind)
        self.window_cat_helper_tensor(
            self.recent_img_and_grads, img_and_grads, recent_ind
        )
        self.window_cat_helper_tensor(self.recent_poses, pose_init, recent_ind)
        self.window_cat_helper_tensor(self.recent_aff_params, aff_init, recent_ind)

        self.reset_iteration_vars(new_kf=False)
        return

    # Remove oldest keyframe and reset anchors
    def initialize_pose_vars(self, pose_init, aff_init):
        num_kf = self.kf_poses.shape[0]
        window_empty = num_kf == 0
        self.window_full = num_kf >= self.cfg["graph"]["num_keyframes"]

        # Add new frame and remove oldest frame if window is full
        kf_start_ind = self.get_kf_start_window_ind()
        normalizeSE3_inplace(pose_init)
        self.window_cat_helper_tensor(self.kf_poses, pose_init, kf_start_ind)
        self.window_cat_helper_tensor(self.kf_aff_params, aff_init, kf_start_ind)

        # Set anchors if window is full or completely empty
        if window_empty or self.window_full:
            self.pose_anchor = self.kf_poses[0:1, ...].clone()
            self.aff_anchor = self.kf_aff_params[0:1, ...].clone()

            # Reset affine frame to be against oldest keyframe in window
            self.kf_aff_params -= self.aff_anchor
            self.aff_anchor = torch.zeros_like(self.aff_anchor)

        return

    def initialize_kf_img_vars_vars(self, rgb, img_and_grads, cov_params_img):
        kf_start_ind = self.get_kf_start_window_ind()
        self.window_cat_helper_tensor(self.rgb, rgb, kf_start_ind)
        self.window_cat_helper_tensor(
            self.kf_img_and_grads, img_and_grads, kf_start_ind
        )
        self.window_cat_helper_tensor(self.cov_params_img, cov_params_img, kf_start_ind)
        return

    # Sparse vars in pixel space (batched per image, dim max num pixels)
    def initialize_sparse_pixel_vars(
        self, pm_first_obs, zm_first_obs, new_depth_dim, Kmm_inv, L_mm, Knm_Kmminv
    ):
        kf_start_ind = self.get_kf_start_window_ind()

        self.window_cat_helper_list(self.depth_dims, new_depth_dim, kf_start_ind)
        self.window_cat_helper_tensor(self.pm_first_obs, pm_first_obs, kf_start_ind)
        self.window_cat_helper_tensor(self.pm, pm_first_obs, kf_start_ind)
        self.window_cat_helper_tensor(self.logzm, torch.log(zm_first_obs), kf_start_ind)

        # Mask of whether landmarks were seen first in an image
        # NOTE: Ones that left window no longer have reference!
        obs_ref_mask = torch.zeros(
            (1, self.cfg["sampling"]["max_num_coords"]),
            device=self.device,
            dtype=torch.bool,
        )
        obs_ref_mask[:, -new_depth_dim:] = True
        self.window_cat_helper_tensor(self.obs_ref_mask, obs_ref_mask, kf_start_ind)

        self.window_cat_helper_tensor(self.Kmm_inv, Kmm_inv, kf_start_ind)
        self.window_cat_helper_tensor(self.L_mm, L_mm, kf_start_ind)
        self.window_cat_helper_tensor(self.Knm_Kmminv, Knm_Kmminv, kf_start_ind)

        return

    # Sparse vars in 3D space (dim number of landmarks)
    def initialize_sparse_landmark_vars(self, corr_mask, P):
        kf_start_ind = self.get_kf_start_window_ind()

        num_kf = self.correspondence_mask.shape[0]
        window_empty = num_kf == 0
        self.window_full = num_kf >= self.cfg["graph"]["num_keyframes"]

        new_depth_dim = P.shape[0]

        if self.correspondence_mask.shape[0] == 0:
            self.correspondence_mask = corr_mask
            self.P_m = P
        else:
            # Add to correspondence mask (num_kf x num_landmarks)

            # Remove columns that show up in no frames
            all_but_old_refs = self.correspondence_mask[kf_start_ind:, :].any(dim=0)
            # Find which tracked
            last_ref_inds = torch.nonzero(self.correspondence_mask[-1, :])
            new_row = torch.zeros_like(self.correspondence_mask[0, :])
            new_row[last_ref_inds[:, 0]] = corr_mask
            new_corr = torch.cat(
                (
                    self.correspondence_mask[kf_start_ind:, all_but_old_refs],
                    new_row[None, all_but_old_refs],
                ),
                dim=0,
            )
            # Add new columns for new depth references
            pad_cols = torch.zeros(
                (new_corr.shape[0], new_depth_dim), device=self.device, dtype=torch.bool
            )
            pad_cols[-1, :] = True
            new_corr = torch.cat((new_corr, pad_cols), dim=1)
            self.correspondence_mask = new_corr

            # Filter out landmarks with no observations left
            remaining_P = self.P_m[all_but_old_refs, :]
            self.P_m.set_(torch.cat((remaining_P, P), dim=0))

        # Anchor landmarks that were connected to any frames that left the window
        # (so ones referenced in now the oldest keyframe in the window)
        if self.window_full:
            anchor_inds = self.correspondence_mask[0, :]
            self.P_m_anchors = self.P_m[anchor_inds, :]

        return

    def get_img_and_grads(self, rgb):
        if self.cfg["color"] == "gray":
            img = TF.rgb_to_grayscale(rgb)
        elif self.cfg["color"] == "rgb":
            img = rgb.clone()

        gx, gy = self.gradient_module(img)
        img_and_grads = torch.cat((img, gx, gy), dim=1)
        return img_and_grads

    def find_kf_from_timestamp(self, kf_timestamp):
        kf_ind = None
        for i in range(len(self.kf_timestamps) - 1, -1, -1):
            if kf_timestamp == self.kf_timestamps[i]:
                kf_ind = i
                break
        return kf_ind

    def get_curr_world_pose(self, pose_curr_kf, kf_ind):
        T_w_curr = get_T_w_curr(self.kf_poses[kf_ind : kf_ind + 1, ...], pose_curr_kf)
        return T_w_curr

    def get_curr_world_aff(self, aff_curr_kf, kf_ind):
        aff_curr = get_aff_w_curr(
            self.kf_aff_params[kf_ind : kf_ind + 1, ...], aff_curr_kf
        )
        return aff_curr

    def load_model(self):
        self.cov_level = -1
        self.network_size = torch.tensor([192, 256], device=self.device)
        self.network_size_list = self.network_size.tolist()

        self.model = DepthCovModule.load_from_checkpoint(
            self.cfg["model_path"], train_size=self.network_size
        )
        self.model.eval()
        self.model.to(self.device)
        self.model.to(torch.float)

    def run_model(self, rgb):
        # Gaussian covs
        rgb_r = TF.resize(
            rgb,
            self.network_size_list,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        ).float()
        with torch.no_grad():
            gaussian_covs = self.model(rgb_r)
            model_level = -1
            cov_params_img = gaussian_covs[model_level].to(dtype=self.dtype)
            cov_params_img = TF.resize(
                cov_params_img,
                rgb.shape[-2:],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )

        return cov_params_img

    def prep_predictor(self, cov_params_img, coords_m):
        b, _, h, w = cov_params_img.shape
        cov_img_size = (h, w)
        device = cov_params_img.device

        coords_m_norm = normalize_coordinates(coords_m, cov_img_size)
        E_m = interpolate_kernel_params(cov_params_img, coords_m_norm)

        # Get test vars
        photo_img_size = self.kf_img_and_grads.shape[-2:]
        coords_n_all = get_test_coords(photo_img_size, device=device, batch_size=b)
        coords_n_norm = normalize_coordinates(
            coords_n_all.to(dtype=self.dtype), cov_img_size
        )
        E_n = interpolate_kernel_params(cov_params_img, coords_n_norm)

        with torch.no_grad():
            level = -1
            K_mm = self.model.cov_modules[level](coords_m_norm, E_m)
            m = coords_m.shape[1]
            K_mm += torch.diag_embed(1e-6 * torch.ones(b, m, device=device))

            L_mm, info = torch.linalg.cholesky_ex(K_mm, upper=False)
            I_mm = (
                torch.eye(m, device=device, dtype=K_mm.dtype)
                .unsqueeze(0)
                .repeat(b, 1, 1)
            )
            K_mm_inv = torch.cholesky_solve(I_mm, L_mm, upper=False)

            K_nm = self.model.cross_cov_modules[level](
                coords_n_norm, E_n, coords_m_norm, E_m
            )
            Knm_Kmminv = K_nm @ K_mm_inv
            Knm_Kmminv = torch.reshape(
                Knm_Kmminv, (b, photo_img_size[0], photo_img_size[1], -1)
            )

        return K_mm_inv, L_mm, Knm_Kmminv

    def window_cat_helper_tensor(self, var, new_var, i):
        var.set_(torch.cat((var[i:, ...], new_var), dim=0))

    def window_cat_helper_list(self, var, new_var, i):
        del var[:i]
        var.append(new_var)

    # Keeps all but first keyframe once capacity is reached
    def get_kf_start_window_ind(self):
        num_max_frames = self.cfg["graph"]["num_keyframes"]
        ind = -num_max_frames + 1
        return ind

    def get_recent_start_window_ind(self):
        num_max_frames = self.cfg["graph"]["num_one_way_frames"]
        ind = -num_max_frames + 1
        return ind

    def reset_iteration_vars(self, new_kf, converged=False):
        self.converged = converged
        if new_kf:
            self.iter = 0
            self.total_err_prev = float("inf")

    def get_safe_ind(self, ind):
        if ind == -1 or ind >= self.kf_poses.shape[0]:
            ind = self.kf_poses.shape[0] - 1
        return ind

    def get_kf_ref_data(self, ind=-1):
        end_ind = self.kf_poses.shape[0]

        ind = max(0, self.kf_poses.shape[0] - self.cfg["track_ref"]["num_keyframes"])

        timestamp = self.kf_timestamps[ind:end_ind]
        rgb = self.rgb[ind:end_ind, :, :, :]
        pose = self.kf_poses[ind:end_ind, ...]
        aff = self.kf_aff_params[ind:end_ind, ...]

        depth_img = self.depth_imgs[ind:end_ind, ...]

        return timestamp, rgb, pose, aff, depth_img

    # NOTE: Cloning all variables so returned tensors are unaffected by new keyframes
    def get_kf_viz_data(self, ind=-1):
        ind = self.get_safe_ind(ind)

        timestamps = self.kf_timestamps.copy()
        rgbs = self.rgb.clone()
        poses = self.kf_poses.clone()

        depth_imgs = self.depth_imgs.clone()

        rec_poses = self.recent_poses.clone()
        kf_pairs = self.kf_pairs.copy()
        one_way_pairs = self.one_way_pairs.copy()

        sparse_coords = swap_coords_xy(self.pm).clone()
        P_sparse = self.P_m.clone()
        obs_ref_mask = self.obs_ref_mask.clone()

        self.last_kf_send_time = time.time()

        return (
            timestamps,
            rgbs,
            poses,
            depth_imgs,
            sparse_coords,
            P_sparse,
            obs_ref_mask,
            rec_poses,
            kf_pairs,
            one_way_pairs,
        )

    def attempt_two_frame_init(self, timestamp, rgb):
        (
            self.is_init,
            T_curr_kf,
            aff_curr_kf,
            sparse_log_depth_kf,
            coords_curr,
            depth_curr,
            mean_log_depth,
        ) = self.two_frame_sfm.handle_frame(rgb, timestamp)
        if self.is_init:
            # Initialize reference keyframe
            self.init_keyframe(
                self.two_frame_sfm.rgb,
                self.two_frame_sfm.cov_params_img,
                self.two_frame_sfm.coords_m,
                self.two_frame_sfm.pose_init,
                sparse_log_depth_kf,
                self.two_frame_sfm.aff_init,
                self.two_frame_sfm.timestamp,
            )

            # Initialize second keyframe
            pose_curr = get_T_w_curr(self.two_frame_sfm.pose_init, T_curr_kf)
            aff_curr = get_aff_w_curr(self.two_frame_sfm.aff_init, aff_curr_kf)
            self.add_keyframe(rgb, pose_curr, aff_curr, timestamp)
            self.init_scale_anchor = mean_log_depth
            self.two_frame_sfm.delete_init_reference()
            kf_updated = True
        else:
            kf_updated = False

        return kf_updated

    def handle_tracking_data(self, data):
        kf_viz_data = None
        kf_updated = False

        if data[0] == "one-way":
            rgb, pose_curr_kf, aff_curr_kf, kf_timestamp, timestamp = data[1:]
            kf_ind = self.find_kf_from_timestamp(kf_timestamp)
            pose_w_init = self.get_curr_world_pose(pose_curr_kf, kf_ind)
            aff_w_init = self.get_curr_world_aff(aff_curr_kf, kf_ind)
            self.add_one_way_frame(rgb, pose_w_init, aff_w_init, timestamp)
        elif data[0] == "keyframe":
            # NOTE: If multiprocessing, Send KF data to visualization before adding new one
            kf_viz_data = self.get_kf_viz_data()
            # Insert new KF for mapping
            rgb, pose_curr_kf, aff_curr_kf, kf_timestamp, timestamp = data[1:]
            kf_ind = self.find_kf_from_timestamp(kf_timestamp)
            pose_w_init = self.get_curr_world_pose(pose_curr_kf, kf_ind)
            aff_w_init = self.get_curr_world_aff(aff_curr_kf, kf_ind)
            self.add_keyframe(rgb, pose_w_init, aff_w_init, timestamp)
            kf_updated = True

        return kf_viz_data, kf_updated

    def prep_geometry_scaffold(self):
        num_kf = self.kf_poses.shape[0]

        # Mask actual correspondences (num_kf x num_landmarks)
        # NOTE: If not updating corr_mask with reprojection info, could compute this only when new KF
        # NOTE: Since assuming same number of points per frame, there should be no padding
        remap_variable_to_batch, paired_landmark_batch_inds = get_batch_remap_function(
            self.correspondence_mask
        )
        landmark_inds, batched_inds = paired_landmark_batch_inds
        point_inds = lin_sys.landmark_to_batched_3d_point_inds(landmark_inds, num_kf)

        first_obs_inds = torch.argmax(
            self.correspondence_mask.int(), dim=0, keepdim=False
        )
        col_inds = torch.arange(first_obs_inds.shape[0], device=self.device)
        first_obs_mask = torch.zeros_like(
            self.correspondence_mask, device=self.device, dtype=torch.bool
        )
        first_obs_mask[first_obs_inds, col_inds] = True
        first_obs_mask = remap_variable_to_batch(first_obs_mask, default_val=False)

        # Preemptively store reinitializations of points in case new ones are behind cameras
        depth_init = self.median_depths[:, None, None] * torch.ones_like(self.logzm)
        init_Pc_m, _ = backprojection(
            self.intrinsics[0, :, :], self.pm_first_obs, depth_init
        )
        init_Pw_m, _, _ = transform_points(self.kf_poses, init_Pc_m)
        # Just use first obs
        init_Pm = init_Pw_m[first_obs_mask, :]

        # 3D point to keyframe vars
        pm, logzm, z_mask, dlogzm_dzm, dzm_dPwm, dlogzm_dTwc, dpm_dPwm, dpm_dTwc = (
            setup_point_to_frame(
                self.P_m,
                self.kf_poses,
                remap_variable_to_batch,
                self.intrinsics,
                reinit_P=init_Pm,
                median_depths=self.median_depths,
            )
        )

        # Reinit landmarks to match up
        reinit_mask = z_mask[first_obs_mask]
        self.P_m[reinit_mask] = init_Pm[reinit_mask]

        return (
            pm,
            dpm_dTwc,
            dpm_dPwm,
            logzm,
            dlogzm_dTwc,
            dlogzm_dzm,
            dzm_dPwm,
            point_inds,
        )

    def prep_dense_ref(self, pm, logzm, dlogzm_dTwc, dlogzm_dzm):
        num_kf = self.kf_poses.shape[0]

        # Setup keyframe testing points
        coords_n, _ = subselect_pixels(
            self.kf_img_and_grads,
            self.cfg["photo_construction"]["nonmax_suppression_window"],
        )

        # Index into images
        batch_inds = (
            torch.arange(num_kf, device=self.device)
            .unsqueeze(1)
            .repeat(1, coords_n.shape[1])
        )
        c = self.kf_img_and_grads.shape[1] // 3
        vals_n = self.kf_img_and_grads[
            batch_inds, :c, coords_n[:, :, 0], coords_n[:, :, 1]
        ]

        Knm_Kmminv_test = self.Knm_Kmminv[
            batch_inds, coords_n[:, :, 0], coords_n[:, :, 1], :
        ].clone()

        # Get dense 3D points in world frame
        Pwn, dPwn_dTwc, dPwn_dzm, median_depths, dlogzn_dlogzm, logzn = (
            setup_test_points(
                pm,
                logzm,
                self.kf_poses,
                Knm_Kmminv_test,
                coords_n,
                self.intrinsics,
                dlogzm_dTwc,
                dlogzm_dzm,
            )
        )

        return vals_n, Pwn, dPwn_dTwc, dPwn_dzm, median_depths, dlogzn_dlogzm, logzn

    def setup_system(self, point_inds):
        device = self.device

        num_kf = self.kf_poses.shape[0]
        num_recent = self.recent_poses.shape[0]
        num_landmarks = self.P_m.shape[0]

        kf_dim = 8 * num_kf
        recent_dim = 8 * num_recent
        geo_dim = 3 * num_landmarks
        dim = kf_dim + recent_dim + geo_dim
        H = torch.zeros((dim, dim), device=self.device, dtype=self.dtype)
        g = torch.zeros((dim,), device=self.device, dtype=self.dtype)

        kf_inds = torch.reshape(torch.arange(kf_dim, device=device), (num_kf, -1))

        if num_recent > 0:
            recent_inds = torch.reshape(
                torch.arange(recent_dim, device=device), (num_recent, -1)
            )
            # Align indices to system
            recent_inds += torch.max(kf_inds) + 1
            landmark_ind_start = torch.max(recent_inds) + 1
        else:
            recent_inds = torch.empty((0), device=self.device, dtype=torch.long)
            landmark_ind_start = torch.max(kf_inds) + 1

        landmark_inds = point_inds + landmark_ind_start
        landmark_inds_flat = (
            torch.reshape(torch.arange(geo_dim, device=device), (num_landmarks, 3))
            + landmark_ind_start
        )

        kf_pose_inds = kf_inds[:, :6]
        kf_aff_inds = kf_inds[:, 6:]

        return (
            H,
            g,
            kf_inds,
            recent_inds,
            landmark_inds,
            kf_pose_inds,
            kf_aff_inds,
            landmark_ind_start,
            landmark_inds_flat,
        )

    def store_vars(self, pm, logzm, Knm_Kmminv):
        # Store variables needed for correspondence
        self.pm = pm
        self.logzm = logzm

        log_depth_imgs = torch.permute(Knm_Kmminv @ logzm[:, None, :, :], (0, 3, 1, 2))
        self.depth_imgs = torch.exp(log_depth_imgs)

        b, _, h, w = self.depth_imgs.shape
        self.median_depths = torch.median(self.depth_imgs.view(b, h * w), dim=1).values

    def iterate(self):
        pm, dpm_dTwc, dpm_dPwm, logzm, dzm_dTwc, dlogzm_dzm, dzm_dPwm, point_inds = (
            self.prep_geometry_scaffold()
        )
        dlogzm_dTwc = dlogzm_dzm @ dzm_dTwc
        dlogzm_dPwm = dlogzm_dzm @ dzm_dPwm

        vals_n, Pwn, dPwn_dTwc, dPwn_dzm, median_depths, dlogzn_dlogzm, logzn = (
            self.prep_dense_ref(pm, logzm, dlogzm_dTwc, dlogzm_dzm)
        )

        self.store_vars(pm, logzm, self.Knm_Kmminv)

        (
            H,
            g,
            kf_inds,
            recent_inds,
            landmark_inds,
            kf_pose_inds,
            kf_aff_inds,
            landmark_ind_start,
            landmark_inds_flat,
        ) = self.setup_system(point_inds)

        mean_sq_photo_err, self.kf_pairs, self.one_way_pairs = create_photo_system(
            self.kf_poses,
            self.kf_aff_params,
            self.recent_poses,
            self.recent_aff_params,
            Pwn,
            dPwn_dTwc,
            dPwn_dzm,
            dzm_dPwm,
            median_depths,
            vals_n,
            self.kf_img_and_grads,
            self.recent_img_and_grads,
            self.kf_timestamps,
            self.recent_timestamps,
            self.intrinsics,
            H,
            g,
            self.cfg["photo_construction"],
            kf_inds,
            recent_inds,
            landmark_inds,
        )

        # Small prior to be close to median depth (mostly for pixels that have no overlap)
        log_median_depths = torch.log(self.median_depths[:, None, None])
        gp_prior_error = gp_ml_cost(
            logzm,
            log_median_depths,
            self.L_mm,
            dlogzm_dPwm,
            dlogzm_dTwc,
            landmark_inds,
            kf_pose_inds,
            H,
            g,
            sigma=1e0,
        )  # Sigma tunes how much of prior we want, too much makes planar

        # Small prior to be close to median depth (mostly for pixels that have no overlap)
        log_depth_error = log_depth_prior(
            logzm,
            log_median_depths,
            dlogzm_dPwm,
            dlogzm_dTwc,
            self.obs_ref_mask,
            landmark_inds,
            kf_pose_inds,
            H,
            g,
            mode="first_mean",
            sigma_first=1e0,
            sigma_all=1e-0,
        )

        pixel_error = pixel_prior_cost(
            pm,
            self.pm_first_obs,
            dpm_dPwm,
            dpm_dTwc,
            self.obs_ref_mask,
            landmark_inds,
            kf_pose_inds,
            H,
            g,
            mode="first",
            pixel_sigma_first=1e-2,
            pixel_sigma_all=3.33e-1,
        )

        pose_prior_err = linearize_pose_prior(
            self.kf_poses[0:1, ...],
            self.pose_anchor[0:1, ...],
            H,
            g,
            [kf_pose_inds[0, 0], kf_pose_inds[0, -1] + 1],
            sigma=self.cfg["sigmas"]["pose_prior"],
        )

        aff_scale_inds = [kf_aff_inds[0, 0], kf_aff_inds[0, 0] + 1]
        aff_scale_err = linearize_scalar_prior(
            self.kf_aff_params[0, 0:1, :],
            self.aff_anchor[0, 0:1, :],
            H,
            g,
            aff_scale_inds,
            sigma=self.cfg["sigmas"]["scale_prior"],
        )
        aff_bias_inds = [kf_aff_inds[0, 1], kf_aff_inds[0, 1] + 1]
        aff_bias_err = linearize_scalar_prior(
            self.kf_aff_params[0, 1:2, :],
            self.aff_anchor[0, 1:2, :],
            H,
            g,
            aff_bias_inds,
            sigma=self.cfg["sigmas"]["scale_prior"],
        )

        scale_err = 0.0
        fixed_landmark_err = 0.0
        # Priors on landmarks that had frames using it marginalized!
        if self.window_full:
            # All landmarks that are used in the new first frame of the window must be constrained
            # These landmarks were optimized by the keyframe just dropped from photometric error
            # To ensure consistency, must freeze these (actually marginalize + FEJ but this is simpler for now!)
            fix_mask = self.correspondence_mask[0, :]
            curr_P = self.P_m[fix_mask, :]
            fix_P_inds = landmark_inds_flat[fix_mask, :]
            fixed_landmark_err = linearize_multi_scalar_prior(
                curr_P.flatten(),
                self.P_m_anchors.flatten(),
                H,
                g,
                fix_P_inds.flatten(),
                sigma=self.cfg["sigmas"]["scale_prior"],
            )

        else:
            # Prior on mean log depth constrains scale initially
            mean_log_depth_prior = self.init_scale_anchor  # Matches TwoFrameSfm
            m = logzm.shape[1]
            scale_err = mean_log_depth_cost(
                logzm[0:1, ...],
                self.Knm_Kmminv[0:1, ...].view(1, -1, m),
                mean_log_depth_prior,
                dlogzm_dPwm[0:1, ...],
                dlogzm_dTwc[0:1, ...],
                landmark_inds[0:1, ...],
                kf_pose_inds[0:1, ...],
                H,
                g,
                self.cfg["sigmas"]["mean_depth_prior"],
            )

        total_err = (
            mean_sq_photo_err
            + gp_prior_error
            + log_depth_error
            + pixel_error
            + pose_prior_err
            + aff_scale_err
            + aff_bias_err
            + scale_err
            + fixed_landmark_err
        )

        # Solve and update
        delta = lin_sys.solve_system(H, g)

        (
            self.kf_poses,
            self.kf_aff_params,
            self.recent_poses,
            self.recent_aff_params,
            self.P_m,
        ) = lin_sys.update_vars(
            delta,
            self.kf_poses,
            self.kf_aff_params,
            kf_inds,
            self.recent_poses,
            self.recent_aff_params,
            recent_inds,
            self.P_m,
            landmark_ind_start,
        )

        # Evaluate convergence
        self.iter += 1
        delta_norm = torch.norm(delta)
        abs_decrease = self.total_err_prev - total_err
        rel_decrease = abs_decrease / self.total_err_prev
        # print(delta_norm, abs_decrease, rel_decrease)
        # print(total_err, self.total_err_prev)
        # if self.iter >= self.cfg["term_criteria"]["max_iter"] \
        #     or delta_norm < self.cfg["term_criteria"]["delta_norm"] \
        #     or abs_decrease < self.cfg["term_criteria"]["abs_tol"] \
        #     or rel_decrease < self.cfg["term_criteria"]["rel_tol"]:
        #   self.converged = True
        #   print("Mapping converged?", self.poses.shape[0], self.recent_poses.shape[0])

        self.total_err_prev = total_err

        return self.converged
