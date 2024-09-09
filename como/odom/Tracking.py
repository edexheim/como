import torch
import torchvision.transforms.functional as TF

from como.odom.frontend.photo_tracking import photo_tracking_pyr, precalc_jacobians
from como.geometry.affine_brightness import get_aff_w_curr, get_rel_aff
from como.geometry.transforms import get_T_w_curr, get_rel_pose, transform_points
from como.geometry.camera import backprojection, projection
from como.geometry.lie_algebra import invertSE3
from como.utils.config import str_to_dtype
from como.utils.image_processing import (
    ImageGradientModule,
    ImagePyramidModule,
    IntrinsicsPyramidModule,
    DepthPyramidModule,
)
from como.utils.coords import swap_coords_xy, get_test_coords, fill_image

from como.utils.multiprocessing import init_gpu


class Tracking:
    def __init__(self, cfg, intrinsics, img_size):
        super().__init__()

        self.cfg = cfg
        self.device = cfg["device"]
        self.dtype = str_to_dtype(cfg["dtype"])

        self.intrinsics = intrinsics
        self.img_size = img_size

    def track(self, data):
        raise NotImplementedError

    def setup(self):
        init_gpu(self.device)
        
        self.reset()
        return

    def reset(self):
        self.mapping_init = False
        self.init_basic_vars()
        self.init_kf_vars()
        self.reset_one_way_vars()
        self.T_w_rec_last = None
        return

    def init_basic_vars(self):
        self.intrinsics = self.intrinsics.to(device=self.device, dtype=self.dtype)

        start_level = self.cfg["pyr"]["start_level"]
        end_level = self.cfg["pyr"]["end_level"]
        depth_interp_mode = self.cfg["pyr"]["depth_interp_mode"]

        intrinsics_pyr_module = IntrinsicsPyramidModule(
            start_level, end_level, self.device
        )
        self.intrinsics_pyr = intrinsics_pyr_module(self.intrinsics, [1.0, 1.0])

        if self.cfg["color"] == "gray":
            c = 1
        elif self.cfg["color"] == "rgb":
            c = 3

        self.gradient_module = ImageGradientModule(
            channels=c, device=self.device, dtype=self.dtype
        )
        self.img_pyr_module = ImagePyramidModule(
            c, start_level, end_level, self.device, dtype=self.dtype
        )
        self.depth_pyr_module = DepthPyramidModule(
            start_level, end_level, depth_interp_mode, self.device
        )

    def reset_one_way_vars(self):
        self.num_one_way_since_kf = 0

        self.last_one_way_empty_pixels = 0

        self.last_flow_rmse = 0.0
        self.last_flow_wo_rot_rmse = 0.0

    def get_curr_world_pose(self):
        T_w_curr = get_T_w_curr(self.T_w_kf, self.T_curr_kf)
        return T_w_curr

    def get_curr_world_aff(self):
        aff_curr = get_aff_w_curr(self.aff_w_kf, self.aff_curr_kf)
        return aff_curr

    def prep_tracking_img(self, rgb):
        if self.cfg["color"] == "gray":
            img_tracking = TF.rgb_to_grayscale(rgb)
        elif self.cfg["color"] == "rgb":
            img_tracking = rgb.clone()
        img_pyr = self.img_pyr_module(img_tracking)
        return img_pyr

    def get_img_gradients(self, img_pyr):
        img_and_grads = []
        for l in range(len(img_pyr)):
            gx, gy = self.gradient_module(img_pyr[l])
            img_and_grads_level = torch.cat((img_pyr[l], gx, gy), dim=1)
            img_and_grads.append(img_and_grads_level)
        return img_and_grads

    # These are variables relative to a reference keyframe
    # Affine brightness parameters are not global for that frame!
    def init_kf_vars(self):
        self.T_curr_kf = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.aff_curr_kf = torch.zeros((1, 2, 1), device=self.device, dtype=self.dtype)
        self.last_one_way_num_pixels = self.img_size[-1] * self.img_size[-2]

        self.last_kf_sent_ts = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.kf_received_ts = torch.zeros(1, device=self.device, dtype=self.dtype)

    def check_keyframe(self, median_depth, num_reproj_depth, T_curr_kf):
        new_kf = False

        num_kf_pixels = self.vals_pyr[-1].shape[1]

        # Need to have received new kf from mapping to avoid immediately setting keyframe
        if self.last_kf_sent_ts <= self.kf_received_ts:
            kf_dist = torch.linalg.norm(T_curr_kf[:, :3, 3])
            if kf_dist > self.cfg["keyframing"]["kf_depth_motion_ratio"] * median_depth:
                new_kf = True
            elif (
                self.cfg["keyframing"]["kf_num_pixels_frac"]
                > num_reproj_depth / num_kf_pixels
            ):
                new_kf = True
        # else:
        #   print("Keyframe ", self.last_kf_sent_ts, " still not received, continue tracking against kf ", self.kf_received_ts)

        return new_kf

    def check_one_way_frame(self, median_depth, num_reproj_depth, T_curr_kf, T_w_curr):
        new_one_way_frame = False

        # Make threshold larger if waiting for keyframe to come soon
        extra_count = 0
        if self.last_kf_sent_ts > self.kf_received_ts:
            extra_count = 1

        # Modify threshold depending on how many num one way frames
        thresh_scale_kf = (1.0 + self.num_one_way_since_kf + extra_count) / (
            1.0 + self.cfg["keyframing"]["one_way_freq"]
        )

        dist_thresh = self.cfg["keyframing"]["kf_depth_motion_ratio"] * median_depth
        num_kf_pixels = self.vals_pyr[-1].shape[1]
        pixel_thresh = (
            1 - self.cfg["keyframing"]["kf_num_pixels_frac"]
        ) * num_kf_pixels

        # Number of empty pixels from KF reference
        num_empty_pixels = num_kf_pixels - num_reproj_depth

        # Thresholds wrt KF params
        kf_dist = torch.linalg.norm(T_curr_kf[:, :3, 3])
        if kf_dist > thresh_scale_kf * dist_thresh:
            new_one_way_frame = True
        elif num_empty_pixels > thresh_scale_kf * pixel_thresh:
            new_one_way_frame = True

        if new_one_way_frame:
            self.last_one_way_empty_pixels = num_empty_pixels
            self.T_w_rec_last = T_w_curr

        return new_one_way_frame

    def get_reproj_last_kf(self, T_curr_kf):
        P_last_kf = self.P_pyr[-1][None, -1, :, :]
        P_curr, _, _ = transform_points(T_curr_kf, P_last_kf)
        p_proj, _ = projection(self.intrinsics_pyr[-1], P_curr)
        coords_proj = swap_coords_xy(p_proj)
        depth_curr = P_curr[:, :, 2:3]

        # Mask out valid coords and depths
        def get_valid_reproj_mask(p, depth, img_size):
            valid_x = torch.logical_and(p[:, :, 0] > 0, p[:, :, 0] < img_size[-1] - 1)
            valid_y = torch.logical_and(p[:, :, 1] > 0, p[:, :, 1] < img_size[-2] - 1)
            valid_mask = torch.logical_and(valid_x, valid_y)
            valid_mask = torch.logical_and(valid_mask, depth[:, :, 0] > 0.0)
            return valid_mask

        mask = get_valid_reproj_mask(p_proj, depth_curr, self.img_size)
        coords_filt = coords_proj[mask, :]
        depth_filt = depth_curr[mask, :]
        reproj_depth = fill_image(coords_filt, depth_filt, self.img_size)
        return reproj_depth

    # Assumes KF data image sizes is same as what goes into tracking
    def update_kf_reference(self, kf_data):
        timestamps, kf_rgb, kf_pose, kf_aff, depth = kf_data

        # Update curr frame to kf variables
        if timestamps[-1] > self.kf_received_ts and self.mapping_init:
            num_kf = kf_pose.shape[0]

            self.T_w_f = get_T_w_curr(self.T_w_kf, self.T_curr_kf)
            self.T_curr_kf = get_rel_pose(self.T_w_f, kf_pose[num_kf - 1 : num_kf])

            self.aff_w_f = get_aff_w_curr(self.aff_w_kf, self.aff_curr_kf)
            self.aff_curr_kf = get_rel_aff(self.aff_w_f, kf_aff[num_kf - 1 : num_kf])

            # Don't have this info but assume full image
            self.reset_one_way_vars()

        elif not self.mapping_init:
            self.mapping_init = True
            self.last_kf_sent_ts = timestamps[-1]

        # Completely new keyframe, update photometric vars
        if timestamps[-1] != self.kf_received_ts:
            # Photometric
            img_pyr = self.prep_tracking_img(kf_rgb)
            self.coords_pyr = []
            self.vals_pyr = []
            self.img_grads_pyr = []
            for i in range(len(img_pyr)):
                gx, gy = self.gradient_module(img_pyr[i])

                num_kf = img_pyr[i].shape[0]
                test_coords = get_test_coords(
                    img_pyr[i].shape[-2:], device=self.device, batch_size=num_kf
                )
                num_coords = test_coords.shape[1]

                batch_inds = (
                    torch.arange(num_kf, device=self.device)
                    .unsqueeze(1)
                    .repeat(1, num_coords)
                )
                vals = img_pyr[i][
                    batch_inds, :, test_coords[:, :, 0], test_coords[:, :, 1]
                ]
                self.vals_pyr.append(vals)  # (B,N,C)

                gx = gx[batch_inds, :, test_coords[:, :, 0], test_coords[:, :, 1]]
                gy = gy[batch_inds, :, test_coords[:, :, 0], test_coords[:, :, 1]]
                dI_dw = torch.stack((gx, gy), dim=-1)  # (B,N,C,2)
                self.img_grads_pyr.append(dI_dw)  # (B,N,2C)
                self.coords_pyr.append(test_coords)

        # Compute variables involving geometry regardless
        self.P_pyr = []
        self.dI_dT_pyr = []
        self.mask_pyr = []
        depth_pyr = self.depth_pyr_module(depth)
        for i in range(len(depth_pyr)):
            test_coords = self.coords_pyr[i]

            num_kf = test_coords.shape[0]
            num_coords = test_coords.shape[1]
            batch_inds = (
                torch.arange(num_kf, device=self.device)
                .unsqueeze(1)
                .repeat(1, num_coords)
            )

            depths = depth_pyr[i][
                batch_inds, 0, test_coords[:, :, 0], test_coords[:, :, 1]
            ]
            depths = depths.unsqueeze(-1)

            test_coords_xy = swap_coords_xy(test_coords)
            P, _ = backprojection(self.intrinsics_pyr[i], test_coords_xy, depths)

            rel_poses = (
                invertSE3(kf_pose[num_kf - 1 : num_kf]) @ kf_pose
            )  # Transform points from any kf to last kf
            P_all, _, _ = transform_points(rel_poses, P)  # (B,N,3)

            # Only use points that project close to camera boundaries and in front
            # NOTE: In theory this can be outside the FoV of the camera, but want to avoid very bad points
            p_all, _ = projection(self.intrinsics_pyr[i], P_all)

            def get_valid_reproj_mask(p, depth, img_size, img_border, depth_thresh):
                valid_x = torch.logical_and(
                    p[:, :, 0] >= -img_border,
                    p[:, :, 0] <= img_size[-1] - 1 + img_border,
                )
                valid_y = torch.logical_and(
                    p[:, :, 1] >= -img_border,
                    p[:, :, 1] <= img_size[-2] - 1 + img_border,
                )
                valid_mask = torch.logical_and(valid_x, valid_y)
                valid_mask = torch.logical_and(
                    valid_mask, depth[:, :, 0] > depth_thresh
                )
                return valid_mask

            mask = get_valid_reproj_mask(
                p_all,
                P_all[
                    :,
                    :,
                    2:3,
                ],
                depth_pyr[i].shape[-2:],
                img_border=50,
                depth_thresh=1e-4,
            )

            dI_dT = precalc_jacobians(
                self.img_grads_pyr[i], P_all, self.vals_pyr[i], self.intrinsics_pyr[i]
            )

            self.P_pyr.append(P_all)
            self.dI_dT_pyr.append(dI_dT)
            self.mask_pyr.append(mask)

        num_kf = kf_pose.shape[0]
        self.kf_received_ts = timestamps[-1]
        self.T_w_kf = kf_pose[num_kf - 1 : num_kf]
        self.aff_w_kf = kf_aff[num_kf - 1 : num_kf]

    def handle_frame(self, data):
        timestamp, rgb = data

        # Track against reference
        img_pyr = self.prep_tracking_img(rgb)
        self.T_curr_kf, self.aff_curr_kf = photo_tracking_pyr(
            self.T_curr_kf,
            self.aff_curr_kf,
            self.vals_pyr,
            self.P_pyr,
            self.dI_dT_pyr,
            self.mask_pyr,
            self.intrinsics_pyr,
            img_pyr,
            self.cfg["sigmas"]["photo"],
            self.cfg["term_criteria"],
        )

        # Send tracked pose
        T_w_curr = self.get_curr_world_pose()

        track_data_viz = (timestamp, T_w_curr.clone())

        # Decide if keyframe or one-way frame for mapping
        track_data_map = None

        reproj_depth = self.get_reproj_last_kf(self.T_curr_kf)
        valid_depth_mask = ~torch.isnan(reproj_depth)
        num_valid_reproj_depth = torch.count_nonzero(valid_depth_mask)
        median_depth = torch.median(reproj_depth[valid_depth_mask])

        new_kf = self.check_keyframe(
            median_depth, num_valid_reproj_depth, self.T_curr_kf
        )
        if new_kf:
            track_data_map = (
                "keyframe",
                rgb.clone(),
                self.T_curr_kf,
                self.aff_curr_kf,
                self.kf_received_ts,
                timestamp,
            )
            # Need this to know whether tracking against older keyframe
            self.last_kf_sent_ts = timestamp
        else:
            # Try to see if add one way frame
            new_one_way_frame = self.check_one_way_frame(
                median_depth, num_valid_reproj_depth, self.T_curr_kf, T_w_curr
            )
            if new_one_way_frame:
                track_data_map = (
                    "one-way",
                    rgb.clone(),
                    self.T_curr_kf,
                    self.aff_curr_kf,
                    self.kf_received_ts,
                    timestamp,
                )

                self.last_rec_sent_ts = timestamp
                self.num_one_way_since_kf += 1

        return track_data_viz, track_data_map
