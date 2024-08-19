import open3d.visualization.gui as gui
import numpy as np

from como.gui.GuiWindow import GuiWindow
from como.utils.multiprocessing import transfer_data

from como.odom.sequential.TrackingSeq import TrackingSeq
from como.odom.sequential.MappingSeq import MappingSeq
from como.utils.o3d import rgb_depth_to_pcd


class ComoSeq(GuiWindow):
    def __init__(self, viz_cfg, slam_cfg, dataset):
        super().__init__(viz_cfg, slam_cfg, dataset)

    def setup_slam_processes(self, slam_cfg):
        # Setup SLAM processes
        intrinsics = self.get_intrinsics()
        img_size = self.get_img_size()

        self.tracking = TrackingSeq(slam_cfg["tracking"], intrinsics, img_size)
        self.mapping = MappingSeq(slam_cfg["mapping"], intrinsics)

        self.tracking.setup()
        self.mapping.setup()

    def start_slam_processes(self):
        self.tracking_done = False
        self.mapping_done = False

    def shutdown_slam_processes(self):
        print("Done.")

    def signal_slam_end(self):
        self.tracking_done = True
        self.mapping_done = True

    def load_data(self, it):
        timestamp, rgb = next(it)
        return timestamp, rgb

    def iter(self, timestamp, rgb):
        # Send input data to tracking and visualization
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_curr_image_render(rgb)
        )
        # Track if init, otherwise send raw data to mapping for initialization
        if self.mapping.is_init:
            track_data_in = (timestamp, rgb.clone())
            track_data_in = transfer_data(
                track_data_in, self.tracking.device, self.tracking.dtype
            )
            track_data_viz, track_data_map = self.tracking.track(track_data_in)
            # Handle tracking viz data
            track_data_viz = transfer_data(track_data_viz, self.device, self.dtype)
            tracked_timestamp, tracked_pose = track_data_viz
            # Record data
            self.timestamps.append(tracked_timestamp)
            self.est_poses = np.concatenate((self.est_poses, tracked_pose))
            # Visualize tracked pose
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_pose_render(tracked_pose)
            )
            # Visualize background using shaders if using
            if self.render_val == "Phong":
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.render_o3d_image()
                )
        else:
            track_data_map = ("init", timestamp, rgb.clone())

        ## Handle tracking map data
        kf_viz_data, kf_ref_data = self.mapping.map(track_data_map)
        # Update tracking kf ref
        if kf_ref_data is not None:
            kf_ref_data = transfer_data(
                kf_ref_data, self.tracking.device, self.tracking.dtype
            )
            self.tracking.update_kf_reference(kf_ref_data)
        # Visualization and bookkeeping
        if kf_viz_data is not None:
            kf_viz_data = transfer_data(kf_viz_data, self.device, self.dtype)
            (
                kf_timestamps,
                kf_rgbs,
                kf_poses,
                kf_depths,
                kf_sparse_coords,
                P_sparse,
                obs_ref_mask,
                one_way_poses,
                kf_pairs,
                one_way_pairs,
                K_mm, corr_mask
            ) = kf_viz_data
            # Storing variables to save later
            self.update_kf_vars(kf_timestamps, kf_rgbs, kf_depths, kf_poses, P_sparse)

            pcd = None
            kf_normals = None
            if self.render_val == "Point Cloud":
                pcd, kf_normals = rgb_depth_to_pcd(
                    kf_rgbs,
                    kf_depths,
                    kf_poses,
                    self.get_intrinsics(),
                    self.cfg["cos_thresh"],
                )

            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda: self.update_keyframe_render(
                    kf_timestamps,
                    kf_rgbs,
                    kf_poses,
                    kf_depths,
                    kf_sparse_coords,
                    P_sparse,
                    obs_ref_mask,
                    one_way_poses,
                    kf_pairs,
                    one_way_pairs,
                    pcd,
                    kf_normals,
                    K_mm, corr_mask
                ),
            )

        return
