import torch  # Must import before Open3D when using CUDA!

import open3d.visualization.gui as gui
import numpy as np

import time

from como.gui.GuiWindow import GuiWindow
from como.odom.multiprocessing.TrackingMp import TrackingMp
from como.odom.multiprocessing.MappingMp import MappingMp
from como.utils.multiprocessing import TupleTensorQueue, release_data, init_gpu
from como.utils.o3d import rgb_depth_to_pcd


class ComoMp(GuiWindow):
    def __init__(self, viz_cfg, slam_cfg, dataset):
        super().__init__(viz_cfg, slam_cfg, dataset)

    def setup_slam_processes(self, slam_cfg):
        # Setup SLAM processes
        intrinsics = self.get_intrinsics()
        img_size = self.get_img_size()
        self.waitev = torch.multiprocessing.Event()
        self.tracking = TrackingMp(
            slam_cfg["tracking"], intrinsics, img_size, self.waitev
        )
        self.mapping = MappingMp(slam_cfg["mapping"], intrinsics, self.waitev)
        # Setup queues
        rgb_queue = TupleTensorQueue(
            self.tracking.device, self.tracking.dtype, maxsize=5
        )
        pose_viz_queue = TupleTensorQueue(self.device, self.dtype)  # Only want recent
        frame_queue = TupleTensorQueue(
            self.mapping.device, self.mapping.dtype, maxsize=1
        )
        kf_ref_queue = TupleTensorQueue(
            self.tracking.device, self.tracking.dtype
        )  # Only want recent
        kf_viz_queue = TupleTensorQueue(self.device, self.dtype)  # Only want recent

        self.rgb_queue = rgb_queue
        self.tracking.rgb_queue = rgb_queue
        self.tracking.tracking_pose_queue = pose_viz_queue
        self.tracking_pose_queue = pose_viz_queue
        self.tracking.frame_queue = frame_queue
        self.mapping.frame_queue = frame_queue
        self.mapping.kf_ref_queue = kf_ref_queue
        self.tracking.kf_ref_queue = kf_ref_queue
        self.mapping.kf_viz_queue = kf_viz_queue
        self.kf_viz_queue = kf_viz_queue

        # Warmup GPU for main process (others handle their own in run)
        init_gpu(self.device)

        # For real-time handling if dataset
        self.start_data_time = time.time()

        self.tracking.setup()
        self.mapping.setup()

    def start_slam_processes(self):
        self.tracking_done = False
        self.mapping_done = False

        print("Starting tracking and mapping processes...")
        self.tracking.start()
        self.mapping.start()
        print("Done.")

    def shutdown_slam_processes(self):
        self.waitev.set()
        print("Joining mapping...")
        self.mapping.join()
        print("Joining tracking...")
        self.tracking.join()
        print("Done.")

    def signal_slam_end(self):
        # Send end signal to queues
        self.rgb_queue.push(("end",))

        # Wait for response
        while not self.tracking_done:
            track_data_viz = self.tracking_pose_queue.pop_until_latest(block=True)
            if track_data_viz is not None:
                if track_data_viz[0] == "end":
                    self.tracking_done = True
            release_data(track_data_viz)

        while not self.mapping_done:
            kf_viz_data = self.kf_viz_queue.pop_until_latest(block=True)
            if kf_viz_data is not None:
                if kf_viz_data[0] == "end":
                    self.mapping_done = True
            release_data(kf_viz_data)

    def load_data(self, it):
        timestamp_next, rgb = next(it)
        # Real-time handling
        end_data_time = time.time()
        real_diff = end_data_time - self.start_data_time
        ts_diff = timestamp_next - self.timestamp
        sleep_time = max(0.0, ts_diff - real_diff)
        self.timestamp = timestamp_next
        # print(sleep_time)
        if not self.is_live:
            time.sleep(sleep_time)
        self.start_data_time = time.time()

        return self.timestamp, rgb

    def iter(self, timestamp, rgb):
        # Send input data to tracking and visualization
        track_data_in = (timestamp, rgb.clone())
        self.rgb_queue.push(track_data_in)  # Blocking until queue has spot
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_curr_image_render(rgb)
        )

        # Receive pose from tracking
        track_data_viz = self.tracking_pose_queue.pop_until_latest(
            block=False, timeout=0.01
        )
        if track_data_viz is not None:
            if track_data_viz[0] == "end":
                self.tracking_done = True
            else:
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
        release_data(track_data_viz)

        # Receive keyframes from mapping
        kf_viz_data = self.kf_viz_queue.pop_until_latest(block=False, timeout=0.01)
        if kf_viz_data is not None:
            if kf_viz_data[0] == "end":
                self.mapping_done = True
            else:
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
                ) = kf_viz_data

                self.update_kf_vars(
                    kf_timestamps, kf_rgbs, kf_depths, kf_poses, P_sparse
                )

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
                    ),
                )
        release_data(kf_viz_data)

        return
