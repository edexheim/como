import torch  # Must import before Open3D when using CUDA!

import open3d.visualization.gui as gui
import numpy as np

import time

import os

from como.gui.GuiWindow import GuiWindow
from como.odom.multiprocessing.TrackingMp import TrackingMp
from como.odom.multiprocessing.MappingMp import MappingMp
from como.utils.multiprocessing import TupleTensorQueue, release_data, init_gpu
from como.utils.o3d import rgb_depth_to_pcd


class ComoMp(GuiWindow):
    def __init__(self, viz_cfg, slam_cfg, dataset):
        super().__init__(viz_cfg, slam_cfg, dataset)

        # print(os.environ["OMP_WAIT_POLICY"])

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
            self.tracking.device, self.tracking.dtype, maxsize=30
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

        # time.sleep(0.01)

        # t1 = time.time()

        # TODO: Skip frame if full?

        # Send input data to tracking and visualization
        track_data_in = (timestamp, rgb.clone())
        self.rgb_queue.push(track_data_in)  # Blocking until queue has spot

        # t2 = time.time()
        
        # self.update_curr_image_lock.acquire()
        update_curr_image_done = self.update_curr_image_done
        # self.update_curr_image_lock.release()
        if update_curr_image_done:
            # self.update_curr_image_lock.acquire()
            self.update_curr_image_done = False
            # self.update_curr_image_lock.release()
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_curr_image_render(rgb)
            )

        # t3 = time.time()

        def reset_scene():
            # Clear geometries and background
            self.widget3d.scene.clear_geometry()
            self.widget3d.scene.set_background([1, 1, 1, 0])
            
            # Clear queues
            self.tracking_pose_queue.pop_until_latest(block=True, timeout=0.001)
            self.kf_viz_queue.pop_until_latest(block=True, timeout=0.001)

        # Receive pose from tracking
        # print("Track viz queue: ", self.tracking_pose_queue.qsize())
        track_data_viz = self.tracking_pose_queue.pop_until_latest(
            block=True, timeout=0.001
        )

        # t4 = time.time()

        if track_data_viz is not None:
            if track_data_viz[0] == "reset":
                print("Viz tracking reset")
                # Clear geometries and background
                gui.Application.instance.post_to_main_thread(
                    self.window, reset_scene)
            elif track_data_viz[0] == "end":
                self.tracking_done = True
            else:
                tracked_timestamp, tracked_pose = track_data_viz
                # Record data
                # self.timestamps.append(tracked_timestamp)
                # self.est_poses = np.concatenate((self.est_poses, tracked_pose))
                self.last_pose = tracked_pose[0].clone().numpy()
                
                # Visualize tracked pose
                # self.update_pose_render_lock.acquire()
                update_pose_render_done = self.update_pose_render_done
                # self.update_pose_render_lock.release()
                if update_pose_render_done:
                    # self.update_pose_render_lock.acquire()
                    self.update_pose_render_done = False
                    # self.update_pose_render_lock.release()
                    gui.Application.instance.post_to_main_thread(
                        self.window, lambda: self.update_pose_render(tracked_pose)
                    )

                # Visualize background using shaders if using
                if self.render_val == "Phong":
                    # self.render_o3d_lock.acquire()
                    render_o3d_done = self.render_o3d_done
                    # self.render_o3d_lock.release()
                    if render_o3d_done:
                        # self.render_o3d_lock.acquire()
                        self.render_o3d_done = False
                        # self.render_o3d_lock.release()
                        # Visualize background using shaders if using
                        gui.Application.instance.post_to_main_thread(
                            self.window, lambda: self.render_o3d_image()
                        )
        
        release_data(track_data_viz)

        # t5 = time.time()

        # Receive keyframes from mapping
        # print("KF viz queue: ", self.kf_viz_queue.qsize())
        kf_viz_data = self.kf_viz_queue.pop_until_latest(block=True, timeout=0.001)

        # t6 = time.time()

        if kf_viz_data is not None:
            if kf_viz_data[0] == "reset":
                print("Viz mapping reset")
                # Clear geometries and background
                gui.Application.instance.post_to_main_thread(
                    self.window, reset_scene)
                
            elif kf_viz_data[0] == "end":
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

                # self.update_keyframe_lock.acquire()
                update_keyframe_done = self.update_keyframe_done
                # self.update_keyframe_lock.release()
                if update_keyframe_done:
                    # self.update_keyframe_lock.acquire()
                    self.update_keyframe_done = False
                    # self.update_keyframe_lock.release()
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

        # time.sleep(0.01)

        # t7 = time.time()

        # print("Idx: ", self.idx, "RGB push: ", t2-t1)
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)
        # print(t5-t4)
        # print(t6-t5)
        # print(t7-t6)

        return
