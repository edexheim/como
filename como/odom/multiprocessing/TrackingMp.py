import torch
import torch.multiprocessing as mp

from como.odom.Tracking import Tracking
from como.utils.multiprocessing import release_data

import time

import os

class TrackingMp(Tracking, mp.Process):
    def __init__(self, cfg, intrinsics, img_size, waitev):
        super().__init__(cfg, intrinsics, img_size)
        self.waitev = waitev

        # print(os.environ["OMP_WAIT_POLICY"])

        # torch.set_num_threads(1)

    def check_failure(self, track_data_map, track_data_viz):
        failed = False
        if track_data_map is not None:
            # Check for NaN
            for data in track_data_map:
                if torch.is_tensor(data):
                    if data.isnan().any():
                        failed = True
                        print("Track data map NaN")
                        break
            
            for data in track_data_viz:
                if torch.is_tensor(data):
                    if data.isnan().any():
                        failed = True
                        print("Track data viz NaN")
                        break
            
            if failed:
                print("Reset Tracking start")
                # Reset
                self.reset()

                # Empty all queues
                # Receivers
                self.kf_ref_queue.pop_until_latest(block=False, timeout=0.01)
                self.rgb_queue.pop_until_latest(block=False, timeout=0.01)

                # Propagate up queue
                print("Pushing resets to queues")
                self.frame_queue.push(("reset",))
                self.tracking_pose_queue.push(("reset",))

                print("Reset Tracking done")
        
        return failed

    def run(self):
        while True:
            # t1 = time.time()
            # time.sleep(0.01)

            # print("KF ref queue: ", self.kf_ref_queue.qsize())
            kf_data = self.kf_ref_queue.pop_until_latest(block=True, timeout=0.001)
            
            # t2 = time.time()

            if kf_data is not None:
                if kf_data[0] == "reset":
                    self.reset()
                elif kf_data[0] == "end":
                    self.tracking_pose_queue.push(("end",))
                    break
                else:
                    # t3 = time.time()
                    self.update_kf_reference(kf_data)
                    # t4 = time.time()
                    # print("Tracking Update KF ref: ", t3-t2)
            release_data(kf_data)

            # Get new RGB
            # print("rgb queue: ", self.rgb_queue.qsize())
            rgb_size = self.rgb_queue.qsize()
            # data = self.rgb_queue.pop(block=True, timeout=0.001)
            data = self.rgb_queue.pop_until_latest(block=True, timeout=0.001)
            if data is not None:
                if data[0] == "end":
                    # Signal mapping queue
                    self.frame_queue.push(("end",))
                elif not self.mapping_init:
                    timestamp, rgb = data
                    self.frame_queue.push(("init", timestamp, rgb.clone()))
                else:
                    self.track(data, rgb_size)
            release_data(data)

            # t4 = time.time()

            # print("Tracking Get data: ", t2-t1)
            # print("Tracking Track total: ", t4-t3)

        self.waitev.wait()

        return

    def track(self, data, rgb_size):
        # t5 = time.time()

        track_data_viz, track_data_map = self.handle_frame(data)

        # t6 = time.time()

        # print("Tracking handle_frame: ", t6-t5, " queue size: ", rgb_size)

        if not self.check_failure(track_data_map, track_data_viz):
            # Send data to viz and mapping
            # t7 = time.time()
            self.tracking_pose_queue.push(track_data_viz)
            # t8 = time.time()
            if track_data_map is not None:
                self.frame_queue.push(track_data_map)
            # t9 = time.time()

            # print("Tracking pose_queue: ", t8-t7)
            # print("Tracking frame_queue: ", t9-t8)

        return
