import torch
import torch.multiprocessing as mp

from como.odom.Tracking import Tracking
from como.utils.multiprocessing import release_data

import time

class TrackingMp(Tracking, mp.Process):
    def __init__(self, cfg, intrinsics, img_size, waitev):
        super().__init__(cfg, intrinsics, img_size)
        self.waitev = waitev

    def check_failure(self, track_data_map):
        failed = False
        if track_data_map is not None:
            # Check for NaN
            for data in track_data_map:
                if torch.is_tensor(data):
                    if data.isnan().any():
                        failed = True
                        break
            
            if failed:
                print("Reset Tracking")
                # Reset
                self.reset()

                # Propagate up queue
                self.frame_queue.push(("reset",))
                self.tracking_pose_queue.push(("reset",))
        
        return failed

    def run(self):
        while True:
            # t1 = time.time()

            kf_data = self.kf_ref_queue.pop_until_latest(block=False, timeout=0.01)
            
            # t2 = time.time()

            if kf_data is not None:
                if kf_data[0] == "reset":
                    self.reset()
                elif kf_data[0] == "end":
                    self.tracking_pose_queue.push(("end",))
                    break
                else:
                    self.update_kf_reference(kf_data)
            release_data(kf_data)

            # t3 = time.time()

            # Get new RGB
            data = self.rgb_queue.pop(timeout=0.001)
            if data is not None:
                if data[0] == "end":
                    # Signal mapping queue
                    self.frame_queue.push(("end",))
                elif not self.mapping_init:
                    timestamp, rgb = data
                    self.frame_queue.push(("init", timestamp, rgb.clone()))
                else:
                    self.track(data)
            release_data(data)

            # t4 = time.time()

            # print("Tracking")
            # print("Get data: ", t2-t1)
            # print("Update KF ref: ", t3-t2)
            # print("Track total: ", t4-t3)

        self.waitev.wait()

        return

    def track(self, data):
        # t5 = time.time()

        track_data_viz, track_data_map = self.handle_frame(data)

        # t6 = time.time()

        # print("handle_frame: ", t6-t5)

        if not self.check_failure(track_data_map):
            # Send data to viz and mapping
            # t7 = time.time()
            self.tracking_pose_queue.push(track_data_viz)
            # t8 = time.time()
            if track_data_map is not None:
                self.frame_queue.push(track_data_map)
            # t9 = time.time()

            # print("pose_queue: ", t8-t7)
            # print("frame_queue: ", t9-t8)

        return
