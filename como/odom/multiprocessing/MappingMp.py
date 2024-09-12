import time
import torch
import torch.multiprocessing as mp

from como.utils.multiprocessing import release_data
from como.odom.Mapping import Mapping

import os


class MappingMp(Mapping, mp.Process):
    def __init__(self, cfg, intrinsics, waitev):
        super().__init__(cfg, intrinsics)
        self.waitev = waitev

        # print(os.environ["OMP_WAIT_POLICY"])

        # torch.set_num_threads(1)

    def check_failure(self, kf_ref_data):
        # Check for NaN
        failed = False

        if kf_ref_data is not None:
            for data in kf_ref_data:
                if torch.is_tensor(data):
                    if data.isnan().any():
                        failed = True
                        break
            
            if failed:
                print("Reset Mapping")
                # Reset
                self.reset()

                # Propagate up queue
                self.frame_queue.push(("reset",))
                self.kf_viz_queue.push(("reset",))
        
        return failed

    def run(self):
        while True:
            # time.sleep(0.01)
            
            kf_updated = False
            if not self.is_init:
                # print("frame queue: ", self.frame_queue.qsize())
                data = self.frame_queue.pop_until_latest(block=True, timeout=0.001)
                if data is not None and data[0] == "init":
                    timestamp, rgb = data[1:]
                    kf_updated = self.attempt_two_frame_init(timestamp, rgb)
                release_data(data)
            else:
                # Handle one frame at a time
                # t1 = time.time()
                # print("frame queue: ", self.frame_queue.qsize())
                data = self.frame_queue.pop(block=True, timeout=0.001)
                
                # t2 = time.time()
                # print("Mapping frame_queue pop: ", t2-t1)
                
                if data is not None:
                    # t3 = time.time()
                    kf_viz_data, kf_updated = self.handle_tracking_data(data)
                    # t4 = time.time()
                    # print("Mapping handle_tracking_data: ", t4-t3)
                    if kf_viz_data is not None:
                        # t5 = time.time()
                        self.kf_viz_queue.push(kf_viz_data)
                        # t6 = time.time()
                        # print("Mapping kf_viz_queue push: ", t6-t5)
                    elif data[0] == "reset":
                        self.reset()
                    elif data[0] == "end":
                        break
                release_data(data)

            # Mapping iteration
            if self.is_init and not self.converged:
                # t7 = time.time()
                self.converged = self.iterate()
                kf_updated = True
                # t8 = time.time()
                # print("Mapping iterate: ", t8-t7)

            # Send updated mapping data if not sent for awhile
            curr_time = time.time()
            if self.is_init and (curr_time - self.last_kf_send_time > 1.0):
                kf_viz_data = self.get_kf_viz_data()
                self.kf_viz_queue.push(kf_viz_data)

            # Send updated keyframe data to queue
            if kf_updated:
                # t9 = time.time()
                kf_ref_data = self.get_kf_ref_data()
                self.kf_ref_queue.push(kf_ref_data)
                # t10 = time.time()
                # print("Mapping kf_ref_queue push: ", t10-t9)

        self.kf_ref_queue.push(("end",))
        self.kf_viz_queue.push(("end",))

        self.waitev.wait()

        return
