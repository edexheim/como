import time
import torch.multiprocessing as mp

from como.utils.multiprocessing import release_data
from como.odom.Mapping import Mapping


class MappingMp(Mapping, mp.Process):
    def __init__(self, cfg, intrinsics, waitev):
        super().__init__(cfg, intrinsics)
        self.waitev = waitev

    def run(self):
        while True:
            kf_updated = False
            if not self.is_init:
                data = self.frame_queue.pop_until_latest(block=True, timeout=0.01)
                if data is not None and data[0] == "init":
                    timestamp, rgb = data[1:]
                    kf_updated = self.attempt_two_frame_init(timestamp, rgb)
                release_data(data)
            else:
                # Handle one frame at a time
                data = self.frame_queue.pop(timeout=0.01)
                if data is not None:
                    kf_viz_data, kf_updated = self.handle_tracking_data(data)
                    if kf_viz_data is not None:
                        self.kf_viz_queue.push(kf_viz_data)
                    elif data[0] == "end":
                        break
                release_data(data)

            # Mapping iteration
            if self.is_init and not self.converged:
                self.converged = self.iterate()
                kf_updated = True

            # Send updated mapping data if not sent for awhile
            curr_time = time.time()
            if self.is_init and (curr_time - self.last_kf_send_time > 1.0):
                kf_viz_data = self.get_kf_viz_data()
                self.kf_viz_queue.push(kf_viz_data)

            # Send updated keyframe data to queue
            if kf_updated:
                kf_ref_data = self.get_kf_ref_data()
                self.kf_ref_queue.push(kf_ref_data)

        self.kf_ref_queue.push(("end",))
        self.kf_viz_queue.push(("end",))

        self.waitev.wait()

        return
