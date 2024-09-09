import torch
import torch.multiprocessing as mp

from como.odom.Tracking import Tracking
from como.utils.multiprocessing import release_data


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
            kf_data = self.kf_ref_queue.pop_until_latest(block=True, timeout=0.01)
            if kf_data is not None:
                if kf_data[0] == "reset":
                    self.reset()
                elif kf_data[0] == "end":
                    self.tracking_pose_queue.push(("end",))
                    break
                else:
                    self.update_kf_reference(kf_data)
            release_data(kf_data)

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

        self.waitev.wait()

        return

    def track(self, data):
        track_data_viz, track_data_map = self.handle_frame(data)

        if not self.check_failure(track_data_map):
            # Send data to viz and mapping
            self.tracking_pose_queue.push(track_data_viz)
            if track_data_map is not None:
                self.frame_queue.push(track_data_map)

        return
