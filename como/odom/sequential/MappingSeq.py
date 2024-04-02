import time

from como.utils.multiprocessing import transfer_data
from como.odom.Mapping import Mapping


class MappingSeq(Mapping):
    def __init__(self, cfg, intrinsics):
        super().__init__(cfg, intrinsics)

    def map(self, data):
        kf_viz_data = None
        kf_ref_data = None

        kf_updated = False

        # Handle incoming data
        if data is not None:
            data = transfer_data(data, self.device, self.dtype)

            if not self.is_init:
                if data[0] == "init":
                    timestamp, rgb = data[1:]
                    kf_updated = self.attempt_two_frame_init(timestamp, rgb)
            else:
                kf_viz_data, kf_updated = self.handle_tracking_data(data)

        # Mapping iteration
        if self.is_init and not self.converged:
            self.converged = self.iterate()
            kf_updated = True

        # Send updated mapping data if not sent for awhile
        curr_time = time.time()
        if self.is_init and (curr_time - self.last_kf_send_time > 1.0):
            kf_viz_data = self.get_kf_viz_data()

        # Send updated keyframe data after iteration
        if data is not None:
            if data[0] == "keyframe":
                kf_viz_data = self.get_kf_viz_data()

        # Send updated keyframe data
        if kf_updated:
            kf_ref_data = self.get_kf_ref_data()

        return kf_viz_data, kf_ref_data
