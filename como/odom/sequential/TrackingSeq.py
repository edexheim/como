from como.odom.Tracking import Tracking


class TrackingSeq(Tracking):
    def __init__(self, cfg, intrinsics, img_size):
        super().__init__(cfg, intrinsics, img_size)

    def track(self, data):
        return self.handle_frame(data)
