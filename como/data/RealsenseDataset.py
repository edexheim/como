import torch
from torch.utils.data import IterableDataset
import torchvision.transforms.functional as TF

import cv2
import pyrealsense2 as rs
import numpy as np

import time

from como.geometry.camera import resize_intrinsics


class RealsenseDataset(IterableDataset):
    def __init__(self, img_size, cfg):
        super().__init__()
        self.is_live = True
        self.img_size = img_size
        self.cfg = cfg

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.save_traj_name = "realsense_" + timestr

        self.start()

    def start(self):
        config = rs.config()
        config.enable_stream(
            stream_type=rs.stream.color,
            width=self.cfg["width"],
            height=self.cfg["height"],
            framerate=self.cfg["fps"],
        )

        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)

        rgb_sensor = profile.get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.enable_auto_exposure, True)
        rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        rgb_sensor.set_option(rs.option.exposure, 100)

        rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        rgb_intrinsics = rgb_profile.get_intrinsics()

        size_orig = torch.tensor([rgb_intrinsics.height, rgb_intrinsics.width])
        image_scale_factors = torch.tensor(self.img_size) / size_orig

        intrinsics_orig = torch.tensor(
            [
                [rgb_intrinsics.fx, 0.0, rgb_intrinsics.ppx],
                [0.0, rgb_intrinsics.fy, rgb_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ]
        )
        distortion = np.asarray(rgb_intrinsics.coeffs)

        ## NOTE: With 0 distortion, getOptimalNewCameraMatrix gives different K,
        # and initUndistortRectifyMap will have a map with values at the borders...

        # Setup distortion
        if distortion is not None:
            orig_img_size = [size_orig[1].item(), size_orig[0].item()]
            K = intrinsics_orig.numpy()
            # alpha = 0.0 means invalid pixels are cropped, while 1.0 means all original pixels are present in new image
            K_u, validPixROI = cv2.getOptimalNewCameraMatrix(
                K, distortion, orig_img_size, alpha=0, newImgSize=orig_img_size
            )
            # TODO: What type to use for maps?
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, distortion, None, K_u, orig_img_size, cv2.CV_32FC1
            )
            intrinsics_orig = torch.from_numpy(K_u)
        else:
            self.map1, self.map2 = None, None
            intrinsics_orig = intrinsics_orig

        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

    def shutdown(self):
        self.pipeline.stop()

    def __len__(self):
        return 1.0e10

    def __iter__(self):
        return self

    def __next__(self):
        frameset = self.pipeline.wait_for_frames()

        timestamp = frameset.get_timestamp()
        timestamp /= 1000.0  # original in ms

        rgb_frame = frameset.get_color_frame()
        rgb_np = np.asanyarray(rgb_frame.get_data())

        # Undistort
        if self.map1 is not None:
            rgb_np_u = cv2.remap(rgb_np, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            rgb_np_u = rgb_np
        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(
            rgb_np_u, new_img_size, interpolation=cv2.INTER_LINEAR
        )
        rgb = TF.to_tensor(rgb_np_resized)

        return timestamp, rgb
