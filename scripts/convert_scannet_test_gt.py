import argparse
import numpy as np
import os
import glob
import re

from como.utils.io import save_traj


def convert_traj(traj_dir):
    poses_list = glob.glob(os.path.join(traj_dir, "pose/*.txt"))
    poses_list = sorted(
        poses_list, key=lambda x: int(re.findall("\d+", x.rsplit("/", 1)[-1])[0])
    )

    T_wc = np.zeros((0, 4, 4))
    timestamps = []
    for i, file_name in enumerate(poses_list):
        pose_np = np.loadtxt(file_name)
        pose_np = np.expand_dims(pose_np, axis=0)
        if np.isfinite(pose_np).all():
            T_wc = np.concatenate((T_wc, pose_np))
            timestamps.append((1.0 / 30.0) * i)

    traj_path_new = traj_dir + "traj_tum.txt"
    save_traj(traj_path_new, timestamps, T_wc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to convert ScanNet GT to TUM format."
    )
    parser.add_argument("traj_dir", type=str, help="Path to config file.")

    args = parser.parse_args()

    convert_traj(args.traj_dir)
