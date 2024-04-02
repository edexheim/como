import argparse
import numpy as np

from como.utils.io import save_traj


def convert_traj(traj_dir):
    traj_path = traj_dir + "traj.txt"
    print(traj_path)

    T_cw_flat = np.loadtxt(traj_path)
    num_poses = T_cw_flat.shape[0]
    T_wc = np.reshape(T_cw_flat, (num_poses, 4, 4))

    timestamps = (1.0 / 30.0) * np.arange(num_poses)

    traj_path_new = traj_dir + "traj_tum.txt"
    save_traj(traj_path_new, timestamps, T_wc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to convert replica GT to TUM format."
    )
    parser.add_argument("traj_dir", type=str, help="Path to config file.")

    args = parser.parse_args()

    convert_traj(args.traj_dir)
