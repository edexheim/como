import torch
import open3d.visualization.gui as gui

from como.odom.multiprocessing.ComoMp import ComoMp
from como.data.dataset_factory import get_dataset

import yaml
import argparse


def main(dataset):
    torch.manual_seed(0)

    ## Parameters
    with open("./config/open3d_viz.yml", "r") as file:
        viz_cfg = yaml.safe_load(file)
    # Odometry setup
    with open("./config/como.yml", "r") as file:
        slam_cfg = yaml.safe_load(file)

    # Open3D visualization setup
    app = gui.Application.instance
    app.initialize()
    viz_window = ComoMp(viz_cfg, slam_cfg, dataset)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    img_size = [192, 256]
    dataset = get_dataset(args.dataset_type, img_size, args.dataset_dir)

    main(dataset)
