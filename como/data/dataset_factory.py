import yaml

from como.data.odom_datasets import (
    ReplicaDataset,
    TumOdometryDataset,
    ScanNetOdometryDataset,
)
from como.data.RealsenseDataset import RealsenseDataset


def get_dataset(dataset_type, img_size, dataset_dir):
    if dataset_type == "replica":
        dataset = ReplicaDataset(dataset_dir, img_size)
    elif dataset_type == "tum":
        dataset = TumOdometryDataset(dataset_dir, img_size)
    elif dataset_type == "scannet":
        dataset = ScanNetOdometryDataset(dataset_dir, img_size)
    elif dataset_type == "realsense":
        with open("./config/realsense.yml", "r") as file:
            rs_cfg = yaml.safe_load(file)
        dataset = RealsenseDataset(img_size, rs_cfg)
    else:
        raise ValueError("dataset_type mode: " + dataset_type + " is not implemented.")

    return dataset
