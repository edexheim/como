import torch


def safe_sqrt(x):
    return torch.sqrt(x + 1e-8)
