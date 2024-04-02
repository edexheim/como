import torch


def str_to_dtype(str):
    if str == "float":
        return torch.float
    elif str == "double":
        return torch.double
    else:
        raise ValueError("Cannot convert : " + str + " to tensor type.")
