import torch


def squared_error(r):
    w = torch.ones_like(r)
    return w


def huber(r):
    k = 1.345
    unit = torch.ones((1), dtype=r.dtype, device=r.device)

    r_abs = torch.abs(r)
    mask = r_abs < k
    w = torch.where(mask, unit, k / r_abs)
    return w


def tukey(r, t=4.6851):
    zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)

    r_abs = torch.abs(r)
    tmp = 1 - torch.square(r_abs / t)
    tmp2 = tmp * tmp
    w = torch.where(r_abs < t, tmp2, zero)
    return w
