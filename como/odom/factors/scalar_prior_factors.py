import torch


def linearize_scalar_prior(scalar, meas, H, g, Dscalar, sigma):
    info_sqrt = 1.0 / sigma
    info = info_sqrt * info_sqrt
    # h(x) - z
    r = (scalar - meas)[:, 0]
    total_err = info * torch.square(r)
    # Gradient g = -Jt @ r
    gs = -r
    g[Dscalar[0] : Dscalar[1]] += info * gs
    # H = Jt @ J
    Hs = 1.0
    H[Dscalar[0] : Dscalar[1], Dscalar[0] : Dscalar[1]] += info * Hs

    return total_err


def linearize_multi_scalar_prior(scalar, meas, H, g, inds, sigma):
    info_sqrt = 1.0 / sigma
    info = info_sqrt * info_sqrt

    # h(x) - z
    r = scalar - meas
    total_err = info * torch.sum(torch.square(r))
    # Gradient g = -Jt @ r
    gs = -r
    g[inds] += info * gs
    # H = Jt @ J
    Hs = 1.0
    H[inds, inds] += info * Hs

    return total_err
