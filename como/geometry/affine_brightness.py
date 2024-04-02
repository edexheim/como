import torch


# w means global reference for affine parameters
def get_aff_w_curr(aff_w_ref, aff_curr_ref):
    aff_params_new = aff_w_ref.clone()
    aff_params_new[:, 0, :] += aff_curr_ref[:, 0, :]
    aff_params_new[:, 1, :] += aff_curr_ref[:, 1, :] * torch.exp(aff_curr_ref[:, 0, :])
    return aff_params_new


def get_rel_aff(aff1, aff2):
    aff_rel = torch.empty_like(aff1)
    aff_rel[:, 0, :] = aff1[:, 0, :] - aff2[:, 0, :]
    aff_rel[:, 1, :] = torch.exp(-aff_rel[:, 0, :]) * (aff1[:, 1, :] - aff2[:, 1, :])
    return aff_rel
