#pragma once

#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DEVICE(x, y) TORCH_CHECK(x.device().type() == y.device().type(), #x " must be be on same device as " #y)

torch::Tensor cross_covariance_cpu(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale);
torch::Tensor cross_covariance_gpu(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale);
torch::Tensor cross_covariance(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale);

void get_new_chol_obs_info_cpu(
    torch::Tensor L, torch::Tensor obs_info, torch::Tensor var,
    torch::Tensor k_ni, torch::Tensor k_id, float k_ii, int N);
void get_new_chol_obs_info_gpu(
    torch::Tensor L, torch::Tensor obs_info, torch::Tensor var,
    torch::Tensor k_ni, torch::Tensor k_id, float k_ii, int N);
void get_new_chol_obs_info(
    torch::Tensor L, torch::Tensor obs_info, torch::Tensor var,
    torch::Tensor k_ni, torch::Tensor k_id, float k_ii, int N);