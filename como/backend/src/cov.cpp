#include "cov.h"
#include <c10/cuda/CUDAGuard.h>

// Unary operations
torch::Tensor cross_covariance(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale) {
  // CHECK_CONTIGUOUS(x1);
  // CHECK_CONTIGUOUS(E1);
  // CHECK_CONTIGUOUS(x2);
  // CHECK_CONTIGUOUS(E2);

  if (x1.device().type() == torch::DeviceType::CPU
    && E1.device().type() == torch::DeviceType::CPU
    && x2.device().type() == torch::DeviceType::CPU
    && E2.device().type() == torch::DeviceType::CPU) {
    return cross_covariance_cpu(x1, E1, x2, E2, scale);
  } 
  else if (x1.device().type() == torch::DeviceType::CUDA
    && E1.device().type() == torch::DeviceType::CUDA
    && x2.device().type() == torch::DeviceType::CUDA
    && E2.device().type() == torch::DeviceType::CUDA) {
    #ifdef BACKEND_WITH_CUDA
      const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
      return cross_covariance_gpu(x1, E1, x2, E2, scale);
    #else
      TORCH_CHECK(false, "Backend was not compiled with CUDA support")
    #endif
  }
  else {
    TORCH_CHECK(false, "All variables must be on same device.")
  }
  return {};
}

void get_new_chol_obs_info(
    torch::Tensor L, torch::Tensor obs_info, torch::Tensor var,
    torch::Tensor k_ni, torch::Tensor k_id, float k_ii, int N) {
  
  CHECK_CONTIGUOUS(L);
  CHECK_CONTIGUOUS(obs_info);
  CHECK_CONTIGUOUS(var);
  CHECK_CONTIGUOUS(k_ni);
  CHECK_CONTIGUOUS(k_id); 

  if (L.device().type() == torch::DeviceType::CPU
    && obs_info.device().type() == torch::DeviceType::CPU
    && k_ni.device().type() == torch::DeviceType::CPU
    && k_id.device().type() == torch::DeviceType::CPU) {
    return get_new_chol_obs_info_cpu(L, obs_info, var, k_ni, k_id, k_ii, N);
  } 
  else if (L.device().type() == torch::DeviceType::CUDA
    && obs_info.device().type() == torch::DeviceType::CUDA
    && k_ni.device().type() == torch::DeviceType::CUDA
    && k_id.device().type() == torch::DeviceType::CUDA) {
    #ifdef BACKEND_WITH_CUDA
      const at::cuda::OptionalCUDAGuard device_guard(device_of(L));
      return get_new_chol_obs_info_gpu(L, obs_info, var, k_ni, k_id, k_ii, N);
    #else
      TORCH_CHECK(false, "Backend was not compiled with CUDA support")
    #endif
  }
  else {
    TORCH_CHECK(false, "All variables must be on same device.")
  }
  return;
}