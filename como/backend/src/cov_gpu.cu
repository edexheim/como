#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include <torch/extension.h>
#include <ATen/Functions.h>

#include "kernel_functions.h"

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)

typedef cub::BlockReduce<float, THREADS> BlockReduce;

template <typename scalar_t>
__global__ void cross_cov_kernel(
  const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> x1,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> E1,
  const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> x2,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> E2,
  scalar_t scale,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> K12) 
{

  const int b = blockIdx.x;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  const int i = p / x2.size(1);
  const int j = p % x2.size(1);

  if (i < x1.size(1) && j < x2.size(1)) {
    // Probabily product quadratic
    // Position diff
    scalar_t diff_x = x1[b][i][0] - x2[b][j][0];
    scalar_t diff_y = x1[b][i][1] - x2[b][j][1];
    // Sum covariance
    scalar_t E_00 = E1[b][i][0][0] + E2[b][j][0][0];
    scalar_t E_01 = E1[b][i][0][1] + E2[b][j][0][1];
    // float E_10 = E1[b][i][1][0] + E2[b][j][1][0];
    scalar_t E_11 = E1[b][i][1][1] + E2[b][j][1][1];
    // Determinant
    scalar_t E_det_inv = 1.0/(E_00*E_11 - E_01*E_01);
    // Quadratic of inverse sum
    scalar_t Q = (E_11 * diff_x * diff_x) - 2 * (E_01 * diff_x * diff_y) + (E_00 * diff_y * diff_y);
    Q *= 0.5*E_det_inv;

    // Probability product constant
    scalar_t E1_det = E1[b][i][0][0]*E1[b][i][1][1] - E1[b][i][0][1]*E1[b][i][1][0];
    scalar_t E2_det = E2[b][j][0][0]*E2[b][j][1][1] - E2[b][j][0][1]*E2[b][j][1][0];
    scalar_t C = 2.0 * pow(E1_det*E2_det, static_cast<scalar_t>(0.25)) * safe_sqrt(E_det_inv);

    K12[b][i][j] = scale * C * matern(Q);
  }
}

torch::Tensor cross_covariance_gpu(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale) {
  // x1 (B, N, 2)
  // E1 (B, N, 2, 2)
  // x2 (B, M, 2)
  // E2 (B, M, 2, 2)

  const int batch_size = x1.size(0);
  const int num_points1 = x1.size(1);
  const int num_points2 = x2.size(1);

  const dim3 blocks(batch_size, NUM_BLOCKS(num_points1*num_points2));
  const dim3 threads(THREADS);

  auto opts = x1.options();
  torch::Tensor K12 = torch::empty({batch_size, num_points1, num_points2}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.type(), "cross_cov_kernel", ([&] {
    cross_cov_kernel<<<blocks, threads>>>(
      x1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      E1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      x2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      E2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      static_cast<scalar_t>(scale),
      K12.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
    
  return K12;
}

__global__ void forward_sub_kernel(
  float k_ii,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> L,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k_ni,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> l_ni) 
{
  const int b = blockIdx.x;
  const int n = threadIdx.x; // row that we are summing over
  const int N = blockDim.x;

  float sum = k_ni[b][n][0];
  for (int i = 0; i < N; i++) { // looping over columns
    if (i == n) {
      l_ni[b][n][0] = sum/L[b][n][n];
    }
    __syncthreads();
    if (i < n) {
      sum -= L[b][n][i] * l_ni[b][i][0];
    }
  }
}

__global__ void solve_chol_diag_kernel(
  float k_ii,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> l_ni,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> l_ii,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> l_ii_inv,
  int N) 
{
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int b = blockIdx.x;
  const int n = threadIdx.x;

  float squared_val = 0.0;
  if (n < N) {
    squared_val = l_ni[b][n][0] * l_ni[b][n][0];
  }

  float sum_of_squares = BlockReduce(temp_storage).Sum(squared_val);
  if (n == 0) {
    l_ii[b][0][0] = sqrt(k_ii - sum_of_squares);
    l_ii_inv[b][0][0] = 1.0/l_ii[b][0][0];
  }
}

__global__ void update_chol_kernel(
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> L,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k_ni,
  float k_ii) 
{
  __shared__ float sum_of_squares;

  const int b = blockIdx.x;
  const int n = threadIdx.x; // row that we are summing over
  const int N = blockDim.x;

  float sum = k_ni[b][n][0];
  for (int i = 0; i < N; i++) { // looping over columns
    if (i == n) { // Only one of these will run per thread so ok to update global variable
      if (n == 0)
        sum_of_squares = 0.0;
      L[b][N][n] = sum/L[b][n][n];
      sum_of_squares += L[b][N][n] * L[b][N][n];
    }
    __syncthreads();
    if (i < n) {
      sum -= L[b][n][i] * L[b][N][i];
    }
  }

  if (n == N-1) {
    L[b][N][N] = sqrt(k_ii - sum_of_squares);
  }
}

__global__ void obs_info_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k_id,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> L,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> obs_info,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> var,
  int N) 
{
  const int b = blockIdx.y;
  const int d = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (d < obs_info.size(2)) {
    float sum = k_id[b][0][d];
    for (int i = 0; i < N; i++) {
      sum -= obs_info[b][i][d] * L[b][N][i];
    }

    float obs_info_new = sum/L[b][N][N];
    obs_info[b][N][d] = obs_info_new;    
    var[b][d] -= obs_info_new*obs_info_new;
  }
}

void get_new_chol_obs_info_gpu(
    torch::Tensor L, torch::Tensor obs_info, torch::Tensor var,
    torch::Tensor k_ni, torch::Tensor k_id, float k_ii, int N) {
  // L (B, n, n)
  // obs_info (B, n, d)
  // k_ni (B, n, 1)
  // k_id (B, 1, d)
  // var (B, d)

  // Typically n << d (inducing points vs. image pixels), so we loop over sum across n in CUDA kernel

  const int batch_size = L.size(0);
  const int d = obs_info.size(2);

  auto opts = L.options();

  update_chol_kernel<<<batch_size, N>>>(
    L.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    k_ni.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    k_ii);

  const dim3 blocks_obs(NUM_BLOCKS(d), batch_size);
  const dim3 threads_obs(THREADS);
  obs_info_kernel<<<blocks_obs, threads_obs>>>(
    k_id.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    L.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    obs_info.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    var.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    N);
    
  return;
}