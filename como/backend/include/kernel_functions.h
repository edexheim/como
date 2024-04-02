#pragma once

#define sqrt3 1.73205080757

__forceinline__ __host__ __device__ float safe_sqrt(float x) {
  return sqrt(x + 1e-8);
}

// matern with v=3/2
__forceinline__ __host__ __device__ float matern(float Q) {
  float tmp = sqrt3 * safe_sqrt(Q);
  float k_v_3_2 = (1 + tmp) * exp(-tmp);
  return k_v_3_2;
}