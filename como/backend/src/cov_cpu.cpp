#include <torch/extension.h>
#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>

#include <vector>

inline float safe_sqrt(float x) {
  return sqrt(x + 1e-8);
}

inline float matern(float Q) {
  float tmp = 1.73205080757 * safe_sqrt(Q);
  float k_v_3_2 = (1 + tmp) * exp(-tmp);
  return k_v_3_2;
}

torch::Tensor cross_covariance_cpu(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale)
{
  // x1 (B, N, 2)
  // E1 (B, N, 2, 2)
  // x2 (B, M, 2)
  // E2 (B, M, 2, 2)

  const int batch_size = x1.size(0);
  const int num_points1 = x1.size(1);
  const int num_points2 = x2.size(1);

  auto opts = x1.options();
  torch::Tensor K12 = torch::empty({batch_size, num_points1, num_points2}, opts);

  auto x1_a = x1.accessor<float,3>();
  auto E1_a = E1.accessor<float,4>();
  auto x2_a = x2.accessor<float,3>();
  auto E2_a = E2.accessor<float,4>();
  auto K12_a = K12.accessor<float,3>();

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t i = 0; i < num_points1; i++) {
      for (size_t j = 0; j < num_points2; j++) {
        float diff_x = x1_a[b][i][0] - x2_a[b][j][0];
        float diff_y = x1_a[b][i][1] - x2_a[b][j][1];
        // Sum covariance
        float E_00 = E1_a[b][i][0][0] + E2_a[b][j][0][0];
        float E_01 = E1_a[b][i][0][1] + E2_a[b][j][0][1];
        // float E_10 = E1_a[b][i][1][0] + E2_a[b][j][1][0];
        float E_11 = E1_a[b][i][1][1] + E2_a[b][j][1][1];
        // Determinant
        float E_det_inv = 1.0/(E_00*E_11 - E_01*E_01);
        // Quadratic of inverse sum
        float Q = (E_11 * diff_x * diff_x) - 2 * (E_01 * diff_x * diff_y) + (E_00 * diff_y * diff_y);
        Q *= 0.5*E_det_inv;

        // Probability product constant
        float E1_det = E1_a[b][i][0][0]*E1_a[b][i][1][1] - E1_a[b][i][0][1]*E1_a[b][i][1][0];
        float E2_det = E2_a[b][j][0][0]*E2_a[b][j][1][1] - E2_a[b][j][0][1]*E2_a[b][j][1][0];
        float C = 2.0 * pow(E1_det*E2_det, static_cast<float>(0.25)) * safe_sqrt(E_det_inv);

        K12_a[b][i][j] = scale * C * matern(Q);
      }
    }
  }

  return K12;
}

void get_new_chol_obs_info_cpu(
    torch::Tensor L, torch::Tensor obs_info, torch::Tensor var,
    torch::Tensor k_ni, torch::Tensor k_id, float k_ii, int N) {
  
  auto L_sub = L.index({torch::indexing::Slice(), torch::indexing::Slice(0, N), torch::indexing::Slice(0, N)});
  torch::Tensor l_ni = at::linalg_solve_triangular(L_sub, k_ni, /*upper=*/false, /*left=*/true);
  torch::Tensor l_ii = at::sqrt(k_ii - at::sum(at::square(l_ni), /*dim=*/1, /*keepdim=*/true));
  auto obs_info_sub = obs_info.index({torch::indexing::Slice(), torch::indexing::Slice(0, N), torch::indexing::Slice()});
  torch::Tensor obs_info_new = at::div(k_id - at::sum(at::mul(l_ni, obs_info_sub),  /*dim=*/1, /*keepdim=*/true), l_ii);

  // Add rows to L and obs_info
  L.index_put_({torch::indexing::Slice(), torch::indexing::Slice(N, N+1), torch::indexing::Slice(0, N)}, l_ni.transpose(1,2));
  L.index_put_({torch::indexing::Slice(), N, N}, l_ii);
  obs_info.index_put_({torch::indexing::Slice(), torch::indexing::Slice(N, N+1), torch::indexing::Slice()}, obs_info_new);

  // Remove uncertainty from newest point
  var -= (obs_info_new * obs_info_new).squeeze(1);

  return;
}