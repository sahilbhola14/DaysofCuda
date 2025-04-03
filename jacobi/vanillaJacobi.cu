// solve the laplace equaiton \nabla^2 T = 0 with boundary
// T(0, x) = 1; T(y, 0) = T(y, 1) = T(1, x) = 0

#include <cuda.h>

#include <iostream>

#define cudaCheck(ans) gpuCheck((ans), __FILE__, __LINE__);

void gpuCheck(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("<CUDA ERROR>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

int getGridDim(const int N, const int blockDim) {
  return (N + blockDim - 1) / blockDim;
}

__global__ void jacobiVanillaKernel(const int x_res, const int y_res,
                                    float *state, float *past_state,
                                    float *rhs) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float left, right, top, bottom;
  int linear_id = row * (x_res - 2) + col;

  __shared__ int Nx;
  __shared__ int Ny;
  __shared__ float inv_dx_sq;
  __shared__ float inv_dy_sq;
  __shared__ float inv_coeff;

  if (threadIdx.x == 0) {
    Nx = (x_res - 2);
    Ny = (y_res - 2);
    inv_dx_sq =
        static_cast<float>(1.0 / ((1.0 / (x_res - 1)) * (1.0 / (x_res - 1))));
    inv_dy_sq =
        static_cast<float>(1.0 / ((1.0 / (y_res - 1)) * (1.0 / (y_res - 1))));
    inv_coeff = static_cast<float>(1.0 / (2 * (inv_dx_sq + inv_dy_sq)));
  }
  __syncthreads();

  // boundary check
  if ((row < (y_res - 2)) && (col < (x_res - 2))) {
    // Top
    if (row > 0) {
      top = past_state[linear_id - Nx];
    } else {
      top = 1.0f;
    }

    // Bottom
    if (row < (Ny - 1)) {
      bottom = past_state[linear_id + Nx];
    } else {
      bottom = 0.0f;
    }

    // Left
    if (col > 0) {
      left = past_state[linear_id - 1];
    } else {
      left = 0.0f;
    }

    // Right
    if (col < (Nx - 1)) {
      right = past_state[linear_id + 1];
    } else {
      right = 0.0f;
    }

    // Update
    state[linear_id] =
        inv_coeff * (inv_dx_sq * (left + right) + inv_dy_sq * (top + bottom) -
                     rhs[linear_id]);
  }
}

void computeTrueSolution(const int x_res, const int y_res, double *state_true) {
  const int modes = 200;
  int state_dim = (x_res - 2) * (y_res - 2);
  double dx = 1.0 / (x_res - 1);  // Grid resolution in x-direction
  double dy = 1.0 / (y_res - 1);  // Grid resolution in y-direction
  int Nx = (x_res - 2);
  double x, y;
  int row, col;
  // Initialize True state with zero
  for (int i = 0; i < state_dim; i++) {
    state_true[i] = 0.0;
  }

  for (int k = 0; k < modes; k++) {
    for (int i = 0; i < state_dim; i++) {
      row = i / Nx;
      col = i % Nx;
      x = (col + 1) * dx;
      y = (row + 1) * dy;

      state_true[i] +=
          (((2.0 * (pow(-1, k + 1) - 1.0)) /
            ((k + 1) * M_PI * sinh((k + 1) * M_PI))) *
           sin((k + 1) * M_PI * x) * sinh((k + 1) * M_PI * (y - 1)));
    }
  }
}

void solveVanillaJacobi(const int x_res, const int y_res, const double etol,
                        const int num_iter) {
  // Instantiation
  int state_dim = (x_res - 2) * (y_res - 2);
  size_t size = state_dim * sizeof(float);
  float *h_state = new float[state_dim];
  float *h_rhs = new float[state_dim];
  float *d_state, *d_past_state, *d_rhs;
  double *state_true = new double[state_dim];

  // Initialize the state on Host
  for (int i = 0; i < state_dim; i++) {
    h_state[i] = 1.0f;
    h_rhs[i] = 0.0f;
  }

  // Initialize on device
  cudaCheck(cudaMalloc((void **)&d_state, size));
  cudaCheck(cudaMalloc((void **)&d_past_state, size));
  cudaCheck(cudaMalloc((void **)&d_rhs, size));

  cudaCheck(cudaMemcpy(d_state, h_state, size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_past_state, h_state, size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_rhs, h_rhs, size, cudaMemcpyHostToDevice));

  // Kernel properties
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(getGridDim(x_res - 2, blockDim.x),
               getGridDim(y_res - 2, blockDim.y));
  // Launch Kernel
  for (int iter = 0; iter < 5000; iter++) {
    jacobiVanillaKernel<<<gridDim, blockDim>>>(x_res, y_res, d_state,
                                               d_past_state, d_rhs);
    cudaCheck(
        cudaMemcpy(d_past_state, d_state, size, cudaMemcpyDeviceToDevice));
  }
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(h_state, d_state, size, cudaMemcpyDeviceToHost));
  // Check Error
  computeTrueSolution(x_res, y_res, state_true);
  // Compute Error between true and computed
  double error = 0.0;
  for (int i = 0; i < state_dim; i++) {
    error += pow(state_true[i] - h_state[i], 2);
  }
  printf("Error (L2 Norm) in computation: %.3e\n", error);

  cudaFree(d_state);
  cudaFree(d_past_state);
  cudaFree(d_rhs);
}

int main() {
  int x_res = 33;
  int y_res = 65;
  double etol = 1e-8;
  int num_iter = 5000;

  solveVanillaJacobi(x_res, y_res, etol, num_iter);
  return 0;
}
