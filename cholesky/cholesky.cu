// Computes the right looking Cholesky Decomposion
#include <cuda.h>

#include <cassert>
#include <iostream>

const int N = 10;
#define cudaCheck(ans) gpuAssert((ans), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

int getGridDim(const int n, const int blockSize) {
  return (n + blockSize - 1) / blockSize;
}

// Create a SPD tri-diagonal matrix
void create_matrix(float *a) {
  int diag_idx;
  for (int i = 0; i < N; i++) {
    diag_idx = i * N + i;
    a[diag_idx] = 4.0f;
    if (i > 0) a[diag_idx - 1] = 1.0f;
    if (i < N - 1) a[diag_idx + 1] = 1.0f;
  }
}

// Compute Vanilla Cholesky (Not Parallalizable)
void computeVanillaCholesky(float *a, float *l) {
  for (int i = 0; i < N; i++) {
    // Compute the diagonals L_ii
    l[i * N + i] = a[i * N + i];

    float sum = 0.0f;
    for (int k = 0; k < i; k++) {
      sum += l[i * N + k] * l[i * N + k];
    }
    l[i * N + i] -= sum;
    l[i * N + i] = std::sqrt(l[i * N + i]);

    // Compute the off-diagonals L_ji for all j > i and j < N
    for (int j = i + 1; j < N; j++) {
      sum = 0.0f;
      for (int k = 0; k < i; k++) {
        sum += l[j * N + k] * l[i * N + k];
      }
      l[j * N + i] = (a[j * N + i] - sum) / l[i * N + i];
    }
  }
}

// Normalize the Diagonal
__global__ void normalizeDiagonalKernel(float *a, const int row) {
  a[row * N + row] = sqrtf(a[row * N + row]);  // Float sqrt
}

// Scale Off diagonal
__global__ void scaleOffDiagonalKernel(float *a, const int row) {
  int gid = (blockIdx.x * blockDim.x + threadIdx.x) + row + 1;
  if (gid < N) {
    a[gid * N + row] = a[gid * N + row] / a[row * N + row];
  }
}

// Rank one update Kernel
__global__ void rankOneUpdateKernel(float *a, const int global_row) {
  // Offset to get the global A idx
  int row = (blockIdx.y * blockDim.y + threadIdx.y) + (global_row + 1);
  int col = (blockIdx.x * blockDim.x + threadIdx.x) + (global_row + 1);
  if ((row < N) && (col < N)) {
    /* printf("global_row: %d, row: %d, col: %d | a: %f | l_start: %f l_i:
     * %f\n", global_row, row, col, a[row * N + col], a[(global_row + 1) * N +
     * global_row], a[(global_row + 1 + (row - (global_row + 1)))*N +
     * global_row]); */
    // Start Idx of the Off diagonal Vector: ((global_row + 1) * N + global_row)
    // Li index: ((global_row + 1 + (row - (global_row + 1))) * N + global_row)
    // Lj index: ((global_row + 1 + (col - (global_row + 1))) * N + global_row)
    a[row * N + col] -=
        a[(global_row + 1 + (row - (global_row + 1))) * N + global_row] *
        a[(global_row + 1 + (col - (global_row + 1))) * N + global_row];
  }
}

// Compute Right Looking Cholesky
void computeRightLookingCholesky(float *h_a, float *h_l) {
  // Initialize
  float *d_a;
  size_t size = N * N * sizeof(float);
  // Allocate Memory on Device
  cudaCheck(cudaMalloc((void **)&d_a, size));

  // Transfer Memory to Device
  cudaCheck(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

  // Compute Cholesky on the Device
  for (int row = 0; row < N; row++) {
    // Normalized diagonal A_ii
    normalizeDiagonalKernel<<<1, 1>>>(d_a, row);

    // Scale the elements below the diagonal
    scaleOffDiagonalKernel<<<getGridDim(N - (row + 1), 16), 16>>>(d_a, row);

    // Rank One update to the Lower Rigth Block
    dim3 blockDim(4, 4, 1);
    dim3 gridDim(getGridDim(N - (row + 1), blockDim.x),
                 getGridDim(N - (row + 1), blockDim.x), 1);
    if (gridDim.x > 0 && gridDim.y > 0) {
      rankOneUpdateKernel<<<gridDim, blockDim>>>(d_a, row);
      cudaCheck(cudaGetLastError());
    }
  }

  // Transfer (All operations are performed in-place (replaceing d_a with the
  // decomposition), but just for keeping a copy of h_a, copied to h_l, we can
  // copy it to h_a as well.

  cudaCheck(cudaMemcpy(h_l, d_a, size,
                       cudaMemcpyDeviceToHost));  // If not want in-place
  /* cudaCheck(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost)); // If want
   * in-place */

  // Zero out the Upper Triangular
  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      if (j > k) {
        h_l[k * N + j] = 0.0f;  // Not in-place
                                /* h_a[k * N + j] = 0.0f; // In-place */
      }
    }
  }
}

int main() {
  float *h_a = new float[N * N];
  float *h_l = new float[N * N];
  // Initialize the matrix on the host memory
  create_matrix(h_a);
  // Serial Vanilla Cholesky (Host)
  /* computeVanillaCholesky(h_a, h_l); */

  // Right Looking Cholesky (Device)
  computeRightLookingCholesky(h_a, h_l);

  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      printf("%.4f ,", h_l[k * N + j]);
    }
    printf("\n");
  }

  // Free Host
  delete[] h_a;
  delete[] h_l;
}
