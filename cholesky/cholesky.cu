// Computes the right looking Cholesky Decomposion
#include <cuda.h>

#include <cassert>
#include <iostream>

const int N = 5;
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
void computeCholesky(float *a, float *l) {
  for (int i = 0; i < N; i++) {
    // Compute the diagonals
    l[i * N + i] = a[i * N + i];
    float sum = 0.0f;
    for (int k = 0; k < i; k++) {
      sum += l[i * N + k] * l[i * N + k];
    }
    l[i * N + i] -= sum;
    l[i * N + i] = std::sqrt(l[i * N + i]);

    // Compute the off-diagonals
    for (int j = i + 1; j < N; j++) {
      sum = 0.0f;
      for (int k = 0; k < i; k++) {
        sum += l[j * N + k] * l[i * N + k];
      }
      l[j * N + i] = (a[j * N + i] - sum) / l[i * N + i];
    }
  }
}

int main() {
  float *h_a = new float[N * N];
  float *h_l = new float[N * N];
  // Initialize the matrix on the host memory
  create_matrix(h_a);
  // Serial Vanilla Cholesky (Host)
  computeVanillaCholesky(h_a, h_l);
  // Right Looking Cholesky (Device)
  /* computeRightLookingCholesky(h_a); */

  // Free Host
  delete[] h_a;
  delete[] h_l;
}
