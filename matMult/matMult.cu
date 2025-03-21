// Perform matrix-matrix multiplicatin of A \times B
#include <cuda.h>

#include <iostream>

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

__global__ void vanillaMatMultKernel(const int m, const int n, const int t,
                                     float *A, float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < m) && (col < t)) {
    float dot_product = 0.0f;
    for (int i = 0; i < n; i++) {
      dot_product += A[row * n + i] * B[i * t + col];
    }
    C[row * t + col] = dot_product;
  }
}

void vanillaMatMult() {
  const int m = 32;  // Number of rows of A
  const int n = 60;  // Number of cols of A
  const int t = 64;  // Number of cols of B
  int size_A = m * n * sizeof(float);
  int size_B = n * t * sizeof(float);
  int size_C = m * t * sizeof(float);
  float *h_A = new float[m * n];
  float *h_B = new float[n * t];
  float *h_C = new float[m * t];
  float *d_A, *d_B, *d_C;

  // Allocate the host data
  for (int i = 0; i < m * n; i++) {
    h_A[i] = 1.0f;
  }
  for (int i = 0; i < n * t; i++) {
    h_B[i] = 2.0f;
  }

  // Allocate the device memory
  cudaCheck(cudaMalloc((void **)&d_A, size_A));
  cudaCheck(cudaMalloc((void **)&d_B, size_B));
  cudaCheck(cudaMalloc((void **)&d_C, size_C));
  cudaCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  // Kernel parameters
  dim3 blockDim(16, 16, 1);  // Each thread would compute one element of output
  dim3 gridDim(getGridDim(t, blockDim.x), getGridDim(m, blockDim.y), 1);

  // Launch the kernel
  vanillaMatMultKernel<<<gridDim, blockDim>>>(m, n, t, d_A, d_B, d_C);
  cudaCheck(cudaGetLastError());

  // Transfer data to host
  cudaCheck(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  // Error check
  float error = 0.0f;
  for (int i = 0; i < m * t; i++) {
    error += (h_C[i] - n * 2.0);
  }
  printf("Error: %f\n", error);

  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  // Vanilla Kernel
  vanillaMatMult();

  return 0;
}
