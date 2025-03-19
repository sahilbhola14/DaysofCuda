// A \times b, where A \in R^{m \times n}

#include <cuda.h>

#include <iostream>

#define cudaCheck(ans) gpuAssert((ans), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

int getGridDim(const int n, const int blockSize) {
  return (n + blockSize - 1) / blockSize;
}

__global__ void matVecMultVanillaKernel(const int m, const int n, float *A,
                                        float *B, float *C) {
  // Each thread processes the {gid} index of the output C
  // Global id
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < m) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum = sum + A[gid * n + i] * B[i];
    }
    C[gid] = sum;
  }
}

int main() {
  // Intialization
  const int m = 50;  // Number of rows
  const int n = 32;  // Number of columns
  const int size = m * n * sizeof(float);
  float *h_A = new float[m * n];
  float *h_B = new float[n];
  float *h_C = new float[m];
  float *d_A, *d_B, *d_C;

  // Allocate data to the host values
  for (int i = 0; i < m * n; i++) {
    h_A[i] = 1.0f;
  }
  for (int i = 0; i < n; i++) {
    h_B[i] = 1.0f;
  }

  // Allocation of device memory
  cudaCheck(cudaMalloc((void **)&d_A, size));
  cudaCheck(cudaMalloc((void **)&d_B, n * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_C, m * sizeof(float)));

  // Move the data to device
  cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));

  /* // Kernel specification */
  int blockDim = 16;  // 1D block with 16 threads
  int gridDim = getGridDim(m, blockDim);

  // Launch the Kernel
  matVecMultVanillaKernel<<<gridDim, blockDim>>>(m, n, d_A, d_B, d_C);
  cudaCheck(cudaGetLastError());

  // Copy from device to host
  cudaCheck(cudaMemcpy(h_C, d_C, m * sizeof(float), cudaMemcpyDeviceToHost));

  // Check the error
  float error = 0.0f;
  for (int i = 0; i < m; i++) {
    error = error + (h_C[i] - n);
  }
  printf("Error: %f\n", error);

  /* // Cuda Free */
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
