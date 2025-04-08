// Compute the Transpose of a Matrix A of size M * N
#include <cuda.h>

#include <iostream>

const int M = 5;  // Number of rows in A
const int N = 3;  // Number of cols in A
#define cudaCheck(ans) gpuCheck((ans), __FILE__, __LINE__);

inline void gpuCheck(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

int getGridDim(const int N, const int blockDim) {
  return (N + blockDim - 1) / blockDim;
}

void initialize_matrix(float *a) {
  for (int i = 0; i < M * N; i++) {
    a[i] = i / N;
  }
}

__global__ void computeTransposeKernel(float *a, float *a_t) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < N) && (col < M)) {
    a_t[row * M + col] = a[col * N + row];
  }
}

void compute_transpose(float *h_a, float *h_a_t) {
  // Initialize
  float *d_a, *d_a_t;
  size_t size = M * N * sizeof(float);
  // Allocation
  cudaCheck(cudaMalloc((void **)&d_a, size));
  cudaCheck(cudaMalloc((void **)&d_a_t, size));
  // Transfer
  cudaCheck(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  // Kernel properties (Map the transpose matrix, not the input matrix)
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(getGridDim(M, blockDim.x), getGridDim(N, blockDim.y), 1);
  // Launch Kernel
  computeTransposeKernel<<<gridDim, blockDim>>>(d_a, d_a_t);
  cudaCheck(cudaGetLastError());
  // Transfer to the Host
  cudaCheck(cudaMemcpy(h_a_t, d_a_t, size, cudaMemcpyDeviceToHost));
  // Free
  cudaFree(d_a);
  cudaFree(d_a_t);
}

int main() {
  float *a = new float[M * N];
  float *a_t = new float[M * N];

  // Initialize the matrix
  initialize_matrix(a);
  // Compute the transpose
  compute_transpose(a, a_t);
  // Check the Error
  float error = 0.0f;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      error += pow(a[i * N + j] - a_t[j * M + i], 2);
    }
  }
  printf("Error: %f\n", error);

  delete[] a;
  delete[] a_t;
}
