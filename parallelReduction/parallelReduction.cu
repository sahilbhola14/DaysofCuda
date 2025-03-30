#include <cuda.h>

#include <iostream>

int getGridDim(const int N, const int blockDim) {
  return (N + blockDim - 1) / blockDim;
}

__global__ void parallelReductionKernel(int N, float *a, float *sum) {
  extern __shared__ float cache[];  // Define the shared memory
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // Load the data
  if (gid < N) {
    cache[tid] = a[gid];  // Load from global to shared memory
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();  // Make sure the data is loaded to the cache

  // Partial sum
  for (int i = 1; i < blockDim.x; i *= 2) {
    if (tid % (2 * i) == 0) {
      cache[tid] = cache[tid] + cache[tid + i];
    }
    __syncthreads();  // Make sure every thread is done computing
  }

  // Save
  if (tid == 0) *sum = cache[tid];
}

int main() {
  const int N = 16;  // Size of the vector
  float *h_a = new float[N];
  float *h_sum = new float;
  float *d_a, *d_sum;
  size_t size = N * sizeof(float);
  // Allocate host data
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
  }
  // Allocate Device memory
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_sum, sizeof(float));
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  // Kernel properties
  int blockDim = 16;  // block dimension
  int gridDim = getGridDim(N, 16);
  parallelReductionKernel<<<gridDim, blockDim, blockDim * sizeof(float)>>>(
      N, d_a, d_sum);

  // Copy
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

  printf("sum: %f \n", *h_sum);

  // Free
  cudaFree(d_a);
  return 0;
}
