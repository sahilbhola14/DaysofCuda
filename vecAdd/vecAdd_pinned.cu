// description: add two vector of size n
// blockDim: number of threads per block
// gridDim: number of grids
// Using pinned memory to improve the Memcpy overhead.
#include <cassert>
#include <iostream>
#include <vector>

#define cuda_check(ans) gpu_assert((ans), __FILE__, __LINE__)

inline void gpu_assert(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("<CUDA Error>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

// kernel for adding
__global__ void vec_add_kernel(const int n, const float *a, const float *b,
                               float *res) {
  // params
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  // kernel launch
  if (gid < n) {
    res[gid] = a[gid] + b[gid];
  }
}

// vector initialization
void initialize_vector(const int n, float *a) {
  for (int i = 0; i < n; ++i) {
    a[i] = 1.0f;
  }
}

// vector printing
void print_vector(const int n, const float *a) {
  for (int i = 0; i < n; i++) {
    printf("%f\n", a[i]);
  }
}

// grid dim
int get_grid_dim(const int n, const int dim) { return (n + dim - 1) / dim; }

// check result
void check_result(const float *a, const int n, const float target) {
  for (int i = 0; i < n; ++i) {
    assert(a[i] == target);
  }
}

int main() {
  // parameters
  const int n = 100000;
  const int size = n * sizeof(float);
  // initialize
  float *h_a, *h_b, *h_res;
  // using pinned memory to enable Asys transfer
  cuda_check(cudaMallocHost(&h_a, size));
  cuda_check(cudaMallocHost(&h_b, size));
  cuda_check(cudaMallocHost(&h_res, size));
  // initialize the vector
  initialize_vector(n, h_a);
  initialize_vector(n, h_b);
  // device pointer
  float *d_a, *d_b, *d_res;
  cuda_check(cudaMalloc((void **)&d_a, size));
  cuda_check(cudaMalloc((void **)&d_b, size));
  cuda_check(cudaMalloc((void **)&d_res, size));
  // copy the data
  cuda_check(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
  // kernel launch
  dim3 blockDim(32 * 4);
  dim3 gridDim(get_grid_dim(n, blockDim.x));
  printf("Number of threads per block: %d\n", blockDim.x);
  printf("Number of grids: %d\n", gridDim.x);
  vec_add_kernel<<<gridDim, blockDim>>>(n, d_a, d_b, d_res);
  cuda_check(cudaGetLastError());
  // transfer
  cuda_check(cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost));
  // check
  check_result(h_res, n, 2.0f);
  // print
  // print_vector(n, h_res);
  // free
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_res);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_res);
  return 0;
}
