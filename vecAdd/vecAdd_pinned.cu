// description: add two vector of size n
// blockDim: number of threads per block
// gridDim: number of grids
// Using pinned memory to improve the Memcpy overhead.
// just using pinned memory does not result in Asyn transfer, but enalbes it
// cudaEvent is just a flag, it frees up the CPU to do other tasks. However,
// using streamSynchronize would have resulted in the CPU to wait.
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
  const int n = 100000000;
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
  // streams so the two H2D copies can run concurrently on separate copy engines
  cudaStream_t stream_a, stream_b;
  cuda_check(cudaStreamCreate(&stream_a));
  cuda_check(cudaStreamCreate(&stream_b));
  cudaEvent_t b_ready;
  cuda_check(cudaEventCreate(&b_ready));
  // copy the data (async, pinned host memory required for true async behavior)
  cuda_check(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream_a));
  cuda_check(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream_b));
  // let stream_a's kernel wait on stream_b's copy instead of blocking the host
  cuda_check(cudaEventRecord(b_ready, stream_b));
  cuda_check(cudaStreamWaitEvent(stream_a, b_ready, 0));
  // kernel launch
  dim3 blockDim(32 * 4);
  dim3 gridDim(get_grid_dim(n, blockDim.x));
  printf("Number of threads per block: %d\n", blockDim.x);
  printf("Number of grids: %d\n", gridDim.x);
  vec_add_kernel<<<gridDim, blockDim, 0, stream_a>>>(n, d_a, d_b, d_res);
  cuda_check(cudaGetLastError());
  // transfer back asynchronously on the same stream (ordered after the kernel)
  cuda_check(
      cudaMemcpyAsync(h_res, d_res, size, cudaMemcpyDeviceToHost, stream_a));
  cuda_check(cudaStreamSynchronize(stream_a));
  // check
  check_result(h_res, n, 2.0f);
  // print
  // print_vector(n, h_res);
  // free
  cudaEventDestroy(b_ready);
  cudaStreamDestroy(stream_a);
  cudaStreamDestroy(stream_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_res);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_res);
  return 0;
}
