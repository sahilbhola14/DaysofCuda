// Perform matrix-matrix multiplicatin of A \times B
#include <cuda.h>

#include <cassert>
#include <iostream>

const int M = 16;  // Number of rows of A
const int N = 16;  // Number of cols of A
const int T = 16;  // Number of cols of B

#define TILE_SIZE 16  // Tile size for tiledMatMult
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

__global__ void tiledMatMultKernel(const int m, const int n, const int t,
                                   float *A, float *B, float *C) {
  // Initialize __shared__ memory
  __shared__ float tile_A[TILE_SIZE * TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * TILE_SIZE + ty;  // Row of C that is evaluated the thread
  int col = bx * TILE_SIZE + tx;  // Col of C that is evaluated the thread

  float dot_product = 0.0f;
  // Loop over the blocks
  for (int bid = 0; bid < (n / TILE_SIZE); bid++) {
    // Load the data from A and B to __shared __
    // Each thread EXACTLY maps one value from A and B. A tile of the size
    // blockDim is loaded. Say [C_1, C_2, C_3, C_4] is the block, then threads
    // loads [A_1,..,A_4]

    // row * N takes to the start fo the row
    // bid * TILE_SIZE takes to the start of the block
    // tx takes to the corresponding value
    int idx_A = row * N + bid * TILE_SIZE + tx;

    // (bid * TILE_SIZE + ty) * T: Skips all the rows and takes to the start
    // col: offsets the column value
    int idx_B = (bid * TILE_SIZE + ty) * T + col;

    // Linear Indexing for the tile
    int idx_tile = ty * TILE_SIZE + tx;

    tile_A[idx_tile] = A[idx_A];  // Copy from DRAM to __shared__
    tile_B[idx_tile] = B[idx_B];  // Copy from DRAM to __shared__

    __syncthreads();  // All threads in block must have copied their data

    for (int k = 0; k < TILE_SIZE; k++) {
      dot_product += tile_A[ty * TILE_SIZE + tx] * tile_B[k * TILE_SIZE + col];
    }

    __syncthreads();  // All threads must be done using __shared__ data

    C[row * t + col] = dot_product;
  }
}

void initializeMatrix(float *A, float *B) {
  // Allocate the host data
  for (int i = 0; i < M * N; i++) {
    A[i] = 1.0f;
  }
  for (int i = 0; i < N * T; i++) {
    B[i] = 2.0f;
  }
}

void vanillaMatMult() {
  int size_A = M * N * sizeof(float);
  int size_B = N * T * sizeof(float);
  int size_C = M * T * sizeof(float);
  float *h_A = new float[M * N];
  float *h_B = new float[N * T];
  float *h_C = new float[M * T];
  float *d_A, *d_B, *d_C;

  // Intialize
  initializeMatrix(h_A, h_B);

  // Allocate the device memory
  cudaCheck(cudaMalloc((void **)&d_A, size_A));
  cudaCheck(cudaMalloc((void **)&d_B, size_B));
  cudaCheck(cudaMalloc((void **)&d_C, size_C));
  cudaCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  // Kernel parameters
  dim3 blockDim(16, 16, 1);  // Each thread would compute one element of output
  dim3 gridDim(getGridDim(T, blockDim.x), getGridDim(M, blockDim.y), 1);

  // Launch the kernel
  vanillaMatMultKernel<<<gridDim, blockDim>>>(M, N, T, d_A, d_B, d_C);
  cudaCheck(cudaGetLastError());

  // Transfer data to host
  cudaCheck(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  // Error check
  float error = 0.0f;
  for (int i = 0; i < M * T; i++) {
    error += (h_C[i] - N * 2.0);
  }
  printf("Error: %f\n", error);

  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void tiledMatMult() {
  // Tiled Matrix Multiplication
  // Each thread loads data from the multiplicands to the __shared__
  // Initialize
  int size_A = M * N * sizeof(float);
  int size_B = N * T * sizeof(float);
  int size_C = M * T * sizeof(float);
  float *h_A = new float[M * N];
  float *h_B = new float[N * T];
  float *h_C = new float[M * T];
  float *d_A, *d_B, *d_C;

  assert((M % TILE_SIZE == 0) &&
         "Currently assumed that matrix align perfectly");
  assert((N % TILE_SIZE == 0) &&
         "Currently assumed that matrix align perfectly");
  assert((T % TILE_SIZE == 0) &&
         "Currently assumed that matrix align perfectly");

  // Allocate Host
  initializeMatrix(h_A, h_B);
  // Allocate Device
  cudaCheck(cudaMalloc((void **)&d_A, size_A));
  cudaCheck(cudaMalloc((void **)&d_B, size_B));
  cudaCheck(cudaMalloc((void **)&d_C, size_C));
  // Copy Host to Device
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
  // Kernel paramters (BlockDim dosent necessarily have to be tile_size.)
  dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
  dim3 gridDim(getGridDim(T, blockDim.x), getGridDim(M, blockDim.y), 1);
  // Kernel Launch
  tiledMatMultKernel<<<gridDim, blockDim>>>(M, N, T, d_A, d_B, d_C);
  cudaCheck(cudaGetLastError());

  // Transfer Data
  cudaCheck(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  // Error check
  float error = 0.0f;
  for (int i = 0; i < M * T; i++) {
    std::cout << h_C[i] << std::endl;
    error += (h_C[i] - N * 2.0);
  }
  printf("Error: %f\n", error);

  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  // Vanilla Kernel
  /* vanillaMatMult(); */
  // Tiled Kernel
  tiledMatMult();

  return 0;
}
