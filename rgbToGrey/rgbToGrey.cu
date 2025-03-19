// Convert an RGB image of size (3, m, n) to Grey Scale
#include <cuda.h>

#include <iostream>

#define cudaCheck(ans) gpuCheck((ans), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

int getGridDim(const int n, const int blockDim) {
  return (n + blockDim - 1) / blockDim;
}

__global__ void rgbToGreyKernel(const int m, const int n,
                                unsigned char *input_image,
                                unsigned char *output_image) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < m) && (col < n)) {
    int linear_idx_inp = row * n + col;  // Linear Index of the output image
        output_image[linear_idx] = input_image[linear_idx
  }
}

int main() {
  // Initialization
  const int m = 60;  // Number of rows
  const int n = 72;  // Number of cols
  const int inp_size = 3 * m * n * sizeof(unsigned char);
  const int out_size = m * n * sizeof(unsigned char);
  unsigned char *h_input_image =
      new unsigned char[3 * m * n];  // Contains 3 channels (RGB)
  unsigned char *h_output_image =
      new unsigned char[m * n];  // Contains single channel
  unsigned char *d_input_image, *d_output_image;

  // Generate a toy image
  for (int i = 0; i < m * n; i++) {
    h_input_image[i * 3] = 100;      // R
    h_input_image[i * 3 + 1] = 50;   // G
    h_input_image[i * 3 + 2] = 220;  // G
  }

  // Allocate Device memory
  cudaCheck(cudaMalloc((void **)&d_input_image, input_size));
  cudaCheck(cudaMalloc((void **)&d_output_image, output_size));

  // Transfer data to Device
  cudaMemcpy(d_input_image, h_input_image, inp_size, cudaMemcpyHostToDevice);

  // Kernel specification
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(getGridDim(n, blockDim.x), getGridDim(m, blockDim.y), 1);

  // Kernel Launch

  // Free
  cudaFree(d_input_image);
  cudaFree(d_output_image);

  return 0;
}
