// Convert an RGB image of size (3, m, n) to Grey Scale
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

int getGridDim(const int n, const int blockDim) {
  return (n + blockDim - 1) / blockDim;
}

__global__ void rgbToGreyKernel(const int m, const int n,
                                unsigned char *input_image,
                                unsigned char *output_image) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < m) && (col < n)) {
    int linear_idx_out = row * n + col;  // Linear Index of the output image
    int linear_idx_in =
        linear_idx_out *
        3;  // Linear Index of the input image (start of the index)
    unsigned char r = input_image[linear_idx_in];
    unsigned char g = input_image[linear_idx_in + 1];
    unsigned char b = input_image[linear_idx_in + 2];
    output_image[linear_idx_out] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

int main() {
  // Initialization
  const int m = 60;  // Number of rows
  const int n = 72;  // Number of cols
  const int input_size = 3 * m * n * sizeof(unsigned char);
  const int output_size = m * n * sizeof(unsigned char);
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
  cudaCheck(cudaMemcpy(d_input_image, h_input_image, input_size,
                       cudaMemcpyHostToDevice));

  // Kernel specification
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(getGridDim(n, blockDim.x), getGridDim(m, blockDim.y), 1);

  // Kernel Launch
  rgbToGreyKernel<<<gridDim, blockDim>>>(m, n, d_input_image, d_output_image);
  cudaCheck(cudaGetLastError());

  // Transfer the data
  cudaCheck(cudaMemcpy(h_output_image, d_output_image, output_size,
                       cudaMemcpyDeviceToHost));

  // Error check
  float error = 0.0f;
  unsigned char true_value = 0.21f * 100 + 0.71f * 50 + 0.07f * 220;
  for (int i = 0; i < m * n; i++) {
    error += (h_output_image[i] - true_value);
  }
  printf("Error: %f\n", error);

  // Free
  cudaFree(d_input_image);
  cudaFree(d_output_image);

  return 0;
}
