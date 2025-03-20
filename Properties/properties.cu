// Sec 3.6 of PMPP
#include <cuda.h>

#include <iostream>

int main() {
  // Get Device count
  int dev_count;
  cudaGetDeviceCount(&dev_count);
  printf("Number of CUDA devices: %d\n", dev_count);
  // Get Device properties
  cudaDeviceProp dev_prop;
  for (int i = 0; i < dev_count; i++) {
    printf("-----\n");
    printf("Properties for Device: %d\n", i);
    cudaGetDeviceProperties(&dev_prop, i);
    printf("Max threads per block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("Number of SMs: %d\n", dev_prop.multiProcessorCount);
    printf("Max threads per SMs: %d\n", dev_prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per SMs: %d\n", dev_prop.maxBlocksPerMultiProcessor);
    printf("Max threads in x-, y-, and z-direction: (%d, %d, %d)\n",
           dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1],
           dev_prop.maxThreadsDim[2]);
    printf("Max blocks in x-, y-, and z-direction: (%d, %d, %d)\n",
           dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
           dev_prop.maxGridSize[2]);
    printf("Warp size: %d\n", dev_prop.warpSize);
  }
  return 0;
}
