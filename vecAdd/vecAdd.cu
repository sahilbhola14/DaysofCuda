// Add two vector of size n
#include <cuda.h>
#include <iostream>

#define cudaCheck(ans) gpuAssert((ans), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess){
        printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}


__global__ void vecAddKernel(int n, float *A, float *B, float *C){
    // blockDim: Number of threads per block
    // blockIdx: Block Idx
    // threadIdx: Thread Idx in the local block
    int tid = threadIdx.x; // Local id of the thread
    int gid = blockIdx.x * blockDim.x + tid; // Global id of the thread
    if (gid < n) C[gid] = A[gid] + B[gid];
}

int main(){

    // Variable definitions
    int n = 1024; // Size of vector
    int size = n * sizeof(float); // Size of the vector in bytes
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];
    float *d_A, *d_B, *d_C;

    // Allocation of host memory
    for (int i = 0; i<n; i++){
        h_A[i] = static_cast<float>(1.0);
        h_B[i] = static_cast<float>(4.0);
    }

    // Allocation of device memory (args: void type address to a pointer, size)
    cudaCheck(cudaMalloc((void**)&d_A, size));
    cudaCheck(cudaMalloc((void**)&d_B, size));
    cudaCheck(cudaMalloc((void**)&d_C, size));

    // Copy the data to the Device (Target, Source, Size, Direction)
    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Invoke the Kernel
    int blockSize = 256; // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;
    printf("Running the kernel with %d threads per block and %d blocks\n", blockSize, numBlocks+1);
    vecAddKernel<<<numBlocks, blockSize>>>(n, d_A, d_B, d_C);
    cudaCheck(cudaGetLastError());
    
    // Copy the result back to host
    cudaCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Check the error
    float error = 0.0;
    for (int i = 0; i<n; i++){
        error = error + (h_C[i] - 5.0);
    }
    std::cout << "Error in vector addition : " << error << std::endl;

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
