// Add two vector of size n
#include <cuda.h>
#include <iostream>

#define cudaCheck(ans) gpuAssert((ans), __FILE__, __LINE__);

void gpuAssert(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess){
        printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}


__global__ void vecAdd(int n, float *A, float *B, float *C){
    // Enter the kernel here
}

int main(){

    // Variable definitions
    int n = 10; // Size of vector
    int size = n * sizeof(float); // Size of the vector in bytes
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocation of memory
    cudaCheck(cudaMalloc((void**)&d_A, size));
    cudaCheck(cudaMalloc((void**)&d_B, size));
    cudaCheck(cudaMalloc((void**)&d_C, size));

    // Copy the data to the Device
    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Invoke the Kernel

    // Check the Kernel Launch
    
    // Copy the result back to host
    cudaCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
