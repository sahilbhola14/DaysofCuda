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


__global__ void vecAdd(int n, float *A, float *B, float *C){
    // Enter the kernel here
}

int main(){

    // Variable definitions
    int n = 10; // Size of vector
    int size = n * sizeof(float); // Size of the vector in bytes
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];
    float *d_A, *d_B, *d_C;

    // Allocation of host memory
    for (int i = 0; i<n; i++){
        h_A[i] = static_cast<float>(1.0);
        h_B[i] = static_cast<float>(1.0);
    }

    // Allocation of device memory
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
