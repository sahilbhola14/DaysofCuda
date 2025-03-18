// Add two m \times n matrix
#include <cuda.h>
#include <iostream>

#define cudaCheck(ans) gpuAssert((ans), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess){
        printf("<Error>: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

int getGridDim(const int n, const int blockDim){
    return (n + blockDim -1 ) / blockDim;
}

__global__ void matAdd(const int m, const int n, float *A, float *B, float *C){
    // Global index of the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < m ) && (col < n)){
        // Compute the linear index
        int linear_idx = row * n + col;
        C[linear_idx] = A[linear_idx] + B[linear_idx];
    }
}


int main(){
    // Declations
    int n = 40; // Number of columns
    int m = 30; // Number of rows
    int size = m * n * sizeof(float); // Size of the matrix in bytes
    float *h_A = new float[m * n]; // Matrix A
    float *h_B = new float[m * n]; // Matrix B
    float *h_C = new float[m * n]; // Matrix C
    float *d_A, *d_B, *d_C;

    // Allocation of host memory
    for (int i = 0; i < m * n; i++){
        h_A[i] = static_cast<float>(1.0);
        h_B[i] = static_cast<float>(2.0);
    }

    // Allocation of device memory
    cudaCheck(cudaMalloc((void **)&d_A, size));
    cudaCheck(cudaMalloc((void **)&d_B, size));
    cudaCheck(cudaMalloc((void **)&d_C, size));
    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel specification
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(getGridDim(n, blockDim.x), getGridDim(m, blockDim.y), 1);

    // Launch the kernel
    matAdd<<<gridDim, blockDim>>>(m, n, d_A, d_B, d_C);
    cudaCheck(cudaGetLastError());

    // Copy the data back
    cudaCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Check the Error
    float error = 0.0;
    for (int i = 0; i < m * n; i++){
        error = error + (h_C[i] - 3.0);
    }

    printf("Error: %f\n", error);

    //Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

