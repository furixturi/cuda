#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// CUDA error checking macro (preprocessor macros has context of file/line info)
#define CUDA_CHECK(err) __cudaCheck((err), __FILE__, __LINE__) 
inline void __cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel for vector addition
__global__
void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Host function to add two vectors
void vectorAdd(const float* A, const float* B, float* C, int N) {
    size_t size = N * sizeof(float);

    // Allocate device memory
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, size));
    CUDA_CHECK(cudaMalloc(&B_d, size));
    CUDA_CHECK(cudaMalloc(&C_d, size));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    // Configure launch parameters and launch kernel
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerGrid);
    // Launch kernel
    vectorAddKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

bool verifyVectorAdd(const float* A, const float* B, const float* C, int N) {
    for (int i = 0; i < N; ++i) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5f) { // Allow a small tolerance for floating point comparison
            std::cerr << "Verification failed at index " << i << ": "
                      << "C[" << i << "] = " << C[i] << ", "
                      << "expected = " << expected << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* A_h = (float*)malloc(size);
    float* B_h = (float*)malloc(size);
    float* C_h = (float*)malloc(size);

    // Initialize inputs
    for (int i = 0; i < N; ++i) {
        A_h[i] = static_cast<float>(i); // safely cast int i to float
        B_h[i] = static_cast<float>(N-i);
    }

    // Perform vector addition
    vectorAdd(A_h, B_h, C_h, N);

    // Check results
    bool success = verifyVectorAdd(A_h, B_h, C_h, N);
    std::cout << (success ? "Vector addition successful!" : "Vector addition failed!") << std::endl;
    
    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h);

    return success ? 0 : 1;
}