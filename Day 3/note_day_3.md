# Day 3

## Lecture 3 notes (continued)

- [recording](https://mediaspace.illinois.edu/media/t/1_joyw26bq) 25:00~
- [deck](https://lumetta.web.engr.illinois.edu/408-Sum24/slide-copies/ece408-lecture2-CUDA-introduction-Sum24.pdf) P15~

### Partial Overview of CUDA Memories

![partial overview of GPU memory](media/GPU_mem.png)
- Device code (GPU code) can
  - R/W per-thread **registers**
    - register size: 64K per processor, divided among thread blocks on the same SM (streaming multiprocessor), 10-100 registers per thread
  - R/W per-grid **global GPU memory**
    - accessible to all threads in the grid, but not guaranteed to be instantly visible to all threads
- Host code (CPU code) can
  - **Allocate memory** for per-grid global memory
  - **Transfer data to/from** per-grid **global GPU memory** 
    - use RDMA to move data across PCIe
- CUDA device memory management API functions
  - `cudaMalloc`
    - allocate in **device global memory**
    - accepts two parameters
      - address to store the pointer to the allocated memory (ess. a pointer to a pointer)
      - size in bytes of allocated memory
  - `cudaFree`
    - **free allocated device global memory**
    - one parameter: **pointer** to the memory to free (returned from `cudaMalloc`)
  - Code example
```c
// host code
void vecAdd(float* A, float* B, float* C, intN)
{
    int size = N * sizeof(float); // size of memory to allocate for each float 
    float *A_d, *B_d, *C_d; // pointers to the floats on device

    // 1. Allocate device memory for A, B, and C
    cudaMalloc((void **) &A_d, size); // parameter 1 is a pointer to the pointer that points to the device memory
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // 1.1 Copy A, B to device memory
    // 2. Kernel invocation

    // 3. Transfer C from device to host
    // 3.1 Free device memory for A, B, C
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}