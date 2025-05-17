# Day 6 - 2025/05/14

## Lecture 4 note 

- [Session 4 recording](https://mediaspace.illinois.edu/media/t/1_z883mlnv)
- [Deck 3](https://lumetta.web.engr.illinois.edu/408-Sum24/slide-copies/ece408-lecture3-CUDA%20parallelism-model-Sum24.pdf)

### Lecture 4 learning items
- multi-dimensional logical organization of CUDA threads
- control structures in a kernel (such as loop)
- concepts
  - thread scheduling
  - latency tolerance
  - hardware occupancy


### Thread assignment details
- How many threads are launched where N = 1000, block_size = 256?
  - floor((N + block_size - 1) / block_size) * block_size = 1024
    - 24 threads will have no real data to add, that's why we need to check `i < N`
- Thread coarsening: each thread launching has some overhead
  - let 1 thread handle 2 additions
```c
// kernel launch configuration: set gridDim to be twice the blockDim (threads per block)
vecAdd<<<ceil(N/(2*256.0)), 256>>>

// in kernel code
i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
if (i < N) C[i] = A[i] + B[i]
```
  - for performance, make sure consecutive threads access consecutive memory locations

### 2D image kernel examples
- Test the boundaries
- greyscale
- blur

### CUDA Threads Blocks
- All threads in a block execute the same kernel program (SPMD)
- Programmer declares block size (num of threads per block)
  - block size: 1 to 1024 concurrent threads
  - block shape: 1D, 2D, or 3D
  - grid size is more flexible, can be quite large (e.g. X: 64 million, Y and Z: 64K)
- Kernel code uses thread index and block index to select work and address shared data
  - i = blockIdx.x(or y, z) * blockDim.x + threadIdx.x
- Threads within the same block
  - have thread index numbers
  - **share data** and **synchronize** while doing their share of work
- Threads in different blocks **cannot** cooperate
  - Thread blocks are scheduled to the SMs(streaming multiprocessors), the scheduling algorithm is proprietary
    - historically, it is the linearization of x, y, z blocks of the grid
- Blocks execute in **arbitrary order!**
  - do synchronization using the global shared memory can easily create deadlocks

### Compute Capabilities are GPU-dependent
- E.g.
  - shared memory / SM: Kepler: 16/48KB, Maxwell: 64KB
  - register file size / SM: 256KB
  - active blocks / SM: Kepler: 16, Maxwell: 32
