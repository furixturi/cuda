# Day 4

## Lecture 4 notes
- [Session 4 recording](https://mediaspace.illinois.edu/media/t/1_z883mlnv)
- [Deck 3](https://lumetta.web.engr.illinois.edu/408-Sum24/slide-copies/ece408-lecture3-CUDA%20parallelism-model-Sum24.pdf)

### Lecture 3 review
- More on CUDA Function Delarations
  - `__device__ float DeviceFunc()`
    - executed on: device
    - callable from: device (e.g., from a kernel)
  - `__global__ void KernelFunc()`
    - executed on: device
    - callable from: host
    - has to return `void`
  - `__host__ float HostFunc()`
    -  executed on: host
    -  callable from: host
  -  `__device__` and `__host__` can be used together in front of one function, two versions (one for device and one for host) of it will be compiled. 
     -  So you can write a utility function that can be used both on device and host

- Compiling A CUDA program
  - Integrated C programs with CUDA extensions ->
    - NVCC Compiler ->
      - Host Code
        - Host C Compiler/Linker
      - Device Code (PTX: intermediate device code, not actual GPU ISA machine code which is completely proprietary)
        - NVIDIA device driver then does Just-in-Time compilation for architecture and micro-architecture specific optimization (only then you know the micro architecture it will be run on)
          - e.g. llvm
    - -> Heterogeneous Computing Platform with CPUs and GPUs

### Lecture 4 learning items
- multi-dimensional logical organization of CUDA threads
- control structures in a kernel (such as loop)
- concepts
  - thread scheduling
  - latency tolerance
  - hardware occupancy