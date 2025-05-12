# vector_add coding note

## Build and run

### Compile

```bash
nvcc -o vector_add vector_add.cu
```

### Run

```bash
./vector_add
```

## Error check of CUDA device memory management API

Cuda memory management API returns an error code. We can wrap the API calls with an error checking macro to catch each error immediately.

Why we use a macro instead of a function for error checking:

- `macro`s has access to context like line numbers and file names:
  - `__FILE__`
  - `__LINE__`
- no function call overhead
- works with all CUDA API calls naturally

A good pattern is to use an inline function in the macro and give the function the file and line number context.
```cpp
#define CUDA_CHECK(err) __cudaCheck((err), __FILE__, __LINE__)
inline void __cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorSring(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
```
In this way, we get
- full file/line context from macro
- function encapsulation that gives us better IDE and debugging support

## Kernel launch parameters 

### How is threads per block (dimBlock) chosen 


### How is blocks per grid (dimGrid) calculated

### Using dim3


## How does CUDA handle more elements than threads

Logically all threads are executed in parallel, but in reality all of the thread blocks will be scheduled and allocated to a fixed number of streaming multi-processors (SMs). Each SM will executed a number of thread blocks in parallel that fits. 

tldr: CUDA does not launch all threads simultaneously, the GPU only runs as many as it can handle, the rest are queued and run as resources become available.