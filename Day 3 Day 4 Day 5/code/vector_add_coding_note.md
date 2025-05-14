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
Successfully running should show
```
Vector addition successful!
```
### Profiling

1. Generate the profile with `nsys`
```bash
nsys profile -o vector_add_report ./vector_add
```
This generates a report `vector_add_report_nsys-rep`

2. To see the report in CLI
```bash
nsys stats --force-export=true vector_add_report.nsys-rep
```
This generates a report `vector_add_report.sqlite` file and then prints out the table.

** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ---------------------
     63.5        462069872        698    661991.2     44549.0      1120   70308400    2945358.5  ioctl                
     33.4        242978049         13  18690619.2  13991479.0      4920  100188201   27096210.2  poll                 
      2.5         18254598         59    309400.0     13060.0      4800    2922330     626163.4  open64               
      0.3          2491911         34     73291.5     11395.0      5600    1625828     275791.9  mmap64               
      0.1           843598         10     84359.8     83439.0     48889     121458      19336.0  sem_timedwait        
      0.1           664121          4    166030.3    148828.0    134828     231637      44283.9  pthread_create       
      0.0           262424         32      8200.8      6330.0      2450      25850       5154.3  fopen                
      0.0           152109         15     10140.6      4500.0      1980      71699      17385.4  mmap                 
      0.0            92609         46      2013.2        45.0        40      87539      12897.9  fgets                
      0.0            75199         29      2593.1      2620.0      1380       4520        715.3  fclose               
      0.0            66198         66      1003.0       930.0       640       2240        298.5  fcntl                
      0.0            60040         11      5458.2      4950.0      1550       8440       1980.1  write                
      0.0            44250          7      6321.4      5570.0      2790       8880       2272.7  open                 
      0.0            40610         14      2900.7      2555.0      1210       4410       1029.1  read                 
      0.0            32900          6      5483.3      5185.0      2270      10620       3109.9  munmap               
      0.0            20050          2     10025.0     10025.0      8580      11470       2043.5  socket               
      0.0            19170          2      9585.0      9585.0      8580      10590       1421.3  fread                
      0.0            12620          1     12620.0     12620.0     12620      12620          0.0  connect              
      0.0            11160          1     11160.0     11160.0     11160      11160          0.0  putc                 
      0.0             9999          1      9999.0      9999.0      9999       9999          0.0  pipe2                
      0.0             8230         64       128.6       130.0        30        370         61.4  pthread_mutex_trylock
      0.0             6460          8       807.5       785.0       650       1150        157.0  dup                  
      0.0             5250          1      5250.0      5250.0      5250       5250          0.0  fwrite               
      0.0             3430          1      3430.0      3430.0      3430       3430          0.0  bind                 
      0.0             1360          1      1360.0      1360.0      1360       1360          0.0  listen               
      0.0              790          7       112.9        40.0        30        470        161.2  fflush               

Processing [vector_add_report.sqlite] with [/opt/nvidia/nsight-systems/2023.2.3/host-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------
     89.8        258862784          3  86287594.7   1616948.0    587072  256658764  147546659.3  cudaMalloc            
      9.6         27557673          1  27557673.0  27557673.0  27557673   27557673          0.0  cudaLaunchKernel      
      0.5          1503120          3    501040.0    221937.0    220607    1060576     484572.8  cudaMemcpy            
      0.1           267336          3     89112.0     63049.0     59049     145238      48647.7  cudaFree              
      0.0             1330          1      1330.0      1330.0      1330       1330          0.0  cuModuleGetLoadingMode

Processing [vector_add_report.sqlite] with [/opt/nvidia/nsight-systems/2023.2.3/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                             Name                            
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------------------------------
    100.0             5087          1    5087.0    5087.0      5087      5087          0.0  vectorAddKernel(const float *, const float *, float *, int)

Processing [vector_add_report.sqlite] with [/opt/nvidia/nsight-systems/2023.2.3/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------
     71.2           848963      1  848963.0  848963.0    848963    848963          0.0  [CUDA memcpy DtoH]
     28.8           342848      2  171424.0  171424.0    169856    172992       2217.5  [CUDA memcpy HtoD]

Processing [vector_add_report.sqlite] with [/opt/nvidia/nsight-systems/2023.2.3/host-linux-x64/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
      8.389      2     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy HtoD]
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy DtoH]

This shows 
- CUDA kernel `vectorAddKernel` ran for 5087 ns
- Most of the time was not spent in the kernel but in memory allocation and launching the kernel (which is typical when running a light workload on a large GPU)
  - cudaMalloc 89.8%
  - cudaLaunchKernel 9.6%
- Memory transfer
  - H2D (host to device): 8.4MB, 343 ns
  - D2H (device to host): 4.2MB, 849 ns
- System calls
  - OS runtime summary: most time spent in `ioctl` and `poll`, normal, as they are often used under the hood by CUDA runtime and drivers for synchronization and memory management
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