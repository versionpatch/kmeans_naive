# Implementation of Naive K-Means and K-Means++ in CUDA, optimized for high dimensional data.   

High throughput computation (about 11 TFlops on RTX 3080) of the distance matrix by block tiling, memory access coalescing and better use of the cache.   
K-Means++ implemented with a high memory throughput kernel (700 GB/s on RTX 3080) for distance computation and Cuda Thrust for parallel prefix sum and point selection.   
Parallel reduction for argmin.    
Accumulation step is implemented using locks. May be inefficient in some cases. Maybe doing a counting sort then summing yields better performance, but this is not implemented.   
