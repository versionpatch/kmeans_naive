# Implementation of Naive K-Means in CUDA, optimized for high dimensional data.

High throughput computation of the distance matrix by block tiling, memory access coalescing and better use of the cache.
Parallel reduction for argmin.
Accumulation step is implemented using locks. May be inefficient in some cases. Maybe doing a counting sort then summing yields better performance, but this is not implemented.
