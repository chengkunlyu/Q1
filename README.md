# Q1 – CUDA Histogram Computation

This repository implements a parallel histogram kernel in CUDA, optimized with warp-private shared memory bins.

## Build
```bash
nvcc -O3 -std=c++14 -arch=sm_70 q1_histogram_cuda.cu -o q1_histo

## Run
./q1_histo             # Default: N = 16,777,216 elements
./q1_histo 33554432    # Custom N

## Output
Histogram: N=16777216, NBINS=256
Time: 0.249 ms | Throughput: 67423.87 Melem/s | Reads: 0.02 GB in 0.249 ms | sum=16777216
Histogram: N=33554432, NBINS=256
Time: 0.286 ms | Throughput: 117369.16 Melem/s | Reads: 0.03 GB in 0.286 ms | sum=33554432
