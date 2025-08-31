// nvcc -O3 -std=c++14 -arch=sm_70 q1_histogram_cuda.cu -o q1_histo
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

#ifndef NBINS
#define NBINS 256
#endif

#define CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

__device__ __forceinline__ int padded_bins() { return NBINS + (NBINS/32); }

template<int BLOCK_THREADS>
__global__ void histo_warp_priv_kernel(const uint8_t* __restrict__ data,
                                       size_t n,
                                       unsigned int* __restrict__ g_bins) {
  constexpr int WARP_SIZE = 32;
  const int warp_id  = threadIdx.x / WARP_SIZE;
  const int lane_id  = threadIdx.x % WARP_SIZE;
  const int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;

  extern __shared__ unsigned int smem[];
  unsigned int* warp_hist = smem + warp_id * padded_bins();

  // init warp-local hist
  for (int b = lane_id; b < padded_bins(); b += WARP_SIZE) warp_hist[b] = 0u;
  __syncthreads();

  // accumulate
  const size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint8_t key = data[i];
    int p = key + (key/32);           // bank-conflict padding
    atomicAdd(&warp_hist[p], 1u);
  }
  __syncthreads();

  // reduce & flush (one global atomic per bin per block)
  for (int b = lane_id; b < NBINS; b += WARP_SIZE) {
    int p = b + (b/32);
    unsigned int sum = 0u;
    #pragma unroll
    for (int w = 0; w < WARPS_PER_BLOCK; ++w)
      sum += smem[w * padded_bins() + p];
    if (warp_id == 0) atomicAdd(&g_bins[b], sum);
  }
}

static void run_hist(const uint8_t* d_data, size_t n, unsigned int* d_bins,
                     int blocks, int threads) {
  size_t shmem = (threads / 32) * (NBINS + NBINS/32) * sizeof(unsigned int);
  switch (threads) {
    case 128: histo_warp_priv_kernel<128><<<blocks,128,shmem>>>(d_data,n,d_bins); break;
    case 256: histo_warp_priv_kernel<256><<<blocks,256,shmem>>>(d_data,n,d_bins); break;
    case 512: histo_warp_priv_kernel<512><<<blocks,512,shmem>>>(d_data,n,d_bins); break;
    default:  histo_warp_priv_kernel<256><<<blocks,256,shmem>>>(d_data,n,d_bins); break;
  }
}

int main(int argc, char** argv) {
  // cfg: N elements, bins = NBINS(compile-time)
  size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : (1ull<<24); // ~16.7M default
  printf("Histogram: N=%zu, NBINS=%d\n", N, NBINS);

  // host random input
  std::vector<uint8_t> h(N);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> uni(0, NBINS-1);
  for (size_t i=0;i<N;++i) h[i]=(uint8_t)uni(rng);

  // device buffers
  uint8_t *d_data=nullptr; unsigned int *d_bins=nullptr;
  CHECK(cudaMalloc(&d_data, N*sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_bins, NBINS*sizeof(unsigned int)));
  CHECK(cudaMemcpy(d_data, h.data(), N, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_bins, 0, NBINS*sizeof(unsigned int)));

  int dev=0; cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, dev));
  int blocks = std::max(1, prop.multiProcessorCount * 6);
  int threads = 256;

  // time kernel
  cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  run_hist(d_data, N, d_bins, blocks, threads);
  CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
  float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));

  // light sanity: sum of bins == N
  std::vector<unsigned int> hist(NBINS);
  CHECK(cudaMemcpy(hist.data(), d_bins, NBINS*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  unsigned long long s=0; for (int i=0;i<NBINS;++i) s+=hist[i];

  double gb_read = (double)N / (1<<30); // GB of reads (bytes→GB)
  double gpselem = (double)N / (ms/1e3); // elements/sec
  printf("Time: %.3f ms | Throughput: %.2f Melem/s | Reads: %.2f GB in %.3f ms | sum=%llu\n",
         ms, gpselem/1e6, gb_read, ms, s);

  CHECK(cudaFree(d_data)); CHECK(cudaFree(d_bins));
  return 0;
}
