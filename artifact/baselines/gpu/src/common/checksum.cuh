// Device-side checksum of a (rows x n) block of a distance matrix, same definition as ours:
//   sum  = Float64 sum of all finite d(i,j) (including d(i,i)=0)
//   cnt  = number of unreachable pairs
// "Unreachable" = value >= thresh (thresh is the code's own infinity sentinel; for float codes
// pass 1e30f so +inf / FLT_MAX / 1e38-style sentinels all count).
#pragma once
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>

#define GB_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
  printf("STATUS cuda_error %s\n", cudaGetErrorString(e_)); fflush(stdout); exit(3); } } while (0)

namespace gb {

template <typename T>
__global__ void checksum_kernel(const T* __restrict__ D, int64_t ld, int64_t rows, int64_t ncols,
                                T thresh, double* out_sum, unsigned long long* out_cnt) {
  double s = 0.0; unsigned long long c = 0;
  int64_t total = rows * ncols;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t i = idx / ncols, j = idx - i * ncols;
    T v = D[i * ld + j];
    if (v >= thresh || v != v) c++; else s += (double)v;
  }
  // warp reduce
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_down_sync(0xffffffff, s, o);
    c += __shfl_down_sync(0xffffffff, c, o);
  }
  __shared__ double ss[32]; __shared__ unsigned long long sc[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { ss[wid] = s; sc[wid] = c; }
  __syncthreads();
  if (wid == 0) {
    int nw = blockDim.x >> 5;
    s = lane < nw ? ss[lane] : 0.0; c = lane < nw ? sc[lane] : 0ull;
    for (int o = 16; o > 0; o >>= 1) {
      s += __shfl_down_sync(0xffffffff, s, o);
      c += __shfl_down_sync(0xffffffff, c, o);
    }
    if (lane == 0) { atomicAdd(out_sum, s); atomicAdd(out_cnt, c); }
  }
}

struct Checksum { double sum = 0; unsigned long long cnt = 0; };

// Accumulating device checksum helper (one small allocation, reused).
struct DevChecksum {
  double* d_sum = nullptr; unsigned long long* d_cnt = nullptr;
  DevChecksum() { GB_CUDA(cudaMalloc(&d_sum, sizeof(double))); GB_CUDA(cudaMalloc(&d_cnt, sizeof(unsigned long long))); reset(); }
  ~DevChecksum() { cudaFree(d_sum); cudaFree(d_cnt); }
  void reset(cudaStream_t st = 0) { cudaMemsetAsync(d_sum, 0, sizeof(double), st); cudaMemsetAsync(d_cnt, 0, sizeof(unsigned long long), st); }
  template <typename T>
  void add(const T* D, int64_t ld, int64_t rows, int64_t ncols, T thresh, cudaStream_t st = 0) {
    if (rows <= 0) return;
    int64_t total = rows * ncols;
    int blocks = (int)std::min<int64_t>((total + 255) / 256, 8192);
    checksum_kernel<T><<<blocks, 256, 0, st>>>(D, ld, rows, ncols, thresh, d_sum, d_cnt);
  }
  Checksum get(cudaStream_t st = 0) {
    Checksum c; GB_CUDA(cudaStreamSynchronize(st));
    GB_CUDA(cudaMemcpy(&c.sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    GB_CUDA(cudaMemcpy(&c.cnt, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return c;
  }
};

// Per-row checksums (for sampled / extrapolated runs, compared against CPU Dijkstra).
template <typename T>
__global__ void row_checksum_kernel(const T* __restrict__ D, int64_t ld, int64_t ncols, T thresh,
                                    double* row_sum, unsigned long long* row_cnt) {
  int64_t i = blockIdx.x;
  double s = 0.0; unsigned long long c = 0;
  for (int64_t j = threadIdx.x; j < ncols; j += blockDim.x) {
    T v = D[i * ld + j];
    if (v >= thresh || v != v) c++; else s += (double)v;
  }
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_down_sync(0xffffffff, s, o);
    c += __shfl_down_sync(0xffffffff, c, o);
  }
  __shared__ double ss[32]; __shared__ unsigned long long sc[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { ss[wid] = s; sc[wid] = c; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k) { s += ss[k]; c += sc[k]; }
    row_sum[i] = s; row_cnt[i] = c;
  }
}

}  // namespace gb
