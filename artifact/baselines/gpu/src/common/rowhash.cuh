// Position-sensitive per-row hash of a distance matrix, same definition as ours
// (<dir>/<graph>.rowhash, line i = rowhash(i) in hex; <dir> = ROWHASH_DIR, or GB_ROWHASH_DIR after gpu_fingerprint.py):
//   uint64 arithmetic mod 2^64; i = source, j = target, 1-based ORIGINAL labels;
//   v(i,j) = d(i,j) as an integer if finite, else 0xFFFFFFFF;
//   rowhash(i) = sum_j v(i,j) * (j * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019).
// Usage (device rows in any order / any column labelling):
//   gb::RowHasher H(graph_name);                       // loads the reference file (if present)
//   H.hash_rows<T>(D, ld, rows, ncols, thresh, d_row_label, d_col_label, st);   // labels 0-based, nullable
//   auto r = H.compare(host_row_labels);                // rows_hashed, rows_bad
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "checksum.cuh"

namespace gb {

static const char* ROWHASH_DIR = "rowhash";

template <typename T>
__global__ void rowhash_kernel(const T* __restrict__ D, int64_t ld, int64_t ncols, T thresh,
                               const int32_t* __restrict__ col_label, unsigned long long* out) {
  const int64_t r = blockIdx.x;
  const unsigned long long C1 = 0x9E3779B97F4A7C15ull, C2 = 0x632BE59BD9B4E019ull;
  unsigned long long acc = 0;
  for (int64_t j = threadIdx.x; j < ncols; j += blockDim.x) {
    T x = D[r * ld + j];
    unsigned long long v = (x >= thresh || x != x) ? 0xFFFFFFFFull : (unsigned long long)(long long)x;
    unsigned long long jl = (unsigned long long)(col_label ? col_label[j] : j) + 1ull;
    acc += v * (jl * C1 + C2);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  __shared__ unsigned long long s[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) s[wid] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k) acc += s[k];
    out[r] = acc;
  }
}

struct RowHashResult { long long rows_hashed = 0, rows_bad = 0; bool have_ref = false; };

struct RowHasher {
  std::vector<unsigned long long> ref;  // ref[i] for 0-based original label i
  std::vector<unsigned long long> got;  // per produced row, in production order
  bool have_ref = false;
  explicit RowHasher(const std::string& graph) {
    std::ifstream f(std::string(ROWHASH_DIR) + "/" + graph + ".rowhash");
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
      size_t a = line.find_first_not_of(" \t");
      if (a == std::string::npos) continue;
      ref.push_back(std::stoull(line.substr(a), nullptr, 16));
    }
    have_ref = !ref.empty();
  }
  // hash `rows` device rows (row r at D + r*ld); appends to got
  template <typename T>
  void hash_rows(const T* D, int64_t ld, int64_t rows, int64_t ncols, T thresh, const int32_t* d_col_label = nullptr,
                 cudaStream_t st = 0) {
    if (rows <= 0) return;
    unsigned long long* d_out;
    GB_CUDA(cudaMalloc(&d_out, rows * sizeof(unsigned long long)));
    rowhash_kernel<T><<<(unsigned)rows, 256, 0, st>>>(D, ld, ncols, thresh, d_col_label, d_out);
    size_t off = got.size(); got.resize(off + rows);
    GB_CUDA(cudaMemcpyAsync(got.data() + off, d_out, rows * sizeof(unsigned long long), cudaMemcpyDeviceToHost, st));
    GB_CUDA(cudaStreamSynchronize(st));
    cudaFree(d_out);
  }
  // row_label[k] = 0-based original source label of the k-th hashed row (empty: k itself)
  RowHashResult compare(const std::vector<int64_t>& row_label = {}) const {
    RowHashResult r; r.have_ref = have_ref;
    r.rows_hashed = (long long)got.size();
    if (!have_ref) return r;
    for (size_t k = 0; k < got.size(); ++k) {
      int64_t i = row_label.empty() ? (int64_t)k : row_label[k];
      if (i < 0 || i >= (int64_t)ref.size() || ref[i] != got[k]) r.rows_bad++;
    }
    return r;
  }
};

}  // namespace gb
