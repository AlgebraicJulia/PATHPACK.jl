#!/usr/bin/env python3
"""Add a per-row fingerprint to the GPU baselines' shared verification code. Usage: gpu_fingerprint.py SRC_DIR

SRC_DIR is a copy of the earlier comparison's driver sources (src/common, src/fastapsp, ...), whose check is a row hash of
the exact integer distances (src/common/rowhash.cuh, src/fastapsp/host_rowhash.hpp). With the benchmarks' real-valued weights
distances are not integers, and two correct Float32 codes round differently, so a hash cannot compare them. This adds,
next to the hash and outside every timer (the drivers hash after their timers stop):

  GB_FP_PROBES=<file of vertex ids, 1-based>  GB_FP_OUT=<file>
      for every row the driver produced: "label unreachable sum p_1 ... p_P" (label 1-based; unreachable: entries at or
      above the driver's infinity; sum: of the other entries, in double; p_k: the entry in probe column k, %.9g, Inf),
      which bench/gpu_baselines.jl compares with ours under the benchmarks' relative tolerance. (Fast-APSP: appended, so the
      job removes the file before a run; its MPI driver writes one file per rank, GB_FP_OUT.rank<r>.)
  GB_ROWHASH_DIR=<dir>   where the reference row hashes are read (default: ./rowhash).
"""
import sys

src = sys.argv[1]

def patch(path, edits):
    s = open(path).read()
    for old, new in edits:
        assert s.count(old) == 1, f"{path}: anchor not found once: {old[:70]!r}"
        s = s.replace(old, new)
    open(path, "w").write(s)

DEV_FP = r'''
// [artifact] per-row fingerprint (GB_FP_PROBES, GB_FP_OUT): unreachable count, sum of the reachable entries (double),
// and the entries of the probe columns (by 0-based label: probe_pos[label] = probe index or -1)
template <typename T>
__global__ void fp_kernel(const T* __restrict__ D, int64_t ld, int64_t ncols, T thresh, const int32_t* __restrict__ col_label,
                          const int32_t* __restrict__ probe_pos, int P, double* sums, unsigned long long* ninf, float* probes) {
  const int64_t r = blockIdx.x;
  double s = 0; unsigned long long c = 0;
  for (int64_t j = threadIdx.x; j < ncols; j += blockDim.x) {
    T x = D[r * ld + j];
    const bool inf = (x >= thresh || x != x);
    if (inf) c++; else s += (double)x;
    const int64_t lab = col_label ? col_label[j] : j;
    const int q = probe_pos[lab];
    if (q >= 0) probes[r * P + q] = inf ? __int_as_float(0x7f800000) : (float)x;
  }
  for (int o = 16; o > 0; o >>= 1) { s += __shfl_down_sync(0xffffffff, s, o); c += __shfl_down_sync(0xffffffff, c, o); }
  __shared__ double ss[32]; __shared__ unsigned long long sc[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { ss[wid] = s; sc[wid] = c; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k) { s += ss[k]; c += sc[k]; }
    sums[r] = s; ninf[r] = c;
  }
}

struct RowHashResult { long long rows_hashed = 0, rows_bad = 0; bool have_ref = false; };
'''

CTOR_OLD = '''  explicit RowHasher(const std::string& graph) {
    std::ifstream f(std::string(ROWHASH_DIR) + "/" + graph + ".rowhash");'''
CTOR_NEW = '''  // [artifact] fingerprints (see fp_kernel)
  std::vector<int32_t> fp_labels; std::string fp_out; int32_t* d_probe_pos = nullptr; int64_t probe_n = 0;
  std::vector<double> fp_sum; std::vector<unsigned long long> fp_inf; std::vector<float> fp_probe;
  explicit RowHasher(const std::string& graph) {
    if (const char* pf = getenv("GB_FP_PROBES")) {
      if (const char* po = getenv("GB_FP_OUT")) {
        std::ifstream pfs(pf); long v; while (pfs >> v) fp_labels.push_back((int32_t)(v - 1));
        fp_out = po;
      }
    }
    const char* rd = getenv("GB_ROWHASH_DIR");
    std::ifstream f(std::string(rd ? rd : ROWHASH_DIR) + "/" + graph + ".rowhash");'''

HASH_OLD = '''    GB_CUDA(cudaStreamSynchronize(st));
    cudaFree(d_out);
  }'''
HASH_NEW = '''    GB_CUDA(cudaStreamSynchronize(st));
    cudaFree(d_out);
    if (!fp_out.empty() && !fp_labels.empty()) {               // [artifact] fingerprints of the same rows
      const int P = (int)fp_labels.size();
      if (!d_probe_pos || probe_n < ncols) {
        if (d_probe_pos) cudaFree(d_probe_pos);
        std::vector<int32_t> pos(ncols, -1);
        for (int k = 0; k < P; ++k) if (fp_labels[k] >= 0 && fp_labels[k] < ncols) pos[fp_labels[k]] = k;
        GB_CUDA(cudaMalloc(&d_probe_pos, ncols * sizeof(int32_t)));
        GB_CUDA(cudaMemcpy(d_probe_pos, pos.data(), ncols * sizeof(int32_t), cudaMemcpyHostToDevice));
        probe_n = ncols;
      }
      double* d_s; unsigned long long* d_c; float* d_p;
      GB_CUDA(cudaMalloc(&d_s, rows * sizeof(double))); GB_CUDA(cudaMalloc(&d_c, rows * sizeof(unsigned long long)));
      GB_CUDA(cudaMalloc(&d_p, rows * P * sizeof(float)));
      fp_kernel<T><<<(unsigned)rows, 256, 0, st>>>(D, ld, ncols, thresh, d_col_label, d_probe_pos, P, d_s, d_c, d_p);
      size_t o = fp_sum.size(); fp_sum.resize(o + rows); fp_inf.resize(o + rows); fp_probe.resize((o + rows) * P);
      GB_CUDA(cudaMemcpyAsync(fp_sum.data() + o, d_s, rows * sizeof(double), cudaMemcpyDeviceToHost, st));
      GB_CUDA(cudaMemcpyAsync(fp_inf.data() + o, d_c, rows * sizeof(unsigned long long), cudaMemcpyDeviceToHost, st));
      GB_CUDA(cudaMemcpyAsync(fp_probe.data() + o * P, d_p, rows * P * sizeof(float), cudaMemcpyDeviceToHost, st));
      GB_CUDA(cudaStreamSynchronize(st));
      cudaFree(d_s); cudaFree(d_c); cudaFree(d_p);
    }
  }'''

CMP_OLD = '''    r.rows_hashed = (long long)got.size();
    if (!have_ref) return r;'''
CMP_NEW = '''    r.rows_hashed = (long long)got.size();
    if (!fp_out.empty() && !fp_labels.empty()) {               // [artifact] write the fingerprints, by row label
      const size_t P = fp_labels.size();
      if (FILE* f = fopen(fp_out.c_str(), "w")) {
        for (size_t k = 0; k < fp_sum.size(); ++k) {
          long long i = row_label.empty() ? (long long)k : (long long)row_label[k];
          fprintf(f, "%lld %llu %.17g", i + 1, fp_inf[k], fp_sum[k]);
          for (size_t q = 0; q < P; ++q) { float x = fp_probe[k * P + q]; if (x != x || x > 3e38f) fprintf(f, " Inf"); else fprintf(f, " %.9g", (double)x); }
          fprintf(f, "\\n");
        }
        fclose(f);
      }
    }
    if (!have_ref) return r;'''

patch(f"{src}/common/rowhash.cuh", [
    ("struct RowHashResult { long long rows_hashed = 0, rows_bad = 0; bool have_ref = false; };\n", DEV_FP.lstrip("\n")),
    (CTOR_OLD, CTOR_NEW), (HASH_OLD, HASH_NEW), (CMP_OLD, CMP_NEW),
    ("#include <fstream>\n", "#include <fstream>\n#include <cstdlib>\n"),
])

# ---- host version (Fast-APSP keeps its result in host memory) ----
HOST_FP = r'''
// [artifact] per-row fingerprints (GB_FP_PROBES, GB_FP_OUT; MPI: one file per rank), as rowhash.cuh's: unreachable
// entries (>= 5e7, Fast-APSP's infinity is 1e8), sum of the others in double, the probe columns' entries
inline void write_fp(const float* D, int64_t nrows, int64_t n, const int32_t* row_label) {
  const char* pf = getenv("GB_FP_PROBES"); const char* po = getenv("GB_FP_OUT");
  if (!pf || !po) return;
  std::vector<int64_t> probes; { std::ifstream f(pf); long v; while (f >> v) probes.push_back(v - 1); }
  std::string out(po);
  if (const char* rk = getenv("OMPI_COMM_WORLD_RANK")) out += std::string(".rank") + rk;
  else if (const char* rk2 = getenv("PMIX_RANK")) out += std::string(".rank") + rk2;
  std::vector<double> sums(nrows); std::vector<unsigned long long> ninf(nrows);
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t r = 0; r < nrows; ++r) {
    double s = 0; unsigned long long c = 0; const float* row = D + r * n;
    for (int64_t j = 0; j < n; ++j) { float x = row[j]; if (x >= 5e7f || x != x) c++; else s += (double)x; }
    sums[r] = s; ninf[r] = c;
  }
  FILE* f = fopen(out.c_str(), "a"); if (!f) return;     // (appended: the MPI driver writes once per task)
  for (int64_t r = 0; r < nrows; ++r) {
    fprintf(f, "%lld %llu %.17g", (long long)row_label[r] + 1, ninf[r], sums[r]);
    for (int64_t q : probes) { float x = (q >= 0 && q < n) ? D[r * n + q] : 0.0f; if (x >= 5e7f || x != x) fprintf(f, " Inf"); else fprintf(f, " %.9g", (double)x); }
    fprintf(f, "\n");
  }
  fclose(f);
}

'''
patch(f"{src}/fastapsp/host_rowhash.hpp", [
    ("#include <vector>\n", "#include <vector>\n#include <cstdio>\n#include <cstdlib>\n"),
    ('  std::ifstream f("rowhash/" + graph + ".rowhash");',
     '  const char* rd = getenv("GB_ROWHASH_DIR");\n  std::ifstream f(std::string(rd ? rd : "rowhash") + "/" + graph + ".rowhash");'),
    ("// rows [0, nrows) of D (row stride n); row_label[r] = 0-based original source label. Returns #bad rows.\n",
     HOST_FP + "// rows [0, nrows) of D (row stride n); row_label[r] = 0-based original source label. Returns #bad rows.\n"),
])
# the drivers call it after their timers, with or without reference hashes (MPI: once per task, appended)
patch(f"{src}/fastapsp/fap_driver.cpp", [
    ("  long long rh_bad = ref.empty() ? -1 : fap_rh::check_rows(Q, n, n, ident.data(), ref);",
     "  fap_rh::write_fp(Q, n, n, ident.data());   // [artifact]\n  long long rh_bad = ref.empty() ? -1 : fap_rh::check_rows(Q, n, n, ident.data(), ref);"),
])
patch(f"{src}/fastapsp/fap_mpi_driver.cpp", [
    ("    if (!ref.empty()) {   // per-row hashes;",
     "    { auto labw = G.getSubGraphIndex(tasks[i]); fap_rh::write_fp(D.data(), (int64_t)(D.size() / G.num_vertexs), G.num_vertexs, labw.data()); }   // [artifact]\n    if (!ref.empty()) {   // per-row hashes;"),
])
print(f"patched {src}")
