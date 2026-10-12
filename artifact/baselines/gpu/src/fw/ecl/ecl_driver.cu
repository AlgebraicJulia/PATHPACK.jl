// ECL-APSP v1.0 (Liu & Burtscher, Texas State, BSD-3) inside our harness.
//   ecl_apsp <graph.mtx> [--budget 120]
// Kernels: ecl_kernels.cuh (generated from ECL-APSP_10.cu by make_ecl_kernels.py: 64-bit indices
// only).  Host flow identical to its FW_gpu_64(): init1/init2, FW0_64, then per x FWrowcol_64 +
// FWrem_64.  int32 distances, INF = INT_MAX/2, 3 upper^2 ints on the device (AdjMat, krows, kcols).
// Removed: its serial O(n^3) FW_cpu check and its D2H copy (we checksum on the device instead).
// Timing as in fwbench.cu: gpu_compute = init + FW (graph on device -> distances on device),
// inproc = alloc/H2D + init + FW, e2e = process wall (job script).  If the FW would exceed
// --budget seconds (estimated from measured rounds), enough rounds to time ~5 s are run and the
// rest is extrapolated (every round costs the same: (sub-1)^2 64x64x64 tiles).
#include "../../common/mtx.hpp"
#include "../../common/checksum.cuh"
#include "../../common/result.hpp"
#include "../../common/rowhash.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

struct ECLgraph { int nodes; int edges; int* nindex; int* nlist; int* eweight; };
#include "ecl_kernels.cuh"

__global__ void f2i(const float* w, int* o, int64_t m) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += (int64_t)gridDim.x * blockDim.x) o[i] = (int)w[i];
}

int main(int argc, char** argv) {
  double T0 = gb::now_s();
  std::string path = argv[1]; double budget = 120.0; int force_rounds = -1;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--budget")) budget = atof(argv[++i]);
    else if (!strcmp(argv[i], "--rounds")) force_rounds = atoi(argv[++i]);
  }
  gb::Result R; R.code = "fw_eclapsp"; R.graph = gb::basename_noext(path);
  double t = gb::now_s();
  GB_CUDA(cudaFree(0));
  cudaDeviceProp prop; GB_CUDA(cudaGetDeviceProperties(&prop, 0)); R.gpu = prop.name;
  { // warm-up: all kernels once on a 128 x 128 problem
    int up = 128; int *a, *kr, *kc; ECLgraph g{0, 0, nullptr, nullptr, nullptr};
    GB_CUDA(cudaMalloc(&a, up * up * 4)); GB_CUDA(cudaMalloc(&kr, up * up * 4)); GB_CUDA(cudaMalloc(&kc, up * up * 4));
    init1<<<16, ThreadsPerBlock>>>(up, a, up); init2<<<1, ThreadsPerBlock>>>(g, a, up);
    FW0_64<<<1, ThreadsPerBlock>>>(a, up, kr, kc);
    for (int x = 0; x < 2; ++x) { FWrowcol_64<<<2, ThreadsPerBlock>>>(a, up, kr, kc, x, 1); FWrem_64<<<1, ThreadsPerBlock>>>(a, up, kr, kc, x, 1); }
    GB_CUDA(cudaDeviceSynchronize()); cudaFree(a); cudaFree(kr); cudaFree(kc);
    gb::DevChecksum c; c.add<int>(nullptr, 0, 0, 0, INT_MAX / 2);
  }
  double t_ctx = gb::now_s() - t;
  double t_read; gb::Csr G = gb::read_mtx(path, &t_read);
  const int n = (int)G.n; R.n = n; R.nnz = G.nnz_file;
  const int sub = (n + tile - 1) / tile, upper = sub * tile, subm1 = sub - 1;

  double ti0 = gb::now_s();
  std::vector<int> rp32(n + 1); for (int i = 0; i <= n; ++i) rp32[i] = (int)G.rowptr[i];
  ECLgraph dg{n, (int)G.m, nullptr, nullptr, nullptr};
  float* dwf;
  GB_CUDA(cudaMalloc(&dg.nindex, sizeof(int) * (n + 1)));
  GB_CUDA(cudaMalloc(&dg.nlist, sizeof(int) * std::max<int64_t>(G.m, 1)));
  GB_CUDA(cudaMalloc(&dg.eweight, sizeof(int) * std::max<int64_t>(G.m, 1)));
  GB_CUDA(cudaMalloc(&dwf, sizeof(float) * std::max<int64_t>(G.m, 1)));
  int *dA = nullptr, *dKr = nullptr, *dKc = nullptr;
  size_t bytes = sizeof(int) * (size_t)upper * upper;
  if (cudaMalloc(&dA, bytes) != cudaSuccess || cudaMalloc(&dKr, bytes) != cudaSuccess || cudaMalloc(&dKc, bytes) != cudaSuccess) {
    R.status = "oom"; R.note = "3 x upper^2 int allocation failed"; R.phase("ctx_warmup", t_ctx); R.phase("read", t_read);
    R.extra.push_back({"bytes_needed", std::to_string(3 * bytes)});
    R.e2e_inproc = gb::now_s() - T0; R.print(); return 0;
  }
  GB_CUDA(cudaMemcpy(dg.nindex, rp32.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
  GB_CUDA(cudaMemcpy(dg.nlist, G.col.data(), sizeof(int) * G.m, cudaMemcpyHostToDevice));
  GB_CUDA(cudaMemcpy(dwf, G.w.data(), sizeof(float) * G.m, cudaMemcpyHostToDevice));
  f2i<<<1024, 256>>>(dwf, dg.eweight, G.m);  // integer weights (exact)
  GB_CUDA(cudaDeviceSynchronize());
  double t_h2d = gb::now_s() - ti0;

  cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
  cudaEventRecord(e0);
  init1<<<(int)std::min<int64_t>(((int64_t)upper * upper + ThreadsPerBlock - 1) / ThreadsPerBlock, 1 << 20), ThreadsPerBlock>>>(n, dA, upper);
  init2<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(dg, dA, upper);
  cudaEventRecord(e1);
  FW0_64<<<1, ThreadsPerBlock>>>(dA, upper, dKr, dKc);
  auto round = [&](int x) {
    FWrowcol_64<<<2 * subm1, ThreadsPerBlock>>>(dA, upper, dKr, dKc, x, subm1);
    FWrem_64<<<subm1 * subm1, ThreadsPerBlock>>>(dA, upper, dKr, dKc, x, subm1);
  };
  int nb = sub > 1 ? sub : 0;
  int probe = std::min(nb, force_rounds > 0 ? force_rounds : 2);
  for (int x = 0; x < probe; ++x) round(x);
  cudaEventRecord(e2); cudaEventSynchronize(e2);
  GB_CUDA(cudaGetLastError());
  float ms_init, ms_probe; cudaEventElapsedTime(&ms_init, e0, e1); cudaEventElapsedTime(&ms_probe, e1, e2);
  double per = probe > 0 ? ms_probe / 1e3 / probe : 0;
  int done = probe; bool extrap = false;
  if (force_rounds > 0 && force_rounds < nb) extrap = true;
  else if (per * nb > budget) {
    extrap = true;
    int target = std::min(nb, std::max(probe + 1, (int)ceil(5.0 / per)));
    for (int x = done; x < target; ++x) round(x);
    done = target; if (done == nb) extrap = false;
  } else { for (int x = done; x < nb; ++x) round(x); done = nb; }
  cudaEventRecord(e3); cudaEventSynchronize(e3);
  GB_CUDA(cudaGetLastError());
  float ms_rest; cudaEventElapsedTime(&ms_rest, e2, e3);
  double t_fw_meas = (ms_probe + ms_rest) / 1e3;
  double t_fw = (extrap && done > 0) ? t_fw_meas / done * nb : t_fw_meas;
  double t_inproc_meas = gb::now_s() - ti0;
  double tc = gb::now_s();
  if (!extrap) {
    gb::DevChecksum cs; cs.add<int>(dA, upper, n, n, INT_MAX / 2);
    gb::Checksum c = cs.get(); R.have_cs = true; R.cs_sum = c.sum; R.cs_cnt = c.cnt;
  }
  double t_cs = gb::now_s() - tc;
  double th = gb::now_s();
  if (!extrap) {  // position-sensitive per-row hashes (AdjMat row = source, column = target)
    gb::RowHasher H(R.graph);
    H.hash_rows<int>(dA, upper, n, n, INT_MAX / 2, nullptr);
    gb::RowHashResult hr = H.compare();
    R.extra.push_back({"rows_hashed", std::to_string(hr.rows_hashed)});
    R.extra.push_back({"rowhash_bad", hr.have_ref ? std::to_string(hr.rows_bad) : std::string("null")});
    R.extra.push_back({"rowhash_ok", hr.have_ref ? (hr.rows_bad == 0 ? "true" : "false") : "null"});
  }
  R.phase("rowhash", gb::now_s() - th);
  R.gpu_compute = ms_init / 1e3 + t_fw;
  R.inproc = extrap ? t_h2d + ms_init / 1e3 + t_fw : t_inproc_meas;
  R.extrapolated = extrap;
  R.phase("ctx_warmup", t_ctx); R.phase("read", t_read); R.phase("alloc_h2d", t_h2d);
  R.phase("init (ECL 'gpu init time')", ms_init / 1e3); R.phase("fw (ECL 'gpu comp time')", t_fw); R.phase("checksum", t_cs);
  R.extra.push_back({"npad", std::to_string(upper)});
  R.extra.push_back({"rounds_total", std::to_string(nb)});
  R.extra.push_back({"rounds_run", std::to_string(done)});
  R.extra.push_back({"fw_measured_s", std::to_string(t_fw_meas)});
  R.extra.push_back({"rate_Tupd_s", std::to_string((double)upper * upper * upper / t_fw / 1e12)});
  R.e2e_inproc = gb::now_s() - T0;
  R.print();
  return 0;
}
