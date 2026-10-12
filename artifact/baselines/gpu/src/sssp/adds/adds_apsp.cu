// APSP by running ADDS (Wang, Fussell, Lin, PPoPP'21; github.com/UCLA-VAST/adds @ 9ef8166) from every source.
// Built twice: ads_int (default; int32 distances, 32-bit atomicMin) and ads_float (-DADDS_FLOAT;
// {parent,dist} 64-bit CAS). Their kernel.cu is #included unchanged except that its main() is renamed;
// only main()/gg_main() are replaced by the loop below, which repeats their per-source sequence:
//   kernel<<<>>>(init node_data)  ->  cudaMemset(work_count)  ->  wl.reinit()  ->  gg_main_pipe_1_wrapper()
// (their RUN_LOOP structure). The delta heuristic input (profiler_kernel -> ave_wt -> wl.set_param) is a pure
// function of the graph, so it is computed once per graph (inside gpu_compute) instead of once per source.
// Row s of the n x n device matrix receives the distances from source s (int32 for ads_int, float for ads_float).
//
// Usage: adds_apsp_<int|float> graph.mtx
// Env: SSSP_BUDGET (s, default 100): run all n sources if the calibrated estimate fits, else a sample;
//      SSSP_SAMPLE_TARGET (s, default 40): sampled-run target time; SSSP_K: force a k-source sample.
#include "mtx.hpp"
#include "checksum.cuh"
#include "dijkstra.hpp"
#include "result.hpp"
#include "rowhash.cuh"
#include <climits>
#include <cfloat>

#define main adds_orig_main
#include "kernel.cu"
#undef main

#ifdef ADDS_FLOAT
typedef float row_t;
static const row_t ROW_THRESH = 1e30f;
static const char* CODE = "adds_float";
__global__ void extract_row(const node_data_type* __restrict__ nd, float* __restrict__ row, unsigned n) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) row[i] = nd[i].dist;
}
#else
typedef int row_t;
static const row_t ROW_THRESH = INT32_MAX;
static const char* CODE = "adds_int";
#endif


// position-sensitive row-hash verification against <GB_ROWHASH_DIR>/<graph>.rowhash
static void add_rowhash_fields(gb::Result& R, const gb::RowHasher& H, const std::vector<int64_t>& labels) {
  auto rr = H.compare(labels);
  R.extra.push_back({"rows_hashed", std::to_string(rr.rows_hashed)});
  R.extra.push_back({"rowhash_bad", rr.have_ref ? std::to_string(rr.rows_bad) : std::string("null")});
  R.extra.push_back({"rowhash_ok", rr.have_ref ? (rr.rows_bad == 0 ? "true" : "false") : "null"});
  if (rr.have_ref && rr.rows_bad > 0 && R.status == "ok") R.status = "rowhash_mismatch";
}

static double envd(const char* k, double d) { const char* v = getenv(k); return v ? atof(v) : d; }

int main(int argc, char** argv) {
  double T0 = gb::now_s();
  if (argc < 2) { fprintf(stderr, "usage: %s graph.mtx\n", argv[0]); return 1; }
  std::string path = argv[1];
  double budget = envd("SSSP_BUDGET", 100), target = envd("SSSP_SAMPLE_TARGET", 40);
  long long forceK = (long long)envd("SSSP_K", 0);
  gb::Result R; R.code = CODE; R.graph = gb::basename_noext(path);

  double t_read; gb::Csr G = gb::read_mtx(path, &t_read);
  const unsigned n = (unsigned)G.n;
  R.n = G.n; R.nnz = G.nnz_file;

  double t = gb::now_s();
  GB_CUDA(cudaSetDevice(0)); GB_CUDA(cudaFree(0));
  double t_ctx = gb::now_s() - t;
  cudaDeviceProp prop; GB_CUDA(cudaGetDeviceProperties(&prop, 0)); R.gpu = prop.name;

  // ---- host graph in memory -> device graph (their CSRGraph) ----
  double t_in0 = gb::now_s();
  CSRGraphTy hg, gg;
  hg.nnodes = n; hg.nedges = (index_type)G.m;
  hg.allocOnHost();
  for (unsigned i = 0; i <= n; ++i) hg.row_start[i] = (index_type)G.rowptr[i];
  for (int64_t e = 0; e < G.m; ++e) { hg.edge_dst[e] = (index_type)G.col[e]; hg.edge_data[e] = (edge_data_type)G.w[e]; }
  snprintf(hg.file_name, sizeof(hg.file_name), "%s", R.graph.c_str());
  hg.copy_to_gpu(gg);
  GB_CUDA(cudaDeviceSynchronize());
  double t_h2d = gb::now_s() - t_in0;

  t = gb::now_s();
  row_t* D = nullptr;
  cudaError_t ae = cudaMalloc(&D, (size_t)n * n * sizeof(row_t));
  if (ae != cudaSuccess) { printf("STATUS oom n*n alloc: %s\n", cudaGetErrorString(ae)); R.status = "oom"; R.print(); return 2; }
  GB_CUDA(cudaDeviceSynchronize());
  double t_alloc = gb::now_s() - t;

  // ---- their gg_main() preamble (launch config, worklist, delta) ----
  t = gb::now_s();
  int num_threads = TB_SIZE, num_tb_per_sm, max_num_threads, min_grid_size;
  GB_CUDA(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &max_num_threads, sssp_kernel));
  // Their check compares TB_SIZE with cudaOccupancyMaxPotentialBlockSize, i.e. the *occupancy-optimal* block
  // size, which on Turing happened to be >= TB_SIZE; on sm_100/sm_120 it returns 640 for a kernel that can
  // still launch 768/448 threads. We keep their TB_SIZE and check the real launch limit instead.
  cudaFuncAttributes fattr; GB_CUDA(cudaFuncGetAttributes(&fattr, sssp_kernel));
  printf("TB_SIZE %d, occupancy-optimal block %d, kernel max threads/block %d, regs %d\n", num_threads, max_num_threads, fattr.maxThreadsPerBlock, fattr.numRegs);
  if (num_threads > fattr.maxThreadsPerBlock) { printf("STATUS fail max threads %d < %d\n", fattr.maxThreadsPerBlock, num_threads); return 1; }
  GB_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_tb_per_sm, sssp_kernel, num_threads, 0));
  int num_sm = prop.multiProcessorCount;
  int num_tb = num_tb_per_sm * (num_sm - 1);
  int num_warps = num_threads / 32;
  printf("sm count %d, tb per sm %d, tb size %d, total tb %d (MAX_TB %d)\n", num_sm, num_tb_per_sm, num_threads, num_tb, MAX_TB);
  unsigned suggest_size = (unsigned)((float)hg.nedges * WL_SIZE_MUL) + (NUM_BAG * BLOCK_SIZE * 8);
  suggest_size = min(suggest_size, 536870912);
  unsigned* work_count;
  GB_CUDA(cudaMalloc((void**)&work_count, num_tb * num_threads * sizeof(unsigned)));
  worklist wl;
  wl.alloc(num_tb, num_warps, suggest_size, hg.nnodes, hg.nedges);
  int other_num_tb = num_tb_per_sm * num_sm;
  float ave_degree = (float)hg.nedges / (float)hg.nnodes, ave_wt;
  {
    int total_warp = other_num_tb * 1024 / 32;
    int warp_edge_interval = hg.nedges / (total_warp * I_TIME * N_TIME);
    std::vector<float> host_bk_wt(other_num_tb);
    float* bk_wt; GB_CUDA(cudaMalloc((void**)&bk_wt, other_num_tb * sizeof(float)));
    profiler_kernel<<<other_num_tb, 1024>>>(gg, bk_wt, warp_edge_interval);
    GB_CUDA(cudaMemcpy(host_bk_wt.data(), bk_wt, other_num_tb * sizeof(float), cudaMemcpyDeviceToHost));
    ave_wt = 0; for (int i = 0; i < other_num_tb; i++) ave_wt += host_bk_wt[i];
    ave_wt /= other_num_tb;
    cudaFree(bk_wt);
  }
  wl.set_param(ave_wt, ave_degree);
  GB_CUDA(cudaDeviceSynchronize());
  double t_setup = gb::now_s() - t;

  const bool adds_debug = envd("ADDS_DEBUG", 0) != 0;
  long long adds_runs = 0;
  auto run_src = [&](unsigned s, row_t* dst) {
    kernel<<<other_num_tb, 1024>>>(gg, s);
    cudaMemset(work_count, 0, num_tb * num_threads * sizeof(unsigned));
    wl.reinit();
    gg_main_pipe_1_wrapper(hg, gg, s, num_tb, num_threads, wl, work_count);  // ends with cudaDeviceSynchronize
#ifdef ADDS_FLOAT
    extract_row<<<(n + 255) / 256, 256>>>(gg.node_data, dst, n);
#else
    cudaMemcpyAsync(dst, gg.node_data, (size_t)n * sizeof(row_t), cudaMemcpyDeviceToDevice);
#endif
    if (adds_debug) {  // ADDS_DEBUG=1: check for an asynchronous error after every source
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("STATUS cuda_error after source %u (run #%lld): %s\n", s, adds_runs, cudaGetErrorString(e)); fflush(stdout); exit(3); }
    }
    adds_runs++;
  };

  // ---- calibration / warm-up (not timed into the result) ----
  int64_t W = std::min<int64_t>(n, 16);
  auto ws = gb::sample_sources(n, W, 777);
  t = gb::now_s();
  for (int64_t q = 0; q < W; ++q) run_src((unsigned)ws[q], D + (size_t)q * n);
  GB_CUDA(cudaDeviceSynchronize());
  double t_calib = gb::now_s() - t;
  double est = t_calib / W;
  bool full = (forceK <= 0) && (est * n <= budget);
  int64_t k = full ? n : (forceK > 0 ? std::min<int64_t>(forceK, n) : std::min<int64_t>(n, std::max<int64_t>(64, (int64_t)(target / est))));
  if (k >= n) { full = true; k = n; }
  printf("calibration: %.6f s/source over %lld sources -> %s, k=%lld\n", est, (long long)W, full ? "full" : "sampled", (long long)k);
  std::vector<int64_t> sources;
  if (full) { sources.resize(n); for (unsigned i = 0; i < n; ++i) sources[i] = i; }
  else sources = gb::sample_sources(n, k);

  // ---- timed loop ----
  GB_CUDA(cudaDeviceSynchronize());
  t = gb::now_s();
  for (int64_t q = 0; q < k; ++q) run_src((unsigned)sources[q], D + (size_t)(full ? sources[q] : q) * n);
  GB_CUDA(cudaDeviceSynchronize());
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { printf("STATUS cuda_error %s\n", cudaGetErrorString(le)); return 3; }
  double t_comp = gb::now_s() - t;

  // ---- checksum / verification ----
  t = gb::now_s();
  R.phase("read", t_read); R.phase("ctx", t_ctx); R.phase("graph_h2d", t_h2d); R.phase("alloc_nn", t_alloc);
  R.phase("setup", t_setup); R.phase("calib_per_source", est);
  R.phase("calibration", t_calib);  // harness warm-up/calibration (subtract from process e2e)
  if (full) {
    gb::DevChecksum cs; cs.add<row_t>(D, n, n, n, ROW_THRESH);
    auto c = cs.get(); R.have_cs = true; R.cs_sum = c.sum; R.cs_cnt = c.cnt;
    gb::RowHasher H(R.graph); H.hash_rows<row_t>(D, n, n, n, ROW_THRESH);
    add_rowhash_fields(R, H, {});
    R.phase("sssp_all", t_comp);
    R.gpu_compute = t_setup + t_comp;
  } else {
    double* d_rs; unsigned long long* d_rc;
    GB_CUDA(cudaMalloc(&d_rs, k * sizeof(double))); GB_CUDA(cudaMalloc(&d_rc, k * sizeof(unsigned long long)));
    gb::row_checksum_kernel<row_t><<<(unsigned)k, 256>>>(D, n, n, ROW_THRESH, d_rs, d_rc);
    std::vector<double> rs(k); std::vector<unsigned long long> rc(k);
    GB_CUDA(cudaMemcpy(rs.data(), d_rs, k * sizeof(double), cudaMemcpyDeviceToHost));
    GB_CUDA(cudaMemcpy(rc.data(), d_rc, k * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    double tc = gb::now_s();
    auto ref = gb::dijkstra_rows(G, sources);
    double t_dij = gb::now_s() - tc;
    long long bad = 0;
    for (int64_t q = 0; q < k; ++q) if (rs[q] != ref[q].sum || rc[q] != ref[q].cnt) {
      if (bad < 5) printf("ROW MISMATCH src %lld: gpu (%.17g,%llu) cpu (%.17g,%llu)\n", (long long)sources[q], rs[q], rc[q], ref[q].sum, ref[q].cnt);
      bad++;
    }
    R.rows_checked = k; R.rows_bad = bad; R.extrapolated = true; R.sources_run = k;
    { gb::RowHasher H(R.graph); H.hash_rows<row_t>(D, n, k, n, ROW_THRESH); add_rowhash_fields(R, H, sources); }
    R.phase("sssp_sample", t_comp); R.phase("cpu_dijkstra_verify", t_dij);
    R.gpu_compute = t_setup + t_comp * (double)n / (double)k;
    if (bad) R.status = "row_mismatch";
  }
  double t_cs = gb::now_s() - t;
  R.phase("checksum", t_cs);
  R.inproc = t_h2d + t_alloc + R.gpu_compute;
  R.e2e_inproc = gb::now_s() - T0;
  if (!full) {  // process time of a sampled run is not an all-pairs e2e: report an extrapolated one instead
    R.e2e_inproc = -1;
    R.extra.push_back({"e2e_extrap", std::to_string(t_read + t_ctx + R.inproc)});
  }
  // canonical (FAIRNESS.md): host CSR in memory -> all distances on the device; includes format conversion,
  // graph build, all allocations (n x n output, workspaces), transfers and kernels; excludes process start,
  // file read, CUDA context creation, and the harness warm-up/calibration. Extrapolated to n for sampled runs.
  R.extra.push_back({"canonical", std::to_string(R.inproc)});
  R.extra.push_back({"sources_total", std::to_string(n)});
  R.print();
  wl.free(); cudaFree(work_count); cudaFree(D);
  return 0;
}
