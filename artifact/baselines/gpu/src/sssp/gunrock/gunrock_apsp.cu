// APSP by running Gunrock's SSSP (github.com/gunrock/gunrock main = "essentials" API) from every source.
// Gunrock has no batched / multi-source SSSP (sssp.hxx is single-source; only BC/PPR batch sources).
// Per source we call the public API gunrock::sssp::run(G, param, result, context) with
// result.distances = row s of the n x n device matrix, exactly as examples/algorithms/sssp/sssp.cu does in
// its multi-source loop. (Re-using one problem/enactor across sources is NOT safe in this Gunrock version:
// vector_frontier_t::push_back appends after the storage's current size while enact() only resets
// num_elements, so the 2nd+ run starts from a stale frontier -- we hit this and dropped that path.)
// Default options (advance load balance = block_mapped, bypass filter, no uniquify), float weights.
//
// Usage: gunrock_apsp graph.mtx      Env: SSSP_BUDGET, SSSP_SAMPLE_TARGET, SSSP_K (see adds_apsp.cu)
#include "mtx.hpp"
#include "checksum.cuh"
#include "dijkstra.hpp"
#include "result.hpp"
#include "rowhash.cuh"
#include <gunrock/algorithms/sssp.hxx>

using namespace gunrock;
using namespace memory;
using vertex_t = int;
using edge_t = int;
using weight_t = float;


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
  gb::Result R; R.code = "gunrock"; R.graph = gb::basename_noext(path);

  double t_read; gb::Csr H = gb::read_mtx(path, &t_read);
  const int64_t n = H.n;
  R.n = H.n; R.nnz = H.nnz_file;

  double t = gb::now_s();
  GB_CUDA(cudaSetDevice(0)); GB_CUDA(cudaFree(0));
  double t_ctx = gb::now_s() - t;
  cudaDeviceProp prop; GB_CUDA(cudaGetDeviceProperties(&prop, 0)); R.gpu = prop.name;

  // ---- host CSR -> device graph ----
  double t_in0 = gb::now_s();
  using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.number_of_rows = (vertex_t)n; csr.number_of_columns = (vertex_t)n; csr.number_of_nonzeros = (edge_t)H.m;
  {
    std::vector<edge_t> rp(n + 1); for (int64_t i = 0; i <= n; ++i) rp[i] = (edge_t)H.rowptr[i];
    csr.row_offsets = rp;
    csr.column_indices = H.col;
    csr.nonzero_values = H.w;
  }
  graph::graph_properties_t properties; properties.directed = true; properties.symmetric = false; properties.weighted = true;
  auto G = graph::build<memory_space_t::device>(properties, csr);
  auto context = std::make_shared<gcuda::multi_context_t>(0);
  GB_CUDA(cudaDeviceSynchronize());
  double t_h2d = gb::now_s() - t_in0;

  t = gb::now_s();
  weight_t* D = nullptr;
  cudaError_t ae = cudaMalloc(&D, (size_t)n * n * sizeof(weight_t));
  if (ae != cudaSuccess) { printf("STATUS oom n*n alloc: %s\n", cudaGetErrorString(ae)); R.status = "oom"; R.print(); return 2; }
  GB_CUDA(cudaDeviceSynchronize());
  double t_alloc = gb::now_s() - t;

  // ---- per-run state ----
  t = gb::now_s();
  thrust::device_vector<vertex_t> predecessors(n);
  gunrock::options_t options;  // library defaults
  GB_CUDA(cudaDeviceSynchronize());
  double t_setup = gb::now_s() - t;

  double gr_ms = 0;  // gunrock's own enact() timer, summed
  auto run_src = [&](int64_t s, weight_t* row) {
    sssp::param_t<vertex_t> param((vertex_t)s, options);
    sssp::result_t<vertex_t, weight_t> result(row, predecessors.data().get(), (vertex_t)n);
    gr_ms += sssp::run(G, param, result, context);
  };

  // ---- calibration / warm-up ----
  int64_t W = std::min<int64_t>(n, 16);
  auto ws = gb::sample_sources(n, W, 777);
  t = gb::now_s();
  for (int64_t q = 0; q < W; ++q) run_src(ws[q], D + (size_t)q * n);
  GB_CUDA(cudaDeviceSynchronize());
  double t_calib = gb::now_s() - t;
  double est = t_calib / W;
  bool full = (forceK <= 0) && (est * n <= budget);
  int64_t k = full ? n : (forceK > 0 ? std::min<int64_t>(forceK, n) : std::min<int64_t>(n, std::max<int64_t>(64, (int64_t)(target / est))));
  if (k >= n) { full = true; k = n; }
  printf("calibration: %.6f s/source over %lld sources -> %s, k=%lld\n", est, (long long)W, full ? "full" : "sampled", (long long)k);
  std::vector<int64_t> sources;
  if (full) { sources.resize(n); for (int64_t i = 0; i < n; ++i) sources[i] = i; }
  else sources = gb::sample_sources(n, k);

  // ---- timed loop ----
  gr_ms = 0;
  GB_CUDA(cudaDeviceSynchronize());
  t = gb::now_s();
  for (int64_t q = 0; q < k; ++q) run_src(sources[q], D + (size_t)(full ? sources[q] : q) * n);
  GB_CUDA(cudaDeviceSynchronize());
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { printf("STATUS cuda_error %s\n", cudaGetErrorString(le)); return 3; }
  double t_comp = gb::now_s() - t;

  t = gb::now_s();
  R.phase("read", t_read); R.phase("ctx", t_ctx); R.phase("graph_h2d", t_h2d); R.phase("alloc_nn", t_alloc);
  R.phase("setup", t_setup); R.phase("calib_per_source", est);
  R.phase("calibration", t_calib);  // harness warm-up/calibration (subtract from process e2e)
  R.phase("gunrock_enact_sum", gr_ms / 1000.0);
  const weight_t TH = 1e30f;
  if (full) {
    gb::DevChecksum cs; cs.add<weight_t>(D, n, n, n, TH);
    auto c = cs.get(); R.have_cs = true; R.cs_sum = c.sum; R.cs_cnt = c.cnt;
    gb::RowHasher Hh(R.graph); Hh.hash_rows<weight_t>(D, n, n, n, TH);
    add_rowhash_fields(R, Hh, {});
    R.phase("sssp_all", t_comp);
    R.gpu_compute = t_setup + t_comp;
  } else {
    double* d_rs; unsigned long long* d_rc;
    GB_CUDA(cudaMalloc(&d_rs, k * sizeof(double))); GB_CUDA(cudaMalloc(&d_rc, k * sizeof(unsigned long long)));
    gb::row_checksum_kernel<weight_t><<<(unsigned)k, 256>>>(D, n, n, TH, d_rs, d_rc);
    std::vector<double> rs(k); std::vector<unsigned long long> rc(k);
    GB_CUDA(cudaMemcpy(rs.data(), d_rs, k * sizeof(double), cudaMemcpyDeviceToHost));
    GB_CUDA(cudaMemcpy(rc.data(), d_rc, k * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    double tc = gb::now_s();
    auto ref = gb::dijkstra_rows(H, sources);
    double t_dij = gb::now_s() - tc;
    long long bad = 0;
    for (int64_t q = 0; q < k; ++q) if (rs[q] != ref[q].sum || rc[q] != ref[q].cnt) {
      if (bad < 5) printf("ROW MISMATCH src %lld: gpu (%.17g,%llu) cpu (%.17g,%llu)\n", (long long)sources[q], rs[q], rc[q], ref[q].sum, ref[q].cnt);
      bad++;
    }
    R.rows_checked = k; R.rows_bad = bad; R.extrapolated = true; R.sources_run = k;
    { gb::RowHasher Hh(R.graph); Hh.hash_rows<weight_t>(D, n, k, n, TH); add_rowhash_fields(R, Hh, sources); }
    R.phase("sssp_sample", t_comp); R.phase("cpu_dijkstra_verify", t_dij);
    R.gpu_compute = t_setup + t_comp * (double)n / (double)k;
    if (bad) R.status = "row_mismatch";
  }
  R.phase("checksum", gb::now_s() - t);
  R.inproc = t_h2d + t_alloc + R.gpu_compute;
  R.e2e_inproc = gb::now_s() - T0;
  if (!full) { R.e2e_inproc = -1; R.extra.push_back({"e2e_extrap", std::to_string(t_read + t_ctx + R.inproc)}); }
  // canonical (FAIRNESS.md): host CSR in memory -> all distances on the device; includes format conversion,
  // graph build, all allocations (n x n output, workspaces), transfers and kernels; excludes process start,
  // file read, CUDA context creation, and the harness warm-up/calibration. Extrapolated to n for sampled runs.
  R.extra.push_back({"canonical", std::to_string(R.inproc)});
  R.extra.push_back({"sources_total", std::to_string(n)});
  R.print();
  cudaFree(D);
  return 0;
}
