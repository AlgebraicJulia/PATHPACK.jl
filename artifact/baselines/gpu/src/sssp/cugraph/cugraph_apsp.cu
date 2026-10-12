// APSP with RAPIDS cuGraph (libcugraph 26.08, conda rapidsai channel), two modes:
//   CUGRAPH_MODE=od   : cugraph::od_shortest_distances(origins = a batch of B consecutive vertices,
//                       destinations = all n vertices), batches of B origins until all n rows are done;
//                       each B x n result block is copied into rows [o, o+B) of the n x n device matrix.
//                       B from CUGRAPH_OD_BATCH (default: as many rows as fit in ~CUGRAPH_OD_BYTES = 2 GiB).
//   CUGRAPH_MODE=sssp : loop of cugraph::sssp(source s) writing distances straight into row s.
// Memory: RMM pool resource on top of cudaMalloc (the RAPIDS-recommended setup; CUGRAPH_POOL=0 disables).
// Graph: create_graph_from_edgelist<int32,int32,store_transposed=false,multi_gpu=false>, renumber=false,
// float weights, vertex list 0..n-1 (keeps isolated vertices).
// Usage: cugraph_apsp graph.mtx      Env: SSSP_BUDGET, SSSP_SAMPLE_TARGET, SSSP_K (see adds_apsp.cu)
#include "mtx.hpp"
#include "checksum.cuh"
#include "dijkstra.hpp"
#include "result.hpp"
#include "rowhash.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <limits>

using vertex_t = int32_t;
using edge_t = int32_t;
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
  std::string mode = getenv("CUGRAPH_MODE") ? getenv("CUGRAPH_MODE") : "od";
  double budget = envd("SSSP_BUDGET", 100), target = envd("SSSP_SAMPLE_TARGET", 40);
  long long forceK = (long long)envd("SSSP_K", 0);
  bool use_pool = envd("CUGRAPH_POOL", 1) != 0;
  gb::Result R; R.code = "cugraph_" + mode; R.graph = gb::basename_noext(path);

  double t_read; gb::Csr H = gb::read_mtx(path, &t_read);
  const int64_t n = H.n, m = H.m;
  R.n = n; R.nnz = H.nnz_file;

  double t = gb::now_s();
  GB_CUDA(cudaSetDevice(0)); GB_CUDA(cudaFree(0));
  double t_ctx = gb::now_s() - t;
  cudaDeviceProp prop; GB_CUDA(cudaGetDeviceProperties(&prop, 0)); R.gpu = prop.name;

  // ---- n x n output first (plain cudaMalloc), then the RMM pool on what is left ----
  t = gb::now_s();
  weight_t* D = nullptr;
  cudaError_t ae = cudaMalloc(&D, (size_t)n * n * sizeof(weight_t));
  if (ae != cudaSuccess) { printf("STATUS oom n*n alloc: %s\n", cudaGetErrorString(ae)); R.status = "oom"; R.print(); return 2; }
  size_t freeb, totb; GB_CUDA(cudaMemGetInfo(&freeb, &totb));
  if (use_pool) {
    size_t init = std::min<size_t>((size_t)(freeb * 0.5), (size_t)8 << 30) & ~(size_t)255;
    size_t maxs = (size_t)(freeb * 0.9) & ~(size_t)255;
    rmm::mr::set_current_device_resource(rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{}, init, maxs});
  }
  GB_CUDA(cudaDeviceSynchronize());
  double t_alloc = gb::now_s() - t;

  // ---- host CSR -> device edge list -> cugraph graph ----
  double t_in0 = gb::now_s();
  raft::handle_t handle;
  auto stream = handle.get_stream();
  std::vector<vertex_t> hs(m), hd(m), hv(n);
  for (int64_t i = 0; i < n; ++i) { hv[i] = (vertex_t)i; for (int64_t e = H.rowptr[i]; e < H.rowptr[i + 1]; ++e) { hs[e] = (vertex_t)i; hd[e] = H.col[e]; } }
  rmm::device_uvector<vertex_t> srcs(m, stream), dsts(m, stream), verts(n, stream);
  rmm::device_uvector<weight_t> wts(m, stream);
  raft::update_device(srcs.data(), hs.data(), m, stream);
  raft::update_device(dsts.data(), hd.data(), m, stream);
  raft::update_device(wts.data(), H.w.data(), m, stream);
  raft::update_device(verts.data(), hv.data(), n, stream);
  handle.sync_stream();
  double t_h2d = gb::now_s() - t_in0;

  // first call of graph creation on a 2-vertex dummy graph: loads the (lazily loaded) graph-construction
  // kernels outside the timed region, like the 1-row od first call below (FAIRNESS.md: first call excluded).
  double t_fc_graph;
  {
    double t0 = gb::now_s();
    rmm::device_uvector<vertex_t> s1(1, stream), d1(1, stream), v1(2, stream);
    rmm::device_uvector<weight_t> w1(1, stream);
    vertex_t hs1 = 0, hd1 = 1, hv1[2] = {0, 1}; weight_t hw1 = 1.f;
    raft::update_device(s1.data(), &hs1, 1, stream); raft::update_device(d1.data(), &hd1, 1, stream);
    raft::update_device(v1.data(), hv1, 2, stream); raft::update_device(w1.data(), &hw1, 1, stream);
    std::vector<cugraph::arithmetic_device_uvector_t> p1; p1.push_back(std::move(w1));
    auto tmp = cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
        handle, std::make_optional(std::move(v1)), std::move(s1), std::move(d1), std::move(p1),
        cugraph::graph_properties_t{false, false}, false);
    handle.sync_stream();
    t_fc_graph = gb::now_s() - t0;
  }
  t = gb::now_s();
  std::vector<cugraph::arithmetic_device_uvector_t> props;
  props.push_back(std::move(wts));
  auto [graph, edge_props, renumber_map] =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
          handle, std::make_optional(std::move(verts)), std::move(srcs), std::move(dsts), std::move(props),
          cugraph::graph_properties_t{false, false}, false);
  auto gv = graph.view();
  auto& wprop = std::get<cugraph::edge_property_t<edge_t, weight_t>>(edge_props[0]);
  auto wv = wprop.view();
  handle.sync_stream();
  double t_graph = gb::now_s() - t;
  if ((int64_t)gv.number_of_vertices() != n) { printf("STATUS fail nv %lld != n %lld\n", (long long)gv.number_of_vertices(), (long long)n); return 1; }

  // ---- unit of work: od batch (B consecutive origins) or one sssp ----
  int64_t B = 1;
  rmm::device_uvector<vertex_t> all_v(n, stream);
  {
    std::vector<vertex_t> tmp(n); for (int64_t i = 0; i < n; ++i) tmp[i] = (vertex_t)i;
    raft::update_device(all_v.data(), tmp.data(), n, stream);
  }
  if (mode == "od") {
    double bytes = envd("CUGRAPH_OD_BYTES", (double)(2ull << 30));
    B = (int64_t)envd("CUGRAPH_OD_BATCH", 0);
    if (B <= 0) B = std::max<int64_t>(1, std::min<int64_t>(n, (int64_t)(bytes / (4.0 * n))));
  }
  // od: a "unit" is a chunk of B consecutive origins; on RMM out_of_memory the chunk size is halved and
  // the chunk retried (kron-type graphs need far more than B*n*4 bytes of od workspace).
  long long n_oom = 0;
  auto run_unit = [&](int64_t o0, int64_t cnt, weight_t* rows) -> bool {
    if (mode == "od") {
      try {
        auto res = cugraph::od_shortest_distances<vertex_t, edge_t, weight_t, false>(
            handle, gv, wv, raft::device_span<vertex_t const>(all_v.data() + o0, cnt),
            raft::device_span<vertex_t const>(all_v.data(), n), std::numeric_limits<weight_t>::max(), false);
        GB_CUDA(cudaMemcpyAsync(rows, res.data(), (size_t)cnt * n * sizeof(weight_t), cudaMemcpyDeviceToDevice, stream));
      } catch (std::bad_alloc& e) {
        handle.sync_stream(); cudaGetLastError();
        printf("OOM with batch %lld: %.160s\n", (long long)cnt, e.what());
        n_oom++;
        return false;
      }
    } else {
      cugraph::sssp<vertex_t, edge_t, weight_t, false>(handle, gv, wv, rows, (vertex_t*)nullptr, (vertex_t)o0,
                                                        std::numeric_limits<weight_t>::max(), false);
    }
    return true;
  };
  // process rows [o0, o0+cnt) with OOM back-off; returns false if even B=1 fails
  auto run_rows = [&](int64_t o0, int64_t cnt) -> bool {
    int64_t o = o0, end = o0 + cnt;
    while (o < end) {
      int64_t c = std::min<int64_t>(B, end - o);
      if (run_unit(o, c, D + (size_t)o * n)) { o += c; continue; }
      if (B == 1) return false;
      B = std::max<int64_t>(1, B / 2);
    }
    return true;
  };

  // ---- calibration / warm-up: od = first chunk (finds a B that fits), sssp = 16 sampled sources ----
  // SSSP_FORCE_FULL=1 skips it (cold, single pass: used for od, which finishes every graph of the suite;
  // the od batch size is then the tuned value from CUGRAPH_OD_BATCH / CUGRAPH_OD_BYTES).
  const bool force_full = envd("SSSP_FORCE_FULL", 0) != 0;
  double est_row = -1;
  t = gb::now_s();
  double t_cal0 = t;
  if (force_full) {
    // first call on a single origin row (loads cuGraph's kernels lazily; reported as "first_call", not timed)
    if (!run_rows(0, 1)) { printf("STATUS oom od batch 1\n"); R.status = "oom"; R.print(); return 2; }
    handle.sync_stream(); GB_CUDA(cudaDeviceSynchronize());
  } else if (mode == "od") {
    if (!run_rows(0, std::min<int64_t>(B, n))) { printf("STATUS oom od batch 1\n"); R.status = "oom"; R.print(); return 2; }
    handle.sync_stream(); GB_CUDA(cudaDeviceSynchronize());
    // time one clean chunk of the final B (the first call above may include OOM retries / warm-up)
    t = gb::now_s();
    int64_t c = std::min<int64_t>(B, n);
    if (!run_rows(0, c)) { printf("STATUS oom\n"); R.status = "oom"; R.print(); return 2; }
    handle.sync_stream(); GB_CUDA(cudaDeviceSynchronize());
    est_row = (gb::now_s() - t) / c;
  } else {
    int64_t W = std::min<int64_t>(n, 16);
    auto wu = gb::sample_sources(n, W, 777);
    for (int64_t q = 0; q < W; ++q) run_unit(wu[q], 1, D + (size_t)wu[q] * n);
    handle.sync_stream(); GB_CUDA(cudaDeviceSynchronize());
    est_row = (gb::now_s() - t) / W;
  }
  double t_calib = gb::now_s() - t_cal0;
  bool full = force_full || ((forceK <= 0) && (est_row * n <= budget));
  int64_t k = full ? n : (forceK > 0 ? std::min<int64_t>(forceK, n) : std::min<int64_t>(n, std::max<int64_t>(64, (int64_t)(target / est_row))));
  if (k >= n) { full = true; k = n; }
  // sampled od: whole chunks of B consecutive origins at spread-out chunk positions
  std::vector<std::pair<int64_t, int64_t>> chunks;  // (start, count)
  if (full) chunks.push_back({0, n});
  else if (mode == "od") {
    int64_t nch = (n + B - 1) / B, want = std::min<int64_t>(nch, std::max<int64_t>(1, (k + B - 1) / B));
    for (auto u : gb::sample_sources(nch, want)) chunks.push_back({u * B, std::min<int64_t>(B, n - u * B)});
  } else {
    for (auto s0 : gb::sample_sources(n, k)) chunks.push_back({s0, 1});
  }
  printf("mode %s batch %lld: calibration %.6f s/row -> %s, chunks=%zu\n", mode.c_str(), (long long)B, est_row,
         full ? "full" : "sampled", chunks.size());

  // ---- timed loop ----
  handle.sync_stream(); GB_CUDA(cudaDeviceSynchronize());
  long long oom_before = n_oom;
  t = gb::now_s();
  int64_t rows_done = 0;
  for (auto& ch : chunks) {
    if (!run_rows(ch.first, ch.second)) { printf("STATUS oom\n"); R.status = "oom"; R.print(); return 2; }
    rows_done += ch.second;
  }
  handle.sync_stream(); GB_CUDA(cudaDeviceSynchronize());
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { printf("STATUS cuda_error %s\n", cudaGetErrorString(le)); return 3; }
  double t_comp = gb::now_s() - t;
  R.extra.push_back({"oom_retries_timed", std::to_string(n_oom - oom_before)});

  t = gb::now_s();
  R.phase("read", t_read); R.phase("ctx", t_ctx); R.phase("alloc_nn_pool", t_alloc); R.phase("edgelist_h2d", t_h2d);
  R.phase("create_graph", t_graph); R.phase("first_call_graph", t_fc_graph); R.phase("calib_per_source", est_row); R.phase("batch_rows", (double)B);
  R.phase(force_full ? "first_call" : "calibration", t_calib);  // harness warm-up/calibration (incl. od OOM search); subtract from process e2e
  const weight_t TH = 1e30f;
  if (full) {
    gb::DevChecksum cs; cs.add<weight_t>(D, n, n, n, TH);
    auto c = cs.get(); R.have_cs = true; R.cs_sum = c.sum; R.cs_cnt = c.cnt;
    gb::RowHasher Hh(R.graph); Hh.hash_rows<weight_t>(D, n, n, n, TH);
    add_rowhash_fields(R, Hh, {});
    R.phase("sssp_all", t_comp);
    R.gpu_compute = t_graph + t_comp;
  } else {
    std::vector<int64_t> sources;
    for (auto& ch : chunks) for (int64_t r = 0; r < ch.second; ++r) sources.push_back(ch.first + r);
    int64_t k = (int64_t)sources.size();
    double* d_rs; unsigned long long* d_rc;
    GB_CUDA(cudaMalloc(&d_rs, k * sizeof(double))); GB_CUDA(cudaMalloc(&d_rc, k * sizeof(unsigned long long)));
    std::vector<double> rs(k); std::vector<unsigned long long> rc(k);
    // rows are at D + s*n (not compacted): one checksum launch per chunk
    int64_t off = 0;
    for (auto& ch : chunks) {
      gb::row_checksum_kernel<weight_t><<<(unsigned)ch.second, 256>>>(D + (size_t)ch.first * n, n, n, TH, d_rs + off, d_rc + off);
      off += ch.second;
    }
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
    { gb::RowHasher Hh(R.graph);
      for (auto& ch : chunks) Hh.hash_rows<weight_t>(D + (size_t)ch.first * n, n, ch.second, n, TH);
      add_rowhash_fields(R, Hh, sources); }
    R.phase("sssp_sample", t_comp); R.phase("cpu_dijkstra_verify", t_dij);
    R.gpu_compute = t_graph + t_comp * (double)n / (double)rows_done;
    if (bad) R.status = "row_mismatch";
  }
  R.phase("checksum", gb::now_s() - t);
  R.inproc = t_alloc + t_h2d + R.gpu_compute;
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
