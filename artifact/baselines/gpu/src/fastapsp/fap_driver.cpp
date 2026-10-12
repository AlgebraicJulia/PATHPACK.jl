// Benchmark driver for FastAPSP (ICS'23).  Same calls as app/singleNodeExample.cpp, except that it
// always calls G.solve() (all K subgraphs -> full n x n distance matrix on the host); the example
// only solves subgraphs 1..3 when n > 20000 ("split" demo mode).
// Run from a directory containing graph/weight-undirected/<name>.mtx (FastAPSP's own reader).
// Usage: fap_driver -f <name> -k <K> [-direct false] [-weight true] [-code fastapsp] [-mtx our.mtx]
//
// Times (FAIRNESS.md):
//   canonical = conv + metis + precondition + solve: our CSR in host memory -> every distance resident
//               in host memory (FastAPSP's output place), in FastAPSP's row order (rows grouped by METIS
//               part, boundary vertices first; columns in original order).  conv = fap_conv.hpp (needs -mtx).
//   canonical_relabeled = canonical + relabel (parallel host row gather into original order, 16 threads,
//               fresh n x n buffer).  FastAPSP has no relabel of its own; this one is ours.
//   gpu_compute = sum of FastAPSP's GPU stage wall times (SSSP, FW, min-plus), incl. their per-call
//               cudaMalloc / H2D / D2H (FastAPSP interleaves host work; there is no device-only span).
//   Excluded: process start, file read (FastAPSP's reader), CUDA context (created up front here).
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "fap/fap.h"
#include "fap/fap_timers.h"
#include "../common/result.hpp"
#include "host_rowhash.hpp"
#include "fap_conv.hpp"

int main(int argc, char** argv) {
  double t_start = fap::fap_now();
  std::string file, code = "fastapsp", mtx; int K = 0; bool directed = false, weighted = true;
  for (int i = 1; i + 1 < argc; i++) {
    if (!strcmp(argv[i], "-f")) file = argv[i + 1];
    else if (!strcmp(argv[i], "-k")) K = std::stoi(argv[i + 1]);
    else if (!strcmp(argv[i], "-direct")) directed = !strcmp(argv[i + 1], "true");
    else if (!strcmp(argv[i], "-weight")) weighted = !strcmp(argv[i + 1], "true");
    else if (!strcmp(argv[i], "-code")) code = argv[i + 1];
    else if (!strcmp(argv[i], "-mtx")) mtx = argv[i + 1];
  }
  gb::Result R; R.code = code; R.graph = file;
  // CUDA context up front (FastAPSP would otherwise create it inside its first SSSP call).
  double t0 = fap::fap_now();
  cudaFree(0);
  double t_ctx = fap::fap_now() - t0;
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); R.gpu = prop.name;

  // our CSR in host memory (file read: excluded), then the timed conversion to FastAPSP's CSR
  double t_ourread = 0, t_conv = -1; fap_conv::FapCsr F; bool have_conv = !mtx.empty();
  if (have_conv) {
    gb::Csr C = gb::read_mtx(mtx, &t_ourread);
    double tc0 = fap::fap_now();
    F = fap_conv::convert(C);
    t_conv = fap::fap_now() - tc0;
  }

  fap::fapGraph G(file, directed, weighted, K);   // reads file (read) + METIS (metis)
  bool conv_ok = have_conv && fap_conv::identical(F, G);
  { fap_conv::FapCsr tmp; std::swap(F, tmp); }   // free before the run
  G.preCondition();                                // boundary detection + reordering (pre)
  G.solve();                                       // stages 1-3 for all K subgraphs (solve)
  cudaDeviceSynchronize();
  auto& T = fap::fap_timers();
  const int64_t n = G.num_vertexs;
  const float* D = G.distData();
  auto st2ed = G.getMapping();                     // row R of dist = original vertex st2ed[R]

  // relabel to original row order (timed separately; not part of FastAPSP)
  double tr0 = fap::fap_now();
  std::unique_ptr<float[]> Do(new float[(size_t)n * n]);
#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < n; ++r) memcpy(Do.get() + (size_t)st2ed[r] * n, D + (size_t)r * n, (size_t)n * sizeof(float));
  double t_relabel = fap::fap_now() - tr0;

  // verification on the relabeled matrix: checksum + position-sensitive per-row hashes (identity labels)
  double tc = fap::fap_now();
  double s = 0; unsigned long long c = 0;
  const float* Q = Do.get(); const int64_t N = n * n;
#pragma omp parallel for reduction(+ : s, c) schedule(static)
  for (int64_t i = 0; i < N; ++i) { float v = Q[i]; if (v >= 5e7f || v != v) c++; else s += (double)v; }
  auto ref = fap_rh::load_ref(file);
  std::vector<int32_t> ident(n); std::iota(ident.begin(), ident.end(), 0);
  long long rh_bad = ref.empty() ? -1 : fap_rh::check_rows(Q, n, n, ident.data(), ref);
  double t_verify = fap::fap_now() - tc;
  Do.reset();

  R.n = n; R.nnz = G.num_edges;
  R.gpu_compute = T.sssp + T.floyd + T.minplus;
  R.inproc = T.metis + T.pre + T.solve;            // FastAPSP CSR -> full host result (no conversion)
  R.e2e_inproc = fap::fap_now() - t_start - t_verify;
  double canonical = (have_conv ? t_conv : 0) + T.metis + T.pre + T.solve;
  R.extra.push_back({"canonical", have_conv ? std::to_string(canonical) : std::string("null")});
  R.extra.push_back({"canonical_relabeled", have_conv ? std::to_string(canonical + t_relabel) : std::string("null")});
  R.extra.push_back({"output", "\"host, FastAPSP row order (METIS parts, boundary first), original column order\""});
  R.phase("ctx", t_ctx); R.phase("read_fastapsp", T.read); R.phase("read_ours", t_ourread); R.phase("conv", t_conv);
  R.phase("metis", T.metis); R.phase("precondition", T.pre);
  R.phase("solve", T.solve); R.phase("sssp_gpu", T.sssp); R.phase("submat_build_host", T.build);
  R.phase("floyd_gpu", T.floyd); R.phase("mat1_host", T.mat1); R.phase("minplus_gpu", T.minplus);
  R.phase("decode_host", T.decode); R.phase("relabel_host", t_relabel); R.phase("verify_host", t_verify);
  R.have_cs = true; R.cs_sum = s; R.cs_cnt = c;
  R.extra.push_back({"conv_identical", have_conv ? (conv_ok ? "true" : "false") : std::string("null")});
  R.extra.push_back({"rows_hashed", std::to_string(ref.empty() ? 0 : (long long)n)});
  R.extra.push_back({"rowhash_bad", ref.empty() ? std::string("null") : std::to_string(rh_bad)});
  R.extra.push_back({"rowhash_ok", ref.empty() ? std::string("null") : (rh_bad == 0 ? "true" : "false")});
  R.extra.push_back({"K", std::to_string(K)});
  R.extra.push_back({"boundary_vertices", std::to_string(T.bdy_total)});
  R.extra.push_back({"max_subgraph", std::to_string(T.max_sub)});
  R.note = "result on host (n x n dist + n x n path, path always computed); gpu_compute = sum of GPU stage wall times incl. per-call transfers";
  R.print();
  return 0;
}
