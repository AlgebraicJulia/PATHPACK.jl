// Multi-GPU driver for FastAPSP = app/MoreTaskMpiExample.cpp (one MPI rank per GPU, K subgraph
// tasks shuffled and split across ranks by fapGraphMoreTaskMPI::loadBalance) plus timing/checksum.
// Differences from the example (no algorithmic change):
//  * rank r uses GPU (local_rank % ndev): the example relies on an external per-rank GPU binding
//    (srun --gpu-bind in app/scripts/run.sh);
//  * after each solveSubGraph() the rank checksums that task's rows (sub_vertexs x n, host) before the
//    next task overwrites them (the example keeps only the last task's rows), and the checksums are
//    reduced to rank 0 (excluded from timings);
//  * timers: inproc = METIS (rank 0) + precondition/bcast + max over ranks of the rank's task time
//    (each rank's per-task checksum time excluded).
// Usage: srun ... fap_mpi_driver -f <name> -k <K> -direct false -weight true -code fastapsp_2gpu
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include "fap/fap.h"
#include "fap/fap_more_task_mpi.h"
#include "fap/fap_timers.h"
#include "../common/result.hpp"
#include "host_rowhash.hpp"
#include "fap_conv.hpp"

int main(int argc, char** argv) {
  double t_start = fap::fap_now();
  MPI_Init(&argc, &argv);
  int P, me; MPI_Comm_size(MPI_COMM_WORLD, &P); MPI_Comm_rank(MPI_COMM_WORLD, &me);
  std::string file, code = "fastapsp_mpi", mtx; int K = 0; bool directed = false, weighted = true;
  for (int i = 1; i + 1 < argc; i++) {
    if (!strcmp(argv[i], "-f")) file = argv[i + 1];
    else if (!strcmp(argv[i], "-k")) K = std::stoi(argv[i + 1]);
    else if (!strcmp(argv[i], "-direct")) directed = !strcmp(argv[i + 1], "true");
    else if (!strcmp(argv[i], "-weight")) weighted = !strcmp(argv[i + 1], "true");
    else if (!strcmp(argv[i], "-code")) code = argv[i + 1];
    else if (!strcmp(argv[i], "-mtx")) mtx = argv[i + 1];
  }
  int lr = 0; const char* e = getenv("OMPI_COMM_WORLD_LOCAL_RANK"); if (!e) e = getenv("SLURM_LOCALID");
  if (e) lr = atoi(e);
  int ndev = 1; cudaGetDeviceCount(&ndev);
  double t0 = fap::fap_now();
  cudaSetDevice(lr % ndev); cudaFree(0);
  double t_ctx = fap::fap_now() - t0;
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, lr % ndev);

  // rank 0: our CSR (file read excluded) -> timed conversion to FastAPSP's CSR (see fap_conv.hpp)
  double t_conv = -1; bool conv_ok = false; fap_conv::FapCsr F;
  if (me == 0 && !mtx.empty()) {
    double tr; gb::Csr C = gb::read_mtx(mtx, &tr);
    double tc0 = fap::fap_now(); F = fap_conv::convert(C); t_conv = fap::fap_now() - tc0;
  }
  fap::fapGraphMoreTaskMPI G(file, directed, weighted, K);   // every rank: read + METIS (as in the example)
  if (me == 0 && !mtx.empty()) { conv_ok = fap_conv::identical(F, G); fap_conv::FapCsr tmp; std::swap(F, tmp); }
  G.init(P, K);
  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = fap::fap_now();
  if (me == 0) { G.preCondition(); G.loadBalance(P); }
  G.Meta_MPI_Bcast(MPI_COMM_WORLD);
  double t_bc = fap::fap_now() - t1;
  auto tasks = G.getCurrentTask(me);
  auto ref = fap_rh::load_ref(file);
  long long rh_rows = 0, rh_bad = 0;
  double cs_s = 0, t_cs = 0; unsigned long long cs_c = 0; long long rows = 0;
  double ts = fap::fap_now();
  for (size_t i = 0; i < tasks.size(); i++) {
    G.solveSubGraph(tasks[i], true);
    double tc = fap::fap_now();
    auto D = G.getSubGraphDistance(tasks[i]);   // copy of this task's rows (sub_vertexs x n)
    rows += (long long)(D.size() / G.num_vertexs);
#pragma omp parallel for reduction(+ : cs_s, cs_c) schedule(static)
    for (int64_t q = 0; q < (int64_t)D.size(); ++q) { float v = D[q]; if (v >= 5e7f || v != v) cs_c++; else cs_s += (double)v; }
    if (!ref.empty()) {   // per-row hashes; row r of this task = original vertex getSubGraphIndex(task)[r]
      auto lab = G.getSubGraphIndex(tasks[i]);
      int64_t nr = (int64_t)(D.size() / G.num_vertexs);
      rh_bad += fap_rh::check_rows(D.data(), nr, G.num_vertexs, lab.data(), ref);
      rh_rows += nr;
    }
    t_cs += fap::fap_now() - tc;
  }
  double t_solve_rank = fap::fap_now() - ts - t_cs;
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  auto& T = fap::fap_timers();
  double gpu_rank = T.sssp + T.floyd + T.minplus;
  double S = 0, mx_solve = 0, mx_gpu = 0, mn_solve = 0; unsigned long long C = 0; long long R = 0;
  MPI_Reduce(&cs_s, &S, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&cs_c, &C, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rows, &R, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  long long RH = 0, RB = 0;
  MPI_Reduce(&rh_rows, &RH, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rh_bad, &RB, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_solve_rank, &mx_solve, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_solve_rank, &mn_solve, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&gpu_rank, &mx_gpu, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (me == 0) {
    double t_all = t_bc + mx_solve;          // precondition/bcast + slowest rank's tasks (checksums excluded)
    gb::Result Rr; Rr.code = code; Rr.graph = file; Rr.gpu = prop.name;
    Rr.n = G.num_vertexs; Rr.nnz = G.num_edges;
    Rr.gpu_compute = mx_gpu;                 // max over ranks of GPU-stage wall times (incl. per-call transfers)
    Rr.inproc = T.metis + t_all;             // METIS + precondition/bcast + all ranks' tasks
    Rr.extra.push_back({"canonical", t_conv >= 0 ? std::to_string(t_conv + T.metis + t_all) : std::string("null")});
    Rr.extra.push_back({"conv_identical", t_conv >= 0 ? (conv_ok ? "true" : "false") : std::string("null")});
    Rr.extra.push_back({"output", "\"host, distributed over the ranks (each rank keeps only its current task's rows), FastAPSP row order\""});
    Rr.phase("conv", t_conv);
    Rr.e2e_inproc = fap::fap_now() - t_start - t_cs;
    Rr.phase("ctx", t_ctx); Rr.phase("read", T.read); Rr.phase("metis", T.metis);
    Rr.phase("precondition_bcast", t_bc); Rr.phase("tasks_wall_all_ranks", t_all - t_bc);
    Rr.phase("solve_rank_max", mx_solve); Rr.phase("solve_rank_min", mn_solve);
    Rr.phase("sssp_gpu_rank0", T.sssp); Rr.phase("floyd_gpu_rank0", T.floyd); Rr.phase("minplus_gpu_rank0", T.minplus);
    Rr.have_cs = true; Rr.cs_sum = S; Rr.cs_cnt = C;
    Rr.extra.push_back({"rows_hashed", std::to_string(RH)});
    Rr.extra.push_back({"rowhash_bad", ref.empty() ? std::string("null") : std::to_string(RB)});
    Rr.extra.push_back({"rowhash_ok", ref.empty() ? std::string("null") : ((RB == 0 && RH == G.num_vertexs) ? "true" : "false")});
    Rr.extra.push_back({"K", std::to_string(K)});
    Rr.extra.push_back({"ranks", std::to_string(P)});
    Rr.extra.push_back({"gpus_visible", std::to_string(ndev)});
    Rr.extra.push_back({"rows_total", std::to_string(R)});
    Rr.note = "MPI multi-GPU mode (MoreTaskMpiExample), one rank per GPU; rows stay distributed on the ranks' hosts";
    if (R != G.num_vertexs) Rr.status = "rows_mismatch";
    Rr.print();
  }
  MPI_Finalize();
  return 0;
}
