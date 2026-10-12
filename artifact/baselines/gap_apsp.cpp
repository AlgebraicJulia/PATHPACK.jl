// All-pairs baseline: GAP's Delta-stepping kernel (DeltaStep from GAP's sssp.cc, unmodified) from every source, each
// result copied into its row of an n x n output that is allocated and first-touched before the timer, exactly like the
// output of our closure and of our Dijkstra baseline. The n runs are spread over THREADS threads, one run per thread at a
// time (each run single-threaded). Usage: gap_apsp GRAPH.wel DELTA THREADS [DUMP]   (DUMP: row 0, for verification)
#define main gap_sssp_main
#include "sssp.cc"
#undef main
#include <omp.h>
#include <algorithm>
#include <cstdio>

int main(int argc, char* argv[]) {
  if (argc < 4) { fprintf(stderr, "usage: gap_apsp GRAPH.wel DELTA THREADS [DUMP]\n"); return 1; }
  const WeightT delta = atof(argv[2]); const int nt = atoi(argv[3]);
  char a0[] = "gap_apsp", a1[] = "-f"; char* fargv[] = {a0, a1, argv[1], nullptr};
  CLBase cli(3, fargv, "gap_apsp"); if (!cli.ParseArgs()) return 1;
  WeightedBuilder b(cli); WGraph g = b.MakeGraph();
  const int64_t n = g.num_nodes();
  WeightT* out = new WeightT[size_t(n) * size_t(n)];                 // the n x n output
  #pragma omp parallel for schedule(static) num_threads(nt)
  for (int64_t s = 0; s < n; s++) std::fill(out + size_t(s) * n, out + size_t(s + 1) * n, WeightT(0));   // first touch
  omp_set_max_active_levels(1);                                      // each run single-threaded
  { pvector<WeightT> w = DeltaStep(g, 0, delta, false); }           // warm-up
  Timer t; t.Start();
  #pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
  for (int64_t s = 0; s < n; s++) {
    pvector<WeightT> d = DeltaStep(g, NodeID(s), delta, false);
    std::copy(d.begin(), d.end(), out + size_t(s) * n);
  }
  t.Stop();
  if (argc > 4) { FILE* f = fopen(argv[4], "w"); for (int64_t i = 0; i < n; i++) fprintf(f, "%.17g\n", out[i] >= kDistInf ? 1.0 / 0.0 : out[i]); fclose(f); }
  printf("gap_apsp: n=%ld threads=%d delta=%g total_s=%.6f\n", long(n), nt, double(delta), t.Seconds());
  delete[] out; return 0;
}
