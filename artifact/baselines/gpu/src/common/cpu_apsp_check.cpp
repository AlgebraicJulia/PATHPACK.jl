// CPU reference: full APSP checksum via Dijkstra from every source (or the first K sources).
// Usage: cpu_apsp_check file.mtx [K]
#include "mtx.hpp"
#include "dijkstra.hpp"
#include <cstdio>
int main(int argc, char** argv) {
  double tr; gb::Csr G = gb::read_mtx(argv[1], &tr);
  int64_t K = argc > 2 ? atoll(argv[2]) : G.n;
  std::vector<int64_t> src; for (int64_t i = 0; i < K; ++i) src.push_back(i);
  double t0 = gb::now_s();
  auto rows = gb::dijkstra_rows(G, src);
  double s = 0; unsigned long long c = 0;
  for (auto& r : rows) { s += r.sum; c += r.cnt; }
  printf("n=%lld m=%lld nnz_file=%lld read=%.3fs dijkstra=%.3fs rows=%lld sum=%.17g cnt=%llu\n",
         (long long)G.n, (long long)G.m, (long long)G.nnz_file, tr, gb::now_s() - t0, (long long)K, s, c);
}
