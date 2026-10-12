// Convert our MTX (general, both directions stored, dups -> min, self loops dropped; gb::read_mtx)
// into FastAPSP's input: graph/weight-undirected/<name>.mtx with each undirected edge once
// (FastAPSP's "-direct false -weight true" reader adds both directions itself).
// If the graph is not symmetric (pattern + weights), writes graph/weight-directed/<name>.mtx with
// all directed edges instead and prints DIRECTED (the driver must then be run with -direct true).
// Usage: mtx2fap in.mtx outdir/graph name
#include "../common/mtx.hpp"
#include <cstdio>
#include <algorithm>
#include <sys/stat.h>

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: mtx2fap in.mtx graphdir name\n"); return 1; }
  double tr; gb::Csr G = gb::read_mtx(argv[1], &tr);
  std::string gdir = argv[2], name = argv[3];
  bool sym = true;
  for (int64_t i = 0; i < G.n && sym; ++i)
    for (int64_t e = G.rowptr[i]; e < G.rowptr[i + 1]; ++e) {
      int32_t j = G.col[e];
      auto b = G.col.begin() + G.rowptr[j], en = G.col.begin() + G.rowptr[j + 1];
      auto it = std::lower_bound(b, en, (int32_t)i);
      if (it == en || *it != i || G.w[it - G.col.begin()] != G.w[e]) { sym = false; break; }
    }
  std::string sub = gdir + (sym ? "/weight-undirected" : "/weight-directed");
  mkdir(gdir.c_str(), 0775); mkdir(sub.c_str(), 0775);
  std::string out = sub + "/" + name + ".mtx";
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { perror(out.c_str()); return 2; }
  int64_t cnt = 0;
  for (int64_t i = 0; i < G.n; ++i)
    for (int64_t e = G.rowptr[i]; e < G.rowptr[i + 1]; ++e) if (!sym || G.col[e] > i) cnt++;
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n%lld %lld %lld\n",
          (long long)G.n, (long long)G.n, (long long)cnt);
  for (int64_t i = 0; i < G.n; ++i)
    for (int64_t e = G.rowptr[i]; e < G.rowptr[i + 1]; ++e)
      if (!sym || G.col[e] > i) fprintf(f, "%lld %d %.9g\n", (long long)(i + 1), G.col[e] + 1, (double)G.w[e]);
  fclose(f);
  printf("%s n=%lld m_directed=%lld written=%lld %s -> %s\n", name.c_str(), (long long)G.n,
         (long long)G.m, (long long)cnt, sym ? "UNDIRECTED" : "DIRECTED", out.c_str());
  return 0;
}
